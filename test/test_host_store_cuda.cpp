#include <gtest/gtest.h>
#include <cuda_runtime_api.h>
#include <atomic>
#include <chrono>
#include <thread>
#include <algorithm>
#include <numeric>
#include "cache/transfer_plan.h"
#include "base/block_allocator.h"
#include "cache/host_store.h"

namespace {
struct Gate { std::atomic<bool> entered{false}, release{false}; };
void CUDART_CB WaitForGate(void* data) {
  auto* gate = static_cast<Gate*>(data);
  gate->entered.store(true);
  while (!gate->release.load()) std::this_thread::yield();
}
}
TEST(HostStoreCudaTest, CancellationDoesNotFreeInFlightSourceOrPinnedDestination) {
  int devices=0;
  if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) GTEST_SKIP();
  base::BlockAllocator pool(1,4,1,8,base::DataType::kDataTypeFp32,base::DeviceType::kDeviceCUDA);
  const auto block=pool.allocate();
  auto payload=pool.get_block_payload_ptrs(block);
  ASSERT_EQ(cudaMemset(payload.key,0x31,payload.key_value_bytes),cudaSuccess);
  ASSERT_EQ(cudaMemset(payload.value,0x72,payload.key_value_bytes),cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(),cudaSuccess);
  cache::PageSchema schema;
  ASSERT_TRUE(cache::MakeKVPageSchema(1,1,4,1,8,pool.storage_spec(),&schema));
  cache::LayoutDescriptor layout;
  ASSERT_TRUE(cache::MakePackedLayout(schema,"host-pinned",1,{0,1},{},&layout));
  auto pinned=base::PinnedCPUDeviceAllocatorFactory::get_instance();
  cache::HostStore store(layout.total_bytes,1,[pinned](size_t n) {
    return std::shared_ptr<uint8_t>(static_cast<uint8_t*>(pinned->allocate(n)),
                                    [pinned](uint8_t* p) { pinned->release(p); });
  });
  cache::HostTicket ticket;
  ASSERT_TRUE(store.reserve(schema,layout,&ticket));
  base::BlockLease source_pin;
  ASSERT_TRUE(pool.acquire_lease(pool.handle(block),base::BlockLeaseKind::kIO,&source_pin));
  cudaStream_t stream=nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),cudaSuccess);
  cudaEvent_t fence=nullptr;
  ASSERT_EQ(cudaEventCreateWithFlags(&fence,cudaEventDisableTiming),cudaSuccess);
  Gate gate;
  ASSERT_EQ(cudaLaunchHostFunc(stream,WaitForGate,&gate),cudaSuccess);
  auto* target=store.begin_copy(ticket);
  const auto copy_key=cudaMemcpyAsync(target,payload.key,payload.key_value_bytes,cudaMemcpyDeviceToHost,stream);
  const auto copy_value=cudaMemcpyAsync(target+payload.key_value_bytes,payload.value,payload.key_value_bytes,cudaMemcpyDeviceToHost,stream);
  const auto recorded=cudaEventRecord(fence,stream);
  // No ASSERT exits while stream is gated: always release and drain it.
  EXPECT_EQ(copy_key,cudaSuccess);
  EXPECT_EQ(copy_value,cudaSuccess);
  EXPECT_EQ(recorded,cudaSuccess);
  const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
  while (!gate.entered.load() && std::chrono::steady_clock::now()<deadline) std::this_thread::yield();
  EXPECT_TRUE(gate.entered.load());
  EXPECT_EQ(cudaEventQuery(fence),cudaErrorNotReady);
  EXPECT_TRUE(store.cancel(ticket));
  pool.free(block);
  EXPECT_EQ(pool.allocate(),-1);
  EXPECT_EQ(store.bytes_used(),layout.total_bytes);
  cache::HostReadLease read;
  EXPECT_FALSE(store.acquire(ticket,&read));
  gate.release.store(true);
  const auto drained=cudaStreamSynchronize(stream);
  EXPECT_EQ(drained,cudaSuccess);
  if (drained!=cudaSuccess) std::terminate();
  for (size_t i=0;i<payload.key_value_bytes;++i) {
    EXPECT_EQ(target[i],0x31);
    EXPECT_EQ(target[payload.key_value_bytes+i],0x72);
  }
  EXPECT_TRUE(store.complete(ticket,true));
  source_pin.reset();
  EXPECT_EQ(store.bytes_used(),0u);
  EXPECT_EQ(pool.num_free_blocks(),1);
  EXPECT_EQ(cudaEventDestroy(fence),cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(stream),cudaSuccess);
}

TEST(HostStoreCudaTest, BoundScatterPlainAndFp8RoundTripAtDifferentBlockAddresses) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) GTEST_SKIP();
  for (auto mode : {base::BlockStorageMode::kPlain,
                    base::BlockStorageMode::kFp8E4M3PerTokenHead}) {
    base::BlockAllocator pool(2, 4, 2, 8, base::DataType::kDataTypeBf16,
                              base::DeviceType::kDeviceCUDA, mode);
    const auto source_block = pool.allocate();
    const auto destination_block = pool.allocate();
    const auto src = pool.get_block_payload_ptrs(source_block);
    const auto dst = pool.get_block_payload_ptrs(destination_block);
    cache::PageSchema schema;
    ASSERT_TRUE(cache::MakeKVPageSchema(1, 1, 4, 2, 8, pool.storage_spec(), &schema));
    std::vector<size_t> order(schema.components.size());
    std::iota(order.begin(), order.end(), 0);
    cache::LayoutDescriptor source_layout, host_layout, target_layout;
    ASSERT_TRUE(cache::MakePackedLayout(schema, "gpu-source", 1, order,
        std::vector<int64_t>(order.size(), source_block), &source_layout));
    std::reverse(order.begin(), order.end());
    ASSERT_TRUE(cache::MakePackedLayout(schema, "host", 2, order, {}, &host_layout));
    ASSERT_TRUE(cache::MakePackedLayout(schema, "gpu-target", 3, order,
        std::vector<int64_t>(order.size(), destination_block), &target_layout));
    auto pinned = base::PinnedCPUDeviceAllocatorFactory::get_instance();
    std::shared_ptr<uint8_t> host(static_cast<uint8_t*>(pinned->allocate(host_layout.total_bytes)),
        [pinned](uint8_t* p) { pinned->release(p); });
    auto address = [](const base::KVBlockPayloadPtrs& payload, cache::ComponentKind kind) {
      switch (kind) {
        case cache::ComponentKind::kKey: return payload.key;
        case cache::ComponentKind::kValue: return payload.value;
        case cache::ComponentKind::kKeyScale: return payload.key_scale;
        case cache::ComponentKind::kValueScale: return payload.value_scale;
      }
      return static_cast<void*>(nullptr);
    };
    cache::ConstComponentEndpoint source{{}, &source_layout}, host_source{{}, &host_layout};
    cache::MutableComponentEndpoint host_target{{}, &host_layout}, target{{}, &target_layout};
    for (const auto& component : schema.components) {
      auto* from = static_cast<uint8_t*>(address(src, component.kind));
      auto* to = static_cast<uint8_t*>(address(dst, component.kind));
      ASSERT_NE(from, to);
      const auto pattern = 41 + static_cast<int>(component.kind) * 19;
      ASSERT_EQ(cudaMemset(from, pattern, component.byte_size()), cudaSuccess);
      source.components.push_back({component, from, component.byte_size()});
      target.components.push_back({component, to, component.byte_size()});
    }
    for (const auto& entry : host_layout.components) {
      host_source.components.push_back({entry.component, host.get() + entry.offset_bytes, entry.size_bytes});
      host_target.components.push_back({entry.component, host.get() + entry.offset_bytes, entry.size_bytes});
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);  // producer fence
    cache::TransferPlanner planner;
    cache::TransferPlan plan;
    cache::BoundTransfer pack, unpack;
    ASSERT_TRUE(planner.plan(schema, schema, &plan));
    ASSERT_TRUE(cache::BoundTransfer::BindComponents(plan, schema, source, host_target, &pack));
    for (const auto& op : pack.operations()) {
      ASSERT_EQ(cudaMemcpy(op.destination, op.source, op.bytes, cudaMemcpyDeviceToHost), cudaSuccess);
    }
    ASSERT_TRUE(cache::BoundTransfer::BindComponents(plan, schema, host_source, target, &unpack));
    for (const auto& op : unpack.operations()) {
      ASSERT_EQ(cudaMemcpy(op.destination, op.source, op.bytes, cudaMemcpyHostToDevice), cudaSuccess);
      std::vector<uint8_t> actual(op.bytes);
      ASSERT_EQ(cudaMemcpy(actual.data(), op.destination, op.bytes, cudaMemcpyDeviceToHost), cudaSuccess);
      EXPECT_EQ(actual, std::vector<uint8_t>(op.bytes,
          static_cast<uint8_t>(41 + static_cast<int>(op.component.kind) * 19)));
      EXPECT_EQ(op.destination_virtual_block_id, destination_block);
    }
    pool.free(source_block);
    pool.free(destination_block);
  }
}
