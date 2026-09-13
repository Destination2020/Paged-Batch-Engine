#include <gtest/gtest.h>
#include <cuda_runtime_api.h>
#include <atomic>
#include <thread>
#include "cache/page_migration.h"
#include "base/kv_cache_manager.h"
namespace {
struct MigrationGate { std::atomic<bool> release{false}; };
void CUDART_CB HoldMigration(void* data) {
  auto* gate = static_cast<MigrationGate*>(data);
  while (!gate->release.load()) std::this_thread::yield();
}
TEST(PageMigrationCudaTest, GatedTransactionsRetainResourcesAndRestorePlainFp8) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) GTEST_SKIP();
  int device = 0;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  for (auto mode : {base::BlockStorageMode::kPlain, base::BlockStorageMode::kFp8E4M3PerTokenHead}) {
    for (bool cancel : {false, true}) {
      base::BlockAllocator pool(1, 4, 1, 8, base::DataType::kDataTypeBf16,
                                base::DeviceType::kDeviceCUDA, mode);
      cache::PageSchema schema;
      ASSERT_TRUE(cache::MakeKVPageSchema(1, 1, 4, 1, 8, pool.storage_spec(), &schema));
      const auto block = pool.allocate();
      auto payload = pool.get_block_payload_ptrs(block);
      ASSERT_EQ(cudaMemset(payload.key, 0x21, payload.key_value_bytes), cudaSuccess);
      ASSERT_EQ(cudaMemset(payload.value, 0x21, payload.key_value_bytes), cudaSuccess);
      if (payload.scale_bytes) {
        ASSERT_EQ(cudaMemset(payload.key_scale, 0x21, payload.scale_bytes), cudaSuccess);
        ASSERT_EQ(cudaMemset(payload.value_scale, 0x21, payload.scale_bytes), cudaSuccess);
      }
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      const auto bytes = 2 * (payload.key_value_bytes + payload.scale_bytes);
      auto allocator = base::PinnedCPUDeviceAllocatorFactory::get_instance();
      // Allocate before gating: cudaHostAlloc may synchronize outstanding work.
      std::shared_ptr<uint8_t> pinned(static_cast<uint8_t*>(allocator->allocate(bytes)),
          [allocator](uint8_t* p) { allocator->release(p); });
      cache::HostStore host(bytes, 1, [pinned](size_t) { return pinned; });
      cache::PageDirectory directory({&pool}, 1);
      cache::LogicalPageId page;
      ASSERT_TRUE(directory.publish("prefix", schema, {pool.handle(block)}, &page));
      pool.free(block);
      cudaStream_t producer = nullptr;
      cudaEvent_t dependency = nullptr;
      ASSERT_EQ(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking), cudaSuccess);
      ASSERT_EQ(cudaEventCreateWithFlags(&dependency, cudaEventDisableTiming), cudaSuccess);
      {
        cache::PageMigrationEngine engine({&pool}, directory, host, schema, 1, bytes,
            cache::MakeCudaTransferBackend(1, device, dependency));
        MigrationGate gate;
        ASSERT_EQ(cudaLaunchHostFunc(producer, HoldMigration, &gate), cudaSuccess);
        const auto recorded = cudaEventRecord(dependency, producer);
        EXPECT_EQ(recorded, cudaSuccess);
        cache::MigrationId down = 0;
        const auto submitted = engine.submit(page, cache::TransferDirection::kToHost, &down);
        EXPECT_TRUE(submitted);
        // Always release/drain the gate even when an expectation fails.
        if (submitted && recorded == cudaSuccess) {
          engine.poll();
          cache::MigrationCompletion result;
          EXPECT_TRUE(engine.completion(down, &result));
          EXPECT_EQ(result.state, cache::MigrationState::kPending);
          EXPECT_EQ(pool.io_pins(block), 1);
          EXPECT_FALSE(directory.demote_gpu(page));
          EXPECT_EQ(host.bytes_used(), bytes);
          if (cancel) {
            EXPECT_TRUE(engine.cancel(down));
            EXPECT_TRUE(directory.erase(page));
            EXPECT_EQ(pool.allocate(), -1);
          }
        }
        gate.release.store(true);
        ASSERT_EQ(cudaStreamSynchronize(producer), cudaSuccess);
        ASSERT_TRUE(engine.drain());
        ASSERT_TRUE(submitted);
        cache::MigrationCompletion result;
        ASSERT_TRUE(engine.completion(down, &result));
        EXPECT_EQ(result.state, cancel ? cache::MigrationState::kCancelled : cache::MigrationState::kSucceeded);
        for (size_t i = 0; i < bytes; ++i) EXPECT_EQ(pinned.get()[i], 0x21);
        ASSERT_TRUE(engine.consume(down));
        if (cancel) {
          EXPECT_EQ(host.bytes_used(), 0u);
          EXPECT_EQ(pool.num_free_blocks(), 1);
        } else {
          ASSERT_TRUE(directory.demote_gpu(page));
          cache::MigrationId up;
          ASSERT_TRUE(engine.submit(page, cache::TransferDirection::kToGpu, &up));
          ASSERT_TRUE(engine.drain());
          ASSERT_TRUE(engine.completion(up, &result));
          ASSERT_EQ(result.state, cache::MigrationState::kSucceeded);
          std::vector<base::BlockLease> view;
          ASSERT_TRUE(directory.acquire(page, base::BlockLeaseKind::kCompute, &view));
          auto restored = pool.get_block_payload_ptrs(view[0].handle().block_id);
          for (auto* ptr : {restored.key, restored.value, restored.key_scale, restored.value_scale}) {
            if (!ptr) continue;
            const size_t n = (ptr == restored.key || ptr == restored.value) ? restored.key_value_bytes : restored.scale_bytes;
            std::vector<uint8_t> actual(n);
            ASSERT_EQ(cudaMemcpy(actual.data(), ptr, n, cudaMemcpyDeviceToHost), cudaSuccess);
            EXPECT_EQ(actual, std::vector<uint8_t>(n, 0x21));
          }
          ASSERT_TRUE(engine.consume(up));
        }
      }
      EXPECT_EQ(cudaEventDestroy(dependency), cudaSuccess);
      EXPECT_EQ(cudaStreamDestroy(producer), cudaSuccess);
    }
  }
}

TEST(PageMigrationCudaTest, KVManagerRestoresHostOnlyRadixPrefix) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) GTEST_SKIP();
  for (auto mode : {base::BlockStorageMode::kPlain,
                    base::BlockStorageMode::kFp8E4M3PerTokenHead}) {
    std::vector<std::unique_ptr<base::BlockAllocator>> pools;
    pools.push_back(std::make_unique<base::BlockAllocator>(
        3, 4, 1, 8, base::DataType::kDataTypeBf16,
        base::DeviceType::kDeviceCUDA, mode));
    base::HostCacheConfig host{true, 4096, 3, 1};
    base::KVCacheManager manager(4, 1, std::move(pools), host);
    const std::vector<int32_t> prompt{1,2,3,4,5};
    const auto source = manager.register_request();
    ASSERT_TRUE(manager.append_slots(source, 5));
    const auto source_block = manager.get_block_ids(source, 0).front();
    auto payload = manager.allocator_mut(0).get_block_payload_ptrs(source_block);
    for (auto* pointer : {payload.key, payload.value, payload.key_scale, payload.value_scale}) {
      if (!pointer) continue;
      const size_t bytes = (pointer == payload.key || pointer == payload.value)
                               ? payload.key_value_bytes : payload.scale_bytes;
      ASSERT_EQ(cudaMemset(pointer, 0x5c, bytes), cudaSuccess);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_TRUE(manager.commit_kv(source, 5));
    manager.publish_radix_cache(source, prompt);
    manager.free_request(source);
    ASSERT_EQ(manager.demote_radix_cache_to_host(), 1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    manager.service_cache_transfers();
    EXPECT_EQ(manager.num_free_blocks(0), 3);

    const auto restored = manager.register_request_with_radix_cache(prompt);
    ASSERT_EQ(manager.request_restore_state(restored), base::RequestRestoreState::kPending);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    manager.service_cache_transfers();
    ASSERT_EQ(manager.request_restore_state(restored), base::RequestRestoreState::kReady);
    ASSERT_EQ(manager.get_context_len(restored), 4);
    const auto restored_block = manager.get_block_ids(restored, 0).front();
    EXPECT_NE(restored_block, source_block);
    auto restored_payload = manager.allocator_mut(0).get_block_payload_ptrs(restored_block);
    for (auto* pointer : {restored_payload.key, restored_payload.value,
                          restored_payload.key_scale, restored_payload.value_scale}) {
      if (!pointer) continue;
      const size_t bytes = (pointer == restored_payload.key || pointer == restored_payload.value)
                               ? restored_payload.key_value_bytes : restored_payload.scale_bytes;
      std::vector<uint8_t> actual(bytes);
      ASSERT_EQ(cudaMemcpy(actual.data(), pointer, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
      EXPECT_EQ(actual, std::vector<uint8_t>(bytes, 0x5c));
    }
    manager.free_request(restored);
    manager.clear_radix_cache();
  }
}

TEST(PageMigrationCudaTest, ThirtyTwoRequestsShareOnePhysicalRestore) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) GTEST_SKIP();
  std::vector<std::unique_ptr<base::BlockAllocator>> pools;
  pools.push_back(std::make_unique<base::BlockAllocator>(
      4, 4, 1, 8, base::DataType::kDataTypeBf16, base::DeviceType::kDeviceCUDA));
  base::HostCacheConfig host{true, 4096, 4, 1};
  base::KVCacheManager manager(4, 1, std::move(pools), host);
  const std::vector<int32_t> prompt{1,2,3,4,5};
  const auto source = manager.register_request();
  ASSERT_TRUE(manager.append_slots(source, 5));
  ASSERT_TRUE(manager.commit_kv(source, 5));
  manager.publish_radix_cache(source, prompt);
  manager.free_request(source);
  ASSERT_EQ(manager.demote_radix_cache_to_host(), 1);
  ASSERT_TRUE(manager.drain_cache_transfers());
  ASSERT_EQ(manager.num_free_blocks(0), 4);
  const auto submissions_before = manager.transfer_scheduler_stats()->physical_submissions;

  std::vector<base::RequestId> requests;
  for (int i = 0; i < 32; ++i) {
    const auto request = manager.register_request_with_radix_cache(prompt);
    ASSERT_EQ(manager.request_restore_state(request), base::RequestRestoreState::kPending);
    requests.push_back(request);
  }
  const auto* stats = manager.transfer_scheduler_stats();
  ASSERT_NE(stats, nullptr);
  EXPECT_EQ(stats->physical_submissions - submissions_before, 1u);
  EXPECT_EQ(stats->merged_waiters, 31u);
  for (size_t i = 0; i + 1 < requests.size(); ++i) manager.free_request(requests[i]);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_GT(manager.service_cache_transfers(), 0);
  EXPECT_EQ(manager.request_restore_state(requests.back()), base::RequestRestoreState::kReady);
  EXPECT_EQ(manager.get_context_len(requests.back()), 4);
  EXPECT_EQ(manager.transfer_scheduler_stats()->physical_submissions - submissions_before, 1u);
  manager.free_request(requests.back());
  manager.service_cache_transfers();
  EXPECT_FALSE(manager.cache_transfers_pending());
}
}  // namespace
