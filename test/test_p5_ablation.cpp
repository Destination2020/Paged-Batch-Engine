#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "base/block_allocator.h"
#include "base/kv_cache_manager.h"
#include "cache/layout_codec.h"
#include "cache/page_schema.h"
#include "cache/transfer_plan.h"
#include "cache/transfer_scheduler.h"
#include "serving/request_checkpoint.h"

namespace {

using Clock = std::chrono::steady_clock;
double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

class ImmediateExecutor final : public cache::FlightExecutor {
 public:
  cache::FlightExecutionResult submit(cache::FlightId,
      const cache::FlightKey&) noexcept override {
    return cache::FlightExecutionResult::kSucceeded;
  }
  cache::FlightExecutionResult poll(cache::FlightId) noexcept override {
    return cache::FlightExecutionResult::kSucceeded;
  }
  bool cancel(cache::FlightId) noexcept override { return true; }
  cache::FlightExecutionResult drain(cache::FlightId) noexcept override {
    return cache::FlightExecutionResult::kSucceeded;
  }
  void release(cache::FlightId) noexcept override {}
};

std::unique_ptr<base::KVCacheManager> make_bench_manager() {
  std::vector<std::unique_ptr<base::BlockAllocator>> pools;
  for (int layer = 0; layer < 24; ++layer) {
    pools.emplace_back(std::make_unique<base::BlockAllocator>(
        16, 16, 2, 64, base::DataType::kDataTypeBf16,
        base::DeviceType::kDeviceCPU, base::BlockStorageMode::kPlain));
  }
  return std::make_unique<base::KVCacheManager>(16, 24, std::move(pools));
}

TEST(P5AblationTest, FixedBudgetReportsLayoutTransactionAndSingleFlightCosts) {
  cache::PageSchema schema;
  ASSERT_TRUE(cache::MakeKVPageSchema(
      1, 24, 16, 2, 64,
      base::MakeKVCacheStorageSpec(base::DataType::kDataTypeBf16,
                                   base::BlockStorageMode::kPlain),
      &schema));
  std::vector<size_t> source_order(schema.components.size());
  std::iota(source_order.begin(), source_order.end(), 0);
  auto destination_order = source_order;
  std::reverse(destination_order.begin(), destination_order.end());
  cache::LayoutDescriptor source_layout, destination_layout;
  ASSERT_TRUE(cache::MakePackedLayout(schema, "source", 1, source_order, {},
                                      &source_layout));
  ASSERT_TRUE(cache::MakePackedLayout(schema, "destination", 2,
                                      destination_order, {}, &destination_layout));
  std::vector<uint8_t> source(source_layout.total_bytes, 37);
  std::vector<uint8_t> destination(destination_layout.total_bytes, 0);
  cache::TransferPlanner planner;
  cache::TransferPlan plan;
  ASSERT_TRUE(planner.plan(schema, schema, &plan));
  cache::BoundTransfer bound;
  ASSERT_TRUE(cache::BoundTransfer::Bind(
      plan, schema, {source.data(), source.size(), &source_layout},
      {destination.data(), destination.size(), &destination_layout}, &bound));
  constexpr int kLayoutIterations = 2000;
  auto start = Clock::now();
  for (int i = 0; i < kLayoutIterations; ++i)
    std::memcpy(destination.data(), source.data(), source.size());
  const double direct_ms = elapsed_ms(start);
  start = Clock::now();
  for (int i = 0; i < kLayoutIterations; ++i)
    ASSERT_TRUE(cache::ExecuteCpuTransfer(bound));
  const double semantic_ms = elapsed_ms(start);

  auto manager = make_bench_manager();
  auto request = manager->register_request();
  ASSERT_TRUE(manager->append_slots(request, 32));
  ASSERT_TRUE(manager->commit_kv(request, 32));
  serving::SequenceState sequence;
  sequence.request_id = request;
  sequence.client_request_id = 1;
  sequence.prompt_tokens.resize(32, 7);
  sequence.computed_tokens = 32;
  sequence.generation_config.sampling.seed = 123456789;
  constexpr int kSnapshotIterations = 200;
  base::KVRequestSnapshot snapshot;
  start = Clock::now();
  for (int i = 0; i < kSnapshotIterations; ++i)
    ASSERT_TRUE(manager->snapshot_request(request, &snapshot));
  const double snapshot_ms = elapsed_ms(start);
  serving::RequestCheckpointStore store(2);
  start = Clock::now();
  for (int i = 0; i < kSnapshotIterations; ++i) {
    serving::CheckpointTicket ticket;
    ASSERT_TRUE(store.save(sequence, "p5-model", manager.get(), &ticket));
    manager->free_request(request);
    ASSERT_TRUE(store.begin_restore(ticket, "p5-model", manager.get()));
    ASSERT_TRUE(store.commit_restore(ticket, manager.get(), &sequence));
    request = sequence.request_id;
  }
  const double checkpoint_roundtrip_ms = elapsed_ms(start);
  manager->free_request(request);

  constexpr int kSchedulerIterations = 1000;
  uint64_t unique_physical = 0, shared_physical = 0;
  auto schedule_case = [&](bool shared) {
    const auto case_start = Clock::now();
    for (int iteration = 0; iteration < kSchedulerIterations; ++iteration) {
      cache::TransferScheduler scheduler(32, 32, 32, 4,
                                         std::make_unique<ImmediateExecutor>());
      std::vector<cache::WaiterTicket> tickets(32);
      for (uint64_t waiter = 0; waiter < tickets.size(); ++waiter) {
        EXPECT_TRUE(scheduler.submit(
            {shared ? 1 : waiter + 1, cache::TransferTarget::kGpu, 0},
            cache::TransferPriority::kDecode, {}, &tickets[waiter]));
      }
      if (shared) shared_physical += scheduler.stats().physical_submissions;
      else unique_physical += scheduler.stats().physical_submissions;
      for (const auto& ticket : tickets) EXPECT_TRUE(scheduler.consume(ticket.waiter));
    }
    return elapsed_ms(case_start);
  };
  const double unique_ms = schedule_case(false);
  const double shared_ms = schedule_case(true);
  EXPECT_EQ(unique_physical, static_cast<uint64_t>(kSchedulerIterations) * 32);
  EXPECT_EQ(shared_physical, static_cast<uint64_t>(kSchedulerIterations));

  std::cout << "P5_ABLATION layout_bytes=" << source.size()
            << " layout_iterations=" << kLayoutIterations
            << " direct_ms=" << direct_ms
            << " semantic_ms=" << semantic_ms
            << " snapshot_iterations=" << kSnapshotIterations
            << " snapshot_ms=" << snapshot_ms
            << " checkpoint_roundtrip_ms=" << checkpoint_roundtrip_ms
            << " scheduler_iterations=" << kSchedulerIterations
            << " unique_ms=" << unique_ms
            << " shared_ms=" << shared_ms
            << " unique_physical=" << unique_physical
            << " shared_physical=" << shared_physical << std::endl;
}

}  // namespace
