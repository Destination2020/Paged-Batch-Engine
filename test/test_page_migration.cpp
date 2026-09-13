#include <gtest/gtest.h>
#include <cstring>
#include "cache/page_migration.h"
namespace {
class DelayedBackend final : public cache::TransferBackend {
 public:
  bool supports(base::DeviceType device) const override { return device == base::DeviceType::kDeviceCPU; }
  cache::BoundTransfer bound;
  cache::BackendResult next = cache::BackendResult::kPending;
  bool copied = false;
  int submitted = 0, released = 0;
  cache::BackendResult submit(cache::MigrationId, cache::TransferDirection,
                              const cache::BoundTransfer& transfer) noexcept override {
    bound = transfer; ++submitted; copied = false;
    return next;
  }
  cache::BackendResult poll(cache::MigrationId) noexcept override {
    if (next == cache::BackendResult::kSucceeded && !copied) {
      if (!cache::ExecuteCpuTransfer(bound)) return cache::BackendResult::kFailedSafe;
      copied = true;
    }
    return next;
  }
  cache::BackendResult drain(cache::MigrationId id) noexcept override { return poll(id); }
  void release(cache::MigrationId) noexcept override { ++released; bound = {}; }
};
struct Fixture {
  base::BlockAllocator pool{2, 4, 1, 8, base::DataType::kDataTypeFp32, base::DeviceType::kDeviceCPU};
  cache::HostStore host{4096, 4};
  cache::PageDirectory directory{{&pool}, 4};
  cache::PageSchema schema;
  cache::LogicalPageId page = 0;
  int block;
  Fixture() {
    EXPECT_TRUE(cache::MakeKVPageSchema(1, 1, 4, 1, 8, pool.storage_spec(), &schema));
    block = pool.allocate();
    auto ptr = pool.get_block_payload_ptrs(block);
    std::memset(ptr.key, 0x38, ptr.key_value_bytes);
    std::memset(ptr.value, 0x71, ptr.key_value_bytes);
    EXPECT_TRUE(directory.publish("prefix", schema, {pool.handle(block)}, &page));
    pool.free(block);
  }
};
TEST(PageMigrationTest, CpuRoundTripPublishesOnlyCompletedWholePages) {
  Fixture f;
  cache::PageMigrationEngine engine({&f.pool}, f.directory, f.host, f.schema, 1, 4096,
                                    cache::MakeCpuTransferBackend());
  cache::MigrationId down, up;
  ASSERT_TRUE(engine.submit(f.page, cache::TransferDirection::kToHost, &down));
  cache::MigrationCompletion completion;
  ASSERT_TRUE(engine.completion(down, &completion));
  EXPECT_EQ(completion.state, cache::MigrationState::kSucceeded);
  EXPECT_EQ(engine.jobs_used(), 1u);
  EXPECT_FALSE(engine.submit(f.page, cache::TransferDirection::kToHost, &up));
  ASSERT_TRUE(engine.consume(down));
  ASSERT_TRUE(f.directory.demote_gpu(f.page));
  EXPECT_EQ(f.pool.num_free_blocks(), 2);
  ASSERT_TRUE(engine.submit(f.page, cache::TransferDirection::kToGpu, &up));
  ASSERT_TRUE(engine.completion(up, &completion));
  ASSERT_EQ(completion.state, cache::MigrationState::kSucceeded);
  std::vector<base::BlockLease> leases;
  ASSERT_TRUE(f.directory.acquire(f.page, base::BlockLeaseKind::kCompute, &leases));
  auto ptr = f.pool.get_block_payload_ptrs(leases[0].handle().block_id);
  for (size_t i = 0; i < ptr.key_value_bytes; ++i) {
    EXPECT_EQ(static_cast<uint8_t*>(ptr.key)[i], 0x38);
    EXPECT_EQ(static_cast<uint8_t*>(ptr.value)[i], 0x71);
  }
  ASSERT_TRUE(engine.consume(up));
  EXPECT_EQ(engine.bytes_used(), 0u);
}
TEST(PageMigrationTest, CancelRetainsBothEndsUntilLateCopyAndCompletion) {
  Fixture f;
  auto backend = std::make_unique<DelayedBackend>();
  auto* control = backend.get();
  cache::PageMigrationEngine engine({&f.pool}, f.directory, f.host, f.schema, 1, 4096,
                                    std::move(backend));
  cache::MigrationId id;
  ASSERT_TRUE(engine.submit(f.page, cache::TransferDirection::kToHost, &id));
  const auto used = f.host.bytes_used();
  ASSERT_GT(used, 0u);
  ASSERT_TRUE(engine.cancel(id));
  ASSERT_TRUE(f.directory.erase(f.page));
  EXPECT_EQ(f.pool.io_pins(f.block), 1);
  EXPECT_EQ(f.pool.num_free_blocks(), 1);
  EXPECT_FALSE(engine.consume(id));
  engine.poll();
  EXPECT_EQ(f.host.bytes_used(), used);
  EXPECT_EQ(control->released, 0);
  control->next = cache::BackendResult::kSucceeded;
  engine.poll();
  cache::MigrationCompletion result;
  ASSERT_TRUE(engine.completion(id, &result));
  EXPECT_EQ(result.state, cache::MigrationState::kCancelled);
  EXPECT_TRUE(control->copied);
  EXPECT_EQ(f.host.bytes_used(), 0u);
  EXPECT_EQ(f.pool.num_free_blocks(), 2);
  EXPECT_EQ(control->released, 1);
  engine.poll();
  EXPECT_EQ(control->released, 1);
  EXPECT_TRUE(engine.consume(id));
}
TEST(PageMigrationTest, UnknownCompletionQuarantinesUntilExplicitDrain) {
  Fixture f;
  auto backend = std::make_unique<DelayedBackend>();
  auto* control = backend.get();
  cache::PageMigrationEngine engine({&f.pool}, f.directory, f.host, f.schema, 1, 4096,
                                    std::move(backend));
  cache::MigrationId id;
  ASSERT_TRUE(engine.submit(f.page, cache::TransferDirection::kToHost, &id));
  control->next = cache::BackendResult::kUnknown;
  engine.poll();
  cache::MigrationCompletion result;
  ASSERT_TRUE(engine.completion(id, &result));
  EXPECT_EQ(result.state, cache::MigrationState::kQuarantined);
  EXPECT_FALSE(engine.drain());
  EXPECT_FALSE(engine.consume(id));
  EXPECT_GT(f.host.bytes_used(), 0u);
  control->next = cache::BackendResult::kSucceeded;
  engine.poll(); // Poll never treats quarantine as ready.
  EXPECT_GT(f.host.bytes_used(), 0u);
  ASSERT_TRUE(engine.drain());
  ASSERT_TRUE(engine.completion(id, &result));
  EXPECT_EQ(result.state, cache::MigrationState::kFailed);
  EXPECT_EQ(f.host.bytes_used(), 0u);
  EXPECT_EQ(f.pool.io_pins(f.block), 0);
  EXPECT_TRUE(engine.consume(id));
}
TEST(PageMigrationTest, AdmissionAndUnknownPageFailureHaveNoLeaks) {
  Fixture f;
  cache::PageMigrationEngine too_small({&f.pool}, f.directory, f.host, f.schema, 1, 1,
                                       cache::MakeCpuTransferBackend());
  cache::MigrationId id = 99;
  EXPECT_FALSE(too_small.submit(f.page, cache::TransferDirection::kToHost, &id));
  EXPECT_EQ(id, 99u);
  EXPECT_EQ(f.pool.io_pins(f.block), 0);
  EXPECT_EQ(f.host.bytes_used(), 0u);
  cache::PageMigrationEngine engine({&f.pool}, f.directory, f.host, f.schema, 1, 4096,
                                    cache::MakeCpuTransferBackend());
  EXPECT_FALSE(engine.submit(999, cache::TransferDirection::kToHost, &id));
  EXPECT_EQ(f.host.bytes_used(), 0u);
  EXPECT_EQ(engine.jobs_used(), 0u);
  EXPECT_EQ(engine.bytes_used(), 0u);
}
TEST(PageMigrationTest, DeletedDestinationRejectsLateH2DCommit) {
  Fixture f;
  {
    cache::PageMigrationEngine down({&f.pool}, f.directory, f.host, f.schema, 1, 4096,
                                    cache::MakeCpuTransferBackend());
    cache::MigrationId id;
    ASSERT_TRUE(down.submit(f.page, cache::TransferDirection::kToHost, &id));
  }
  ASSERT_TRUE(f.directory.demote_gpu(f.page));
  auto backend = std::make_unique<DelayedBackend>();
  auto* control = backend.get();
  cache::PageMigrationEngine engine({&f.pool}, f.directory, f.host, f.schema, 1, 4096,
                                    std::move(backend));
  cache::MigrationId id;
  ASSERT_TRUE(engine.submit(f.page, cache::TransferDirection::kToGpu, &id));
  EXPECT_EQ(f.pool.num_free_blocks(), 1);
  ASSERT_TRUE(f.directory.erase(f.page));
  EXPECT_GT(f.host.bytes_used(), 0u);
  control->next = cache::BackendResult::kSucceeded;
  engine.poll();
  cache::MigrationCompletion result;
  ASSERT_TRUE(engine.completion(id, &result));
  EXPECT_EQ(result.state, cache::MigrationState::kFailed);
  EXPECT_EQ(f.pool.num_free_blocks(), 2);
  EXPECT_EQ(f.host.bytes_used(), 0u);
  EXPECT_TRUE(engine.consume(id));
}
}  // namespace

TEST(PageMigrationTest, LaterLayerAllocationFailureRollsBackOnlyNewBlocks) {
  base::BlockAllocator a(1, 4, 1, 8, base::DataType::kDataTypeFp32, base::DeviceType::kDeviceCPU);
  base::BlockAllocator b(1, 4, 1, 8, base::DataType::kDataTypeFp32, base::DeviceType::kDeviceCPU);
  cache::HostStore host(4096, 1);
  cache::PageDirectory directory({&a, &b}, 1);
  cache::PageSchema schema;
  ASSERT_TRUE(cache::MakeKVPageSchema(1, 2, 4, 1, 8, a.storage_spec(), &schema));
  const auto x = a.allocate(), y = b.allocate();
  cache::LogicalPageId page;
  ASSERT_TRUE(directory.publish("two-layers", schema, {a.handle(x), b.handle(y)}, &page));
  a.free(x); b.free(y);
  cache::PageMigrationEngine engine({&a, &b}, directory, host, schema, 1, 4096,
                                    cache::MakeCpuTransferBackend());
  cache::MigrationId id;
  ASSERT_TRUE(engine.submit(page, cache::TransferDirection::kToHost, &id));
  ASSERT_TRUE(engine.consume(id));
  ASSERT_TRUE(directory.demote_gpu(page));
  const auto blocker = b.allocate();
  ASSERT_GE(blocker, 0);
  EXPECT_FALSE(engine.submit(page, cache::TransferDirection::kToGpu, &id));
  EXPECT_EQ(a.num_free_blocks(), 1);
  EXPECT_EQ(b.num_free_blocks(), 0);
  EXPECT_EQ(engine.jobs_used(), 0u);
  EXPECT_EQ(engine.bytes_used(), 0u);
  cache::HostReadLease source;
  EXPECT_TRUE(directory.acquire_host(page, &source));
  b.free(blocker);
}

TEST(PageMigrationTest, SafeSubmitFailureAndShutdownRollbackOnlyOwnedResources) {
  Fixture f;
  {
    auto backend = std::make_unique<DelayedBackend>();
    backend->next = cache::BackendResult::kFailedSafe;
    cache::PageMigrationEngine engine({&f.pool}, f.directory, f.host, f.schema, 1, 4096,
                                      std::move(backend));
    cache::MigrationId id;
    ASSERT_TRUE(engine.submit(f.page, cache::TransferDirection::kToHost, &id));
    cache::MigrationCompletion result;
    ASSERT_TRUE(engine.completion(id, &result));
    EXPECT_EQ(result.state, cache::MigrationState::kFailed);
    EXPECT_EQ(f.host.bytes_used(), 0u);
    EXPECT_EQ(f.pool.io_pins(f.block), 0);
    EXPECT_EQ(f.directory.find("prefix", f.schema), f.page);
  }
  {
    auto backend = std::make_unique<DelayedBackend>();
    auto* control = backend.get();
    cache::PageMigrationEngine engine({&f.pool}, f.directory, f.host, f.schema, 1, 4096,
                                      std::move(backend));
    cache::MigrationId id;
    ASSERT_TRUE(engine.submit(f.page, cache::TransferDirection::kToHost, &id));
    control->next = cache::BackendResult::kSucceeded;
    // Destructor cancels the waiter, drains the physical copy, then rolls back.
  }
  EXPECT_EQ(f.host.bytes_used(), 0u);
  EXPECT_EQ(f.pool.io_pins(f.block), 0);
  EXPECT_EQ(f.directory.find("prefix", f.schema), f.page);
  EXPECT_EQ(f.pool.num_free_blocks(), 1);
}
