#include <gtest/gtest.h>
#include "cache/page_directory.h"

TEST(PageDirectoryTest, EvictionRetainsInFlightPageUntilLeaseCompletion) {
  base::BlockAllocator pool(1, 4, 1, 8, base::DataType::kDataTypeFp32,
                           base::DeviceType::kDeviceCPU);
  cache::PageDirectory directory({&pool}, 1);
  cache::PageSchema schema;
  ASSERT_TRUE(cache::MakeKVPageSchema(1, 1, 4, 1, 8, pool.storage_spec(), &schema));
  const auto block = pool.allocate();
  cache::LogicalPageId id = 0;
  ASSERT_TRUE(directory.publish("model-A/prefix-A", schema, {pool.handle(block)}, &id));
  pool.free(block);
  EXPECT_EQ(pool.num_free_blocks(), 0);
  EXPECT_EQ(directory.find("model-A/prefix-A", schema), id);
  EXPECT_EQ(directory.find("model-B/prefix-A", schema), 0u);
  std::vector<base::BlockLease> io;
  ASSERT_TRUE(directory.acquire(id, base::BlockLeaseKind::kIO, &io));
  ASSERT_TRUE(directory.erase(id));
  EXPECT_EQ(pool.num_free_blocks(), 0);
  EXPECT_FALSE(directory.acquire(id, base::BlockLeaseKind::kCompute, &io));
  io.clear();
  EXPECT_EQ(pool.num_free_blocks(), 1);
}

TEST(PageDirectoryTest, BoundedPublicationValidatesAllLayersBeforeRetaining) {
  base::BlockAllocator a(1, 4, 1, 8, base::DataType::kDataTypeFp32,
                        base::DeviceType::kDeviceCPU);
  base::BlockAllocator b(1, 4, 1, 8, base::DataType::kDataTypeFp32,
                        base::DeviceType::kDeviceCPU);
  cache::PageDirectory directory({&a, &b}, 1);
  cache::PageSchema schema;
  ASSERT_TRUE(cache::MakeKVPageSchema(1, 2, 4, 1, 8, a.storage_spec(), &schema));
  const auto x=a.allocate(), y=b.allocate();
  cache::LogicalPageId id=0;
  auto stale = b.handle(y);
  ++stale.generation;
  EXPECT_FALSE(directory.publish("one", schema, {a.handle(x), stale}, &id));
  EXPECT_EQ(directory.size(), 0u);
  ASSERT_TRUE(directory.publish("one", schema, {a.handle(x), b.handle(y)}, &id));
  cache::LogicalPageId duplicate=0;
  ASSERT_TRUE(directory.publish("one", schema, {a.handle(x), b.handle(y)}, &duplicate));
  EXPECT_EQ(id, duplicate);
  EXPECT_FALSE(directory.publish("two", schema, {a.handle(x), b.handle(y)}, &duplicate));
  a.free(x); b.free(y);
  directory.clear();
  EXPECT_EQ(a.num_free_blocks(), 1);
  EXPECT_EQ(b.num_free_blocks(), 1);
}

TEST(PageDirectoryTest, HostResidencyDemotionRestoreAndDeferredEviction) {
  base::BlockAllocator pool(2, 4, 1, 8, base::DataType::kDataTypeFp32,
                           base::DeviceType::kDeviceCPU);
  cache::PageSchema schema;
  ASSERT_TRUE(cache::MakeKVPageSchema(1, 1, 4, 1, 8, pool.storage_spec(), &schema));
  cache::LayoutDescriptor layout;
  ASSERT_TRUE(cache::MakePackedLayout(schema, "host", 1, {1, 0}, {}, &layout));
  cache::HostStore store(layout.total_bytes, 1); // Outlives directory and readers.
  cache::PageDirectory directory({&pool}, 2);
  const auto block = pool.allocate();
  cache::LogicalPageId id;
  ASSERT_TRUE(directory.publish("prefix", schema, {pool.handle(block)}, &id));
  pool.free(block);
  EXPECT_FALSE(directory.demote_gpu(id));
  cache::HostTicket ticket;
  ASSERT_TRUE(store.reserve(schema, layout, &ticket));
  EXPECT_FALSE(directory.attach_host(id, &store, ticket));
  auto* bytes = store.begin_copy(ticket);
  ASSERT_NE(bytes, nullptr);
  std::fill(bytes, bytes + layout.total_bytes, 0x61); // CPU copy completion oracle.
  EXPECT_FALSE(directory.attach_host(id, &store, ticket));
  ASSERT_TRUE(store.complete(ticket, true));
  ASSERT_TRUE(directory.attach_host(id, &store, ticket));
  EXPECT_FALSE(store.evict(ticket));
  std::vector<base::BlockLease> io;
  ASSERT_TRUE(directory.acquire(id, base::BlockLeaseKind::kIO, &io));
  EXPECT_FALSE(directory.demote_gpu(id));
  io.clear();
  ASSERT_TRUE(directory.demote_gpu(id));
  EXPECT_EQ(pool.num_free_blocks(), 2);
  EXPECT_EQ(directory.find("prefix", schema), id);
  EXPECT_FALSE(directory.acquire(id, base::BlockLeaseKind::kCompute, &io));
  cache::HostReadLease reader;
  ASSERT_TRUE(directory.acquire_host(id, &reader));
  EXPECT_EQ(reader.data()[0], 0x61);
  const auto restored = pool.allocate();
  const auto restored_handle = pool.handle(restored);
  EXPECT_FALSE(directory.install_gpu(id, nullptr, ticket, {restored_handle}));
  EXPECT_FALSE(directory.install_gpu(id, &store, ticket + 1, {restored_handle}));
  auto stale = pool.handle(restored);
  ++stale.generation;
  EXPECT_FALSE(directory.install_gpu(id, &store, ticket, {stale}));
  ASSERT_TRUE(directory.install_gpu(id, &store, ticket, {restored_handle}));
  pool.free(restored);
  ASSERT_TRUE(directory.acquire(id, base::BlockLeaseKind::kCompute, &io));
  EXPECT_FALSE(directory.demote_gpu(id));
  io.clear();
  ASSERT_TRUE(directory.demote_gpu(id));
  ASSERT_TRUE(directory.drop_host(id));
  EXPECT_EQ(directory.find("prefix", schema), 0u);
  EXPECT_EQ(directory.size(), 0u);
  EXPECT_EQ(store.bytes_used(), layout.total_bytes);
  cache::HostReadLease late_reader;
  EXPECT_FALSE(store.acquire(ticket, &late_reader));
  EXPECT_EQ(reader.data()[layout.total_bytes - 1], 0x61);
  EXPECT_FALSE(directory.install_gpu(id, &store, ticket, {restored_handle}));
  reader.reset();
  EXPECT_EQ(store.bytes_used(), 0u);
  EXPECT_EQ(store.pages_used(), 0u);
  EXPECT_EQ(pool.num_free_blocks(), 2);
}

TEST(PageDirectoryTest, HostAttachmentRejectsSchemaVersionAndDuplicateTicket) {
  base::BlockAllocator pool(2, 4, 1, 8, base::DataType::kDataTypeFp32,
                           base::DeviceType::kDeviceCPU);
  cache::PageSchema schema;
  ASSERT_TRUE(cache::MakeKVPageSchema(1, 1, 4, 1, 8, pool.storage_spec(), &schema));
  auto other = schema;
  ++other.version;
  cache::LayoutDescriptor layout;
  ASSERT_TRUE(cache::MakePackedLayout(schema, "host", 1, {0, 1}, {}, &layout));
  cache::HostStore store(layout.total_bytes * 2, 2);
  cache::PageDirectory directory({&pool}, 2);
  const auto a = pool.allocate(), b = pool.allocate();
  cache::LogicalPageId one, two;
  ASSERT_TRUE(directory.publish("one", schema, {pool.handle(a)}, &one));
  ASSERT_TRUE(directory.publish("two", schema, {pool.handle(b)}, &two));
  pool.free(a); pool.free(b);
  cache::HostTicket wrong, correct;
  ASSERT_TRUE(store.reserve(other, layout, &wrong));
  ASSERT_NE(store.begin_copy(wrong), nullptr);
  ASSERT_TRUE(store.complete(wrong, true));
  EXPECT_FALSE(directory.attach_host(one, &store, wrong));
  ASSERT_TRUE(store.reserve(schema, layout, &correct));
  ASSERT_NE(store.begin_copy(correct), nullptr);
  ASSERT_TRUE(store.complete(correct, true));
  ASSERT_TRUE(directory.attach_host(one, &store, correct));
  EXPECT_FALSE(directory.attach_host(two, &store, correct));
  ASSERT_TRUE(directory.drop_host(one)); // GPU residency remains indexed.
  EXPECT_EQ(directory.find("one", schema), one);
  EXPECT_EQ(store.pages_used(), 1u);
  ASSERT_TRUE(store.evict(wrong));
  directory.clear();
  EXPECT_EQ(pool.num_free_blocks(), 2);
}
