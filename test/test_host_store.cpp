#include <gtest/gtest.h>
#include <cstring>
#include <numeric>
#include "cache/host_store.h"
#include "cache/transfer_plan.h"

namespace {
struct Fixture {
  cache::PageSchema schema;
  cache::LayoutDescriptor layout;
  Fixture() {
    EXPECT_TRUE(cache::MakeKVPageSchema(1, 2, 4, 1, 8,
        base::MakeKVCacheStorageSpec(base::DataType::kDataTypeFp32, base::BlockStorageMode::kPlain), &schema));
    std::vector<size_t> order(schema.components.size());
    std::iota(order.begin(), order.end(), 0);
    EXPECT_TRUE(cache::MakePackedLayout(schema, "host", 1, order, {}, &layout));
  }
};
}

TEST(HostStoreTest, AdmissionBoundsBytesAndEntriesBeforeAllocation) {
  Fixture f;
  int allocations=0;
  cache::HostStore store(f.layout.total_bytes, 1, [&](size_t n) {
    ++allocations;
    return std::shared_ptr<uint8_t>(new uint8_t[n](), std::default_delete<uint8_t[]>());
  });
  cache::HostTicket ticket=0, rejected=123;
  ASSERT_TRUE(store.reserve(f.schema, f.layout, &ticket));
  EXPECT_FALSE(store.reserve(f.schema, f.layout, &rejected));
  EXPECT_EQ(rejected,123u);
  EXPECT_EQ(allocations,1);
  EXPECT_EQ(store.bytes_used(),f.layout.total_bytes);
  ASSERT_TRUE(store.cancel(ticket));
  EXPECT_EQ(store.bytes_used(),0u);
  ASSERT_TRUE(store.reserve(f.schema,f.layout,&rejected));
  EXPECT_NE(ticket,rejected);
  EXPECT_FALSE(store.complete(ticket,true));
  EXPECT_TRUE(store.cancel(rejected));
}

TEST(HostStoreTest, CancelAfterSubmitRetainsBufferUntilPhysicalCompletion) {
  Fixture f;
  cache::HostStore store(f.layout.total_bytes,1);
  cache::HostTicket ticket;
  ASSERT_TRUE(store.reserve(f.schema,f.layout,&ticket));
  auto* target=store.begin_copy(ticket);
  ASSERT_NE(target,nullptr);
  ASSERT_TRUE(store.cancel(ticket));
  EXPECT_EQ(store.bytes_used(),f.layout.total_bytes);
  cache::HostReadLease read;
  EXPECT_FALSE(store.acquire(ticket,&read));
  // Models a late write after logical cancellation. ASan must see live memory.
  std::memset(target,0xa5,f.layout.total_bytes);
  ASSERT_TRUE(store.complete(ticket,true));
  EXPECT_EQ(store.bytes_used(),0u);
  EXPECT_FALSE(store.acquire(ticket,&read));
  EXPECT_FALSE(store.complete(ticket,true));
}

TEST(HostStoreTest, ReadyPageRoundTripAndReadPinProtectEviction) {
  Fixture f;
  cache::HostStore store(f.layout.total_bytes,1);
  cache::HostTicket ticket;
  ASSERT_TRUE(store.reserve(f.schema,f.layout,&ticket));
  std::vector<uint8_t> source(f.layout.total_bytes), restored(f.layout.total_bytes,0);
  for (size_t i=0;i<source.size();++i) source[i]=static_cast<uint8_t>(i*7);
  cache::TransferPlanner planner;
  cache::TransferPlan plan;
  ASSERT_TRUE(planner.plan(f.schema,f.schema,&plan));
  cache::BoundTransfer pack;
  ASSERT_TRUE(cache::BoundTransfer::Bind(plan,f.schema,{source.data(),source.size(),&f.layout},
      {store.begin_copy(ticket),f.layout.total_bytes,store.layout(ticket)},&pack));
  ASSERT_TRUE(cache::ExecuteCpuTransfer(pack));
  cache::HostReadLease read;
  EXPECT_FALSE(store.acquire(ticket,&read));
  ASSERT_TRUE(store.complete(ticket,true));
  ASSERT_TRUE(store.acquire(ticket,&read));
  EXPECT_FALSE(store.evict(ticket));
  cache::HostReadLease moved=std::move(read);
  EXPECT_FALSE(read);
  cache::BoundTransfer unpack;
  ASSERT_TRUE(cache::BoundTransfer::Bind(plan,f.schema,{moved.data(),f.layout.total_bytes,store.layout(ticket)},
      {restored.data(),restored.size(),&f.layout},&unpack));
  ASSERT_TRUE(cache::ExecuteCpuTransfer(unpack));
  EXPECT_EQ(source,restored);
  moved.reset();
  EXPECT_TRUE(store.evict(ticket));
  EXPECT_EQ(store.bytes_used(),0u);
}

TEST(HostStoreTest, UnknownCompletionQuarantinesCapacity) {
  Fixture f;
  cache::HostStore store(f.layout.total_bytes,1);
  cache::HostTicket ticket, next;
  ASSERT_TRUE(store.reserve(f.schema,f.layout,&ticket));
  ASSERT_NE(store.begin_copy(ticket),nullptr);
  ASSERT_TRUE(store.quarantine(ticket));
  EXPECT_FALSE(store.complete(ticket,true));
  EXPECT_FALSE(store.evict(ticket));
  EXPECT_FALSE(store.cancel(ticket));
  EXPECT_FALSE(store.reserve(f.schema,f.layout,&next));
  cache::HostReadLease read;
  EXPECT_FALSE(store.acquire(ticket,&read));
  EXPECT_EQ(store.bytes_used(),f.layout.total_bytes);
  ASSERT_TRUE(store.reclaim_after_quiescence(ticket));
  EXPECT_EQ(store.bytes_used(),0u);
}

TEST(HostStoreTest, AllocationFailureDoesNotConsumeBudget) {
  Fixture f;
  cache::HostStore store(f.layout.total_bytes,1,[](size_t) { return std::shared_ptr<uint8_t>(); });
  cache::HostTicket ticket=321;
  EXPECT_FALSE(store.reserve(f.schema,f.layout,&ticket));
  EXPECT_EQ(ticket,321u);
  EXPECT_EQ(store.bytes_used(),0u);
  EXPECT_EQ(store.pages_used(),0u);
}
