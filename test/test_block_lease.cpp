#include <gtest/gtest.h>
#include <type_traits>
#include <utility>
#include "base/block_allocator.h"

static_assert(!std::is_copy_constructible_v<base::BlockLease>);
static_assert(std::is_nothrow_move_constructible_v<base::BlockLease>);

TEST(BlockLeaseTest, IOAndComputePreventReuseUntilBothReleased) {
  base::BlockAllocator pool(1, 4, 1, 8, base::DataType::kDataTypeFp32,
                            base::DeviceType::kDeviceCPU);
  const auto block = pool.allocate();
  const auto old = pool.handle(block);
  base::BlockLease compute, io;
  ASSERT_TRUE(pool.acquire_lease(old, base::BlockLeaseKind::kCompute, &compute));
  ASSERT_TRUE(pool.acquire_lease(old, base::BlockLeaseKind::kIO, &io));
  pool.free(block);  // Request/cached owner releases before in-flight copy.
  EXPECT_EQ(pool.allocate(), -1);
  EXPECT_EQ(pool.compute_pins(block), 1);
  EXPECT_EQ(pool.io_pins(block), 1);
  base::BlockLease moved = std::move(io);
  EXPECT_FALSE(io);
  compute.reset();
  EXPECT_EQ(pool.allocate(), -1);
  moved.reset();
  EXPECT_FALSE(pool.is_current(old));
  EXPECT_EQ(pool.num_free_blocks(), 1);
  EXPECT_EQ(pool.allocate(), block);
  const auto current = pool.handle(block);
  EXPECT_GT(current.generation, old.generation);
  EXPECT_FALSE(pool.acquire_lease(old, base::BlockLeaseKind::kIO, &io));
  EXPECT_TRUE(pool.is_current(current));
  pool.free(block);
}

TEST(BlockLeaseTest, RejectsWrongPoolAndMoveAssignmentReleasesOldPin) {
  base::BlockAllocator a(1, 4, 1, 8, base::DataType::kDataTypeFp32,
                         base::DeviceType::kDeviceCPU);
  base::BlockAllocator b(1, 4, 1, 8, base::DataType::kDataTypeFp32,
                         base::DeviceType::kDeviceCPU);
  const auto aid = a.allocate(), bid = b.allocate();
  base::BlockLease first, second;
  EXPECT_FALSE(b.acquire_lease(a.handle(aid), base::BlockLeaseKind::kIO, &first));
  ASSERT_TRUE(a.acquire_lease(a.handle(aid), base::BlockLeaseKind::kIO, &first));
  ASSERT_TRUE(b.acquire_lease(b.handle(bid), base::BlockLeaseKind::kCompute, &second));
  a.free(aid); b.free(bid);
  first = std::move(second);
  EXPECT_EQ(a.num_free_blocks(), 1);
  EXPECT_EQ(b.num_free_blocks(), 0);
  first.reset();
  EXPECT_EQ(b.num_free_blocks(), 1);
}
