#include <gtest/gtest.h>

#include "base/block_allocator.h"

TEST(KVPoolViewTest, BindsPhysicalPoolWithoutOwningAllocationQueue) {
  base::BlockAllocator owner(4, 16, 2, 8, base::DataType::kDataTypeFp32,
                             base::DeviceType::kDeviceCPU);
  const int32_t free_before = owner.num_free_blocks();
  auto view = owner.pool_view();
  ASSERT_TRUE(view.valid());
  EXPECT_EQ(view.device_type(), base::DeviceType::kDeviceCPU);
  EXPECT_EQ(view.key_pool().ptr<void>(), owner.key_pool().ptr<void>());
  EXPECT_EQ(view.value_pool().ptr<void>(), owner.value_pool().ptr<void>());
  EXPECT_EQ(owner.num_free_blocks(), free_before);

  const int32_t block = owner.allocate();
  ASSERT_GE(block, 0);
  EXPECT_EQ(owner.num_free_blocks(), free_before - 1);
  // Copying/importing the descriptor does not affect physical ownership.
  const base::KVPoolView imported = view;
  EXPECT_EQ(imported.key_pool().ptr<void>(), view.key_pool().ptr<void>());
  EXPECT_EQ(owner.num_free_blocks(), free_before - 1);
  owner.free(block);
  EXPECT_EQ(owner.num_free_blocks(), free_before);
}
