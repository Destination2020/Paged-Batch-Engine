#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include "base/block_allocator.h"
#include "base/sequence_kv_manager.h"

namespace {

std::vector<std::unique_ptr<base::BlockAllocator>> make_allocators(int32_t num_layers,
                                                                   int32_t num_blocks,
                                                                   int32_t block_size) {
  std::vector<std::unique_ptr<base::BlockAllocator>> allocators;
  allocators.reserve(num_layers);
  for (int32_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    allocators.push_back(std::make_unique<base::BlockAllocator>(
        num_blocks, block_size, 1, 8, base::DataType::kDataTypeFp32,
        base::DeviceType::kDeviceCPU));
  }
  return allocators;
}

}  // namespace

TEST(SequenceKVManagerTest, AppendTokensBulkCommitAcrossBlockBoundary) {
  constexpr int32_t kBlockSize = 4;
  constexpr int32_t kNumLayers = 2;

  auto allocators = make_allocators(kNumLayers, 4, kBlockSize);
  base::SequenceKVManager manager(kBlockSize, kNumLayers);

  ASSERT_TRUE(manager.append_tokens(allocators, 3));
  EXPECT_EQ(manager.num_tokens(), 3);
  EXPECT_EQ(manager.page_table(0).num_blocks(), 1);
  EXPECT_EQ(manager.page_table(0).num_tokens(), 3);
  EXPECT_EQ(manager.num_tokens_in_last_block(), 3);
  EXPECT_EQ(manager.current_slot(0), std::make_pair(0, 2));

  ASSERT_TRUE(manager.append_tokens(allocators, 3));
  EXPECT_EQ(manager.num_tokens(), 6);
  EXPECT_EQ(manager.page_table(0).num_blocks(), 2);
  EXPECT_EQ(manager.page_table(1).num_blocks(), 2);
  EXPECT_EQ(manager.page_table(0).num_tokens(), 6);
  EXPECT_EQ(manager.page_table(1).num_tokens(), 6);
  EXPECT_EQ(manager.num_tokens_in_last_block(), 2);
  EXPECT_EQ(manager.current_slot(0), std::make_pair(1, 1));
  EXPECT_EQ(manager.get_slot(0, 4), std::make_pair(1, 0));
  EXPECT_EQ(manager.get_slot(1, 5), std::make_pair(1, 1));
  EXPECT_EQ(allocators[0]->num_free_blocks(), 2);
  EXPECT_EQ(allocators[1]->num_free_blocks(), 2);

  const auto& stats = manager.stats();
  EXPECT_EQ(stats.append_calls, 2);
  EXPECT_EQ(stats.tokens_requested, 6);
  EXPECT_EQ(stats.bulk_append_calls, 2);
  EXPECT_EQ(stats.blocks_allocated, 4);
  EXPECT_EQ(stats.allocation_failures, 0);
  EXPECT_EQ(stats.rollback_count, 0);
}

TEST(SequenceKVManagerTest, AppendTokensFailureRollsBackWithoutMutation) {
  constexpr int32_t kBlockSize = 4;
  constexpr int32_t kNumLayers = 2;

  auto allocators = make_allocators(kNumLayers, 2, kBlockSize);
  base::SequenceKVManager manager(kBlockSize, kNumLayers);

  ASSERT_TRUE(manager.append_tokens(allocators, 4));
  const auto before_block_ids_layer0 = manager.page_table(0).block_ids();
  const auto before_block_ids_layer1 = manager.page_table(1).block_ids();
  const int32_t free_blocks_before_layer0 = allocators[0]->num_free_blocks();
  const int32_t free_blocks_before_layer1 = allocators[1]->num_free_blocks();

  EXPECT_FALSE(manager.append_tokens(allocators, 5));

  EXPECT_EQ(manager.num_tokens(), 4);
  EXPECT_EQ(manager.page_table(0).block_ids(), before_block_ids_layer0);
  EXPECT_EQ(manager.page_table(1).block_ids(), before_block_ids_layer1);
  EXPECT_EQ(manager.page_table(0).num_blocks(), 1);
  EXPECT_EQ(manager.page_table(1).num_blocks(), 1);
  EXPECT_EQ(manager.page_table(0).num_tokens(), 4);
  EXPECT_EQ(manager.page_table(1).num_tokens(), 4);
  EXPECT_EQ(manager.num_tokens_in_last_block(), 4);
  EXPECT_EQ(manager.current_slot(0), std::make_pair(0, 3));
  EXPECT_EQ(allocators[0]->num_free_blocks(), free_blocks_before_layer0);
  EXPECT_EQ(allocators[1]->num_free_blocks(), free_blocks_before_layer1);

  const auto& stats = manager.stats();
  EXPECT_EQ(stats.append_calls, 2);
  EXPECT_EQ(stats.tokens_requested, 9);
  EXPECT_EQ(stats.bulk_append_calls, 2);
  EXPECT_EQ(stats.blocks_allocated, 2);
  EXPECT_EQ(stats.allocation_failures, 1);
  EXPECT_EQ(stats.rollback_count, 1);
}

TEST(SequenceKVManagerTest, TruncateToShrinksLogicalLengthAndReleasesTailBlocks) {
  constexpr int32_t kBlockSize = 4;
  constexpr int32_t kNumLayers = 2;

  auto allocators = make_allocators(kNumLayers, 4, kBlockSize);
  base::SequenceKVManager manager(kBlockSize, kNumLayers);

  ASSERT_TRUE(manager.append_tokens(allocators, 6));
  EXPECT_EQ(manager.num_tokens(), 6);
  EXPECT_EQ(manager.page_table(0).num_blocks(), 2);
  EXPECT_EQ(manager.page_table(1).num_blocks(), 2);
  EXPECT_EQ(allocators[0]->num_free_blocks(), 2);
  EXPECT_EQ(allocators[1]->num_free_blocks(), 2);

  manager.truncate_to(allocators, 3);

  EXPECT_EQ(manager.num_tokens(), 3);
  EXPECT_EQ(manager.shared_prefix_tokens(), 0);
  EXPECT_EQ(manager.private_tokens(), 3);
  EXPECT_EQ(manager.page_table(0).num_tokens(), 3);
  EXPECT_EQ(manager.page_table(1).num_tokens(), 3);
  EXPECT_EQ(manager.page_table(0).num_blocks(), 1);
  EXPECT_EQ(manager.page_table(1).num_blocks(), 1);
  EXPECT_EQ(manager.num_tokens_in_last_block(), 3);
  EXPECT_EQ(manager.current_slot(0), std::make_pair(0, 2));
  EXPECT_EQ(manager.current_slot(1), std::make_pair(0, 2));
  EXPECT_EQ(allocators[0]->num_free_blocks(), 3);
  EXPECT_EQ(allocators[1]->num_free_blocks(), 3);

  ASSERT_TRUE(manager.append_tokens(allocators, 3));
  EXPECT_EQ(manager.num_tokens(), 6);
  EXPECT_EQ(manager.page_table(0).num_blocks(), 2);
  EXPECT_EQ(manager.page_table(1).num_blocks(), 2);
  EXPECT_EQ(manager.get_slot(0, 4), std::make_pair(2, 0));
  EXPECT_EQ(manager.get_slot(1, 5), std::make_pair(2, 1));
}

TEST(SequenceKVManagerTest, TruncateWithSharedPrefixKeepsSharedBlocksPinned) {
  constexpr int32_t kBlockSize = 4;
  constexpr int32_t kNumLayers = 2;

  auto allocators = make_allocators(kNumLayers, 6, kBlockSize);
  base::SequenceKVManager manager(kBlockSize, kNumLayers);

  const std::vector<std::vector<int32_t>> shared_blocks = {{0}, {0}};
  ASSERT_TRUE(allocators[0]->allocate() == 0);
  ASSERT_TRUE(allocators[1]->allocate() == 0);

  manager.adopt_shared_prefix(allocators, shared_blocks, 4);
  ASSERT_TRUE(manager.append_tokens(allocators, 4));

  EXPECT_EQ(manager.shared_prefix_tokens(), 4);
  EXPECT_EQ(manager.private_tokens(), 4);
  EXPECT_EQ(manager.num_tokens(), 8);
  EXPECT_EQ(manager.page_table(0).num_blocks(), 2);
  EXPECT_EQ(manager.page_table(1).num_blocks(), 2);
  EXPECT_EQ(allocators[0]->num_free_blocks(), 4);
  EXPECT_EQ(allocators[1]->num_free_blocks(), 4);

  manager.truncate_to(allocators, 4);

  EXPECT_EQ(manager.num_tokens(), 4);
  EXPECT_EQ(manager.shared_prefix_tokens(), 4);
  EXPECT_EQ(manager.private_tokens(), 0);
  EXPECT_EQ(manager.page_table(0).num_tokens(), 4);
  EXPECT_EQ(manager.page_table(1).num_tokens(), 4);
  EXPECT_EQ(manager.page_table(0).num_blocks(), 1);
  EXPECT_EQ(manager.page_table(1).num_blocks(), 1);
  EXPECT_EQ(manager.get_slot(0, 3), std::make_pair(0, 3));
  EXPECT_EQ(manager.get_slot(1, 3), std::make_pair(0, 3));
  EXPECT_EQ(allocators[0]->num_free_blocks(), 5);
  EXPECT_EQ(allocators[1]->num_free_blocks(), 5);

  EXPECT_DEATH(manager.truncate_to(allocators, 3), "shared prefix");
}
