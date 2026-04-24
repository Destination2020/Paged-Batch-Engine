#include <gtest/gtest.h>
#include "base/compressed_radix_cache_tree.h"

#include <vector>

namespace {

std::vector<int32_t> Tokens(std::initializer_list<int32_t> values) {
  return std::vector<int32_t>(values);
}

std::vector<std::vector<int32_t>> BlockIds(
    std::initializer_list<std::initializer_list<int32_t>> layers) {
  std::vector<std::vector<int32_t>> result;
  for (const auto& layer : layers) {
    result.emplace_back(layer);
  }
  return result;
}

}  // namespace

TEST(CompressedRadixCacheTreeTest, InsertsOneCompressedSegmentForLongPrefix) {
  base::CompressedRadixCacheTree tree(/*block_size=*/2, /*num_layers=*/2);

  const auto insert = tree.insert(
      Tokens({1, 2, 3, 4, 5, 6, 7, 8}),
      BlockIds({{10, 11, 12, 13}, {20, 21, 22, 23}}));

  EXPECT_EQ(insert.existing_prefix_blocks, 0);
  EXPECT_EQ(insert.inserted_blocks, 4);
  ASSERT_NE(insert.leaf, nullptr);
  EXPECT_EQ(insert.leaf->segment_blocks(tree.block_size()), 4);
  EXPECT_EQ(insert.leaf->prefix_blocks(tree.block_size()), 4);
  EXPECT_EQ(tree.node_count(), 2);

  const auto match = tree.match_prefix(Tokens({1, 2, 3, 4, 5, 6, 7, 8}));
  EXPECT_EQ(match.matched_blocks, 4);
  EXPECT_EQ(match.matched_tokens, 8);
  EXPECT_EQ(match.matched_leaf, insert.leaf);
  EXPECT_EQ(match.block_ids_per_layer,
            BlockIds({{10, 11, 12, 13}, {20, 21, 22, 23}}));
}

TEST(CompressedRadixCacheTreeTest, MatchInsideSegmentSplitsNodeAtBlockBoundary) {
  base::CompressedRadixCacheTree tree(/*block_size=*/2, /*num_layers=*/2);

  auto* original_leaf = tree.insert(
      Tokens({1, 2, 3, 4, 5, 6, 7, 8}),
      BlockIds({{10, 11, 12, 13}, {20, 21, 22, 23}})).leaf;
  ASSERT_NE(original_leaf, nullptr);

  const auto match = tree.match_prefix(Tokens({1, 2, 3, 4}));
  ASSERT_NE(match.matched_leaf, nullptr);
  EXPECT_NE(match.matched_leaf, original_leaf);
  EXPECT_EQ(match.matched_blocks, 2);
  EXPECT_EQ(match.matched_tokens, 4);
  EXPECT_EQ(match.matched_leaf->segment_blocks(tree.block_size()), 2);
  EXPECT_EQ(match.matched_leaf->prefix_blocks(tree.block_size()), 2);
  EXPECT_EQ(match.block_ids_per_layer, BlockIds({{10, 11}, {20, 21}}));
  EXPECT_EQ(tree.split_count(), 1);
  EXPECT_EQ(tree.node_count(), 3);

  ASSERT_EQ(match.matched_leaf->children.size(), 1);
  const auto& suffix = match.matched_leaf->children.begin()->second;
  ASSERT_NE(suffix, nullptr);
  EXPECT_EQ(suffix->segment_tokens, Tokens({5, 6, 7, 8}));
  EXPECT_EQ(suffix->segment_block_ids_per_layer,
            BlockIds({{12, 13}, {22, 23}}));
  EXPECT_EQ(suffix->depth_blocks_before, 2);
  EXPECT_EQ(suffix->prefix_blocks(tree.block_size()), 4);
}

TEST(CompressedRadixCacheTreeTest, InsertDivergentSuffixAfterSplitCreatesSibling) {
  base::CompressedRadixCacheTree tree(/*block_size=*/2, /*num_layers=*/1);

  tree.insert(Tokens({1, 2, 3, 4, 5, 6, 7, 8}),
              BlockIds({{10, 11, 12, 13}}));
  const auto prefix_match = tree.match_prefix(Tokens({1, 2, 3, 4}));
  ASSERT_NE(prefix_match.matched_leaf, nullptr);
  ASSERT_EQ(prefix_match.matched_leaf->children.size(), 1);

  const auto insert = tree.insert(Tokens({1, 2, 3, 4, 99, 100}),
                                  BlockIds({{30, 31, 32}}));
  EXPECT_EQ(insert.existing_prefix_blocks, 2);
  EXPECT_EQ(insert.inserted_blocks, 1);
  ASSERT_NE(insert.leaf, nullptr);
  EXPECT_EQ(insert.leaf->segment_tokens, Tokens({99, 100}));
  EXPECT_EQ(insert.leaf->segment_block_ids_per_layer, BlockIds({{32}}));
  EXPECT_EQ(prefix_match.matched_leaf->children.size(), 2);

  const auto branch_match = tree.match_prefix(Tokens({1, 2, 3, 4, 99, 100}));
  EXPECT_EQ(branch_match.matched_blocks, 3);
  EXPECT_EQ(branch_match.block_ids_per_layer, BlockIds({{10, 11, 32}}));

  const auto original_branch_match =
      tree.match_prefix(Tokens({1, 2, 3, 4, 5, 6, 7, 8}));
  EXPECT_EQ(original_branch_match.matched_blocks, 4);
  EXPECT_EQ(original_branch_match.block_ids_per_layer,
            BlockIds({{10, 11, 12, 13}}));
}

TEST(CompressedRadixCacheTreeTest, InsertShorterPrefixSplitsExistingLongLeaf) {
  base::CompressedRadixCacheTree tree(/*block_size=*/2, /*num_layers=*/1);

  tree.insert(Tokens({1, 2, 3, 4, 5, 6, 7, 8}),
              BlockIds({{10, 11, 12, 13}}));

  const auto insert = tree.insert(Tokens({1, 2, 3, 4}),
                                  BlockIds({{30, 31}}));
  ASSERT_NE(insert.leaf, nullptr);
  EXPECT_TRUE(insert.split_performed);
  EXPECT_EQ(insert.existing_prefix_blocks, 2);
  EXPECT_EQ(insert.inserted_blocks, 0);
  EXPECT_EQ(insert.leaf->segment_tokens, Tokens({1, 2, 3, 4}));
  EXPECT_EQ(insert.leaf->segment_block_ids_per_layer, BlockIds({{10, 11}}));
  EXPECT_EQ(tree.split_count(), 1);
  EXPECT_EQ(tree.node_count(), 3);
}

TEST(CompressedRadixCacheTreeTest, IgnoresPartialTrailingBlock) {
  base::CompressedRadixCacheTree tree(/*block_size=*/2, /*num_layers=*/1);

  const auto insert = tree.insert(Tokens({1, 2, 3, 4, 5}),
                                  BlockIds({{10, 11, 12}}));
  ASSERT_NE(insert.leaf, nullptr);
  EXPECT_EQ(insert.inserted_blocks, 2);
  EXPECT_EQ(insert.inserted_tokens, 4);
  EXPECT_EQ(insert.leaf->segment_tokens, Tokens({1, 2, 3, 4}));
  EXPECT_EQ(insert.leaf->segment_block_ids_per_layer, BlockIds({{10, 11}}));

  const auto match = tree.match_prefix(Tokens({1, 2, 3, 4, 5}));
  EXPECT_EQ(match.matched_blocks, 2);
  EXPECT_EQ(match.matched_tokens, 4);
}

TEST(CompressedRadixCacheTreeTest, PinAndUnpinOperateOnWholePath) {
  base::CompressedRadixCacheTree tree(/*block_size=*/2, /*num_layers=*/1);

  tree.insert(Tokens({1, 2, 3, 4, 5, 6, 7, 8}),
              BlockIds({{10, 11, 12, 13}}));
  auto prefix_match = tree.match_prefix(Tokens({1, 2, 3, 4}));
  ASSERT_NE(prefix_match.matched_leaf, nullptr);
  auto full_match = tree.match_prefix(Tokens({1, 2, 3, 4, 5, 6, 7, 8}));
  ASSERT_NE(full_match.matched_leaf, nullptr);

  tree.pin_path(full_match.matched_leaf);
  EXPECT_EQ(prefix_match.matched_leaf->active_ref_count, 1);
  EXPECT_EQ(full_match.matched_leaf->active_ref_count, 1);

  tree.unpin_path(full_match.matched_leaf);
  EXPECT_EQ(prefix_match.matched_leaf->active_ref_count, 0);
  EXPECT_EQ(full_match.matched_leaf->active_ref_count, 0);
}

TEST(CompressedRadixCacheTreeTest, NamespaceCanBeModeledBySeparateTrees) {
  base::CompressedRadixCacheTree namespace_a(/*block_size=*/2, /*num_layers=*/1);
  base::CompressedRadixCacheTree namespace_b(/*block_size=*/2, /*num_layers=*/1);

  namespace_a.insert(Tokens({1, 2, 3, 4}), BlockIds({{10, 11}}));

  EXPECT_EQ(namespace_a.match_prefix(Tokens({1, 2, 3, 4})).matched_blocks, 2);
  EXPECT_EQ(namespace_b.match_prefix(Tokens({1, 2, 3, 4})).matched_blocks, 0);
}

TEST(CompressedRadixCacheTreeTest, RepeatedMatchesCanSplitCompressedPathMultipleTimes) {
  base::CompressedRadixCacheTree tree(/*block_size=*/2, /*num_layers=*/1);

  tree.insert(Tokens({1, 2, 3, 4, 5, 6, 7, 8}),
              BlockIds({{10, 11, 12, 13}}));

  const auto split_at_two = tree.match_prefix(Tokens({1, 2, 3, 4}));
  ASSERT_NE(split_at_two.matched_leaf, nullptr);
  EXPECT_EQ(split_at_two.matched_blocks, 2);
  EXPECT_EQ(tree.split_count(), 1);

  const auto split_at_three = tree.match_prefix(Tokens({1, 2, 3, 4, 5, 6}));
  ASSERT_NE(split_at_three.matched_leaf, nullptr);
  EXPECT_EQ(split_at_three.matched_blocks, 3);
  EXPECT_EQ(split_at_three.block_ids_per_layer, BlockIds({{10, 11, 12}}));
  EXPECT_EQ(tree.split_count(), 2);
  EXPECT_EQ(tree.node_count(), 4);

  ASSERT_EQ(split_at_two.matched_leaf->children.size(), 1);
  const auto& middle = split_at_two.matched_leaf->children.begin()->second;
  ASSERT_NE(middle, nullptr);
  EXPECT_EQ(middle->segment_tokens, Tokens({5, 6}));
  EXPECT_EQ(middle->segment_block_ids_per_layer, BlockIds({{12}}));

  ASSERT_EQ(middle->children.size(), 1);
  const auto& tail = middle->children.begin()->second;
  ASSERT_NE(tail, nullptr);
  EXPECT_EQ(tail->segment_tokens, Tokens({7, 8}));
  EXPECT_EQ(tail->segment_block_ids_per_layer, BlockIds({{13}}));
}

TEST(CompressedRadixCacheTreeTest, SplitPreservesPinnedSuffixPathCounts) {
  base::CompressedRadixCacheTree tree(/*block_size=*/2, /*num_layers=*/1);

  auto full_match = tree.insert(Tokens({1, 2, 3, 4, 5, 6, 7, 8}),
                                BlockIds({{10, 11, 12, 13}}));
  ASSERT_NE(full_match.leaf, nullptr);
  tree.pin_path(full_match.leaf);
  EXPECT_EQ(full_match.leaf->active_ref_count, 1);

  const auto prefix_match = tree.match_prefix(Tokens({1, 2, 3, 4}));
  ASSERT_NE(prefix_match.matched_leaf, nullptr);
  EXPECT_EQ(prefix_match.matched_leaf->active_ref_count, 1);

  ASSERT_EQ(prefix_match.matched_leaf->children.size(), 1);
  auto* suffix = prefix_match.matched_leaf->children.begin()->second.get();
  ASSERT_NE(suffix, nullptr);
  EXPECT_EQ(suffix, full_match.leaf);
  EXPECT_EQ(suffix->active_ref_count, 1);

  tree.unpin_path(full_match.leaf);
  EXPECT_EQ(prefix_match.matched_leaf->active_ref_count, 0);
  EXPECT_EQ(suffix->active_ref_count, 0);
}

TEST(CompressedRadixCacheTreeTest, SharedPrefixPinCountsAccumulateAcrossBranches) {
  base::CompressedRadixCacheTree tree(/*block_size=*/2, /*num_layers=*/1);

  tree.insert(Tokens({1, 2, 3, 4, 5, 6, 7, 8}),
              BlockIds({{10, 11, 12, 13}}));
  const auto shared_prefix = tree.match_prefix(Tokens({1, 2, 3, 4}));
  ASSERT_NE(shared_prefix.matched_leaf, nullptr);

  tree.insert(Tokens({1, 2, 3, 4, 9, 10}),
              BlockIds({{10, 11, 30}}));
  tree.insert(Tokens({1, 2, 3, 4, 11, 12, 13, 14}),
              BlockIds({{10, 11, 40, 41}}));

  auto branch_a = tree.match_prefix(Tokens({1, 2, 3, 4, 5, 6, 7, 8}));
  auto branch_b = tree.match_prefix(Tokens({1, 2, 3, 4, 9, 10}));
  auto branch_c = tree.match_prefix(Tokens({1, 2, 3, 4, 11, 12, 13, 14}));
  ASSERT_NE(branch_a.matched_leaf, nullptr);
  ASSERT_NE(branch_b.matched_leaf, nullptr);
  ASSERT_NE(branch_c.matched_leaf, nullptr);

  tree.pin_path(branch_a.matched_leaf);
  tree.pin_path(branch_b.matched_leaf);
  tree.pin_path(branch_c.matched_leaf);

  EXPECT_EQ(shared_prefix.matched_leaf->active_ref_count, 3);
  EXPECT_EQ(branch_a.matched_leaf->active_ref_count, 1);
  EXPECT_EQ(branch_b.matched_leaf->active_ref_count, 1);
  EXPECT_EQ(branch_c.matched_leaf->active_ref_count, 1);

  tree.unpin_path(branch_b.matched_leaf);
  EXPECT_EQ(shared_prefix.matched_leaf->active_ref_count, 2);
  EXPECT_EQ(branch_b.matched_leaf->active_ref_count, 0);

  tree.unpin_path(branch_a.matched_leaf);
  tree.unpin_path(branch_c.matched_leaf);
  EXPECT_EQ(shared_prefix.matched_leaf->active_ref_count, 0);
}

TEST(CompressedRadixCacheTreeTest, ComplexMultiBranchTreeMatchesCorrectLeaves) {
  base::CompressedRadixCacheTree tree(/*block_size=*/2, /*num_layers=*/2);

  tree.insert(Tokens({1, 2, 3, 4, 5, 6, 7, 8}),
              BlockIds({{10, 11, 12, 13}, {20, 21, 22, 23}}));
  tree.match_prefix(Tokens({1, 2, 3, 4}));
  tree.insert(Tokens({1, 2, 3, 4, 9, 10}),
              BlockIds({{10, 11, 30}, {20, 21, 40}}));
  tree.insert(Tokens({1, 2, 3, 4, 11, 12, 13, 14}),
              BlockIds({{10, 11, 50, 51}, {20, 21, 60, 61}}));
  tree.insert(Tokens({21, 22, 23, 24}),
              BlockIds({{70, 71}, {80, 81}}));

  EXPECT_EQ(tree.root()->children.size(), 2);

  const auto branch1 = tree.match_prefix(Tokens({1, 2, 3, 4, 5, 6, 7, 8}));
  const auto branch2 = tree.match_prefix(Tokens({1, 2, 3, 4, 9, 10}));
  const auto branch3 = tree.match_prefix(Tokens({1, 2, 3, 4, 11, 12, 13, 14}));
  const auto root_sibling = tree.match_prefix(Tokens({21, 22, 23, 24}));

  EXPECT_EQ(branch1.block_ids_per_layer,
            BlockIds({{10, 11, 12, 13}, {20, 21, 22, 23}}));
  EXPECT_EQ(branch2.block_ids_per_layer,
            BlockIds({{10, 11, 30}, {20, 21, 40}}));
  EXPECT_EQ(branch3.block_ids_per_layer,
            BlockIds({{10, 11, 50, 51}, {20, 21, 60, 61}}));
  EXPECT_EQ(root_sibling.block_ids_per_layer,
            BlockIds({{70, 71}, {80, 81}}));
}

TEST(CompressedRadixCacheTreeTest, InsertExactExistingSequenceIsIdempotentAfterSplits) {
  base::CompressedRadixCacheTree tree(/*block_size=*/2, /*num_layers=*/1);

  tree.insert(Tokens({1, 2, 3, 4, 5, 6, 7, 8}),
              BlockIds({{10, 11, 12, 13}}));
  tree.match_prefix(Tokens({1, 2, 3, 4}));
  tree.insert(Tokens({1, 2, 3, 4, 9, 10}),
              BlockIds({{10, 11, 30}}));

  const int32_t nodes_before = tree.node_count();
  const int32_t splits_before = tree.split_count();

  const auto insert = tree.insert(Tokens({1, 2, 3, 4, 5, 6, 7, 8}),
                                  BlockIds({{10, 11, 12, 13}}));
  EXPECT_EQ(insert.existing_prefix_blocks, 4);
  EXPECT_EQ(insert.inserted_blocks, 0);
  EXPECT_EQ(tree.node_count(), nodes_before);
  EXPECT_EQ(tree.split_count(), splits_before);
}

TEST(CompressedRadixCacheTreeTest, ShorterMatchCanResolveInternalBoundaryForFallback) {
  base::CompressedRadixCacheTree tree(/*block_size=*/2, /*num_layers=*/1);

  tree.insert(Tokens({1, 2, 3, 4, 5, 6, 7, 8}),
              BlockIds({{10, 11, 12, 13}}));

  const auto fallback_match = tree.match_prefix(Tokens({1, 2, 3, 4, 5, 6}));
  EXPECT_EQ(fallback_match.matched_blocks, 3);
  EXPECT_EQ(fallback_match.block_ids_per_layer, BlockIds({{10, 11, 12}}));
  ASSERT_NE(fallback_match.matched_leaf, nullptr);
  EXPECT_EQ(fallback_match.matched_leaf->segment_tokens, Tokens({1, 2, 3, 4, 5, 6}));

  const auto full_match = tree.match_prefix(Tokens({1, 2, 3, 4, 5, 6, 7, 8}));
  EXPECT_EQ(full_match.matched_blocks, 4);
  EXPECT_EQ(full_match.block_ids_per_layer, BlockIds({{10, 11, 12, 13}}));
}

TEST(CompressedRadixCacheTreeTest, EmptyAndPartialOnlyInputsReturnRoot) {
  base::CompressedRadixCacheTree tree(/*block_size=*/2, /*num_layers=*/2);

  const auto empty_insert = tree.insert(Tokens({}), BlockIds({{}, {}}));
  EXPECT_EQ(empty_insert.leaf, tree.root());
  EXPECT_EQ(tree.node_count(), 1);

  const auto partial_insert = tree.insert(Tokens({1}), BlockIds({{}, {}}));
  EXPECT_EQ(partial_insert.leaf, tree.root());
  EXPECT_EQ(tree.node_count(), 1);

  const auto empty_match = tree.match_prefix(Tokens({}));
  const auto partial_match = tree.match_prefix(Tokens({1}));
  EXPECT_EQ(empty_match.matched_leaf, tree.root());
  EXPECT_EQ(partial_match.matched_leaf, tree.root());
  EXPECT_EQ(empty_match.matched_blocks, 0);
  EXPECT_EQ(partial_match.matched_blocks, 0);
}

TEST(CompressedRadixCacheTreeTest, ValidatesConstructionAndInputErrors) {
  EXPECT_THROW((base::CompressedRadixCacheTree(/*block_size=*/0, /*num_layers=*/1)),
               std::invalid_argument);
  EXPECT_THROW((base::CompressedRadixCacheTree(/*block_size=*/2, /*num_layers=*/0)),
               std::invalid_argument);

  base::CompressedRadixCacheTree tree(/*block_size=*/2, /*num_layers=*/2);

  EXPECT_THROW(tree.insert(Tokens({1, 2, 3, 4}), BlockIds({{10, 11}})),
               std::invalid_argument);
  EXPECT_THROW(tree.insert(Tokens({1, 2, 3, 4}), BlockIds({{10}, {20, 21}})),
               std::invalid_argument);
  EXPECT_THROW(tree.unpin_path(tree.root()), std::logic_error);
  EXPECT_THROW(tree.pin_path(nullptr), std::invalid_argument);
  EXPECT_THROW(tree.unpin_path(nullptr), std::invalid_argument);
}
