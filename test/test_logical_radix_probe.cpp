#include <gtest/gtest.h>
#include "base/compressed_radix_cache_tree.h"

TEST(LogicalRadixProbeTest, ShortProbePreservesTreeAndWidePageIds) {
  base::LogicalRadixCacheTree tree(2, 1);
  const uint64_t page = UINT64_C(1) << 48;
  tree.insert({1,2,3,4,5,6}, {{page,page+1,page+2}});
  const auto tick=tree.access_tick();
  const auto nodes=tree.node_count();
  const auto hit=tree.probe_prefix({1,2,3,99});
  EXPECT_EQ(hit.matched_tokens,2);
  EXPECT_EQ(hit.block_ids_per_layer[0], (std::vector<uint64_t>{page}));
  EXPECT_EQ(tree.access_tick(),tick);
  EXPECT_EQ(tree.node_count(),nodes);
  EXPECT_EQ(tree.split_count(),0);
  const auto adopted=tree.match_prefix({1,2});
  EXPECT_EQ(adopted.matched_tokens,2);
  EXPECT_EQ(tree.split_count(),1);
  tree.pin_path(adopted.matched_leaf);
  tree.unpin_path(adopted.matched_leaf);
}
