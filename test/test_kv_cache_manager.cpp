#include <gtest/gtest.h>
#include <memory>
#include <initializer_list>
#include <vector>
#include "base/block_allocator.h"
#include "base/kv_cache_manager.h"

namespace {

std::vector<std::unique_ptr<base::BlockAllocator>> make_allocators(
    int32_t num_layers, int32_t num_blocks, int32_t block_size) {
  std::vector<std::unique_ptr<base::BlockAllocator>> allocators;
  allocators.reserve(num_layers);
  for (int32_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    allocators.push_back(std::make_unique<base::BlockAllocator>(
        num_blocks, block_size, 1, 8, base::DataType::kDataTypeFp32,
        base::DeviceType::kDeviceCPU));
  }
  return allocators;
}

std::vector<int32_t> Tokens(std::initializer_list<int32_t> values) {
  return std::vector<int32_t>(values);
}

void AppendPrompt(base::KVCacheManager* manager,
                  base::RequestId request_id,
                  const std::vector<int32_t>& prompt_tokens) {
  ASSERT_NE(manager, nullptr);
  ASSERT_TRUE(manager->append_slots(
      request_id, static_cast<int32_t>(prompt_tokens.size())));
  EXPECT_EQ(manager->get_context_len(request_id),
            static_cast<int32_t>(prompt_tokens.size()));
}

std::vector<int32_t> PublishPromptAndFree(
    base::KVCacheManager* manager,
    const std::vector<int32_t>& prompt_tokens) {
  const base::RequestId request_id = manager->register_request();
  AppendPrompt(manager, request_id, prompt_tokens);
  std::vector<int32_t> block_ids = manager->get_block_ids(request_id, 0);
  manager->publish_radix_cache(request_id, prompt_tokens);
  manager->free_request(request_id);
  return block_ids;
}

}  // namespace

TEST(KVCacheManagerTest, ReusesFreedSlotWithFreshGeneration) {
  constexpr int32_t kBlockSize = 4;
  constexpr int32_t kNumLayers = 2;

  base::KVCacheManager manager(
      kBlockSize, kNumLayers, make_allocators(kNumLayers, 8, kBlockSize));

  EXPECT_EQ(manager.request_slot_capacity(), 0);
  EXPECT_EQ(manager.num_active_requests(), 0);

  const base::RequestId first = manager.register_request();
  EXPECT_TRUE(manager.is_valid_request(first));
  EXPECT_EQ(manager.request_slot_capacity(), 1);
  EXPECT_EQ(manager.num_active_requests(), 1);

  ASSERT_TRUE(manager.append_slots(first, 3));
  EXPECT_EQ(manager.get_context_len(first), 3);

  manager.free_request(first);
  EXPECT_FALSE(manager.is_valid_request(first));
  EXPECT_EQ(manager.request_slot_capacity(), 1);
  EXPECT_EQ(manager.num_active_requests(), 0);

  const base::RequestId second = manager.register_request();
  EXPECT_TRUE(manager.is_valid_request(second));
  EXPECT_NE(first, second);
  EXPECT_EQ(manager.request_slot_capacity(), 1);
  EXPECT_EQ(manager.num_active_requests(), 1);
  EXPECT_EQ(manager.get_context_len(second), 0);

  ASSERT_TRUE(manager.append_slot(second));
  EXPECT_EQ(manager.get_context_len(second), 1);
}

TEST(KVCacheManagerTest, InvalidatesStaleAndMalformedHandles) {
  constexpr int32_t kBlockSize = 4;
  constexpr int32_t kNumLayers = 2;

  base::KVCacheManager manager(
      kBlockSize, kNumLayers, make_allocators(kNumLayers, 8, kBlockSize));

  const base::RequestId first = manager.register_request();
  manager.free_request(first);
  const base::RequestId second = manager.register_request();

  EXPECT_FALSE(manager.is_valid_request(-1));
  EXPECT_FALSE(manager.is_valid_request(first));
  EXPECT_TRUE(manager.is_valid_request(second));
  EXPECT_FALSE(manager.is_valid_request(second + 1));
}

TEST(KVCacheManagerTest, AppendFailsWhenAllocatorCapacityIsExhausted) {
  constexpr int32_t kBlockSize = 4;
  constexpr int32_t kNumLayers = 2;

  base::KVCacheManager manager(
      kBlockSize, kNumLayers, make_allocators(kNumLayers, 3, kBlockSize));

  const base::RequestId active = manager.register_request();
  ASSERT_TRUE(manager.append_slots(active, 12));
  EXPECT_EQ(manager.num_free_blocks(0), 0);

  EXPECT_FALSE(manager.append_slots(active, 4));
  EXPECT_EQ(manager.get_context_len(active), 12);
  EXPECT_EQ(manager.num_free_blocks(0), 0);

  manager.free_request(active);
  EXPECT_EQ(manager.num_free_blocks(0), 3);
}

TEST(KVCacheManagerTest, RadixCacheReusesLongestPrefixWithSeedTail) {
  constexpr int32_t kBlockSize = 2;
  constexpr int32_t kNumLayers = 2;

  base::KVCacheManager manager(
      kBlockSize, kNumLayers, make_allocators(kNumLayers, 10, kBlockSize));

  const auto prompt = Tokens({1, 2, 3, 4, 5, 6});
  const auto cached_blocks = PublishPromptAndFree(&manager, prompt);

  EXPECT_EQ(manager.num_free_blocks(0), 7);
  EXPECT_EQ(manager.radix_cache_node_count(), 2);
  EXPECT_EQ(manager.radix_cache_evictable_blocks(), 3);
  EXPECT_EQ(manager.radix_cache_stats().published_blocks, 3);

  const base::RequestId hit = manager.register_request_with_radix_cache(prompt);
  EXPECT_EQ(manager.get_context_len(hit), 4);
  EXPECT_EQ(manager.get_block_ids(hit, 0).size(), 2);
  EXPECT_EQ(manager.get_block_ids(hit, 0)[0], cached_blocks[0]);
  EXPECT_EQ(manager.get_block_ids(hit, 0)[1], cached_blocks[1]);
  EXPECT_EQ(manager.radix_cache_stats().cache_hits, 1);
  EXPECT_EQ(manager.radix_cache_stats().tokens_reused, 4);

  ASSERT_TRUE(manager.append_slots(hit, 2));
  EXPECT_EQ(manager.get_context_len(hit), 6);
  EXPECT_EQ(manager.get_block_ids(hit, 0).size(), 3);
  EXPECT_NE(manager.get_block_ids(hit, 0)[2], cached_blocks[2]);

  manager.free_request(hit);
  EXPECT_EQ(manager.num_free_blocks(0), 7);
}

TEST(KVCacheManagerTest, RadixCachePartialHitSplitsCompressedNode) {
  constexpr int32_t kBlockSize = 2;
  constexpr int32_t kNumLayers = 1;

  base::KVCacheManager manager(
      kBlockSize, kNumLayers, make_allocators(kNumLayers, 12, kBlockSize));

  const auto long_prompt = Tokens({1, 2, 3, 4, 5, 6, 7, 8});
  const auto cached_blocks = PublishPromptAndFree(&manager, long_prompt);
  ASSERT_EQ(cached_blocks.size(), 4);
  EXPECT_EQ(manager.radix_cache_node_count(), 2);

  const auto divergent_prompt = Tokens({1, 2, 3, 4, 5, 6, 99, 100});
  const base::RequestId divergent =
      manager.register_request_with_radix_cache(divergent_prompt);
  EXPECT_EQ(manager.get_context_len(divergent), 6);
  EXPECT_EQ(manager.radix_cache_split_count(), 1);
  EXPECT_EQ(manager.radix_cache_node_count(), 3);
  EXPECT_EQ(manager.get_block_ids(divergent, 0)[0], cached_blocks[0]);
  EXPECT_EQ(manager.get_block_ids(divergent, 0)[1], cached_blocks[1]);
  EXPECT_EQ(manager.get_block_ids(divergent, 0)[2], cached_blocks[2]);

  ASSERT_TRUE(manager.append_slots(divergent, 2));
  manager.publish_radix_cache(divergent, divergent_prompt);
  EXPECT_EQ(manager.radix_cache_stats().published_blocks, 5);
  EXPECT_EQ(manager.radix_cache_node_count(), 4);

  const int32_t divergent_tail_block = manager.get_block_ids(divergent, 0)[3];
  manager.free_request(divergent);

  const base::RequestId branch_hit =
      manager.register_request_with_radix_cache(divergent_prompt);
  EXPECT_EQ(manager.get_context_len(branch_hit), 6);
  EXPECT_EQ(manager.get_block_ids(branch_hit, 0)[0], cached_blocks[0]);
  EXPECT_EQ(manager.get_block_ids(branch_hit, 0)[1], cached_blocks[1]);
  EXPECT_EQ(manager.get_block_ids(branch_hit, 0)[2], cached_blocks[2]);
  EXPECT_NE(manager.get_block_ids(branch_hit, 0).back(), divergent_tail_block);
  manager.free_request(branch_hit);
}

TEST(KVCacheManagerTest, RadixCachePublishKeepsBlocksResidentAfterRequestFree) {
  constexpr int32_t kBlockSize = 2;
  constexpr int32_t kNumLayers = 2;

  base::KVCacheManager manager(
      kBlockSize, kNumLayers, make_allocators(kNumLayers, 6, kBlockSize));

  const auto prompt = Tokens({10, 11, 12, 13});
  const base::RequestId request = manager.register_request();
  AppendPrompt(&manager, request, prompt);
  EXPECT_EQ(manager.num_free_blocks(0), 4);

  manager.publish_radix_cache(request, prompt);
  manager.free_request(request);

  EXPECT_EQ(manager.num_free_blocks(0), 4);
  EXPECT_EQ(manager.radix_cache_evictable_blocks(), 2);

  manager.clear_radix_cache();
  EXPECT_EQ(manager.num_free_blocks(0), 6);
  EXPECT_EQ(manager.radix_cache_node_count(), 1);
  EXPECT_EQ(manager.radix_cache_evictable_blocks(), 0);
}

TEST(KVCacheManagerTest, RadixCacheCanBeDisabledExplicitly) {
  constexpr int32_t kBlockSize = 2;
  constexpr int32_t kNumLayers = 1;

  base::KVCacheManager manager(
      kBlockSize, kNumLayers, make_allocators(kNumLayers, 8, kBlockSize));
  manager.set_radix_cache_enabled(false);

  EXPECT_FALSE(manager.radix_cache_enabled());

  const auto prompt = Tokens({1, 2, 3, 4});
  const base::RequestId request = manager.register_request_with_radix_cache(prompt);
  EXPECT_EQ(manager.get_context_len(request), 0);

  AppendPrompt(&manager, request, prompt);
  manager.publish_radix_cache(request, prompt);
  manager.free_request(request);

  EXPECT_EQ(manager.radix_cache_node_count(), 1);
  EXPECT_EQ(manager.radix_cache_stats().publish_requests, 0);
  EXPECT_EQ(manager.radix_cache_stats().published_blocks, 0);

  const base::RequestId miss = manager.register_request_with_radix_cache(prompt);
  EXPECT_EQ(manager.get_context_len(miss), 0);
  EXPECT_EQ(manager.radix_cache_stats().cache_hits, 0);
  manager.free_request(miss);
}

TEST(KVCacheManagerTest, RadixCacheDoesNotEvictActivePinnedPath) {
  constexpr int32_t kBlockSize = 2;
  constexpr int32_t kNumLayers = 1;

  base::KVCacheManager manager(
      kBlockSize, kNumLayers, make_allocators(kNumLayers, 4, kBlockSize));

  const auto prompt = Tokens({1, 2, 3, 4, 5, 6});
  PublishPromptAndFree(&manager, prompt);
  EXPECT_EQ(manager.num_free_blocks(0), 1);

  const base::RequestId active = manager.register_request_with_radix_cache(prompt);
  EXPECT_EQ(manager.get_context_len(active), 4);
  EXPECT_EQ(manager.radix_cache_evictable_blocks(), 1);

  const base::RequestId pressure = manager.register_request();
  EXPECT_FALSE(manager.append_slots(pressure, 6));
  EXPECT_EQ(manager.get_context_len(pressure), 0);
  EXPECT_EQ(manager.get_context_len(active), 4);
  EXPECT_EQ(manager.num_free_blocks(0), 2);
  EXPECT_EQ(manager.radix_cache_stats().evicted_blocks, 1);
  EXPECT_EQ(manager.radix_cache_evictable_blocks(), 0);

  manager.free_request(pressure);
  manager.free_request(active);
  EXPECT_EQ(manager.radix_cache_evictable_blocks(), 2);
}

TEST(KVCacheManagerTest, AppendUnderPressureEvictsColdRadixCache) {
  constexpr int32_t kBlockSize = 2;
  constexpr int32_t kNumLayers = 1;

  base::KVCacheManager manager(
      kBlockSize, kNumLayers, make_allocators(kNumLayers, 4, kBlockSize));

  PublishPromptAndFree(&manager, Tokens({1, 2, 3, 4, 5, 6}));
  EXPECT_EQ(manager.num_free_blocks(0), 1);
  EXPECT_EQ(manager.radix_cache_evictable_blocks(), 3);

  const base::RequestId request = manager.register_request();
  ASSERT_TRUE(manager.append_slots(request, 4));
  EXPECT_EQ(manager.get_context_len(request), 4);
  EXPECT_EQ(manager.num_free_blocks(0), 2);
  EXPECT_EQ(manager.radix_cache_stats().evictions, 1);
  EXPECT_EQ(manager.radix_cache_stats().evicted_blocks, 3);
  EXPECT_EQ(manager.radix_cache_node_count(), 1);

  manager.free_request(request);
  EXPECT_EQ(manager.num_free_blocks(0), 4);
}
