#include <gtest/gtest.h>
#include <memory>
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

TEST(KVCacheManagerTest, PrefixCacheRetainsSharedBlocksAcrossRequests) {
  constexpr int32_t kBlockSize = 4;
  constexpr int32_t kNumLayers = 2;

  base::KVCacheManager manager(
      kBlockSize, kNumLayers, make_allocators(kNumLayers, 16, kBlockSize));

  const std::vector<int32_t> prompt_tokens = {
      10, 11, 12, 13, 14, 15, 16, 17,
  };

  const base::RequestId first = manager.register_request_with_prompt(prompt_tokens);
  ASSERT_TRUE(manager.append_slots(first, static_cast<int32_t>(prompt_tokens.size())));
  manager.publish_prefix_cache(first, prompt_tokens);
  manager.free_request(first);

  const auto free_blocks_after_publish = manager.num_free_blocks(0);
  EXPECT_LT(free_blocks_after_publish, 16);

  const base::RequestId second = manager.register_request_with_prompt(prompt_tokens);
  EXPECT_EQ(manager.get_context_len(second), 4);
  EXPECT_EQ(manager.prefix_cache_stats().cache_hits, 1);
  EXPECT_EQ(manager.prefix_cache_stats().tokens_reused, 4);

  manager.free_request(second);
  // The prefix cache keeps one persistent reference to the shared prefix
  // blocks so a future request can reuse them without rebuilding KV.
  EXPECT_EQ(manager.num_free_blocks(0), 15);

  manager.clear_prefix_cache();
  EXPECT_EQ(manager.num_free_blocks(0), 16);
}
