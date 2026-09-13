#include <gtest/gtest.h>
#include <cstring>
#include <memory>
#include <initializer_list>
#include <vector>
#include "base/block_allocator.h"
#include "base/kv_cache_manager.h"
#include "data/multimodal_prefix_key.h"
#include "serving/prefix_builder.h"

namespace base {
struct KVCacheManagerTestPeer {
  static void remove_residency(KVCacheManager* manager, const std::vector<int32_t>& tokens,
                               size_t page) {
    const auto match = manager->radix_cache_.probe_prefix(tokens);
    manager->page_directory_->erase(match.block_ids_per_layer[0].at(page));
  }
  static RequestId advance_to_final_generation(KVCacheManager* manager, RequestId id) {
    auto& slot = manager->lookup_request_slot(id);
    slot.generation = KVCacheManager::kMaxRequestGeneration;
    return KVCacheManager::encode_request_id(KVCacheManager::decode_slot_index(id),
                                           slot.generation);
  }
  static size_t recovery_objects(const KVCacheManager* manager) {
    return manager->recovery_graph_.size();
  }
  static void invalidate_recovery_schema_version(KVCacheManager* manager) {
    cache::RecoveryObject dependency;
    dependency.object_id = "kv-schema";
    dependency.version = "stale-schema-version";
    dependency.kind = cache::RecoveryKind::kPreserveUntilReleased;
    dependency.gpu_serviceable = true;
    ASSERT_TRUE(manager->recovery_graph_.publish(std::move(dependency)));
  }
};
}  // namespace base

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
  EXPECT_TRUE(manager->commit_kv(request_id, manager->get_context_len(request_id)));
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

TEST(KVCacheManagerTest, ReusesOneSlotFor100000Lifetimes) {
  base::KVCacheManager manager(4, 1, make_allocators(1, 2, 4));
  base::RequestId previous = -1;
  for (int i = 0; i < 100000; ++i) {
    const auto id = manager.register_request();
    ASSERT_TRUE(manager.is_valid_request(id));
    ASSERT_FALSE(manager.is_valid_request(previous));
    ASSERT_GT(id, previous);
    ASSERT_EQ(manager.request_slot_capacity(), 1);
    ASSERT_TRUE(manager.append_slot(id));
    manager.free_request(id);
    ASSERT_FALSE(manager.is_valid_request(id));
    previous = id;
  }
  EXPECT_GT(previous, static_cast<int64_t>(INT32_MAX));
  EXPECT_EQ(manager.num_active_requests(), 0);
}

TEST(KVCacheManagerTest, RetiresExhaustedSlotWithoutHandleReuse) {
  base::KVCacheManager manager(4, 1, make_allocators(1, 2, 4));
  const auto original = manager.register_request();
  const auto last = base::KVCacheManagerTestPeer::advance_to_final_generation(&manager,
                                                                            original);
  ASSERT_TRUE(manager.is_valid_request(last));
  ASSERT_FALSE(manager.is_valid_request(original));
  manager.free_request(last);
  const auto next = manager.register_request();
  EXPECT_TRUE(manager.is_valid_request(next));
  EXPECT_FALSE(manager.is_valid_request(last));
  EXPECT_FALSE(manager.is_valid_request(original));
  EXPECT_EQ(manager.request_slot_capacity(), 2);
  manager.free_request(next);
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
  ASSERT_TRUE(manager.commit_kv(divergent, manager.get_context_len(divergent)));
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

  ASSERT_TRUE(manager.commit_kv(request, manager.get_context_len(request)));
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
  ASSERT_TRUE(manager.commit_kv(request, manager.get_context_len(request)));
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

TEST(KVCacheManagerTest, LogicalPagesResolveDifferentLayerBlockIdsAfterSplit) {
  base::KVCacheManager manager(4, 2, make_allocators(2, 8, 4));
  const auto occupied = manager.allocator_mut(1).allocate();
  const auto first = manager.register_request();
  const auto prompt = Tokens({1,2,3,4,5,6,7,8,9});
  ASSERT_TRUE(manager.append_slots(first, 9));
  const auto layer0 = manager.get_block_ids(first, 0);
  const auto layer1 = manager.get_block_ids(first, 1);
  ASSERT_NE(layer0.front(), layer1.front());
  ASSERT_TRUE(manager.commit_kv(first, manager.get_context_len(first)));
  manager.publish_radix_cache(first, prompt);
  manager.free_request(first);
  const auto short_hit = manager.register_request_with_radix_cache(Tokens({1,2,3,4,99}));
  EXPECT_EQ(manager.get_context_len(short_hit), 4);
  EXPECT_EQ(manager.get_block_ids(short_hit, 0).front(), layer0.front());
  EXPECT_EQ(manager.get_block_ids(short_hit, 1).front(), layer1.front());
  manager.free_request(short_hit);
  const auto full_hit = manager.register_request_with_radix_cache(prompt);
  EXPECT_EQ(manager.get_context_len(full_hit), 8);
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(manager.get_block_ids(full_hit, 0)[i], layer0[i]);
    EXPECT_EQ(manager.get_block_ids(full_hit, 1)[i], layer1[i]);
  }
  manager.free_request(full_hit);
  manager.clear_radix_cache();
  manager.allocator_mut(1).free(occupied);
  EXPECT_EQ(manager.num_free_blocks(0), 8);
  EXPECT_EQ(manager.num_free_blocks(1), 8);
}

TEST(KVCacheManagerTest, PublishesOnlyCommittedFullPages) {
  base::KVCacheManager manager(4, 1, make_allocators(1, 8, 4));
  const auto source = manager.register_request();
  const auto prompt = Tokens({1,2,3,4,5,6,7,8,9});
  ASSERT_TRUE(manager.append_slots(source, 9));
  EXPECT_EQ(manager.kv_committed_tokens(source), 0);
  manager.publish_radix_cache(source, prompt);
  const auto miss = manager.register_request_with_radix_cache(prompt);
  EXPECT_EQ(manager.get_context_len(miss), 0);
  manager.free_request(miss);
  EXPECT_FALSE(manager.commit_kv(source, 10));
  ASSERT_TRUE(manager.commit_kv(source, 7));
  manager.publish_radix_cache(source, prompt);
  EXPECT_EQ(manager.published_full_tokens(source), 4);
  EXPECT_FALSE(manager.commit_kv(source, 6));
  const auto hit = manager.register_request_with_radix_cache(prompt);
  EXPECT_EQ(manager.get_context_len(hit), 4);
  EXPECT_EQ(manager.kv_committed_tokens(hit), 4);
  manager.free_request(hit);
  manager.free_request(source);
  EXPECT_FALSE(manager.commit_kv(source, 9));
}

TEST(KVCacheManagerTest, ComputeLeaseSurvivesRequestCancellation) {
  base::KVCacheManager manager(4, 1, make_allocators(1, 1, 4));
  const auto source = manager.register_request();
  ASSERT_TRUE(manager.append_slots(source, 4));
  std::vector<base::BlockLease> pins;
  ASSERT_TRUE(manager.acquire_compute_leases(source, &pins));
  manager.free_request(source);
  EXPECT_EQ(manager.num_free_blocks(0), 0);
  const auto replacement = manager.register_request();
  EXPECT_FALSE(manager.append_slot(replacement));
  EXPECT_FALSE(manager.commit_kv(source, 4));
  pins.clear();
  ASSERT_TRUE(manager.append_slot(replacement));
  manager.free_request(replacement);
}

TEST(KVCacheManagerTest, SharedExternalPrefixCanServeSequentialDecodeRequests) {
  base::KVCacheManager manager(4, 1, make_allocators(1, 4, 4));
  const auto source = manager.register_request();
  ASSERT_TRUE(manager.append_slots(source, 5));
  ASSERT_TRUE(manager.commit_kv(source, 5));
  base::ExternalKVRequestState state;
  ASSERT_TRUE(manager.export_external_request(source, &state));
  for (int repetition = 0; repetition < 2; ++repetition) {
    base::RequestId decode = -1;
    ASSERT_TRUE(manager.restore_external_shared_request(state, &decode));
    EXPECT_EQ(manager.get_context_len(decode), 5);
    manager.free_request(decode);
    EXPECT_TRUE(manager.is_valid_request(source));
  }
  manager.free_request(source);
  EXPECT_EQ(manager.num_free_blocks(0), 4);
}

TEST(KVCacheManagerTest, AllocationIntentHasNoSideEffectsAndRejectsStaleProgress) {
  base::KVCacheManager manager(4, 1, make_allocators(1, 2, 4));
  const auto id = manager.register_request();
  base::KVAllocationIntent first, second;
  ASSERT_TRUE(manager.plan_allocation(id, 4, &first));
  ASSERT_TRUE(manager.plan_allocation(id, 1, &second));
  EXPECT_EQ(manager.num_free_blocks(0), 2);
  EXPECT_EQ(manager.get_context_len(id), 0);
  ASSERT_TRUE(manager.admit_allocation(second));
  EXPECT_FALSE(manager.admit_allocation(first));
  EXPECT_EQ(manager.get_context_len(id), 1);
  manager.free_request(id);
  const auto next = manager.register_request();
  EXPECT_FALSE(manager.admit_allocation(second));
  EXPECT_EQ(manager.get_context_len(next), 0);
  EXPECT_EQ(manager.num_free_blocks(0), 2);
  manager.free_request(next);
}

TEST(KVCacheManagerTest, MissingResidencyUsesOnlyContiguousReadyPrefix) {
  base::KVCacheManager manager(4, 1, make_allocators(1, 8, 4));
  const auto prompt = Tokens({1,2,3,4,5,6,7,8,9});
  PublishPromptAndFree(&manager, prompt);
  base::KVCacheManagerTestPeer::remove_residency(&manager, prompt, 1);
  const auto hit = manager.register_request_with_radix_cache(prompt);
  EXPECT_EQ(manager.get_context_len(hit), 4);
  manager.free_request(hit);
  base::KVCacheManagerTestPeer::remove_residency(&manager, prompt, 0);
  const auto nodes = manager.radix_cache_node_count();
  const auto splits = manager.radix_cache_split_count();
  const auto miss = manager.register_request_with_radix_cache(Tokens({1,2,3,4,99}));
  EXPECT_EQ(manager.get_context_len(miss), 0);
  EXPECT_EQ(manager.radix_cache_node_count(), nodes);
  EXPECT_EQ(manager.radix_cache_split_count(), splits);
  manager.free_request(miss);
  manager.clear_radix_cache();
  EXPECT_EQ(manager.num_free_blocks(0), 8);
}

TEST(KVCacheManagerTest, PartialTailForkSharesThenCopiesOnFirstAppend) {
  base::KVCacheManager manager(4, 2, make_allocators(2, 24, 4));
  const auto source = manager.register_request();
  ASSERT_TRUE(manager.append_slots(source, 6));
  ASSERT_TRUE(manager.commit_kv(source, 6));
  for (int layer = 0; layer < 2; ++layer) {
    const auto tail = manager.get_block_ids(source, layer).back();
    auto payload = manager.allocator_mut(layer).get_block_payload_ptrs(tail);
    std::memset(payload.key, 0x30 + layer, payload.key_value_bytes);
    std::memset(payload.value, 0x50 + layer, payload.key_value_bytes);
  }

  std::vector<base::RequestId> branches;
  const base::BranchTokenBoundaries boundaries{6, 1, 0};
  for (int i = 0; i < 4; ++i) {
    base::RequestId branch = -1;
    base::BranchSnapshot snapshot;
    ASSERT_TRUE(manager.fork_request(source, boundaries, true, 77, &branch, &snapshot));
    EXPECT_EQ(snapshot.valid_tokens, 6);
    EXPECT_EQ(snapshot.first_token, 77);
    EXPECT_EQ(manager.branch_boundaries(branch).sampled_tokens, 1);
    EXPECT_GT(manager.branch_generation(branch), 0u);
    branches.push_back(branch);
  }
  for (int layer = 0; layer < 2; ++layer) {
    const int tail = manager.get_block_ids(source, layer).back();
    EXPECT_EQ(manager.allocator(layer).ref_count(tail), 5);
  }

  const auto shared_full = manager.get_block_ids(source, 0).front();
  const auto shared_tail = manager.get_block_ids(source, 0).back();
  for (auto branch : branches) {
    ASSERT_TRUE(manager.append_slot(branch));
    EXPECT_EQ(manager.get_block_ids(branch, 0).front(), shared_full);
    EXPECT_NE(manager.get_block_ids(branch, 0).back(), shared_tail);
    EXPECT_EQ(manager.append_stats(branch).cow_pages, 2);
    EXPECT_EQ(manager.append_stats(branch).cow_bytes, 512);
    const auto copied = manager.allocator(0).get_block_payload_ptrs(
        manager.get_block_ids(branch, 0).back());
    EXPECT_EQ(static_cast<const uint8_t*>(copied.key)[0], 0x30);
    EXPECT_EQ(static_cast<const uint8_t*>(copied.value)[0], 0x50);
  }
  EXPECT_EQ(manager.allocator(0).ref_count(shared_full), 5);
  EXPECT_EQ(manager.allocator(0).ref_count(shared_tail), 1);

  manager.free_request(branches[0]);
  manager.free_request(branches[2]);
  EXPECT_TRUE(manager.is_valid_request(branches[1]));
  EXPECT_TRUE(manager.is_valid_request(branches[3]));
  EXPECT_EQ(manager.allocator(0).ref_count(shared_full), 3);
  manager.free_request(source);
  EXPECT_EQ(manager.allocator(0).ref_count(shared_full), 2);
  manager.free_request(branches[1]);
  manager.free_request(branches[3]);
  EXPECT_EQ(manager.num_free_blocks(0), 24);
  EXPECT_EQ(manager.num_free_blocks(1), 24);
}

TEST(KVCacheManagerTest, PageAlignedForkNeedsNoCowAndSamplesIndependently) {
  base::KVCacheManager manager(4, 1, make_allocators(1, 12, 4));
  const auto source = manager.register_request();
  ASSERT_TRUE(manager.append_slots(source, 8));
  ASSERT_TRUE(manager.commit_kv(source, 8));
  base::RequestId inherited = -1;
  base::RequestId independent = -1;
  ASSERT_TRUE(manager.fork_request(source, {8, 1, 0}, true, 99, &inherited));
  ASSERT_TRUE(manager.fork_request(source, {8, 0, 1}, false, -1, &independent));
  EXPECT_EQ(manager.get_block_ids(inherited, 0), manager.get_block_ids(source, 0));
  EXPECT_EQ(manager.get_block_ids(independent, 0), manager.get_block_ids(source, 0));
  ASSERT_TRUE(manager.append_slot(inherited));
  ASSERT_TRUE(manager.append_slot(independent));
  EXPECT_EQ(manager.append_stats(inherited).cow_bytes, 0);
  EXPECT_EQ(manager.append_stats(independent).cow_bytes, 0);
  EXPECT_NE(manager.get_block_ids(inherited, 0).back(),
            manager.get_block_ids(independent, 0).back());
  manager.free_request(source);
  manager.free_request(independent);
  manager.free_request(inherited);
  EXPECT_EQ(manager.num_free_blocks(0), 12);
}

TEST(KVCacheManagerTest, MultimodalRadixKeyIncludesFullImageIdentity) {
  base::KVCacheManager manager(4, 1, make_allocators(1, 16, 4));
  std::vector<int32_t> placeholders(12, 151655);
  data::ContentId image_a;
  data::ContentId image_b;
  image_a.digest[0] = 1;
  image_b.digest[31] = 1;
  std::vector<int32_t> key_a;
  std::vector<int32_t> key_b;
  ASSERT_TRUE(data::MakeMultimodalRadixTokens(
      placeholders, {{0, 8, image_a}}, &key_a));
  ASSERT_TRUE(data::MakeMultimodalRadixTokens(
      placeholders, {{0, 8, image_b}}, &key_b));
  EXPECT_NE(key_a, key_b);

  const auto source = manager.register_request();
  ASSERT_TRUE(manager.append_slots(source, 12));
  ASSERT_TRUE(manager.commit_kv(source, 12));
  manager.publish_radix_cache(source, key_a);
  manager.free_request(source);
  const auto same_image = manager.register_request_with_radix_cache(key_a);
  EXPECT_EQ(manager.get_context_len(same_image), 8);
  manager.free_request(same_image);
  const auto different_image = manager.register_request_with_radix_cache(key_b);
  EXPECT_EQ(manager.get_context_len(different_image), 0);
  manager.free_request(different_image);
  manager.clear_radix_cache();
}

TEST(KVCacheManagerTest, PrefixBuilderPublishesAfterTemporaryRequestExits) {
  base::KVCacheManager manager(4, 1, make_allocators(1, 12, 4));
  serving::PrefixBuilder builder(&manager);
  const auto prefix = Tokens({1,2,3,4,5,6,7,8,9});
  ASSERT_TRUE(builder.build(prefix, [&](base::RequestId task) {
    EXPECT_EQ(manager.get_context_len(task), 9);
    return true;
  }));
  EXPECT_EQ(builder.stats().tasks_succeeded, 1);
  EXPECT_EQ(builder.stats().computed_tokens, 9);
  EXPECT_EQ(manager.num_active_requests(), 0);
  const auto later = manager.register_request_with_radix_cache(prefix);
  EXPECT_EQ(manager.get_context_len(later), 8);
  manager.free_request(later);
  manager.clear_radix_cache();
  EXPECT_EQ(manager.num_free_blocks(0), 12);
}

TEST(KVCacheManagerTest, HostOnlyPrefixRestoresAllLayersBeforeAdoption) {
  base::HostCacheConfig host{true, 4096, 8, 2};
  base::KVCacheManager manager(4, 2, make_allocators(2, 8, 4), host);
  const auto prompt = Tokens({1,2,3,4,5,6,7,8,9});
  PublishPromptAndFree(&manager, prompt);
  EXPECT_EQ(manager.demote_radix_cache_to_host(), 2);
  EXPECT_EQ(manager.num_free_blocks(0), 8);
  EXPECT_EQ(manager.num_free_blocks(1), 8);
  const auto request = manager.register_request_with_radix_cache(prompt);
  EXPECT_EQ(manager.request_restore_state(request), base::RequestRestoreState::kPending);
  EXPECT_EQ(manager.get_context_len(request), 0);
  EXPECT_GT(manager.service_cache_transfers(), 0);
  EXPECT_EQ(manager.request_restore_state(request), base::RequestRestoreState::kReady);
  EXPECT_EQ(manager.get_context_len(request), 8);
  EXPECT_EQ(manager.kv_committed_tokens(request), 8);
  EXPECT_EQ(manager.published_full_tokens(request), 8);
  EXPECT_EQ(manager.radix_cache_stats().host_restored_blocks, 2);
  EXPECT_EQ(manager.radix_cache_stats().recovery_graph_demote_decisions, 2);
  EXPECT_EQ(manager.radix_cache_stats().recovery_graph_committed_demotions, 2);
  EXPECT_EQ(manager.radix_cache_stats().recovery_graph_restores, 2);
  EXPECT_GE(base::KVCacheManagerTestPeer::recovery_objects(&manager), 3u);
  EXPECT_EQ(manager.radix_cache_stats().tokens_reused, 8);
  for (int layer = 0; layer < 2; ++layer) {
    EXPECT_EQ(manager.get_block_ids(request, layer).size(), 2u);
  }
  manager.free_request(request);
  manager.clear_radix_cache();
  EXPECT_EQ(manager.num_free_blocks(0), 8);
  EXPECT_EQ(manager.num_free_blocks(1), 8);
}

TEST(KVCacheManagerTest, SharedRestoreTargetsAreReservedOnceForThirtyTwoRequests) {
  base::HostCacheConfig host{true, 4096, 8, 2};
  base::KVCacheManager manager(4, 1, make_allocators(1, 4, 4), host);
  const auto prompt = Tokens({1,2,3,4,5,6,7,8,9});
  PublishPromptAndFree(&manager, prompt);
  ASSERT_EQ(manager.demote_radix_cache_to_host(), 2);
  std::vector<base::RequestId> requests;
  for (int i = 0; i < 32; ++i) {
    requests.push_back(manager.register_request_with_radix_cache(prompt));
    const auto state = manager.request_restore_state(requests.back());
    ASSERT_TRUE(state == base::RequestRestoreState::kPending ||
                state == base::RequestRestoreState::kReady);
    // The synchronous CPU oracle may materialize a target during submit, but
    // it must never reserve once per logical waiter.
    ASSERT_LE(manager.restore_reserved_target_pages(), 2u);
  }
  ASSERT_TRUE(manager.drain_cache_transfers());
  EXPECT_EQ(manager.restore_reserved_target_pages(), 0u);
  EXPECT_EQ(manager.transfer_scheduler_stats()->physical_submissions, 4u);
  for (auto id : requests) {
    EXPECT_EQ(manager.request_restore_state(id), base::RequestRestoreState::kReady);
    EXPECT_EQ(manager.get_context_len(id), 8);
    manager.free_request(id);
  }
}

TEST(KVCacheManagerTest, ReservedRestoreTargetsRejectGrowthWithoutPartialAllocation) {
  base::HostCacheConfig host{true, 4096, 4, 1};
  base::KVCacheManager manager(4, 1, make_allocators(1, 4, 4), host);
  const auto prompt = Tokens({1,2,3,4,5});
  PublishPromptAndFree(&manager, prompt);
  ASSERT_EQ(manager.demote_radix_cache_to_host(), 1);
  const auto restoring = manager.register_request_with_radix_cache(prompt);
  ASSERT_EQ(manager.restore_reserved_target_pages(), 1u);

  const auto first = manager.register_request();
  const auto second = manager.register_request();
  ASSERT_TRUE(manager.append_slots(first, 8));
  ASSERT_TRUE(manager.append_slots(second, 4));
  EXPECT_EQ(manager.schedulable_free_blocks(0), 0);
  EXPECT_FALSE(manager.append_slots(second, 4));
  EXPECT_EQ(manager.get_context_len(second), 4);
  EXPECT_LE(manager.restore_reserved_target_pages(), 1u);

  manager.free_request(first);
  ASSERT_TRUE(manager.drain_cache_transfers());
  EXPECT_EQ(manager.request_restore_state(restoring), base::RequestRestoreState::kReady);
  manager.free_request(second);
  manager.free_request(restoring);
}

TEST(KVCacheManagerTest, CancelledRestoreDrainsWithoutReusingDestinationEarly) {
  base::HostCacheConfig host{true, 4096, 4, 1};
  base::KVCacheManager manager(4, 1, make_allocators(1, 2, 4), host);
  const auto prompt = Tokens({1,2,3,4,5});
  PublishPromptAndFree(&manager, prompt);
  ASSERT_EQ(manager.demote_radix_cache_to_host(), 1);
  const auto request = manager.register_request_with_radix_cache(prompt);
  ASSERT_EQ(manager.request_restore_state(request), base::RequestRestoreState::kPending);
  manager.free_request(request);
  // The cancelled physical job is detached from the recycled request slot and
  // consumed by the owner completion service.
  EXPECT_GT(manager.service_cache_transfers(), 0);
  const auto replacement = manager.register_request();
  EXPECT_TRUE(manager.append_slots(replacement, 4));
  manager.free_request(replacement);
}

TEST(KVCacheManagerTest, AllocationPressureDemotesWithoutErasingLogicalPrefix) {
  base::HostCacheConfig host{true, 4096, 2, 1};
  base::KVCacheManager manager(4, 1, make_allocators(1, 2, 4), host);
  const auto cached_prompt = Tokens({1,2,3,4,5});
  PublishPromptAndFree(&manager, cached_prompt);
  const auto pressure = manager.register_request();
  ASSERT_TRUE(manager.append_slots(pressure, 4));
  EXPECT_EQ(manager.num_free_blocks(0), 0);
  // This allocation invokes the host demotion policy and retries only after
  // the CPU oracle has established physical completion.
  ASSERT_TRUE(manager.append_slots(pressure, 4));
  EXPECT_EQ(manager.radix_cache_stats().host_demoted_blocks, 1);
  EXPECT_EQ(manager.radix_cache_stats().recovery_graph_demote_decisions, 1);
  EXPECT_EQ(manager.radix_cache_stats().recovery_graph_committed_demotions, 1);
  EXPECT_EQ(manager.radix_cache_node_count(), 2);
  manager.free_request(pressure);
  const auto restored = manager.register_request_with_radix_cache(cached_prompt);
  EXPECT_EQ(manager.request_restore_state(restored), base::RequestRestoreState::kPending);
  manager.service_cache_transfers();
  EXPECT_EQ(manager.request_restore_state(restored), base::RequestRestoreState::kReady);
  EXPECT_EQ(manager.get_context_len(restored), 4);
  manager.free_request(restored);
}

TEST(KVCacheManagerTest, RecoveryGraphProtectsActiveConsumerOnActualDemotionPath) {
  base::HostCacheConfig host{true, 4096, 4, 1};
  base::KVCacheManager manager(4, 1, make_allocators(1, 4, 4), host);
  const auto prompt = Tokens({1,2,3,4,5});
  PublishPromptAndFree(&manager, prompt);
  const auto active = manager.register_request_with_radix_cache(prompt);
  ASSERT_EQ(manager.get_context_len(active), 4);
  EXPECT_EQ(manager.demote_radix_cache_to_host(), 0);
  EXPECT_GE(manager.radix_cache_stats().recovery_graph_keep_decisions, 1);
  EXPECT_EQ(manager.radix_cache_stats().host_demoted_blocks, 0);
  manager.free_request(active);
  EXPECT_EQ(manager.demote_radix_cache_to_host(), 1);
  EXPECT_TRUE(manager.drain_cache_transfers());
  EXPECT_EQ(manager.radix_cache_stats().recovery_graph_committed_demotions, 1);
}

TEST(KVCacheManagerTest, RecoveryGraphKeepsLastReplicaWhenHostIsFull) {
  // One packed page is 256 bytes for this schema; 128 bytes cannot admit it.
  base::HostCacheConfig host{true, 128, 4, 1};
  base::KVCacheManager manager(4, 1, make_allocators(1, 4, 4), host);
  const auto prompt = Tokens({1,2,3,4,5});
  PublishPromptAndFree(&manager, prompt);
  EXPECT_EQ(manager.demote_radix_cache_to_host(), 0);
  EXPECT_GE(manager.radix_cache_stats().recovery_graph_keep_decisions, 1);
  EXPECT_EQ(manager.num_free_blocks(0), 3);
}

TEST(KVCacheManagerTest, RecoveryGraphRejectsVersionInvalidationOnDemotionPath) {
  base::HostCacheConfig host{true, 4096, 4, 1};
  base::KVCacheManager manager(4, 1, make_allocators(1, 4, 4), host);
  const auto prompt = Tokens({1,2,3,4,5});
  PublishPromptAndFree(&manager, prompt);
  base::KVCacheManagerTestPeer::invalidate_recovery_schema_version(&manager);
  EXPECT_EQ(manager.demote_radix_cache_to_host(), 0);
  EXPECT_EQ(manager.radix_cache_stats().recovery_graph_publish_rejections, 1);
  EXPECT_EQ(manager.num_free_blocks(0), 3);
}

TEST(KVCacheManagerTest, ClearRetiresActualPageRecoveryObjects) {
  base::HostCacheConfig host{true, 4096, 4, 1};
  base::KVCacheManager manager(4, 1, make_allocators(1, 4, 4), host);
  PublishPromptAndFree(&manager, Tokens({1,2,3,4,5}));
  ASSERT_EQ(manager.demote_radix_cache_to_host(), 1);
  ASSERT_TRUE(manager.drain_cache_transfers());
  ASSERT_EQ(manager.recovery_dependency_object_count(), 2u);  // schema + page
  manager.clear_radix_cache();
  EXPECT_EQ(manager.recovery_dependency_object_count(), 1u);  // immutable schema only
  EXPECT_EQ(manager.radix_cache_node_count(), 1);
  EXPECT_EQ(manager.num_free_blocks(0), 4);
}
