// KVCacheManager implementation
#include "base/kv_cache_manager.h"
#include <algorithm>
#include <glog/logging.h>

namespace base {

RequestId KVCacheManager::encode_request_id(uint32_t slot_idx,
                                            uint32_t generation) {
  CHECK_LT(slot_idx, kMaxRequestSlots) << "slot index overflow";
  CHECK_LE(generation, kMaxRequestGeneration) << "request generation overflow";
  const uint32_t raw = (generation << kRequestSlotBits) | slot_idx;
  CHECK_LE(raw, static_cast<uint32_t>(INT32_MAX));
  return static_cast<RequestId>(raw);
}
 
uint32_t KVCacheManager::decode_slot_index(RequestId id) {
  return static_cast<uint32_t>(id) & kRequestSlotMask;
}

uint32_t KVCacheManager::decode_generation(RequestId id) {
  return static_cast<uint32_t>(id) >> kRequestSlotBits;
}

KVCacheManager::KVCacheManager(int32_t block_size, int32_t num_layers,
                               std::vector<std::unique_ptr<BlockAllocator>> layer_allocators)
    : block_size_(block_size),
      num_layers_(num_layers),
      layer_allocators_(std::move(layer_allocators)),
      radix_cache_(block_size, num_layers) {
  CHECK_GT(block_size_, 0);
  CHECK_GT(num_layers_, 0);
  CHECK_EQ(static_cast<int32_t>(layer_allocators_.size()), num_layers_);
}

RequestId KVCacheManager::register_request() {
  uint32_t slot_idx = 0;
  RequestSlot* slot = nullptr;
  if (!free_slot_indices_.empty()) {
    slot_idx = free_slot_indices_.back();
    free_slot_indices_.pop_back();
    slot = &request_slots_[slot_idx];
    CHECK(!slot->active);
    CHECK_LT(slot->generation, kMaxRequestGeneration)
        << "request slot generation overflow at slot " << slot_idx;
    slot->generation++;
    slot->manager.clear();
    slot->radix_cache_leaf = nullptr;
    slot->radix_cache_shared_tokens = 0;
  } else {
    CHECK_LT(request_slots_.size(), static_cast<size_t>(kMaxRequestSlots))
        << "request slot capacity exhausted";
    slot_idx = static_cast<uint32_t>(request_slots_.size());
    request_slots_.emplace_back(block_size_, num_layers_);
    slot = &request_slots_.back();
  }

  slot->active = true;
  num_active_requests_++;
  return encode_request_id(slot_idx, slot->generation);
}

RequestId KVCacheManager::register_request_with_radix_cache(
    const std::vector<int32_t>& prompt_tokens) {
  const RequestId id = register_request();
  RequestSlot& slot = lookup_request_slot(id);
  maybe_attach_radix_cache(&slot, prompt_tokens);
  return id;
}

void KVCacheManager::free_request(RequestId id) {
  uint32_t slot_idx = 0;
  auto& slot = lookup_request_slot(id, &slot_idx);
  maybe_unpin_radix_cache_path(&slot);
  slot.manager.release_all(layer_allocators_);
  slot.active = false;
  free_slot_indices_.push_back(slot_idx);
  CHECK_GT(num_active_requests_, 0);
  num_active_requests_--;
}

void KVCacheManager::publish_radix_cache(
    RequestId id, const std::vector<int32_t>& prompt_tokens) {
  auto& slot = lookup_request_slot(id);
  if (!radix_cache_enabled_) {
    return;
  }

  radix_cache_stats_.publish_requests++;
  const int32_t full_blocks = full_blocks_for_tokens(
      std::min(static_cast<int32_t>(prompt_tokens.size()), slot.manager.num_tokens()));
  if (full_blocks == 0) {
    return;
  }

  const std::vector<std::vector<int32_t>> block_ids_per_layer =
      full_block_ids_for_request(slot, full_blocks);
  retain_block_ids(block_ids_per_layer, 0, full_blocks);
  const auto result = radix_cache_.insert(prompt_tokens, block_ids_per_layer);
  if (result.existing_prefix_blocks > 0) {
    release_block_ids(block_ids_per_layer, 0, result.existing_prefix_blocks);
  }
  radix_cache_stats_.published_blocks += result.inserted_blocks;
}

void KVCacheManager::clear_radix_cache() {
  for (auto& slot : request_slots_) {
    if (slot.active) {
      maybe_unpin_radix_cache_path(&slot);
    }
  }
  release_block_ids(radix_cache_.clear_and_collect_block_ids());
}

bool KVCacheManager::append_slot(RequestId id) {
  if (lookup_request(id).append_token(layer_allocators_)) {
    return true;
  }
  if (!ensure_free_blocks_available(1)) {
    return false;
  }
  return lookup_request(id).append_token(layer_allocators_);
}

bool KVCacheManager::append_slots(RequestId id, int32_t num_tokens) {
  SequenceKVManager& request = lookup_request(id);
  if (request.append_tokens(layer_allocators_, num_tokens)) {
    return true;
  }

  const int32_t current_tokens = request.num_tokens();
  const int32_t blocks_before = (current_tokens + block_size_ - 1) / block_size_;
  const int32_t blocks_after = (current_tokens + num_tokens + block_size_ - 1) / block_size_;
  const int32_t required_blocks_per_layer = blocks_after - blocks_before;
  if (required_blocks_per_layer <= 0) {
    return false;
  }
  if (!ensure_free_blocks_available(required_blocks_per_layer)) {
    return false;
  }
  return request.append_tokens(layer_allocators_, num_tokens);
}

std::pair<int32_t, int32_t> KVCacheManager::get_slot(
    RequestId id, int32_t layer_idx, int32_t token_pos) const {
  return lookup_request(id).get_slot(layer_idx, token_pos);
}

const std::vector<int32_t>& KVCacheManager::get_block_ids(RequestId id,
                                                           int32_t layer_idx) const {
  return lookup_request(id).page_table(layer_idx).block_ids();
}

int32_t KVCacheManager::get_context_len(RequestId id) const {
  return lookup_request(id).num_tokens();
}

int32_t KVCacheManager::num_tokens_in_last_block(RequestId id) const {
  return lookup_request(id).num_tokens_in_last_block();
}

std::pair<int32_t, int32_t> KVCacheManager::current_slot(RequestId id,
                                                          int32_t layer_idx) const {
  return lookup_request(id).current_slot(layer_idx);
}

std::vector<int32_t> KVCacheManager::build_slot_mapping(
    const std::vector<RequestId>& request_ids, int32_t layer_idx) const {
  std::vector<int32_t> mapping;
  mapping.reserve(request_ids.size());
  for (auto rid : request_ids) {
    auto [block_id, offset] = current_slot(rid, layer_idx);
    mapping.push_back(block_id * block_size_ + offset);
  }
  return mapping;
}

std::pair<std::vector<int32_t>, int32_t> KVCacheManager::build_block_tables(
    const std::vector<RequestId>& request_ids, int32_t layer_idx) const {
  int32_t max_blocks = 0;
  for (auto rid : request_ids) {
    auto& ids = get_block_ids(rid, layer_idx);
    max_blocks = std::max(max_blocks, static_cast<int32_t>(ids.size()));
  }

  int32_t batch_size = static_cast<int32_t>(request_ids.size());
  std::vector<int32_t> table(batch_size * max_blocks, -1);  // pad with -1
  for (int32_t b = 0; b < batch_size; ++b) {
    auto& ids = get_block_ids(request_ids[b], layer_idx);
    for (int32_t i = 0; i < static_cast<int32_t>(ids.size()); ++i) {
      table[b * max_blocks + i] = ids[i];
    }
  }
  return {table, max_blocks};
}

std::vector<int32_t> KVCacheManager::build_seq_lens(
    const std::vector<RequestId>& request_ids) const {
  std::vector<int32_t> lens;
  lens.reserve(request_ids.size());
  for (auto rid : request_ids) {
    lens.push_back(get_context_len(rid));
  }
  return lens;
}

const BlockAllocator& KVCacheManager::allocator(int32_t layer_idx) const {
  CHECK_GE(layer_idx, 0);
  CHECK_LT(layer_idx, num_layers_);
  return *layer_allocators_[layer_idx];
}

BlockAllocator& KVCacheManager::allocator_mut(int32_t layer_idx) {
  CHECK_GE(layer_idx, 0);
  CHECK_LT(layer_idx, num_layers_);
  return *layer_allocators_[layer_idx];
}

int32_t KVCacheManager::num_free_blocks(int32_t layer_idx) const {
  return layer_allocators_[layer_idx]->num_free_blocks();
}

int32_t KVCacheManager::num_total_blocks(int32_t layer_idx) const {
  return layer_allocators_[layer_idx]->num_total_blocks();
}

const KVAppendStats& KVCacheManager::append_stats(RequestId id) const {
  return lookup_request(id).stats();
}

bool KVCacheManager::is_valid_request(RequestId id) const {
  uint32_t slot_idx = 0;
  uint32_t generation = 0;
  if (!try_decode_request_id(id, &slot_idx, &generation)) {
    return false;
  }
  const auto& slot = request_slots_[slot_idx];
  return slot.active && slot.generation == generation;
}

bool KVCacheManager::try_decode_request_id(RequestId id, uint32_t* slot_idx,
                                           uint32_t* generation) const {
  if (id < 0) {
    return false;
  }

  const uint32_t decoded_slot_idx = decode_slot_index(id);
  if (decoded_slot_idx >= request_slots_.size()) {
    return false;
  }

  if (slot_idx != nullptr) {
    *slot_idx = decoded_slot_idx;
  }
  if (generation != nullptr) {
    *generation = decode_generation(id);
  }
  return true;
}

int32_t KVCacheManager::full_blocks_for_tokens(int32_t num_tokens) const {
  CHECK_GE(num_tokens, 0);
  return num_tokens / block_size_;
}

int32_t KVCacheManager::additional_blocks_needed(int32_t current_tokens,
                                                 int32_t appended_tokens) const {
  CHECK_GE(current_tokens, 0);
  CHECK_GE(appended_tokens, 0);
  const int32_t blocks_before =
      (current_tokens + block_size_ - 1) / block_size_;
  const int32_t blocks_after =
      (current_tokens + appended_tokens + block_size_ - 1) / block_size_;
  return blocks_after - blocks_before;
}

int32_t KVCacheManager::min_free_blocks_across_layers() const {
  int32_t min_free_blocks = num_free_blocks(0);
  for (int32_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    min_free_blocks = std::min(min_free_blocks, num_free_blocks(layer_idx));
  }
  return min_free_blocks;
}

std::vector<std::vector<int32_t>> KVCacheManager::full_block_ids_for_request(
    const RequestSlot& slot, int32_t full_blocks) const {
  CHECK_GE(full_blocks, 0);
  std::vector<std::vector<int32_t>> block_ids_per_layer(num_layers_);
  for (int32_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    const auto& block_ids = slot.manager.page_table(layer_idx).block_ids();
    CHECK_GE(static_cast<int32_t>(block_ids.size()), full_blocks);
    block_ids_per_layer[layer_idx].assign(block_ids.begin(),
                                          block_ids.begin() + full_blocks);
  }
  return block_ids_per_layer;
}

void KVCacheManager::release_block_ids(
    const std::vector<std::vector<int32_t>>& block_ids_per_layer) {
  release_block_ids(block_ids_per_layer, 0,
                    block_ids_per_layer.empty()
                        ? 0
                        : static_cast<int32_t>(block_ids_per_layer[0].size()));
}

void KVCacheManager::release_block_ids(
    const std::vector<std::vector<int32_t>>& block_ids_per_layer,
    int32_t start_block,
    int32_t block_count) {
  CHECK_EQ(static_cast<int32_t>(block_ids_per_layer.size()), num_layers_);
  CHECK_GE(start_block, 0);
  CHECK_GE(block_count, 0);
  for (int32_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    CHECK_LE(start_block + block_count,
             static_cast<int32_t>(block_ids_per_layer[layer_idx].size()));
    for (int32_t block_idx = 0; block_idx < block_count; ++block_idx) {
      layer_allocators_[layer_idx]->free(
          block_ids_per_layer[layer_idx][start_block + block_idx]);
    }
  }
}

void KVCacheManager::retain_block_ids(
    const std::vector<std::vector<int32_t>>& block_ids_per_layer,
    int32_t start_block,
    int32_t block_count) {
  CHECK_EQ(static_cast<int32_t>(block_ids_per_layer.size()), num_layers_);
  CHECK_GE(start_block, 0);
  CHECK_GE(block_count, 0);
  for (int32_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    CHECK_LE(start_block + block_count,
             static_cast<int32_t>(block_ids_per_layer[layer_idx].size()));
    for (int32_t block_idx = 0; block_idx < block_count; ++block_idx) {
      layer_allocators_[layer_idx]->incref(
          block_ids_per_layer[layer_idx][start_block + block_idx]);
    }
  }
}

bool KVCacheManager::ensure_free_blocks_available(int32_t required_blocks_per_layer) {
  CHECK_GE(required_blocks_per_layer, 0);
  if (required_blocks_per_layer == 0) {
    return true;
  }

  auto has_capacity = [&]() {
    for (int32_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
      if (num_free_blocks(layer_idx) < required_blocks_per_layer) {
        return false;
      }
    }
    return true;
  };

  while (!has_capacity()) {
    if (evict_radix_cache_blocks(required_blocks_per_layer) == 0) {
      return false;
    }
  }
  return true;
}

void KVCacheManager::maybe_attach_radix_cache(
    RequestSlot* slot, const std::vector<int32_t>& prompt_tokens) {
  CHECK_NE(slot, nullptr);
  if (!radix_cache_enabled_) {
    return;
  }
  radix_cache_stats_.lookup_requests++;

  const int32_t max_reusable_blocks =
      std::max(0, static_cast<int32_t>(prompt_tokens.size()) - 1) / block_size_;
  if (max_reusable_blocks <= 0) {
    radix_cache_stats_.cache_misses++;
    return;
  }
  const int32_t max_reusable_tokens = max_reusable_blocks * block_size_;
  std::vector<int32_t> reusable_tokens(prompt_tokens.begin(),
                                       prompt_tokens.begin() + max_reusable_tokens);
  auto match = radix_cache_.match_prefix(reusable_tokens);
  if (match.matched_blocks <= 0 || match.matched_leaf == radix_cache_.root()) {
    radix_cache_stats_.cache_misses++;
    return;
  }

  slot->manager.adopt_shared_prefix(layer_allocators_, match.block_ids_per_layer,
                                    match.matched_tokens);
  radix_cache_.pin_path(match.matched_leaf);
  slot->radix_cache_leaf = match.matched_leaf;
  slot->radix_cache_shared_tokens = match.matched_tokens;
  radix_cache_stats_.cache_hits++;
  radix_cache_stats_.tokens_reused += match.matched_tokens;
}

void KVCacheManager::maybe_unpin_radix_cache_path(RequestSlot* slot) {
  CHECK_NE(slot, nullptr);
  if (slot->radix_cache_leaf != nullptr) {
    radix_cache_.unpin_path(slot->radix_cache_leaf);
    slot->radix_cache_leaf = nullptr;
    slot->radix_cache_shared_tokens = 0;
  }
}

int32_t KVCacheManager::evict_radix_cache_blocks(int32_t min_blocks_to_evict) {
  CHECK_GE(min_blocks_to_evict, 0);
  int32_t evicted_blocks = 0;
  while (evicted_blocks < min_blocks_to_evict) {
    const auto leaves = radix_cache_.collect_evictable_leaves();
    if (leaves.empty()) {
      break;
    }
    const auto removed_block_ids = radix_cache_.erase_leaf(leaves.front());
    const int32_t removed_blocks =
        removed_block_ids.empty() ? 0 : static_cast<int32_t>(removed_block_ids[0].size());
    release_block_ids(removed_block_ids);
    evicted_blocks += removed_blocks;
    radix_cache_stats_.evictions++;
    radix_cache_stats_.evicted_blocks += removed_blocks;
  }
  return evicted_blocks;
}

int32_t KVCacheManager::radix_cache_evictable_blocks() const {
  return radix_cache_.evictable_blocks();
}

void KVCacheManager::set_radix_cache_enabled(bool enabled) {
  radix_cache_enabled_ = enabled;
}

KVCacheManager::RequestSlot& KVCacheManager::lookup_request_slot(
    RequestId id, uint32_t* slot_idx) {
  return const_cast<RequestSlot&>(
      static_cast<const KVCacheManager*>(this)->lookup_request_slot(id, slot_idx));
}

const KVCacheManager::RequestSlot& KVCacheManager::lookup_request_slot(
    RequestId id, uint32_t* slot_idx) const {
  uint32_t decoded_slot_idx = 0;
  uint32_t generation = 0;
  CHECK(try_decode_request_id(id, &decoded_slot_idx, &generation))
      << "Request " << id << " not found";

  const auto& slot = request_slots_[decoded_slot_idx];
  CHECK(slot.active) << "Request " << id << " is inactive";
  CHECK_EQ(slot.generation, generation)
      << "Request " << id << " generation mismatch";

  if (slot_idx != nullptr) {
    *slot_idx = decoded_slot_idx;
  }
  return slot;
}

SequenceKVManager& KVCacheManager::lookup_request(RequestId id) {
  return lookup_request_slot(id).manager;
}

const SequenceKVManager& KVCacheManager::lookup_request(RequestId id) const {
  return lookup_request_slot(id).manager;
}

}  // namespace base


