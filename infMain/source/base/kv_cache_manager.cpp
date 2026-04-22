// KVCacheManager implementation
#include "base/kv_cache_manager.h"
#include <algorithm>
#include <functional>
#include <sstream>
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
      layer_allocators_(std::move(layer_allocators)) {
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
    slot->shared_prefix_tokens = 0;
    slot->prefix_cache_key.clear();
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

RequestId KVCacheManager::register_request_with_prompt(
    const std::vector<int32_t>& prompt_tokens) {
  const RequestId id = register_request();
  RequestSlot& slot = lookup_request_slot(id);
  maybe_attach_prefix_cache(&slot, prompt_tokens);
  return id;
}

void KVCacheManager::free_request(RequestId id) {
  uint32_t slot_idx = 0;
  auto& slot = lookup_request_slot(id, &slot_idx);
  if (!slot.prefix_cache_key.empty()) {
    auto it = prefix_cache_.find(slot.prefix_cache_key);
    if (it != prefix_cache_.end()) {
      CHECK_GT(it->second.ref_count, 0);
      --it->second.ref_count;
      if (it->second.ref_count == 0) {
        release_prefix_cache_entry(it->second);
        prefix_cache_.erase(it);
      }
    }
    slot.prefix_cache_key.clear();
    slot.shared_prefix_tokens = 0;
  }
  slot.manager.release_all(layer_allocators_);
  slot.active = false;
  free_slot_indices_.push_back(slot_idx);
  CHECK_GT(num_active_requests_, 0);
  num_active_requests_--;
}

void KVCacheManager::publish_prefix_cache(
    RequestId id, const std::vector<int32_t>& prompt_tokens) {
  auto& slot = lookup_request_slot(id);
  maybe_publish_prefix_cache(&slot, prompt_tokens);
}

void KVCacheManager::clear_prefix_cache() {
  for (auto& [key, entry] : prefix_cache_) {
    (void)key;
    release_prefix_cache_entry(entry);
  }
  prefix_cache_.clear();

  for (auto& slot : request_slots_) {
    if (!slot.active) {
      slot.shared_prefix_tokens = 0;
      slot.prefix_cache_key.clear();
    }
  }
}

bool KVCacheManager::append_slot(RequestId id) {
  return lookup_request(id).append_token(layer_allocators_);
}

bool KVCacheManager::append_slots(RequestId id, int32_t num_tokens) {
  return lookup_request(id).append_tokens(layer_allocators_, num_tokens);
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

std::string KVCacheManager::build_prefix_cache_key(
    const std::vector<int32_t>& prompt_tokens, int32_t shared_tokens) const {
  CHECK_GE(shared_tokens, 0);
  CHECK_LE(shared_tokens, static_cast<int32_t>(prompt_tokens.size()));

  uint64_t hash = 1469598103934665603ull;
  auto hash_mix = [&](uint64_t value) {
    hash ^= value;
    hash *= 1099511628211ull;
  };

  hash_mix(static_cast<uint64_t>(shared_tokens));
  for (int32_t idx = 0; idx < shared_tokens; ++idx) {
    hash_mix(static_cast<uint32_t>(prompt_tokens[idx]));
  }

  std::ostringstream os;
  os << shared_tokens << ":" << hash;
  return os.str();
}

int32_t KVCacheManager::cacheable_prefix_tokens(
    const std::vector<int32_t>& prompt_tokens) const {
  const int32_t prompt_tokens_count = static_cast<int32_t>(prompt_tokens.size());
  if (prompt_tokens_count <= block_size_) {
    return 0;
  }

  const int32_t full_prompt_blocks = prompt_tokens_count / block_size_;
  if (full_prompt_blocks <= 1) {
    return 0;
  }

  // Keep one full block private so later decode appends never target a shared
  // tail block. The MVP intentionally trades hit rate for simple correctness.
  return (full_prompt_blocks - 1) * block_size_;
}

void KVCacheManager::release_prefix_cache_entry(const PrefixCacheEntry& entry) {
  for (int32_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    for (int32_t block_id : entry.block_ids_per_layer[layer_idx]) {
      layer_allocators_[layer_idx]->free(block_id);
    }
  }
}

void KVCacheManager::maybe_attach_prefix_cache(
    RequestSlot* slot, const std::vector<int32_t>& prompt_tokens) {
  CHECK_NE(slot, nullptr);
  if (!prefix_cache_enabled_) {
    return;
  }
  prefix_cache_stats_.lookup_requests++;

  const int32_t shared_tokens = cacheable_prefix_tokens(prompt_tokens);
  if (shared_tokens <= 0) {
    prefix_cache_stats_.cache_misses++;
    return;
  }

  const std::string key = build_prefix_cache_key(prompt_tokens, shared_tokens);
  auto it = prefix_cache_.find(key);
  if (it == prefix_cache_.end()) {
    prefix_cache_stats_.cache_misses++;
    return;
  }

  slot->manager.adopt_shared_prefix(layer_allocators_, it->second.block_ids_per_layer,
                                    it->second.shared_tokens);
  slot->shared_prefix_tokens = it->second.shared_tokens;
  slot->prefix_cache_key = key;
  it->second.ref_count++;
  prefix_cache_stats_.cache_hits++;
  prefix_cache_stats_.tokens_reused += it->second.shared_tokens;
}

void KVCacheManager::maybe_publish_prefix_cache(
    RequestSlot* slot, const std::vector<int32_t>& prompt_tokens) {
  CHECK_NE(slot, nullptr);
  if (!prefix_cache_enabled_) {
    return;
  }
  const int32_t shared_tokens = cacheable_prefix_tokens(prompt_tokens);
  if (shared_tokens <= 0 || slot->shared_prefix_tokens >= shared_tokens) {
    return;
  }

  const std::string key = build_prefix_cache_key(prompt_tokens, shared_tokens);
  auto [it, inserted] = prefix_cache_.emplace(key, PrefixCacheEntry{});
  if (inserted) {
    PrefixCacheEntry& entry = it->second;
    entry.shared_tokens = shared_tokens;
    entry.ref_count = 1;  // Cache holds one persistent reference.
    entry.block_ids_per_layer.resize(num_layers_);
    const int32_t shared_blocks = shared_tokens / block_size_;
    for (int32_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
      const auto& block_ids = slot->manager.page_table(layer_idx).block_ids();
      CHECK_GE(static_cast<int32_t>(block_ids.size()), shared_blocks);
      entry.block_ids_per_layer[layer_idx].assign(block_ids.begin(),
                                                  block_ids.begin() + shared_blocks);
      for (int32_t block_id : entry.block_ids_per_layer[layer_idx]) {
        layer_allocators_[layer_idx]->incref(block_id);
      }
    }
  }
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









