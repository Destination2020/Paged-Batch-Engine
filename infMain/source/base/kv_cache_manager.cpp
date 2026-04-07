// KVCacheManager implementation
#include "base/kv_cache_manager.h"
#include <glog/logging.h>

namespace base {

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
  RequestId id = next_id_++;
  requests_.emplace(id, SequenceKVManager(block_size_, num_layers_));
  return id;
}

void KVCacheManager::free_request(RequestId id) {
  auto it = requests_.find(id);
  CHECK(it != requests_.end()) << "Request " << id << " not found";
  it->second.release_all(layer_allocators_);
  requests_.erase(it);
}

bool KVCacheManager::append_slot(RequestId id) {
  auto it = requests_.find(id);
  CHECK(it != requests_.end()) << "Request " << id << " not found";
  return it->second.append_token(layer_allocators_);
}

const std::vector<int32_t>& KVCacheManager::get_block_ids(RequestId id,
                                                           int32_t layer_idx) const {
  auto it = requests_.find(id);
  CHECK(it != requests_.end()) << "Request " << id << " not found";
  return it->second.page_table(layer_idx).block_ids();
}

int32_t KVCacheManager::get_context_len(RequestId id) const {
  auto it = requests_.find(id);
  CHECK(it != requests_.end()) << "Request " << id << " not found";
  return it->second.num_tokens();
}

int32_t KVCacheManager::num_tokens_in_last_block(RequestId id) const {
  auto it = requests_.find(id);
  CHECK(it != requests_.end()) << "Request " << id << " not found";
  return it->second.num_tokens_in_last_block();
}

std::pair<int32_t, int32_t> KVCacheManager::current_slot(RequestId id,
                                                          int32_t layer_idx) const {
  auto it = requests_.find(id);
  CHECK(it != requests_.end()) << "Request " << id << " not found";
  return it->second.current_slot(layer_idx);
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

}  // namespace base
