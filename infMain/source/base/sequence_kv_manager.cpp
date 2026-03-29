// Sequence-level KV cache manager implementation
#include "base/sequence_kv_manager.h"
#include <glog/logging.h>

namespace base {

SequenceKVManager::SequenceKVManager(int32_t block_size, int32_t num_layers)
    : num_layers_(num_layers), block_size_(block_size) {
  CHECK_GT(block_size_, 0) << "block_size must be positive";
  CHECK_GT(num_layers_, 0) << "num_layers must be positive";

  // Initialize page table for each layer
  for (int32_t i = 0; i < num_layers_; ++i) {
    page_tables_.emplace_back(block_size_);
  }
}

bool SequenceKVManager::append_token(
    const std::vector<std::unique_ptr<BlockAllocator>>& layer_allocators) {
  CHECK_EQ(static_cast<int32_t>(layer_allocators.size()), num_layers_)
      << "layer_allocators size must match num_layers";

  const bool need_new_block = (num_tokens_ % block_size_ == 0);
  std::vector<int32_t> newly_allocated_block_ids;
  newly_allocated_block_ids.reserve(num_layers_);

  if (need_new_block) {
    for (int32_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
      int32_t new_block_id = layer_allocators[layer_idx]->allocate();
      if (new_block_id == -1) {
        for (int32_t rollback_layer = 0;
             rollback_layer < static_cast<int32_t>(newly_allocated_block_ids.size());
             ++rollback_layer) {
          layer_allocators[rollback_layer]->free(newly_allocated_block_ids[rollback_layer]);
        }
        LOG(WARNING) << "Failed to allocate block for layer " << layer_idx
                     << ": no free blocks available";
        return false;
      }
      newly_allocated_block_ids.push_back(new_block_id);
      page_tables_[layer_idx].append_block(new_block_id);
    }
  }

  ++num_tokens_;
  return true;
}

int32_t SequenceKVManager::num_tokens_in_last_block() const {
  if (num_tokens_ == 0) {
    return 0;
  }

  int32_t remainder = num_tokens_ % block_size_;
  return remainder == 0 ? block_size_ : remainder;
}

std::pair<int32_t, int32_t> SequenceKVManager::current_slot(int32_t layer_idx) const {
  CHECK_GE(layer_idx, 0);
  CHECK_LT(layer_idx, num_layers_);

  if (num_tokens_ == 0) {
    return {-1, -1};  // No tokens yet
  }

  int32_t last_token_idx = num_tokens_ - 1;
  int32_t block_idx = last_token_idx / block_size_;
  int32_t offset_in_block = last_token_idx % block_size_;

  const auto& block_ids = page_tables_[layer_idx].block_ids();
  CHECK_LT(block_idx, block_ids.size()) << "Block index out of range";

  return {block_ids[block_idx], offset_in_block};
}

void SequenceKVManager::release_all(
    const std::vector<std::unique_ptr<BlockAllocator>>& layer_allocators) {
  CHECK_EQ(static_cast<int32_t>(layer_allocators.size()), num_layers_)
      << "layer_allocators size must match num_layers";

  for (int32_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    auto& allocator = layer_allocators[layer_idx];
    const auto& block_ids = page_tables_[layer_idx].block_ids();
    for (int32_t block_id : block_ids) {
      allocator->free(block_id);
    }
    page_tables_[layer_idx].clear();
  }
  num_tokens_ = 0;
}

void SequenceKVManager::clear() {
  for (auto& pt : page_tables_) {
    pt.clear();
  }
  num_tokens_ = 0;
}

}  // namespace base



