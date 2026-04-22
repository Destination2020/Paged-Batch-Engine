// Sequence-level KV cache manager implementation
#include "base/sequence_kv_manager.h"
#include <glog/logging.h>

namespace base {

namespace {

int32_t blocks_for_tokens(int32_t num_tokens, int32_t block_size) {
  if (num_tokens <= 0) {
    return 0;
  }
  return (num_tokens + block_size - 1) / block_size;
}

}  // namespace

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
  return append_tokens_internal(layer_allocators, 1);
}

bool SequenceKVManager::append_tokens(
    const std::vector<std::unique_ptr<BlockAllocator>>& layer_allocators,
    int32_t num_tokens) {
  CHECK_GT(num_tokens, 0);
  return append_tokens_internal(layer_allocators, num_tokens);
}

std::pair<int32_t, int32_t> SequenceKVManager::get_slot(
    int32_t layer_idx, int32_t token_pos) const {
  CHECK_GE(layer_idx, 0);
  CHECK_LT(layer_idx, num_layers_);
  CHECK_GE(token_pos, 0);
  CHECK_LT(token_pos, num_tokens_);

  int32_t block_idx = token_pos / block_size_;
  int32_t offset = token_pos % block_size_;
  int32_t block_id = page_tables_[layer_idx].block_ids().at(block_idx);
  return {block_id, offset};
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
  stats_ = KVAppendStats{};
}

void SequenceKVManager::adopt_shared_prefix(
    const std::vector<std::unique_ptr<BlockAllocator>>& layer_allocators,
    const std::vector<std::vector<int32_t>>& shared_block_ids,
    int32_t shared_tokens) {
  CHECK_EQ(static_cast<int32_t>(layer_allocators.size()), num_layers_)
      << "layer_allocators size must match num_layers";
  CHECK_EQ(static_cast<int32_t>(shared_block_ids.size()), num_layers_)
      << "shared_block_ids size must match num_layers";
  CHECK_EQ(num_tokens_, 0) << "Shared prefix can only be adopted into an empty sequence";
  CHECK_GE(shared_tokens, 0);
  CHECK_EQ(shared_tokens % block_size_, 0)
      << "Shared prefix must end on a block boundary for safe reuse";

  const int32_t expected_blocks = blocks_for_tokens(shared_tokens, block_size_);
  for (int32_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    CHECK_EQ(static_cast<int32_t>(shared_block_ids[layer_idx].size()), expected_blocks)
        << "Shared prefix block count mismatch at layer " << layer_idx;
    for (int32_t block_id : shared_block_ids[layer_idx]) {
      layer_allocators[layer_idx]->incref(block_id);
      page_tables_[layer_idx].append_block(block_id);
    }
    page_tables_[layer_idx].add_token_count(shared_tokens);
  }
  num_tokens_ = shared_tokens;
}

bool SequenceKVManager::append_tokens_internal(
    const std::vector<std::unique_ptr<BlockAllocator>>& layer_allocators,
    int32_t num_tokens) {
  CHECK_EQ(static_cast<int32_t>(layer_allocators.size()), num_layers_)
      << "layer_allocators size must match num_layers";
  CHECK_GT(num_tokens, 0);

  stats_.append_calls++;
  stats_.tokens_requested += num_tokens;
  if (num_tokens > 1) {
    stats_.bulk_append_calls++;
  }

  const int32_t blocks_before = blocks_for_tokens(num_tokens_, block_size_);
  const int32_t blocks_after = blocks_for_tokens(num_tokens_ + num_tokens, block_size_);
  const int32_t new_blocks_needed = blocks_after - blocks_before;

  if (new_blocks_needed <= 0) {
    num_tokens_ += num_tokens;
    for (auto& page_table : page_tables_) {
      page_table.add_token_count(num_tokens);
    }
    return true;
  }

  std::vector<std::vector<int32_t>> pending_block_ids(num_layers_);
  for (auto& block_ids : pending_block_ids) {
    block_ids.reserve(new_blocks_needed);
  }

  for (int32_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    for (int32_t block_idx = 0; block_idx < new_blocks_needed; ++block_idx) {
      const int32_t block_id = layer_allocators[layer_idx]->allocate();
      if (block_id == -1) {
        for (int32_t rollback_layer = 0; rollback_layer <= layer_idx; ++rollback_layer) {
          for (int32_t allocated_block_id : pending_block_ids[rollback_layer]) {
            layer_allocators[rollback_layer]->free(allocated_block_id);
          }
        }
        stats_.allocation_failures++;
        stats_.rollback_count++;
        LOG(WARNING) << "Failed to allocate " << new_blocks_needed
                     << " KV blocks for layer " << layer_idx
                     << ": no free blocks available";
        return false;
      }
      pending_block_ids[layer_idx].push_back(block_id);
    }
  }

  for (int32_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    for (int32_t block_id : pending_block_ids[layer_idx]) {
      page_tables_[layer_idx].append_block(block_id);
    }
    page_tables_[layer_idx].add_token_count(num_tokens);
  }

  num_tokens_ += num_tokens;
  stats_.blocks_allocated += static_cast<int64_t>(num_layers_) * new_blocks_needed;
  return true;
}

}  // namespace base
