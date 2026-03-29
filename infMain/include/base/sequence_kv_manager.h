// Sequence-level KV cache manager using PageTable
#ifndef KUIPER_INCLUDE_BASE_SEQUENCE_KV_MANAGER_H_
#define KUIPER_INCLUDE_BASE_SEQUENCE_KV_MANAGER_H_

#include <memory>
#include <vector>
#include "base/block_allocator.h"
#include "base/kv_block.h"

namespace base {

// Manages KV cache for a single sequence using paged memory
class SequenceKVManager {
 public:
  explicit SequenceKVManager(int32_t block_size, int32_t num_layers);

  // Append one token's KV to cache
  // Allocates a new block if current block is full
  // Returns true if successful, false if allocator has no free blocks
  bool append_token(const std::vector<std::unique_ptr<BlockAllocator>>& layer_allocators);

  // Get the page table for a specific layer
  const PageTable& page_table(int32_t layer_idx) const { return page_tables_.at(layer_idx); }

  // Get number of tokens stored
  int32_t num_tokens() const { return num_tokens_; }

  // Get number of tokens in the last block
  int32_t num_tokens_in_last_block() const;

  // Get current slot position for writing next KV
  // Returns (block_id, offset_in_block)
  std::pair<int32_t, int32_t> current_slot(int32_t layer_idx) const;

  // Release all blocks back to allocator
  void release_all(const std::vector<std::unique_ptr<BlockAllocator>>& layer_allocators);

  // Clear the manager (for reuse)
  void clear();

 private:
  int32_t num_layers_ = 0;
  int32_t block_size_ = 0;
  int32_t num_tokens_ = 0;
  std::vector<PageTable> page_tables_;
};

}  // namespace base

#endif  // KUIPER_INCLUDE_BASE_SEQUENCE_KV_MANAGER_H_





