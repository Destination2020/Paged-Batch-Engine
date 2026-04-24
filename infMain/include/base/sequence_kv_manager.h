// Sequence-level KV cache manager using PageTable
#ifndef KUIPER_INCLUDE_BASE_SEQUENCE_KV_MANAGER_H_
#define KUIPER_INCLUDE_BASE_SEQUENCE_KV_MANAGER_H_

#include <memory>
#include <vector>
#include "base/block_allocator.h"
#include "base/kv_block.h"

namespace base {

struct KVAppendStats {
  int64_t append_calls = 0;
  int64_t tokens_requested = 0;
  int64_t bulk_append_calls = 0;
  int64_t blocks_allocated = 0;
  int64_t allocation_failures = 0;
  int64_t rollback_count = 0;
};

// Manages KV cache for a single sequence using paged memory
class SequenceKVManager {
 public:
  explicit SequenceKVManager(int32_t block_size, int32_t num_layers);

  // Append one token's KV to cache
  // Allocates a new block if current block is full
  // Returns true if successful, false if allocator has no free blocks
  bool append_token(const std::vector<std::unique_ptr<BlockAllocator>>& layer_allocators);

  // Append multiple tokens at once, allocating blocks as needed
  // Returns false and rolls back on failure
  bool append_tokens(const std::vector<std::unique_ptr<BlockAllocator>>& layer_allocators,
                     int32_t num_tokens);

  // Get slot (block_id, offset) for a specific token position in a layer
  std::pair<int32_t, int32_t> get_slot(int32_t layer_idx, int32_t token_pos) const;

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

  // Truncate the sequence to a target token count. Shared-prefix tokens are
  // treated as immutable lower bound for truncation.
  void truncate_to(const std::vector<std::unique_ptr<BlockAllocator>>& layer_allocators,
                   int32_t new_num_tokens);

  // Clear the manager (for reuse)
  void clear();

  // Adopt a shared prefix represented by existing physical blocks. The shared
  // prefix must end on a block boundary so later appends do not mutate shared
  // blocks.
  void adopt_shared_prefix(const std::vector<std::unique_ptr<BlockAllocator>>& layer_allocators,
                           const std::vector<std::vector<int32_t>>& shared_block_ids,
                           int32_t shared_tokens);

  int32_t shared_prefix_tokens() const { return shared_prefix_tokens_; }
  int32_t private_tokens() const { return num_tokens_ - shared_prefix_tokens_; }

  const KVAppendStats& stats() const { return stats_; }

 private:
  bool append_tokens_internal(
      const std::vector<std::unique_ptr<BlockAllocator>>& layer_allocators,
      int32_t num_tokens);

  int32_t num_layers_ = 0;
  int32_t block_size_ = 0;
  int32_t num_tokens_ = 0;
  int32_t shared_prefix_tokens_ = 0;
  std::vector<PageTable> page_tables_;
  KVAppendStats stats_;
};

}  // namespace base

#endif  // KUIPER_INCLUDE_BASE_SEQUENCE_KV_MANAGER_H_


