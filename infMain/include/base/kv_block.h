// KV Block and PageTable structures for PagedAttention
#ifndef KUIPER_INCLUDE_BASE_KV_BLOCK_H_
#define KUIPER_INCLUDE_BASE_KV_BLOCK_H_

#include <cstdint>
#include <vector>

namespace base {

// Physical block metadata
struct KVBlock {
  int32_t block_id;
  int32_t ref_count;  // For future copy-on-write support

  KVBlock() : block_id(-1), ref_count(0) {}
  explicit KVBlock(int32_t id) : block_id(id), ref_count(1) {}
};

// Page table for a single sequence
// Maps logical token positions to physical blocks
class PageTable {
 public:
  explicit PageTable(int32_t block_size)
      : block_size_(block_size), num_tokens_(0) {}

  // Get the list of physical block IDs
  const std::vector<int32_t>& block_ids() const { return block_ids_; }

  // Get number of blocks currently allocated
  int32_t num_blocks() const { return static_cast<int32_t>(block_ids_.size()); }

  // Get total number of tokens stored
  int32_t num_tokens() const { return num_tokens_; }

  // Get number of tokens in the last block
  int32_t num_tokens_in_last_block() const {
    if (num_tokens_ == 0) return 0;
    int32_t remainder = num_tokens_ % block_size_;
    return remainder == 0 ? block_size_ : remainder;
  }

  // Add a new physical block to the page table
  void append_block(int32_t physical_block_id) {
    block_ids_.push_back(physical_block_id);
  }
  void pop_last_block() {
    if (!block_ids_.empty()) {
      block_ids_.pop_back();
    }
  }
  void replace_last_block(int32_t physical_block_id) {
    if (!block_ids_.empty()) block_ids_.back() = physical_block_id;
  }

  // Increment token count (called after writing KV)
  void increment_token_count() { num_tokens_++; }
  void add_token_count(int32_t count) { num_tokens_ += count; }

  // Truncate this page table to a logical token count and return physical
  // blocks that are no longer referenced by the table.
  std::vector<int32_t> truncate_to_tokens(int32_t new_num_tokens) {
    int32_t new_block_count = 0;
    if (new_num_tokens > 0) {
      new_block_count = (new_num_tokens + block_size_ - 1) / block_size_;
    }

    std::vector<int32_t> removed_blocks;
    if (new_block_count < static_cast<int32_t>(block_ids_.size())) {
      removed_blocks.assign(block_ids_.begin() + new_block_count, block_ids_.end());
      block_ids_.resize(new_block_count);
    }
    num_tokens_ = new_num_tokens;
    return removed_blocks;
  }

  // Clear the page table
  void clear() {
    block_ids_.clear();
    num_tokens_ = 0;
  }

  int32_t block_size() const { return block_size_; }

 private:
  int32_t block_size_;              // Tokens per block (e.g., 16)
  int32_t num_tokens_;              // Total tokens stored
  std::vector<int32_t> block_ids_;  // Physical block IDs in logical order
};

}  // namespace base

#endif  // KUIPER_INCLUDE_BASE_KV_BLOCK_H_
