// Block-based KV cache allocator for PagedAttention
#ifndef KUIPER_INCLUDE_BASE_BLOCK_ALLOCATOR_H_
#define KUIPER_INCLUDE_BASE_BLOCK_ALLOCATOR_H_

#include <queue>
#include <vector>
#include "base/base.h"
#include "tensor/tensor.h"

namespace base {

enum class BlockStorageMode : uint8_t {
  kPlain = 0,
  kFp8E4M3PerTokenHead = 1,
};

// Manages a pool of fixed-size KV cache blocks
class BlockAllocator {
 public:
  BlockAllocator(int32_t num_blocks, int32_t block_size, 
                 int32_t num_kv_heads, int32_t head_size,
                 base::DataType dtype,base::DeviceType device,
                 BlockStorageMode storage_mode = BlockStorageMode::kPlain);

  ~BlockAllocator();

  // Allocate a free block, returns block_id or -1 if no free blocks
  int32_t allocate();

  // Free a block back to the pool
  void free(int32_t block_id);

  // Get pointers to key/value for a specific block
  // Returns (key_ptr, value_ptr)
  std::pair<void*, void*> get_block_ptrs(int32_t block_id) const;

  // Query free blocks
  int32_t num_free_blocks() const { return static_cast<int32_t>(free_queue_.size()); }
  int32_t num_total_blocks() const { return num_blocks_; }

  // Get the underlying pool tensors (for passing to kernels)
  tensor::Tensor& key_pool() { return key_pool_; }
  const tensor::Tensor& key_pool() const { return key_pool_; }
  tensor::Tensor& value_pool() { return value_pool_; }
  const tensor::Tensor& value_pool() const { return value_pool_; }
  tensor::Tensor& key_scale_pool() { return key_scale_pool_; }
  const tensor::Tensor& key_scale_pool() const { return key_scale_pool_; }
  tensor::Tensor& value_scale_pool() { return value_scale_pool_; }
  const tensor::Tensor& value_scale_pool() const { return value_scale_pool_; }

  int32_t block_size() const { return block_size_; }
  int32_t num_kv_heads() const { return num_kv_heads_; }
  int32_t head_size() const { return head_size_; }
  BlockStorageMode storage_mode() const { return storage_mode_; }
  bool uses_fp8_storage() const { return storage_mode_ == BlockStorageMode::kFp8E4M3PerTokenHead; }

 private:
  int32_t num_blocks_;
  int32_t block_size_;
  int32_t num_kv_heads_;
  int32_t head_size_;
  base::DataType dtype_;
  base::DeviceType device_;
  BlockStorageMode storage_mode_ = BlockStorageMode::kPlain;

  // Physical memory pools
  // Layout: [num_blocks, block_size * num_kv_heads * head_size]
  tensor::Tensor key_pool_;
  tensor::Tensor value_pool_;
  // FP8 scale pools, only allocated for kFp8E4M3PerTokenHead mode.
  // Layout: [num_blocks, block_size * num_kv_heads]
  tensor::Tensor key_scale_pool_;
  tensor::Tensor value_scale_pool_;

  std::vector<bool> free_table_;    // free_table_[block_id] = is_free
  std::queue<int32_t> free_queue_;  // Queue of free block IDs
};

}  // namespace base

#endif  // KUIPER_INCLUDE_BASE_BLOCK_ALLOCATOR_H_
