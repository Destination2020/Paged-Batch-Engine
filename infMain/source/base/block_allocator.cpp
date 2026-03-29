// Block-based KV cache allocator implementation
#include "base/block_allocator.h"
#include <glog/logging.h>
#include "base/alloc.h"

namespace base {

BlockAllocator::BlockAllocator(int32_t num_blocks, int32_t block_size, 
                               int32_t num_kv_heads, int32_t head_size, base::DataType dtype,
                               base::DeviceType device)
    : num_blocks_(num_blocks),
      block_size_(block_size),
      num_kv_heads_(num_kv_heads),
      head_size_(head_size),
      dtype_(dtype),
      device_(device) {

  CHECK_GT(num_blocks, 0) << "num_blocks must be positive";
  CHECK_GT(block_size, 0) << "block_size must be positive";
  CHECK_GT(num_kv_heads, 0) << "num_kv_heads must be positive";
  CHECK_GT(head_size, 0) << "head_size must be positive";

  // Allocate physical memory pools
  // Shape: [num_blocks, block_size * num_kv_heads * head_size]
  std::shared_ptr<DeviceAllocator> allocator;
  if (device == DeviceType::kDeviceCUDA) {
    allocator = CUDADeviceAllocatorFactory::get_instance();
  } else {
    allocator = CPUDeviceAllocatorFactory::get_instance();
  }

  key_pool_ = tensor::Tensor(dtype, num_blocks, block_size * num_kv_heads * head_size,
                             true, allocator);
  value_pool_ = tensor::Tensor(dtype, num_blocks, block_size * num_kv_heads * head_size,
                               true, allocator);

  key_pool_.set_device_type(device);
  value_pool_.set_device_type(device);

  // Initialize free table and queue
  free_table_.resize(num_blocks, true);
  for (int32_t i = 0; i < num_blocks; ++i) {
    free_queue_.push(i);
  }

  LOG(INFO) << "BlockAllocator initialized: " << num_blocks << " blocks, "
            << "block_size=" << block_size << ", "
            << "total_memory=" << (key_pool_.byte_size() + value_pool_.byte_size()) / (1024.0 * 1024.0)
            << " MB";
}

BlockAllocator::~BlockAllocator() {
  // Tensors will auto-release memory
}

int32_t BlockAllocator::allocate() {
  if (free_queue_.empty()) {
    return -1;  // No free blocks
  }

  int32_t block_id = free_queue_.front();
  free_queue_.pop();
  free_table_[block_id] = false;
  return block_id;
}

void BlockAllocator::free(int32_t block_id) {
  CHECK_GE(block_id, 0) << "Invalid block_id";
  CHECK_LT(block_id, num_blocks_) << "block_id out of range";
  CHECK(!free_table_[block_id]) << "Double free of block " << block_id;

  free_table_[block_id] = true;
  free_queue_.push(block_id);
}

std::pair<void*, void*> BlockAllocator::get_block_ptrs(int32_t block_id) const {
  CHECK_GE(block_id, 0); 
  CHECK_LT(block_id, num_blocks_);

  // Calculate offset: block_id * (num_layers * block_size * num_kv_heads * head_size)
  //                   + layer_idx * (block_size * num_kv_heads * head_size)
  int64_t block_stride = block_size_ * num_kv_heads_ * head_size_;
  int64_t offset = block_id * block_stride;

  void* key_ptr = nullptr;
  void* value_ptr = nullptr;

  if (dtype_ == DataType::kDataTypeFp32) {
    key_ptr = const_cast<float*>(key_pool_.ptr<float>(offset));
    value_ptr = const_cast<float*>(value_pool_.ptr<float>(offset));
  } else if (dtype_ == DataType::kDataTypeBf16) {
    key_ptr = const_cast<uint16_t*>(key_pool_.ptr<uint16_t>(offset));
    value_ptr = const_cast<uint16_t*>(value_pool_.ptr<uint16_t>(offset));
  } else {
    LOG(FATAL) << "Unsupported dtype: " << static_cast<int>(dtype_);
  }

  return {key_ptr, value_ptr};
}

}  // namespace base
