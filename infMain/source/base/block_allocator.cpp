// Block-based KV cache allocator implementation
#include "base/block_allocator.h"
#include <glog/logging.h>
#include "base/alloc.h"

namespace base {

BlockAllocator::BlockAllocator(int32_t num_blocks, int32_t block_size, 
                               int32_t num_kv_heads, int32_t head_size,
                               const KVCacheStorageSpec& storage_spec,
                               base::DeviceType device)
    : num_blocks_(num_blocks),
      block_size_(block_size),
      num_kv_heads_(num_kv_heads),
      head_size_(head_size),
      device_(device),
      storage_spec_(storage_spec) {

  CHECK_GT(num_blocks, 0) << "num_blocks must be positive";
  CHECK_GT(block_size, 0) << "block_size must be positive";
  CHECK_GT(num_kv_heads, 0) << "num_kv_heads must be positive";
  CHECK_GT(head_size, 0) << "head_size must be positive";
  STATUS_CHECK(ValidateKVCacheStorageSpec(storage_spec_, device_, storage_spec_.logical_dtype));

  // Allocate physical memory pools
  // Shape: [num_blocks, block_size * num_kv_heads * head_size]
  std::shared_ptr<DeviceAllocator> allocator;
  if (device == DeviceType::kDeviceCUDA) {
    allocator = CUDADeviceAllocatorFactory::get_instance();
  } else {
    allocator = CPUDeviceAllocatorFactory::get_instance();
  }

  const base::DataType pool_dtype = storage_spec_.storage_dtype;
  key_pool_ = tensor::Tensor(pool_dtype, num_blocks, block_size * num_kv_heads * head_size,
                             true, allocator);
  value_pool_ = tensor::Tensor(pool_dtype, num_blocks, block_size * num_kv_heads * head_size,
                               true, allocator);

  key_pool_.set_device_type(device);
  value_pool_.set_device_type(device);
  if (storage_spec_.has_scales()) {
    CHECK_EQ(device, DeviceType::kDeviceCUDA)
        << "FP8 KV cache storage is only supported on CUDA.";
    key_scale_pool_ =
        tensor::Tensor(storage_spec_.scale_dtype, num_blocks, block_size * num_kv_heads, true,
                       allocator);
    value_scale_pool_ =
        tensor::Tensor(storage_spec_.scale_dtype, num_blocks, block_size * num_kv_heads, true,
                       allocator);
    key_scale_pool_.set_device_type(device);
    value_scale_pool_.set_device_type(device);
  }

  // Initialize refcounts and free queue.
  ref_counts_.resize(num_blocks, 0);
  for (int32_t i = 0; i < num_blocks; ++i) {
    free_queue_.push(i);
  }

  LOG(INFO) << "BlockAllocator initialized: " << num_blocks << " blocks, "
            << "block_size=" << block_size << ", "
            << "storage_mode=" << BlockStorageModeName(storage_spec_.storage_mode) << ", "
            << "total_memory=" << (key_pool_.byte_size() + value_pool_.byte_size()) / (1024.0 * 1024.0)
            << " MB";
}

BlockAllocator::BlockAllocator(int32_t num_blocks, int32_t block_size,
                               int32_t num_kv_heads, int32_t head_size,
                               base::DataType dtype, base::DeviceType device,
                               BlockStorageMode storage_mode)
    : BlockAllocator(num_blocks, block_size, num_kv_heads, head_size,
                     MakeKVCacheStorageSpec(dtype, storage_mode), device) {}

BlockAllocator::~BlockAllocator() {
  // Tensors will auto-release memory
}

int32_t BlockAllocator::allocate() {
  if (free_queue_.empty()) {
    return -1;  // No free blocks
  }

  int32_t block_id = free_queue_.front();
  free_queue_.pop();
  CHECK_EQ(ref_counts_[block_id], 0) << "Allocated block " << block_id
                                     << " still has outstanding references";
  ref_counts_[block_id] = 1;
  return block_id;
}

void BlockAllocator::incref(int32_t block_id) {
  CHECK_GE(block_id, 0) << "Invalid block_id";
  CHECK_LT(block_id, num_blocks_) << "block_id out of range";
  CHECK_GT(ref_counts_[block_id], 0) << "Cannot incref a free block " << block_id;
  ++ref_counts_[block_id];
}

void BlockAllocator::free(int32_t block_id) {
  CHECK_GE(block_id, 0) << "Invalid block_id";
  CHECK_LT(block_id, num_blocks_) << "block_id out of range";
  CHECK_GT(ref_counts_[block_id], 0) << "Double free of block " << block_id;

  --ref_counts_[block_id];
  if (ref_counts_[block_id] == 0) {
    free_queue_.push(block_id);
  }
}

std::pair<void*, void*> BlockAllocator::get_block_ptrs(int32_t block_id) const {
  CHECK_GE(block_id, 0); 
  CHECK_LT(block_id, num_blocks_);
  CHECK(storage_spec_.storage_mode == BlockStorageMode::kPlain)
      << "get_block_ptrs only supports plain KV storage.";

  // Calculate offset: block_id * (num_layers * block_size * num_kv_heads * head_size)
  //                   + layer_idx * (block_size * num_kv_heads * head_size)
  int64_t block_stride = block_size_ * num_kv_heads_ * head_size_;
  int64_t offset = block_id * block_stride;

  void* key_ptr = nullptr;
  void* value_ptr = nullptr;

  if (storage_spec_.storage_dtype == DataType::kDataTypeFp32) {
    key_ptr = const_cast<float*>(key_pool_.ptr<float>(offset));
    value_ptr = const_cast<float*>(value_pool_.ptr<float>(offset));
  } else if (storage_spec_.storage_dtype == DataType::kDataTypeBf16) {
    key_ptr = const_cast<uint16_t*>(key_pool_.ptr<uint16_t>(offset));
    value_ptr = const_cast<uint16_t*>(value_pool_.ptr<uint16_t>(offset));
  } else {
    LOG(FATAL) << "Unsupported dtype: " << static_cast<int>(storage_spec_.storage_dtype);
  }

  return {key_ptr, value_ptr};
}

}  // namespace base
