// Block-based KV cache allocator for PagedAttention
#ifndef KUIPER_INCLUDE_BASE_BLOCK_ALLOCATOR_H_
#define KUIPER_INCLUDE_BASE_BLOCK_ALLOCATOR_H_

#include <queue>
#include <memory>
#include <vector>
#include "base/base.h"
#include "base/kv_cache_format.h"
#include "tensor/tensor.h"

namespace base {

struct BlockHandle {
  int32_t block_id = -1;
  uint64_t generation = 0;
  uint64_t pool_id = 0;
};

enum class BlockLeaseKind { kCompute, kIO };
class BlockAllocator;

// Owner-thread lease. The allocator must outlive all leases.
class BlockLease {
 public:
  BlockLease() = default;
  ~BlockLease();
  BlockLease(const BlockLease&) = delete;
  BlockLease& operator=(const BlockLease&) = delete;
  BlockLease(BlockLease&& other) noexcept;
  BlockLease& operator=(BlockLease&& other) noexcept;
  explicit operator bool() const { return allocator_ != nullptr; }
  BlockHandle handle() const { return handle_; }
  void reset();
 private:
  friend class BlockAllocator;
  BlockAllocator* allocator_ = nullptr;
  BlockHandle handle_;
  BlockLeaseKind kind_ = BlockLeaseKind::kCompute;
};


struct KVBlockPayloadPtrs {
  void* key = nullptr;
  void* value = nullptr;
  void* key_scale = nullptr;
  void* value_scale = nullptr;
  size_t key_value_bytes = 0;
  size_t scale_bytes = 0;
};

// A process-local binding to memory physically owned by another process.
// Slot IDs in initial_blocks already contain immutable prefix data; only IDs
// in allocatable_blocks may be selected for new writes. The capsule keeps the
// CUDA IPC mapping alive longer than every Tensor view.
struct ExternalKVPoolBinding {
  void* base = nullptr;
  uint64_t bytes = 0;
  int32_t device = -1;
  int32_t num_layers = 0;
  int32_t num_blocks = 0;
  int32_t block_size = 0;
  int32_t num_kv_heads = 0;
  int32_t head_size = 0;
  DataType storage_dtype = DataType::kDataTypeUnknown;
  std::vector<int32_t> initial_blocks;
  std::vector<int32_t> allocatable_blocks;
  std::shared_ptr<void> capsule;

  size_t tensor_bytes_per_layer() const {
    return static_cast<size_t>(num_blocks) * block_size * num_kv_heads *
           head_size * DataTypeSize(storage_dtype);
  }
  size_t layer_stride_bytes() const { return 2 * tensor_bytes_per_layer(); }
  bool valid() const {
    return base != nullptr && capsule != nullptr && device >= 0 && num_layers > 0 &&
           num_blocks > 0 && block_size > 0 && num_kv_heads > 0 && head_size > 0 &&
           storage_dtype != DataType::kDataTypeUnknown &&
           bytes == static_cast<uint64_t>(num_layers) * layer_stride_bytes();
  }
};

// Non-owning binding used by compute kernels. Allocation/free ownership stays
// in BlockAllocator (local mode) or the data-service pool owner (external
// mode); importing a view never creates another free queue.
class KVPoolView {
 public:
  KVPoolView() = default;
  KVPoolView(tensor::Tensor* key, tensor::Tensor* value,
             tensor::Tensor* key_scale, tensor::Tensor* value_scale,
             KVCacheStorageSpec storage_spec, DeviceType device)
      : key_(key), value_(value), key_scale_(key_scale), value_scale_(value_scale),
        storage_spec_(storage_spec), device_(device) {}

  bool valid() const { return key_ != nullptr && value_ != nullptr; }
  bool uses_fp8_storage() const { return storage_spec_.uses_fp8_storage(); }
  DeviceType device_type() const { return device_; }
  const KVCacheStorageSpec& storage_spec() const { return storage_spec_; }
  tensor::Tensor& key_pool() const { return *key_; }
  tensor::Tensor& value_pool() const { return *value_; }
  tensor::Tensor& key_scale_pool() const { return *key_scale_; }
  tensor::Tensor& value_scale_pool() const { return *value_scale_; }

 private:
  tensor::Tensor* key_ = nullptr;
  tensor::Tensor* value_ = nullptr;
  tensor::Tensor* key_scale_ = nullptr;
  tensor::Tensor* value_scale_ = nullptr;
  KVCacheStorageSpec storage_spec_;
  DeviceType device_ = DeviceType::kDeviceUnknown;
};

// Manages a pool of fixed-size KV cache blocks
class BlockAllocator {
 public:
  BlockAllocator(int32_t num_blocks, int32_t block_size, 
                 int32_t num_kv_heads, int32_t head_size,
                 const KVCacheStorageSpec& storage_spec,
                 base::DeviceType device);

  BlockAllocator(int32_t num_blocks, int32_t block_size,
                 int32_t num_kv_heads, int32_t head_size,
                 base::DataType dtype, base::DeviceType device,
                 BlockStorageMode storage_mode = BlockStorageMode::kPlain);

  BlockAllocator(int32_t layer_index, const ExternalKVPoolBinding& binding);

  ~BlockAllocator();

  // Allocate a free block, returns block_id or -1 if no free blocks
  int32_t allocate();

  // Increase the reference count on an allocated block.
  void incref(int32_t block_id);

  int32_t ref_count(int32_t block_id) const;

  // Release one reference to a block. The block returns to the free pool
  // when its refcount drops to zero.
  void free(int32_t block_id);

  BlockHandle handle(int32_t block_id) const;
  bool is_current(BlockHandle handle) const;
  bool acquire_lease(BlockHandle handle, BlockLeaseKind kind, BlockLease* lease);
  int32_t compute_pins(int32_t block_id) const;
  int32_t io_pins(int32_t block_id) const;

  // Get pointers to key/value for a specific block
  // Returns (key_ptr, value_ptr)
  std::pair<void*, void*> get_block_ptrs(int32_t block_id) const;

  KVBlockPayloadPtrs get_block_payload_ptrs(int32_t block_id) const;

  // Bitwise device-local copy used when a shared partial page becomes writable.
  void copy_block_payload(int32_t source_block_id, int32_t destination_block_id) const;

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
  const KVCacheStorageSpec& storage_spec() const { return storage_spec_; }
  BlockStorageMode storage_mode() const { return storage_spec_.storage_mode; }
  bool uses_fp8_storage() const { return storage_spec_.uses_fp8_storage(); }
  base::DataType logical_dtype() const { return storage_spec_.logical_dtype; }
  base::DataType storage_dtype() const { return storage_spec_.storage_dtype; }
  base::DataType scale_dtype() const { return storage_spec_.scale_dtype; }
  base::DeviceType device_type() const { return device_; }
  size_t key_value_bytes_per_block() const;
  size_t scale_bytes_per_block() const;
  uint64_t payload_copy_bytes() const { return payload_copy_bytes_; }
  KVPoolView pool_view() {
    return KVPoolView(&key_pool_, &value_pool_, &key_scale_pool_, &value_scale_pool_,
                      storage_spec_, device_);
  }
  KVPoolView pool_view() const {
    return KVPoolView(const_cast<tensor::Tensor*>(&key_pool_),
                      const_cast<tensor::Tensor*>(&value_pool_),
                      const_cast<tensor::Tensor*>(&key_scale_pool_),
                      const_cast<tensor::Tensor*>(&value_scale_pool_),
                      storage_spec_, device_);
  }

 private:
  friend class BlockLease;
  void release_lease(BlockHandle handle, BlockLeaseKind kind);
  uint64_t pool_id_ = 0;
  std::vector<uint64_t> generations_;
  std::vector<int32_t> compute_pins_;
  std::vector<int32_t> io_pins_;
  int32_t num_blocks_;
  int32_t block_size_;
  int32_t num_kv_heads_;
  int32_t head_size_;
  base::DeviceType device_;
  KVCacheStorageSpec storage_spec_;
  bool external_pool_ = false;
  std::shared_ptr<void> external_pool_capsule_;
  mutable uint64_t payload_copy_bytes_ = 0;

  // Physical memory pools
  // Layout: [num_blocks, block_size * num_kv_heads * head_size]
  tensor::Tensor key_pool_;
  tensor::Tensor value_pool_;
  // FP8 scale pools, only allocated for kFp8E4M3PerTokenHead mode.
  // Layout: [num_blocks, block_size * num_kv_heads]
  tensor::Tensor key_scale_pool_;
  tensor::Tensor value_scale_pool_;

  std::vector<int32_t> ref_counts_; // ref_counts_[block_id] = live references
  std::queue<int32_t> free_queue_;  // Queue of free block IDs
};

}  // namespace base

#endif  // KUIPER_INCLUDE_BASE_BLOCK_ALLOCATOR_H_
