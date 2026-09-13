// Block-based KV cache allocator implementation
#include "base/block_allocator.h"
#include <glog/logging.h>
#include <atomic>
#include <cstring>
#include <limits>
#include <utility>
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
#ifdef KUIPER_CPU_ONLY
    LOG(FATAL) << "CUDA block pool unavailable in CPU-only build";
#else
    allocator = CUDADeviceAllocatorFactory::get_instance();
#endif
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

  static std::atomic<uint64_t> next_pool_id{1};
  pool_id_ = next_pool_id.fetch_add(1);
  CHECK_NE(pool_id_, 0u);
  generations_.resize(num_blocks, 0);
  compute_pins_.resize(num_blocks, 0);
  io_pins_.resize(num_blocks, 0);
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

BlockAllocator::BlockAllocator(int32_t layer_index,
                               const ExternalKVPoolBinding& binding)
    : num_blocks_(binding.num_blocks),
      block_size_(binding.block_size),
      num_kv_heads_(binding.num_kv_heads),
      head_size_(binding.head_size),
      device_(DeviceType::kDeviceCUDA),
      storage_spec_(MakeKVCacheStorageSpec(binding.storage_dtype,
                                           BlockStorageMode::kPlain)),
      external_pool_(true),
      external_pool_capsule_(binding.capsule) {
  CHECK(binding.valid()) << "invalid external KV pool binding";
  CHECK_GE(layer_index, 0);CHECK_LT(layer_index, binding.num_layers);
  const size_t tensor_bytes=binding.tensor_bytes_per_layer();
  auto* layer=static_cast<uint8_t*>(binding.base)+
      static_cast<size_t>(layer_index)*binding.layer_stride_bytes();
  key_pool_=tensor::Tensor(binding.storage_dtype,binding.num_blocks,
      binding.block_size*binding.num_kv_heads*binding.head_size,false,nullptr,layer);
  value_pool_=tensor::Tensor(binding.storage_dtype,binding.num_blocks,
      binding.block_size*binding.num_kv_heads*binding.head_size,false,nullptr,layer+tensor_bytes);
  key_pool_.set_device_type(DeviceType::kDeviceCUDA);
  value_pool_.set_device_type(DeviceType::kDeviceCUDA);

  static std::atomic<uint64_t> next_external_pool_id{uint64_t{1}<<63};
  pool_id_=next_external_pool_id.fetch_add(1);CHECK_NE(pool_id_,0u);
  generations_.resize(num_blocks_,0);compute_pins_.resize(num_blocks_,0);
  io_pins_.resize(num_blocks_,0);ref_counts_.resize(num_blocks_,0);
  std::vector<uint8_t> assigned(num_blocks_,0);
  for(int32_t id:binding.initial_blocks){CHECK_GE(id,0);CHECK_LT(id,num_blocks_);CHECK(!assigned[id]);assigned[id]=1;generations_[id]=1;ref_counts_[id]=1;}
  for(int32_t id:binding.allocatable_blocks){CHECK_GE(id,0);CHECK_LT(id,num_blocks_);CHECK(!assigned[id]);assigned[id]=1;free_queue_.push(id);}
  LOG(INFO)<<"External BlockAllocator bound: layer="<<layer_index
           <<" blocks="<<num_blocks_<<" initial="<<binding.initial_blocks.size()
           <<" allocatable="<<binding.allocatable_blocks.size()
           <<" tensor_bytes="<<tensor_bytes;
}

BlockAllocator::~BlockAllocator() {
  for (int32_t i = 0; i < num_blocks_; ++i) {
    CHECK_EQ(compute_pins_[i], 0) << "allocator destroyed with compute lease";
    CHECK_EQ(io_pins_[i], 0) << "allocator destroyed with I/O lease";
  }
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
  CHECK_LT(generations_[block_id], std::numeric_limits<uint64_t>::max());
  ++generations_[block_id];
  ref_counts_[block_id] = 1;
  return block_id;
}

void BlockAllocator::incref(int32_t block_id) {
  CHECK_GE(block_id, 0) << "Invalid block_id";
  CHECK_LT(block_id, num_blocks_) << "block_id out of range";
  CHECK_GT(ref_counts_[block_id], 0) << "Cannot incref a free block " << block_id;
  ++ref_counts_[block_id];
}

int32_t BlockAllocator::ref_count(int32_t block_id) const {
  CHECK_GE(block_id, 0);
  CHECK_LT(block_id, num_blocks_);
  return ref_counts_[block_id];
}

void BlockAllocator::free(int32_t block_id) {
  CHECK_GE(block_id, 0) << "Invalid block_id";
  CHECK_LT(block_id, num_blocks_) << "block_id out of range";
  CHECK_GT(ref_counts_[block_id], 0) << "Double free of block " << block_id;

  --ref_counts_[block_id];
  if (ref_counts_[block_id] == 0 &&
      generations_[block_id] != std::numeric_limits<uint64_t>::max()) {
    free_queue_.push(block_id);
  }
}

BlockHandle BlockAllocator::handle(int32_t block_id) const {
  CHECK_GE(block_id, 0);
  CHECK_LT(block_id, num_blocks_);
  CHECK_GT(ref_counts_[block_id], 0);
  return {block_id, generations_[block_id], pool_id_};
}

bool BlockAllocator::is_current(BlockHandle h) const {
  return h.pool_id == pool_id_ && h.block_id >= 0 && h.block_id < num_blocks_ &&
         ref_counts_[h.block_id] > 0 && generations_[h.block_id] == h.generation;
}

bool BlockAllocator::acquire_lease(BlockHandle h, BlockLeaseKind kind, BlockLease* lease) {
  if (lease == nullptr || *lease || !is_current(h)) return false;
  incref(h.block_id);
  auto& pins = kind == BlockLeaseKind::kCompute ? compute_pins_ : io_pins_;
  ++pins[h.block_id];
  lease->allocator_ = this;
  lease->handle_ = h;
  lease->kind_ = kind;
  return true;
}

void BlockAllocator::release_lease(BlockHandle h, BlockLeaseKind kind) {
  CHECK(is_current(h));
  auto& pins = kind == BlockLeaseKind::kCompute ? compute_pins_ : io_pins_;
  CHECK_GT(pins[h.block_id], 0);
  --pins[h.block_id];
  free(h.block_id);
}

int32_t BlockAllocator::compute_pins(int32_t id) const {
  CHECK_GE(id, 0); CHECK_LT(id, num_blocks_);
  return compute_pins_[id];
}
int32_t BlockAllocator::io_pins(int32_t id) const {
  CHECK_GE(id, 0); CHECK_LT(id, num_blocks_);
  return io_pins_[id];
}

BlockLease::~BlockLease() { reset(); }
BlockLease::BlockLease(BlockLease&& other) noexcept { *this = std::move(other); }
BlockLease& BlockLease::operator=(BlockLease&& other) noexcept {
  if (this != &other) {
    reset();
    allocator_ = std::exchange(other.allocator_, nullptr);
    handle_ = other.handle_;
    kind_ = other.kind_;
  }
  return *this;
}
void BlockLease::reset() {
  if (allocator_) {
    auto* owner = std::exchange(allocator_, nullptr);
    owner->release_lease(handle_, kind_);
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

size_t BlockAllocator::key_value_bytes_per_block() const {
  return static_cast<size_t>(block_size_) * num_kv_heads_ * head_size_ *
         DataTypeSize(storage_spec_.storage_dtype);
}

size_t BlockAllocator::scale_bytes_per_block() const {
  if (!storage_spec_.has_scales()) {
    return 0;
  }
  return static_cast<size_t>(block_size_) * num_kv_heads_ *
         DataTypeSize(storage_spec_.scale_dtype);
}

KVBlockPayloadPtrs BlockAllocator::get_block_payload_ptrs(int32_t block_id) const {
  CHECK_GE(block_id, 0);
  CHECK_LT(block_id, num_blocks_);

  const size_t key_value_bytes = key_value_bytes_per_block();
  const size_t key_value_offset = static_cast<size_t>(block_id) * key_value_bytes;
  KVBlockPayloadPtrs ptrs;
  ptrs.key = const_cast<uint8_t*>(key_pool_.ptr<uint8_t>(key_value_offset));
  ptrs.value = const_cast<uint8_t*>(value_pool_.ptr<uint8_t>(key_value_offset));
  ptrs.key_value_bytes = key_value_bytes;

  const size_t scale_bytes = scale_bytes_per_block();
  ptrs.scale_bytes = scale_bytes;
  if (scale_bytes > 0) {
    const size_t scale_offset = static_cast<size_t>(block_id) * scale_bytes;
    ptrs.key_scale = const_cast<uint8_t*>(key_scale_pool_.ptr<uint8_t>(scale_offset));
    ptrs.value_scale = const_cast<uint8_t*>(value_scale_pool_.ptr<uint8_t>(scale_offset));
  }
  return ptrs;
}

void BlockAllocator::copy_block_payload(int32_t source_block_id,
                                        int32_t destination_block_id) const {
  CHECK_NE(source_block_id, destination_block_id);
  CHECK_GT(ref_count(source_block_id), 0);
  CHECK_GT(ref_count(destination_block_id), 0);
  const auto source = get_block_payload_ptrs(source_block_id);
  const auto destination = get_block_payload_ptrs(destination_block_id);
  auto copy = [&](const void* src, void* dst, size_t bytes) {
    if (bytes == 0) return;
    if (device_ == DeviceType::kDeviceCPU) {
      std::memcpy(dst, src, bytes);
      return;
    }
#ifndef KUIPER_CPU_ONLY
    CUDADeviceAllocatorFactory::get_instance()->memcpy(
        src, dst, bytes, MemcpyKind::kMemcpyCUDA2CUDA, nullptr, true);
#else
    LOG(FATAL) << "CUDA block copy in CPU-only build";
#endif
  };
  copy(source.key, destination.key, source.key_value_bytes);
  copy(source.value, destination.value, source.key_value_bytes);
  copy(source.key_scale, destination.key_scale, source.scale_bytes);
  copy(source.value_scale, destination.value_scale, source.scale_bytes);
  payload_copy_bytes_ += 2 * source.key_value_bytes + 2 * source.scale_bytes;
}

}  // namespace base
