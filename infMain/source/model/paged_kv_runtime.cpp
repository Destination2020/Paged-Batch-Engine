#include "model/paged_kv_runtime.h"
#include <glog/logging.h>
#include "base/alloc.h"
#include "base/backend_runtime.h"
#include "base/cuda_backend_runtime.h"
#include "op/kernels/cuda/paged_mha_fast_kernel.cuh"
#include "op/kernels/cuda/paged_mha_kernel.cuh"
#include "op/kernels/cuda/paged_mha_prefill_kernel.cuh"
#include "op/kernels/cuda/scatter_kv_kernel.cuh"

namespace model {

namespace {

class UnsupportedPagedKVRuntime : public PagedKVRuntime {
 public:
  UnsupportedPagedKVRuntime(base::DeviceType device_type,
                            std::shared_ptr<base::DeviceContext> device_context)
      : device_type_(device_type), device_context_(std::move(device_context)) {}

  void scatter(const tensor::Tensor& /*key_tensor*/,
               const tensor::Tensor& /*value_tensor*/,
               const base::KVPoolView& /*pool*/,
               const tensor::Tensor& /*slot_mapping*/,
               int32_t /*block_size*/,
               int32_t /*num_kv_heads*/,
               int32_t /*head_size*/,
               int32_t /*batch_tokens*/) const override {
    fail("scatter");
  }

  void scatter_single_token(const tensor::Tensor& /*key_tensor*/,
                            const tensor::Tensor& /*value_tensor*/,
                            const base::KVPoolView& /*pool*/,
                            int32_t /*physical_block_id*/,
                            int32_t /*offset_in_block*/,
                            int32_t /*block_size*/,
                            int32_t /*num_kv_heads*/,
                            int32_t /*head_size*/) const override {
    fail("scatter_single_token");
  }

  bool decode(const base::KVPoolView& /*pool*/,
              const PagedKVDecodeRuntimeArgs& /*args*/) const override {
    fail("decode");
    return false;
  }

  void prefill(const base::KVPoolView& /*pool*/,
               const PagedKVPrefillRuntimeArgs& /*args*/) const override {
    fail("prefill");
  }

 protected:
  [[noreturn]] void fail(const char* method) const {
    LOG(FATAL) << "PagedKVRuntime::" << method
               << " is not implemented for device type " << device_type_
               << ". Runtime kind=" << runtime_kind()
               << ", backend context="
               << (device_context_ != nullptr
                       ? static_cast<int>(device_context_->backend)
                       : static_cast<int>(base::BackendType::kUnknown));
  }

  virtual const char* runtime_kind() const {
    return "unsupported";
  }

 private:
  base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;
  std::shared_ptr<base::DeviceContext> device_context_;
};

class CpuPagedKVRuntime final : public UnsupportedPagedKVRuntime {
 public:
  explicit CpuPagedKVRuntime(std::shared_ptr<base::DeviceContext> device_context)
      : UnsupportedPagedKVRuntime(base::DeviceType::kDeviceCPU,
                                  std::move(device_context)) {}

 private:
  const char* runtime_kind() const override {
    return "cpu-stub";
  }
};

class FutureBackendPagedKVRuntime final : public UnsupportedPagedKVRuntime {
 public:
  FutureBackendPagedKVRuntime(base::DeviceType device_type,
                              std::shared_ptr<base::DeviceContext> device_context)
      : UnsupportedPagedKVRuntime(device_type, std::move(device_context)) {}

 private:
  const char* runtime_kind() const override {
    return "future-backend-stub";
  }
};

class CudaPagedKVRuntime final : public PagedKVRuntime {
 public:
  explicit CudaPagedKVRuntime(std::shared_ptr<base::DeviceContext> device_context)
      : device_context_(std::move(device_context)) {}

  void scatter(
      const tensor::Tensor& key_tensor,
      const tensor::Tensor& value_tensor,
      const base::KVPoolView& pool,
      const tensor::Tensor& slot_mapping,
      int32_t block_size,
      int32_t num_kv_heads,
      int32_t head_size,
      int32_t batch_tokens) const override;

  void scatter_single_token(
      const tensor::Tensor& key_tensor,
      const tensor::Tensor& value_tensor,
      const base::KVPoolView& pool,
      int32_t physical_block_id,
      int32_t offset_in_block,
      int32_t block_size,
      int32_t num_kv_heads,
      int32_t head_size) const override;

  bool decode(const base::KVPoolView& pool,
              const PagedKVDecodeRuntimeArgs& args) const override;

  void prefill(const base::KVPoolView& pool,
               const PagedKVPrefillRuntimeArgs& args) const override;

 private:
  void require_cuda_runtime_or_die() const;

  kernel::CudaConfig* cuda_config_or_die() const;

  std::shared_ptr<base::DeviceAllocator> pinned_host_allocator_or_die() const;

  std::shared_ptr<base::DeviceAllocator> device_allocator_or_die() const;

  void* transfer_or_compute_queue() const;

  void runtime_copy_or_die(const void* src,
                           void* dst,
                           size_t byte_size,
                           base::CopyDirection direction,
                           void* queue,
                           bool need_sync) const;

 private:
  std::shared_ptr<base::DeviceContext> device_context_;
};

void CudaPagedKVRuntime::require_cuda_runtime_or_die() const {
  CHECK(device_context_ != nullptr);
  CHECK(device_context_->backend == base::BackendType::kCUDA)
      << "Current paged KV runtime requires a CUDA backend context.";
  CHECK(device_context_->runtime != nullptr);
}

kernel::CudaConfig* CudaPagedKVRuntime::cuda_config_or_die() const {
  require_cuda_runtime_or_die();
  auto cuda_context = base::cuda_config_from_device_context(device_context_);
  CHECK_NE(cuda_context, nullptr);
  return cuda_context.get();
}

std::shared_ptr<base::DeviceAllocator> CudaPagedKVRuntime::pinned_host_allocator_or_die() const {
  require_cuda_runtime_or_die();
  auto allocator = device_context_->runtime->pinned_host_allocator();
  if (allocator != nullptr) {
    return allocator;
  }
  return base::PinnedCPUDeviceAllocatorFactory::get_instance();
}

std::shared_ptr<base::DeviceAllocator> CudaPagedKVRuntime::device_allocator_or_die() const {
  require_cuda_runtime_or_die();
  auto allocator = device_context_->runtime->device_allocator();
  CHECK(allocator != nullptr) << "Device allocator is not initialized for paged KV runtime";
  return allocator;
}

void* CudaPagedKVRuntime::transfer_or_compute_queue() const {
  CHECK(device_context_ != nullptr);
  return device_context_->transfer_queue != nullptr ? device_context_->transfer_queue
                                                    : device_context_->compute_queue;
}

void CudaPagedKVRuntime::runtime_copy_or_die(const void* src,
                                             void* dst,
                                             size_t byte_size,
                                             base::CopyDirection direction,
                                             void* queue,
                                             bool need_sync) const {
  require_cuda_runtime_or_die();
  base::CopyParams params;
  params.src = src;
  params.dst = dst;
  params.byte_size = byte_size;
  params.direction = direction;
  params.queue = queue;
  params.need_sync = need_sync;
  auto status = device_context_->runtime->copy(params);
  CHECK(status) << status.get_err_msg();
}

void CudaPagedKVRuntime::scatter(
    const tensor::Tensor& key_tensor,
    const tensor::Tensor& value_tensor,
    const base::KVPoolView& pool,
    const tensor::Tensor& slot_mapping,
    int32_t block_size,
    int32_t num_kv_heads,
    int32_t head_size,
    int32_t batch_tokens) const {
  kernel::CudaConfig* config = cuda_config_or_die();
  CHECK(pool.valid());
  if (pool.uses_fp8_storage()) {
    kernel::scatter_kv_batch_to_pages_fp8_e4m3_cu(
        key_tensor,
        value_tensor,
        pool.key_pool(),
        pool.value_pool(),
        pool.key_scale_pool(),
        pool.value_scale_pool(),
        slot_mapping,
        block_size,
        num_kv_heads,
        head_size,
        batch_tokens,
        base::DeviceType::kDeviceCUDA,
        config);
    return;
  }

  kernel::scatter_kv_batch_to_pages_cu(
      key_tensor,
      value_tensor,
      pool.key_pool(),
      pool.value_pool(),
      slot_mapping,
      block_size,
      num_kv_heads,
      head_size,
      batch_tokens,
      base::DeviceType::kDeviceCUDA,
      config);
}

void CudaPagedKVRuntime::scatter_single_token(
    const tensor::Tensor& key_tensor,
    const tensor::Tensor& value_tensor,
    const base::KVPoolView& pool,
    int32_t physical_block_id,
    int32_t offset_in_block,
    int32_t block_size,
    int32_t num_kv_heads,
    int32_t head_size) const {
  kernel::CudaConfig* config = cuda_config_or_die();
  CHECK(pool.valid());
  if (!pool.uses_fp8_storage()) {
    kernel::scatter_kv_to_page_cu(
        key_tensor,
        value_tensor,
        pool.key_pool(),
        pool.value_pool(),
        physical_block_id,
        offset_in_block,
        block_size,
        num_kv_heads,
        head_size,
        base::DeviceType::kDeviceCUDA,
        config);
    return;
  }

  auto pinned_alloc = pinned_host_allocator_or_die();
  auto device_alloc = device_allocator_or_die();
  tensor::Tensor slot_mapping_host(base::DataType::kDataTypeInt32, 1, true, pinned_alloc);
  slot_mapping_host.set_device_type(base::DeviceType::kDeviceCPU);
  slot_mapping_host.index<int32_t>(0) = physical_block_id * block_size + offset_in_block;

  tensor::Tensor slot_mapping_device(base::DataType::kDataTypeInt32, 1, true, device_alloc);
  slot_mapping_device.set_device_type(base::DeviceType::kDeviceCUDA);
  runtime_copy_or_die(slot_mapping_host.ptr<int32_t>(),
                      slot_mapping_device.ptr<int32_t>(),
                      sizeof(int32_t),
                      base::CopyDirection::kHostToDevice,
                      transfer_or_compute_queue(),
                      false);

  scatter(key_tensor, value_tensor, pool, slot_mapping_device,
          block_size, num_kv_heads, head_size, 1);
}

bool CudaPagedKVRuntime::decode(const base::KVPoolView& pool,
                                const PagedKVDecodeRuntimeArgs& args) const {
  CHECK_NE(args.queries, nullptr);
  CHECK_NE(args.outputs, nullptr);
  CHECK_NE(args.block_tables, nullptr);
  CHECK_NE(args.seq_lens, nullptr);
  CHECK_NE(args.partial_out, nullptr);
  CHECK_NE(args.partial_max, nullptr);
  CHECK_NE(args.partial_sum, nullptr);
  kernel::CudaConfig* cuda_config = cuda_config_or_die();

  CHECK(pool.valid());
  if (pool.uses_fp8_storage()) {
    if (!kernel::splitkv_batched_paged_mha_fast_decode_cu(
            args.batch_size, args.head_num, args.head_size, args.kv_mul,
            *args.queries, *args.outputs,
            pool.key_pool(),
            pool.value_pool(),
            *args.block_tables, *args.seq_lens,
            args.max_blocks_per_seq, args.block_size, args.num_kv_heads,
            *args.partial_out,
            *args.partial_max,
            *args.partial_sum,
            base::DeviceType::kDeviceCUDA, cuda_config)) {
      const bool launched = kernel::splitkv_batched_paged_mha_fp8_decode_cu(
          args.batch_size, args.head_num, args.head_size, args.kv_mul,
          *args.queries, *args.outputs,
          pool.key_pool(),
          pool.value_pool(),
          pool.key_scale_pool(),
          pool.value_scale_pool(),
          *args.block_tables, *args.seq_lens,
          args.max_blocks_per_seq, args.block_size, args.num_kv_heads,
          *args.partial_out,
          *args.partial_max,
          *args.partial_sum,
          base::DeviceType::kDeviceCUDA, cuda_config);
      CHECK(launched) << "FP8 KV cache decode launch failed";
    }
    return true;
  }

  if (!kernel::splitkv_batched_paged_mha_fast_decode_cu(
          args.batch_size, args.head_num, args.head_size, args.kv_mul,
          *args.queries, *args.outputs,
          pool.key_pool(),
          pool.value_pool(),
          *args.block_tables, *args.seq_lens,
          args.max_blocks_per_seq, args.block_size, args.num_kv_heads,
          *args.partial_out,
          *args.partial_max,
          *args.partial_sum,
          base::DeviceType::kDeviceCUDA, cuda_config)) {
    kernel::splitkv_batched_paged_mha_decode_cu(
        args.batch_size, args.head_num, args.head_size, args.kv_mul,
        *args.queries, *args.outputs,
        pool.key_pool(),
        pool.value_pool(),
        *args.block_tables, *args.seq_lens,
        args.max_blocks_per_seq, args.block_size, args.num_kv_heads,
        *args.partial_out,
        *args.partial_max,
        *args.partial_sum,
        base::DeviceType::kDeviceCUDA, cuda_config);
  }
  return true;
}

void CudaPagedKVRuntime::prefill(const base::KVPoolView& pool,
                                 const PagedKVPrefillRuntimeArgs& args) const {
  CHECK_NE(args.queries, nullptr);
  CHECK_NE(args.chunk_keys, nullptr);
  CHECK_NE(args.chunk_values, nullptr);
  CHECK_NE(args.outputs, nullptr);
  CHECK_NE(args.block_tables, nullptr);
  CHECK_NE(args.request_indices, nullptr);
  CHECK_NE(args.base_context_lens, nullptr);
  CHECK_NE(args.chunk_row_starts, nullptr);
  CHECK_NE(args.local_token_offsets, nullptr);
  CHECK_NE(args.partial_out, nullptr);
  CHECK_NE(args.partial_max, nullptr);
  CHECK_NE(args.partial_sum, nullptr);
  kernel::CudaConfig* cuda_config = cuda_config_or_die();

  CHECK(pool.valid());
  if (pool.uses_fp8_storage()) {
    kernel::batched_paged_mha_prefill_fp8_cu(
        args.batch_size, args.head_num, args.head_size, args.kv_mul,
        *args.queries, *args.chunk_keys, *args.chunk_values, *args.outputs,
        pool.key_pool(),
        pool.value_pool(),
        pool.key_scale_pool(),
        pool.value_scale_pool(),
        *args.block_tables,
        *args.request_indices,
        *args.base_context_lens,
        *args.chunk_row_starts,
        *args.local_token_offsets,
        *args.partial_out,
        *args.partial_max,
        *args.partial_sum,
        args.max_blocks_per_seq, args.max_prefix_blocks,
        args.block_size, args.num_kv_heads,
        base::DeviceType::kDeviceCUDA, cuda_config);
    return;
  }

  kernel::batched_paged_mha_prefill_cu(
      args.batch_size, args.head_num, args.head_size, args.kv_mul,
      *args.queries, *args.chunk_keys, *args.chunk_values, *args.outputs,
      pool.key_pool(),
      pool.value_pool(),
      *args.block_tables,
      *args.request_indices,
      *args.base_context_lens,
      *args.chunk_row_starts,
      *args.local_token_offsets,
      *args.partial_out,
      *args.partial_max,
      *args.partial_sum,
      args.max_blocks_per_seq, args.max_prefix_blocks,
      args.block_size, args.num_kv_heads,
      base::DeviceType::kDeviceCUDA, cuda_config);
}

}  // namespace

std::shared_ptr<PagedKVRuntime> create_paged_kv_runtime(
    base::DeviceType device_type,
    const std::shared_ptr<base::DeviceContext>& device_context) {
  switch (device_type) {
    case base::DeviceType::kDeviceCPU:
      return std::make_shared<CpuPagedKVRuntime>(device_context);
    case base::DeviceType::kDeviceCUDA:
      return std::make_shared<CudaPagedKVRuntime>(device_context);
    case base::DeviceType::kDeviceHIP:
    case base::DeviceType::kDeviceTPU:
    case base::DeviceType::kDeviceUnknown:
    default:
      return std::make_shared<FutureBackendPagedKVRuntime>(device_type, device_context);
  }
}

}  // namespace model
