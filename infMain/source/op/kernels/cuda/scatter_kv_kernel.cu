// Scatter KV to paged blocks kernel implementation
#include "op/kernels/cuda/scatter_kv_kernel.cuh"
#include "cuda_type_utils.cuh"

namespace kernel {

// Simple kernel to copy K/V to the correct position in paged pool
template <typename T>
__global__ void scatter_kv_kernel(
    const T* __restrict__ key_src,     // [num_kv_heads * head_size]
    const T* __restrict__ value_src,   // [num_kv_heads * head_size]
    T* __restrict__ key_pool,
    T* __restrict__ value_pool,
    int32_t physical_block_id,
    int32_t offset_in_block,
    int32_t block_size,
    int32_t num_kv_heads,
    int32_t head_size) {

  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t total_size = num_kv_heads * head_size;

  if (idx >= total_size) return;

  // Calculate destination offset in pool
  // Pool layout: [num_blocks, block_size * num_kv_heads * head_size]
  int64_t block_stride = block_size * num_kv_heads * head_size;
  int64_t token_stride = num_kv_heads * head_size;

  int64_t dest_offset = physical_block_id * block_stride
                      + offset_in_block * token_stride
                      + idx;

  key_pool[dest_offset] = key_src[idx];
  value_pool[dest_offset] = value_src[idx];
}

void scatter_kv_to_page_cu(
    const tensor::Tensor& key_tensor,
    const tensor::Tensor& value_tensor,
    tensor::Tensor& key_pool,
    tensor::Tensor& value_pool,
    int32_t physical_block_id,
    int32_t offset_in_block,
    int32_t block_size,
    int32_t num_kv_heads,
    int32_t head_size,
    base::DeviceType device_type,
    CudaConfig* config) {

  UNUSED(device_type);
  cudaStream_t stream = config->stream;

  int32_t total_size = num_kv_heads * head_size;
  int32_t block_dim = 256;
  int32_t grid_dim = (total_size + block_dim - 1) / block_dim;

  if (key_tensor.data_type() == base::DataType::kDataTypeFp32) {
    scatter_kv_kernel<float><<<grid_dim, block_dim, 0, stream>>>(
        key_tensor.ptr<float>(),
        value_tensor.ptr<float>(),
        const_cast<float*>(key_pool.ptr<float>()),
        const_cast<float*>(value_pool.ptr<float>()),
        physical_block_id,
        offset_in_block,
        block_size,
        num_kv_heads,
        head_size);
  } else {
    CHECK_EQ(key_tensor.data_type(), base::DataType::kDataTypeBf16);
    scatter_kv_kernel<base::CudaBF16><<<grid_dim, block_dim, 0, stream>>>(
        reinterpret_cast<const base::CudaBF16*>(key_tensor.ptr<uint16_t>()),
        reinterpret_cast<const base::CudaBF16*>(value_tensor.ptr<uint16_t>()),
        reinterpret_cast<base::CudaBF16*>(const_cast<uint16_t*>(key_pool.ptr<uint16_t>())),
        reinterpret_cast<base::CudaBF16*>(const_cast<uint16_t*>(value_pool.ptr<uint16_t>())),
        physical_block_id,
        offset_in_block,
        block_size,
        num_kv_heads,
        head_size);
  }
}

}  // namespace kernel
