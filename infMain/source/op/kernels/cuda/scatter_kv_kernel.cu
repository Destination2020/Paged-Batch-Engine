// Scatter KV to paged blocks kernel implementation
#include "op/kernels/cuda/scatter_kv_kernel.cuh"

#include <cuda_fp8.h>

#include "cuda_type_utils.cuh"

namespace kernel {

namespace {

__device__ inline float warp_max(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
  }
  return value;
}

}  // namespace

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

// Batched scatter KV kernel: each block handles one token in the batch
template <typename T>
__global__ void scatter_kv_batch_kernel(
    const T* __restrict__ key_input,     // [batch_tokens, kv_dim]
    const T* __restrict__ value_input,   // [batch_tokens, kv_dim]
    T* __restrict__ key_pool,
    T* __restrict__ value_pool,
    const int32_t* __restrict__ slot_mapping,  // [batch_tokens]
    int32_t block_size,
    int32_t kv_dim) {

  int32_t token_idx = blockIdx.x;
  int32_t slot = slot_mapping[token_idx];
  int32_t block_id = slot / block_size;
  int32_t offset = slot % block_size;

  int64_t block_stride = static_cast<int64_t>(block_size) * kv_dim;
  int64_t dest_base = block_id * block_stride + offset * kv_dim;

  const T* key_src = key_input + token_idx * kv_dim;
  const T* val_src = value_input + token_idx * kv_dim;
  T* key_dst = key_pool + dest_base;
  T* val_dst = value_pool + dest_base;

  for (int32_t i = threadIdx.x; i < kv_dim; i += blockDim.x) {
    key_dst[i] = key_src[i];
    val_dst[i] = val_src[i];
  }
}

template <typename T>
__global__ void scatter_kv_batch_fp8_e4m3_kernel(
    const T* __restrict__ key_input,         // [batch_tokens, kv_dim]
    const T* __restrict__ value_input,       // [batch_tokens, kv_dim]
    int8_t* __restrict__ key_pool,           // raw fp8 bit-patterns
    int8_t* __restrict__ value_pool,
    float* __restrict__ key_scale_pool,      // [num_blocks, block_size * num_kv_heads]
    float* __restrict__ value_scale_pool,
    const int32_t* __restrict__ slot_mapping,
    int32_t block_size,
    int32_t num_kv_heads,
    int32_t head_size,
    int32_t kv_dim) {
  constexpr float kFp8Max = 448.f;
  constexpr int32_t kThreadsPerWarp = 32;
  constexpr int32_t kMaxWarps = 8;

  const int32_t token_idx = blockIdx.x;
  const int32_t kv_head_idx = blockIdx.y;
  const int32_t slot = slot_mapping[token_idx];
  if (slot < 0) {
    return;
  }

  const int32_t block_id = slot / block_size;
  const int32_t offset_in_block = slot % block_size;
  const int32_t lane = threadIdx.x & (kThreadsPerWarp - 1);
  const int32_t warp_id = threadIdx.x / kThreadsPerWarp;
  const int32_t num_warps = blockDim.x / kThreadsPerWarp;

  __shared__ float warp_k_amax[kMaxWarps];
  __shared__ float warp_v_amax[kMaxWarps];
  __shared__ float k_scale;
  __shared__ float v_scale;
  __shared__ float k_inv_scale;
  __shared__ float v_inv_scale;

  const T* key_src = key_input + static_cast<int64_t>(token_idx) * kv_dim + kv_head_idx * head_size;
  const T* value_src = value_input + static_cast<int64_t>(token_idx) * kv_dim + kv_head_idx * head_size;

  float local_k_amax = 0.f;
  float local_v_amax = 0.f;
  for (int32_t i = threadIdx.x; i < head_size; i += blockDim.x) {
    local_k_amax = fmaxf(local_k_amax, fabsf(scalar_to_float(key_src[i])));
    local_v_amax = fmaxf(local_v_amax, fabsf(scalar_to_float(value_src[i])));
  }

  local_k_amax = warp_max(local_k_amax);
  local_v_amax = warp_max(local_v_amax);
  if (lane == 0) {
    warp_k_amax[warp_id] = local_k_amax;
    warp_v_amax[warp_id] = local_v_amax;
  }
  __syncthreads();

  if (warp_id == 0) {
    float block_k_amax = lane < num_warps ? warp_k_amax[lane] : 0.f;
    float block_v_amax = lane < num_warps ? warp_v_amax[lane] : 0.f;
    block_k_amax = warp_max(block_k_amax);
    block_v_amax = warp_max(block_v_amax);
    if (lane == 0) {
      k_scale = block_k_amax > 0.f ? block_k_amax / kFp8Max : 1.f;
      v_scale = block_v_amax > 0.f ? block_v_amax / kFp8Max : 1.f;
      k_inv_scale = block_k_amax > 0.f ? kFp8Max / block_k_amax : 0.f;
      v_inv_scale = block_v_amax > 0.f ? kFp8Max / block_v_amax : 0.f;
      const int64_t scale_offset =
          static_cast<int64_t>(block_id) * (block_size * num_kv_heads) +
          static_cast<int64_t>(offset_in_block) * num_kv_heads + kv_head_idx;
      key_scale_pool[scale_offset] = k_scale;
      value_scale_pool[scale_offset] = v_scale;
    }
  }
  __syncthreads();

  const int64_t block_stride = static_cast<int64_t>(block_size) * kv_dim;
  const int64_t dst_base =
      static_cast<int64_t>(block_id) * block_stride +
      static_cast<int64_t>(offset_in_block) * kv_dim +
      static_cast<int64_t>(kv_head_idx) * head_size;

  for (int32_t i = threadIdx.x; i < head_size; i += blockDim.x) {
    const float k_val = k_inv_scale > 0.f ? scalar_to_float(key_src[i]) * k_inv_scale : 0.f;
    const float v_val = v_inv_scale > 0.f ? scalar_to_float(value_src[i]) * v_inv_scale : 0.f;
    key_pool[dst_base + i] =
        static_cast<int8_t>(__nv_cvt_float_to_fp8(k_val, __NV_SATFINITE, __NV_E4M3));
    value_pool[dst_base + i] =
        static_cast<int8_t>(__nv_cvt_float_to_fp8(v_val, __NV_SATFINITE, __NV_E4M3));
  }
}

void scatter_kv_batch_to_pages_cu(
    const tensor::Tensor& key_tensor,
    const tensor::Tensor& value_tensor,
    tensor::Tensor& key_pool,
    tensor::Tensor& value_pool,
    const tensor::Tensor& slot_mapping,
    int32_t block_size,
    int32_t num_kv_heads,
    int32_t head_size,
    int32_t batch_tokens,
    base::DeviceType device_type,
    CudaConfig* config) {

  UNUSED(device_type);
  cudaStream_t stream = config->stream;
  int32_t kv_dim = num_kv_heads * head_size;
  int32_t block_dim = 256;

  if (key_tensor.data_type() == base::DataType::kDataTypeFp32) {
    scatter_kv_batch_kernel<float><<<batch_tokens, block_dim, 0, stream>>>(
        key_tensor.ptr<float>(),
        value_tensor.ptr<float>(),
        const_cast<float*>(key_pool.ptr<float>()),
        const_cast<float*>(value_pool.ptr<float>()),
        slot_mapping.ptr<int32_t>(),
        block_size, kv_dim);
  } else {
    CHECK_EQ(key_tensor.data_type(), base::DataType::kDataTypeBf16);
    scatter_kv_batch_kernel<base::CudaBF16><<<batch_tokens, block_dim, 0, stream>>>(
        reinterpret_cast<const base::CudaBF16*>(key_tensor.ptr<uint16_t>()),
        reinterpret_cast<const base::CudaBF16*>(value_tensor.ptr<uint16_t>()),
        reinterpret_cast<base::CudaBF16*>(const_cast<uint16_t*>(key_pool.ptr<uint16_t>())),
        reinterpret_cast<base::CudaBF16*>(const_cast<uint16_t*>(value_pool.ptr<uint16_t>())),
        slot_mapping.ptr<int32_t>(),
        block_size, kv_dim);
  }
}

void scatter_kv_batch_to_pages_fp8_e4m3_cu(
    const tensor::Tensor& key_tensor,
    const tensor::Tensor& value_tensor,
    tensor::Tensor& key_pool,
    tensor::Tensor& value_pool,
    tensor::Tensor& key_scale_pool,
    tensor::Tensor& value_scale_pool,
    const tensor::Tensor& slot_mapping,
    int32_t block_size,
    int32_t num_kv_heads,
    int32_t head_size,
    int32_t batch_tokens,
    base::DeviceType device_type,
    CudaConfig* config) {
  UNUSED(device_type);
  CHECK_NE(config, nullptr);
  CHECK_EQ(key_pool.data_type(), base::DataType::kDataTypeInt8);
  CHECK_EQ(value_pool.data_type(), base::DataType::kDataTypeInt8);
  CHECK_EQ(key_scale_pool.data_type(), base::DataType::kDataTypeFp32);
  CHECK_EQ(value_scale_pool.data_type(), base::DataType::kDataTypeFp32);
  CHECK_EQ(slot_mapping.data_type(), base::DataType::kDataTypeInt32);

  cudaStream_t stream = config->stream;
  const int32_t kv_dim = num_kv_heads * head_size;
  dim3 grid(batch_tokens, num_kv_heads);
  constexpr int32_t kBlockDim = 128;

  if (key_tensor.data_type() == base::DataType::kDataTypeFp32) {
    scatter_kv_batch_fp8_e4m3_kernel<float><<<grid, kBlockDim, 0, stream>>>(
        key_tensor.ptr<float>(),
        value_tensor.ptr<float>(),
        key_pool.ptr<int8_t>(),
        value_pool.ptr<int8_t>(),
        key_scale_pool.ptr<float>(),
        value_scale_pool.ptr<float>(),
        slot_mapping.ptr<int32_t>(),
        block_size,
        num_kv_heads,
        head_size,
        kv_dim);
    return;
  }

  CHECK_EQ(key_tensor.data_type(), base::DataType::kDataTypeBf16);
  scatter_kv_batch_fp8_e4m3_kernel<base::CudaBF16><<<grid, kBlockDim, 0, stream>>>(
      reinterpret_cast<const base::CudaBF16*>(key_tensor.ptr<uint16_t>()),
      reinterpret_cast<const base::CudaBF16*>(value_tensor.ptr<uint16_t>()),
      key_pool.ptr<int8_t>(),
      value_pool.ptr<int8_t>(),
      key_scale_pool.ptr<float>(),
      value_scale_pool.ptr<float>(),
      slot_mapping.ptr<int32_t>(),
      block_size,
      num_kv_heads,
      head_size,
      kv_dim);
}

}  // namespace kernel
