// Updated on March 23, 2026
#include <climits>
#include <base/bf16.h>
#include "op/kernels/cuda/sampler_kernel.cuh"
#include "../kernels_interface.h"
#include "cuda_type_utils.cuh"
#include "argmax_kernel.cuh"

namespace kernel {

template <typename T>
__device__ inline float argmax_value_to_float(T value);

template <>
__device__ inline float argmax_value_to_float<float>(float value) {
  return value;
}

template <>
__device__ inline float argmax_value_to_float<base::CudaBF16>(base::CudaBF16 value) {
  return __bfloat162float(value);
}

__forceinline__ __device__ void warp_reduce_argmax(float& val, size_t& ptr) {
  float tmp_val;
  size_t tmp_ptr;
  unsigned int mask = __ballot_sync(0xFFFFFFFF, true);
  for (unsigned int k = (warpSize >> 1); k > 0; k >>= 1) {
    tmp_val = __shfl_down_sync(mask, val, k, warpSize);
    tmp_ptr = __shfl_down_sync(mask, ptr, k, warpSize);
    if (ptr == SIZE_MAX || tmp_ptr == SIZE_MAX) continue;
    if (tmp_val > val) {
      val = tmp_val;
      ptr = tmp_ptr;
    } else if (tmp_val == val && tmp_ptr < ptr) {
      ptr = tmp_ptr;
    }
  }
}

__forceinline__ __device__ void block_reduce_argmax(float& val, size_t& ptr, float* shared_value,
                                                    size_t* shared_ptr) {
  int lane_id = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;

  warp_reduce_argmax(val, ptr);

  __syncthreads();
  if (lane_id == 0) {
    shared_value[warp_id] = val;
    shared_ptr[warp_id] = ptr;
  }

  __syncthreads();
  if (threadIdx.x < blockDim.x / warpSize) {
    val = shared_value[lane_id];
    ptr = shared_ptr[lane_id];
  } else {
    val = 0;
    ptr = SIZE_MAX;
  }

  if (warp_id == 0) {
    warp_reduce_argmax(val, ptr);
  }
}

template <typename T>
__global__ void argmax_kernel_impl(const T* input_ptr, size_t size, size_t* output_idx) {
  __shared__ size_t shared_max_ptr[32];
  __shared__ float shared_max_value[32];
  const uint32_t tid = threadIdx.x;
  if (tid >= size) {
    return;
  }

  size_t max_index = threadIdx.x;
  float max_value = argmax_value_to_float(input_ptr[max_index]);
  for (size_t i = tid; i < size; i += blockDim.x) {
    const float value = argmax_value_to_float(input_ptr[i]);
    if (value > max_value) {
      max_index = i;
      max_value = value;
    }
  }

  block_reduce_argmax(max_value, max_index, shared_max_value, shared_max_ptr);
  __syncthreads();
  if (threadIdx.x == 0) {
    *output_idx = max_index;
  }
}

template <typename T>
static size_t argmax_kernel_dispatch(const T* input_ptr, size_t size, void* stream) {
  std::shared_ptr<base::DeviceAllocator> alloc_cu =
      base::CUDADeviceAllocatorFactory::get_instance();
  size_t* index = static_cast<size_t*>(alloc_cu->allocate(sizeof(size_t)));
  size_t output_index = 0;
  if (!stream) {
    argmax_kernel_impl<T><<<1, 512>>>(input_ptr, size, index);
    cudaMemcpy(&output_index, index, sizeof(size_t), cudaMemcpyDeviceToHost);
  } else {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    argmax_kernel_impl<T><<<1, 512, 0, stream_>>>(input_ptr, size, index);
    cudaMemcpyAsync(&output_index, index, sizeof(size_t), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
  }
  return output_index;
}

size_t argmax_kernel_cu(const float* input_ptr, size_t size, void* stream) {
  return argmax_kernel_dispatch<float>(input_ptr, size, stream);
}

size_t argmax_kernel_cu_bf16(const uint16_t* input_ptr, size_t size, void* stream) {
  return argmax_kernel_dispatch<base::CudaBF16>(
      reinterpret_cast<const base::CudaBF16*>(input_ptr), size, stream);
}

namespace {

template <typename T>
__global__ void gather_logits_rows_kernel(const T* __restrict__ logits,
                                          const int32_t* __restrict__ row_indices,
                                          T* __restrict__ gathered_logits,
                                          int32_t vocab_size) {
  const int32_t sample_idx = blockIdx.x;
  const int32_t src_row = row_indices[sample_idx];
  const T* src = logits + static_cast<int64_t>(src_row) * vocab_size;
  T* dst = gathered_logits + static_cast<int64_t>(sample_idx) * vocab_size;
  for (int32_t vocab_idx = threadIdx.x; vocab_idx < vocab_size; vocab_idx += blockDim.x) {
    dst[vocab_idx] = src[vocab_idx];
  }
}

__device__ inline void sample_pick_better_argmax(float candidate_value,
                                                 int32_t candidate_index,
                                                 float& best_value,
                                                 int32_t& best_index) {
  if (candidate_index == INT_MAX) {
    return;
  }
  if (best_index == INT_MAX || candidate_value > best_value ||
      (candidate_value == best_value && candidate_index < best_index)) {
    best_value = candidate_value;
    best_index = candidate_index;
  }
}

__device__ inline void sample_warp_reduce_argmax(float& best_value, int32_t& best_index) {
  for (int32_t offset = warpSize / 2; offset > 0; offset >>= 1) {
    const float other_value = __shfl_down_sync(0xffffffffu, best_value, offset);
    const int32_t other_index = __shfl_down_sync(0xffffffffu, best_index, offset);
    sample_pick_better_argmax(other_value, other_index, best_value, best_index);
  }
}

template <typename T>
__global__ void argmax_rows_kernel(const T* __restrict__ logits,
                                   int32_t* __restrict__ token_ids,
                                   int32_t vocab_size) {
  const int32_t sample_idx = blockIdx.x;
  const T* row = logits + static_cast<int64_t>(sample_idx) * vocab_size;

  float best_value = 0.f;
  int32_t best_index = INT_MAX;
  for (int32_t vocab_idx = threadIdx.x; vocab_idx < vocab_size; vocab_idx += blockDim.x) {
    const float value = scalar_to_float(row[vocab_idx]);
    sample_pick_better_argmax(value, vocab_idx, best_value, best_index);
  }

  sample_warp_reduce_argmax(best_value, best_index);

  __shared__ float shared_best_values[32];
  __shared__ int32_t shared_best_indices[32];
  const int32_t lane = threadIdx.x & (warpSize - 1);
  const int32_t warp_id = threadIdx.x / warpSize;
  const int32_t num_warps = blockDim.x / warpSize;

  if (lane == 0) {
    shared_best_values[warp_id] = best_value;
    shared_best_indices[warp_id] = best_index;
  }
  __syncthreads();

  if (warp_id == 0) {
    best_value = (lane < num_warps) ? shared_best_values[lane] : 0.f;
    best_index = (lane < num_warps) ? shared_best_indices[lane] : INT_MAX;
    sample_warp_reduce_argmax(best_value, best_index);
    if (lane == 0) {
      token_ids[sample_idx] = best_index;
    }
  }
}

template <typename T>
__global__ void argmax_selected_rows_kernel(const T* __restrict__ logits,
                                            const int32_t* __restrict__ row_indices,
                                            int32_t* __restrict__ token_ids,
                                            int32_t vocab_size) {
  const int32_t sample_idx = blockIdx.x;
  const int32_t row_idx = row_indices[sample_idx];
  const T* row = logits + static_cast<int64_t>(row_idx) * vocab_size;

  float best_value = 0.f;
  int32_t best_index = INT_MAX;
  for (int32_t vocab_idx = threadIdx.x; vocab_idx < vocab_size; vocab_idx += blockDim.x) {
    const float value = scalar_to_float(row[vocab_idx]);
    sample_pick_better_argmax(value, vocab_idx, best_value, best_index);
  }

  sample_warp_reduce_argmax(best_value, best_index);

  __shared__ float shared_best_values[32];
  __shared__ int32_t shared_best_indices[32];
  const int32_t lane = threadIdx.x & (warpSize - 1);
  const int32_t warp_id = threadIdx.x / warpSize;
  const int32_t num_warps = blockDim.x / warpSize;

  if (lane == 0) {
    shared_best_values[warp_id] = best_value;
    shared_best_indices[warp_id] = best_index;
  }
  __syncthreads();

  if (warp_id == 0) {
    best_value = (lane < num_warps) ? shared_best_values[lane] : 0.f;
    best_index = (lane < num_warps) ? shared_best_indices[lane] : INT_MAX;
    sample_warp_reduce_argmax(best_value, best_index);
    if (lane == 0) {
      token_ids[sample_idx] = best_index;
    }
  }
}

template <typename T>
void launch_gather_logits_rows(const tensor::Tensor& logits,
                               const tensor::Tensor& row_indices,
                               tensor::Tensor& gathered_logits,
                               cudaStream_t stream) {
  constexpr int32_t kThreads = 256;
  const int32_t sample_count = row_indices.get_dim(0);
  gather_logits_rows_kernel<T><<<sample_count, kThreads, 0, stream>>>(
      logits.ptr<T>(), row_indices.ptr<int32_t>(), gathered_logits.ptr<T>(),
      logits.get_dim(1));
}

template <typename T>
void launch_argmax_rows(const tensor::Tensor& logits,
                        tensor::Tensor& token_ids,
                        cudaStream_t stream) {
  constexpr int32_t kThreads = 256;
  const int32_t sample_count = logits.get_dim(0);
  argmax_rows_kernel<T><<<sample_count, kThreads, 0, stream>>>(
      logits.ptr<T>(), token_ids.ptr<int32_t>(), logits.get_dim(1));
}

template <typename T>
void launch_argmax_selected_rows(const tensor::Tensor& logits,
                                 const tensor::Tensor& row_indices,
                                 tensor::Tensor& token_ids,
                                 cudaStream_t stream) {
  constexpr int32_t kThreads = 256;
  const int32_t sample_count = row_indices.get_dim(0);
  argmax_selected_rows_kernel<T><<<sample_count, kThreads, 0, stream>>>(
      logits.ptr<T>(), row_indices.ptr<int32_t>(), token_ids.ptr<int32_t>(),
      logits.get_dim(1));
}

void validate_logits_tensor(const tensor::Tensor& logits) {
  CHECK_EQ(logits.device_type(), base::DeviceType::kDeviceCUDA);
  CHECK_EQ(logits.dims_size(), 2);
  CHECK_GT(logits.get_dim(0), 0);
  CHECK_GT(logits.get_dim(1), 0);
  CHECK(logits.data_type() == base::DataType::kDataTypeFp32 ||
        logits.data_type() == base::DataType::kDataTypeBf16);
}

void validate_row_indices_tensor(const tensor::Tensor& row_indices) {
  CHECK_EQ(row_indices.device_type(), base::DeviceType::kDeviceCUDA);
  CHECK_EQ(row_indices.data_type(), base::DataType::kDataTypeInt32);
  CHECK_EQ(row_indices.dims_size(), 1);
  CHECK_GT(row_indices.get_dim(0), 0);
}

void validate_token_ids_tensor(const tensor::Tensor& token_ids, int32_t sample_count) {
  CHECK_EQ(token_ids.device_type(), base::DeviceType::kDeviceCUDA);
  CHECK_EQ(token_ids.data_type(), base::DataType::kDataTypeInt32);
  CHECK_EQ(token_ids.dims_size(), 1);
  CHECK_EQ(token_ids.get_dim(0), sample_count);
}

}  // namespace

void gather_logits_rows_cu(const tensor::Tensor& logits,
                           const tensor::Tensor& row_indices,
                           tensor::Tensor& gathered_logits,
                           void* stream) {
  validate_logits_tensor(logits);
  validate_row_indices_tensor(row_indices);
  CHECK_EQ(gathered_logits.device_type(), base::DeviceType::kDeviceCUDA);
  CHECK_EQ(gathered_logits.data_type(), logits.data_type());
  CHECK_EQ(gathered_logits.dims_size(), 2);
  CHECK_EQ(gathered_logits.get_dim(0), row_indices.get_dim(0));
  CHECK_EQ(gathered_logits.get_dim(1), logits.get_dim(1));

  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  if (logits.data_type() == base::DataType::kDataTypeFp32) {
    launch_gather_logits_rows<float>(logits, row_indices, gathered_logits, cuda_stream);
  } else {
    launch_gather_logits_rows<base::CudaBF16>(logits, row_indices, gathered_logits, cuda_stream);
  }
}

void argmax_rows_cu(const tensor::Tensor& logits,
                    tensor::Tensor& token_ids,
                    void* stream) {
  validate_logits_tensor(logits);
  validate_token_ids_tensor(token_ids, logits.get_dim(0));

  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  if (logits.data_type() == base::DataType::kDataTypeFp32) {
    launch_argmax_rows<float>(logits, token_ids, cuda_stream);
  } else {
    launch_argmax_rows<base::CudaBF16>(logits, token_ids, cuda_stream);
  }
}

void argmax_selected_rows_cu(const tensor::Tensor& logits,
                             const tensor::Tensor& row_indices,
                             tensor::Tensor& token_ids,
                             void* stream) {
  validate_logits_tensor(logits);
  validate_row_indices_tensor(row_indices);
  validate_token_ids_tensor(token_ids, row_indices.get_dim(0));

  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  if (logits.data_type() == base::DataType::kDataTypeFp32) {
    launch_argmax_selected_rows<float>(logits, row_indices, token_ids, cuda_stream);
  } else {
    launch_argmax_selected_rows<base::CudaBF16>(logits, row_indices, token_ids, cuda_stream);
  }
}

void sample_argmax_rows_cu(const tensor::Tensor& logits,
                           const tensor::Tensor& row_indices,
                           tensor::Tensor& gathered_logits,
                           tensor::Tensor& token_ids,
                           void* stream) {
  gather_logits_rows_cu(logits, row_indices, gathered_logits, stream);
  argmax_rows_cu(gathered_logits, token_ids, stream);
}

}  // namespace kernel
