// Updated on March 23, 2026
#include <base/bf16.h>
#include "../kernels_interface.h"
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

}  // namespace kernel
