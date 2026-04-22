// Updated on March 23, 2026
#ifndef KUIPER_SOURCE_OP_KERNELS_CUDA_CUDA_TYPE_UTILS_CUH_
#define KUIPER_SOURCE_OP_KERNELS_CUDA_CUDA_TYPE_UTILS_CUH_

#include <base/bf16.h>
#include <cuda_fp8.h>

namespace kernel {

template <typename T>
__device__ inline float scalar_to_float(T value);

template <>
__device__ inline float scalar_to_float<float>(float value) {
  return value;
}

template <>
__device__ inline float scalar_to_float<base::CudaBF16>(base::CudaBF16 value) {
  return base::cuda_bf16_to_float(value);
}

template <typename T>
__device__ inline T float_to_scalar(float value);

template <>
__device__ inline float float_to_scalar<float>(float value) {
  return value;
}

template <>
__device__ inline base::CudaBF16 float_to_scalar<base::CudaBF16>(float value) {
  return base::cuda_bf16_from_float(value);
}

template <>
__device__ inline float scalar_to_float<int8_t>(int8_t value) {
  __nv_fp8_e4m3 fp8_value;
  fp8_value.__x = static_cast<uint8_t>(value);
  return static_cast<float>(fp8_value);
}

template <typename T>
__device__ inline float dot_product(const T* lhs, const T* rhs, int32_t size) {
  float sum = 0.f;
  for (int32_t i = 0; i < size; ++i) {
    sum += scalar_to_float(lhs[i]) * scalar_to_float(rhs[i]);
  }
  return sum;
}

}  // namespace kernel

#endif  // KUIPER_SOURCE_OP_KERNELS_CUDA_CUDA_TYPE_UTILS_CUH_
