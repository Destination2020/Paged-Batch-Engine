// Updated on March 23, 2026
#ifndef KUIPER_INCLUDE_BASE_BF16_H_
#define KUIPER_INCLUDE_BASE_BF16_H_

#include <cstdint>
#include <cstring>

#ifdef __CUDACC__
#include <cuda_bf16.h>
#endif

namespace base {

inline uint16_t float_to_bf16_bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

inline float bf16_bits_to_float(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float out = 0.f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

#ifdef __CUDACC__
using CudaBF16 = __nv_bfloat16;

__device__ inline float cuda_bf16_to_float(CudaBF16 value) { return __bfloat162float(value); }

__device__ inline CudaBF16 cuda_bf16_from_float(float value) { return __float2bfloat16_rn(value); }
#endif

}  // namespace base

#endif  // KUIPER_INCLUDE_BASE_BF16_H_
