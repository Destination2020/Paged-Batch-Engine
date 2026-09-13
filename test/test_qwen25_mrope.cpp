#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "base/alloc.h"
#include "base/cuda_config.h"
#include "../infMain/source/op/kernels/cuda/rope_kernel.cuh"
#include "tensor/tensor.h"

TEST(Qwen25MRoPETest, SelectsTemporalHeightAndWidthSections) {
  using base::DataType;
  using base::MemcpyKind;
  using tensor::Tensor;
  constexpr int kHeadSize = 128;
  constexpr int kBatch = 2;
  constexpr int kMaxPosition = 8;
  auto cuda_alloc = base::CUDADeviceAllocatorFactory::get_instance();

  std::vector<float> source(kBatch * kHeadSize);
  for (size_t i = 0; i < source.size(); ++i) {
    source[i] = static_cast<float>(i + 1) / 128.0f;
  }
  std::vector<float> sin_cache(kMaxPosition * kHeadSize);
  std::vector<float> cos_cache(kMaxPosition * kHeadSize);
  for (int pos = 0; pos < kMaxPosition; ++pos) {
    for (int d = 0; d < kHeadSize; ++d) {
      const float frequency = 1.0f / std::pow(1000000.0f, static_cast<float>(d) / kHeadSize);
      sin_cache[pos * kHeadSize + d] = std::sin(pos * frequency);
      cos_cache[pos * kHeadSize + d] = std::cos(pos * frequency);
    }
  }
  // axis-major: temporal, height, width.
  std::vector<int32_t> positions{1, 2, 3, 4, 5, 6};

  Tensor q(DataType::kDataTypeFp32, kBatch, kHeadSize, true, cuda_alloc);
  Tensor k(DataType::kDataTypeFp32, kBatch, kHeadSize, true, cuda_alloc);
  Tensor pos(DataType::kDataTypeInt32, 3, kBatch, true, cuda_alloc);
  Tensor sin(DataType::kDataTypeFp32, kMaxPosition, kHeadSize, true, cuda_alloc);
  Tensor cos(DataType::kDataTypeFp32, kMaxPosition, kHeadSize, true, cuda_alloc);
  cuda_alloc->memcpy(source.data(), q.ptr<float>(), q.byte_size(),
                     MemcpyKind::kMemcpyCPU2CUDA, nullptr, true);
  cuda_alloc->memcpy(source.data(), k.ptr<float>(), k.byte_size(),
                     MemcpyKind::kMemcpyCPU2CUDA, nullptr, true);
  cuda_alloc->memcpy(positions.data(), pos.ptr<int32_t>(), pos.byte_size(),
                     MemcpyKind::kMemcpyCPU2CUDA, nullptr, true);
  cuda_alloc->memcpy(sin_cache.data(), sin.ptr<float>(), sin.byte_size(),
                     MemcpyKind::kMemcpyCPU2CUDA, nullptr, true);
  cuda_alloc->memcpy(cos_cache.data(), cos.ptr<float>(), cos.byte_size(),
                     MemcpyKind::kMemcpyCPU2CUDA, nullptr, true);

  kernel::CudaConfig config;
  ASSERT_EQ(cudaStreamCreate(&config.stream), cudaSuccess);
  kernel::mrope_kernel_batched_cu(kHeadSize, kHeadSize, kHeadSize, q, k, pos, sin, cos,
                                  16, 24, kBatch, config.stream);
  ASSERT_EQ(cudaPeekAtLastError(), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(config.stream), cudaSuccess);

  std::vector<float> actual(source.size());
  cuda_alloc->memcpy(q.ptr<float>(), actual.data(), q.byte_size(),
                     MemcpyKind::kMemcpyCUDA2CPU, nullptr, true);
  for (int token = 0; token < kBatch; ++token) {
    for (int pair = 0; pair < kHeadSize / 2; ++pair) {
      const int axis = pair < 16 ? 0 : (pair < 40 ? 1 : 2);
      const int position = positions[axis * kBatch + token];
      const float sine = sin_cache[position * kHeadSize + pair * 2];
      const float cosine = cos_cache[position * kHeadSize + pair * 2];
      const int first = token * kHeadSize + pair;
      const int second = first + kHeadSize / 2;
      EXPECT_NEAR(actual[first], cosine * source[first] - sine * source[second], 1e-6);
      EXPECT_NEAR(actual[second], cosine * source[second] + sine * source[first], 1e-6);
    }
  }
}
