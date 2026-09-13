// Updated on April 11, 2026
#include <gtest/gtest.h>
#include <vector>
#include <cuda_runtime.h>
#include "base/alloc.h"
#include "base/bf16.h"
#include "op/kernels/cuda/sampler_kernel.cuh"

namespace {

bool HasCudaDevice() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::vector<int32_t> cpu_argmax_rows(const std::vector<float>& logits,
                                     int32_t num_rows,
                                     int32_t vocab_size,
                                     const std::vector<int32_t>& row_indices) {
  std::vector<int32_t> results(row_indices.size(), 0);
  for (size_t sample_idx = 0; sample_idx < row_indices.size(); ++sample_idx) {
    const int32_t row_idx = row_indices[sample_idx];
    const float* row = logits.data() + static_cast<int64_t>(row_idx) * vocab_size;
    int32_t best_idx = 0;
    float best_val = row[0];
    for (int32_t vocab_idx = 1; vocab_idx < vocab_size; ++vocab_idx) {
      if (row[vocab_idx] > best_val) {
        best_val = row[vocab_idx];
        best_idx = vocab_idx;
      }
    }
    results[sample_idx] = best_idx;
  }
  return results;
}

}  // namespace

TEST(test_sampler_cu, mixed_row_argmax_fp32_matches_cpu) {
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  constexpr int32_t kNumRows = 5;
  constexpr int32_t kVocabSize = 8;
  const std::vector<float> logits = {
      -1.f, 0.2f,  0.2f, -0.5f, -0.1f, 0.0f,  0.1f, -0.2f,
       1.f, 2.5f,  1.2f,  2.5f,  0.4f, 0.3f, -2.f, -3.f,
      -4.f, -3.f, -2.f,  -1.f, -0.5f, 0.7f,  0.6f,  0.5f,
       3.f, 1.5f,  3.f,   2.8f, 1.1f, 0.9f,  0.8f,  0.7f,
       0.f, 0.1f,  0.2f,  0.3f, 5.f,  4.9f, 4.8f,  4.7f,
  };
  const std::vector<int32_t> row_indices = {4, 1, 3, 2};
  const std::vector<int32_t> expected =
      cpu_argmax_rows(logits, kNumRows, kVocabSize, row_indices);

  tensor::Tensor logits_tensor(base::DataType::kDataTypeFp32, kNumRows, kVocabSize, true,
                               alloc_cpu);
  for (int32_t idx = 0; idx < kNumRows * kVocabSize; ++idx) {
    logits_tensor.index<float>(idx) = logits[idx];
  }
  logits_tensor.to_cuda();

  tensor::Tensor row_indices_tensor(base::DataType::kDataTypeInt32,
                                    static_cast<int32_t>(row_indices.size()), true, alloc_cpu);
  for (size_t idx = 0; idx < row_indices.size(); ++idx) {
    row_indices_tensor.index<int32_t>(idx) = row_indices[idx];
  }
  row_indices_tensor.to_cuda();

  tensor::Tensor gathered_logits(base::DataType::kDataTypeFp32,
                                 static_cast<int32_t>(row_indices.size()), kVocabSize, true,
                                 alloc_cu);
  tensor::Tensor token_ids(base::DataType::kDataTypeInt32,
                           static_cast<int32_t>(row_indices.size()), true, alloc_cu);

  kernel::sample_argmax_rows_cu(logits_tensor, row_indices_tensor, gathered_logits, token_ids,
                                nullptr);
  token_ids.to_cpu();

  for (size_t idx = 0; idx < expected.size(); ++idx) {
    EXPECT_EQ(token_ids.index<int32_t>(idx), expected[idx]);
  }
}

TEST(test_sampler_cu, direct_selected_row_argmax_fp32_matches_cpu) {
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  constexpr int32_t kNumRows = 5;
  constexpr int32_t kVocabSize = 8;
  const std::vector<float> logits = {
      -1.f, 0.2f,  0.2f, -0.5f, -0.1f, 0.0f,  0.1f, -0.2f,
       1.f, 2.5f,  1.2f,  2.5f,  0.4f, 0.3f, -2.f, -3.f,
      -4.f, -3.f, -2.f,  -1.f, -0.5f, 0.7f,  0.6f,  0.5f,
       3.f, 1.5f,  3.f,   2.8f, 1.1f, 0.9f,  0.8f,  0.7f,
       0.f, 0.1f,  0.2f,  0.3f, 5.f,  4.9f, 4.8f,  4.7f,
  };
  const std::vector<int32_t> row_indices = {4, 1, 3, 2};
  const std::vector<int32_t> expected =
      cpu_argmax_rows(logits, kNumRows, kVocabSize, row_indices);

  tensor::Tensor logits_tensor(base::DataType::kDataTypeFp32, kNumRows, kVocabSize, true,
                               alloc_cpu);
  for (int32_t idx = 0; idx < kNumRows * kVocabSize; ++idx) {
    logits_tensor.index<float>(idx) = logits[idx];
  }
  logits_tensor.to_cuda();

  tensor::Tensor row_indices_tensor(base::DataType::kDataTypeInt32,
                                    static_cast<int32_t>(row_indices.size()), true, alloc_cpu);
  for (size_t idx = 0; idx < row_indices.size(); ++idx) {
    row_indices_tensor.index<int32_t>(idx) = row_indices[idx];
  }
  row_indices_tensor.to_cuda();

  tensor::Tensor token_ids(base::DataType::kDataTypeInt32,
                           static_cast<int32_t>(row_indices.size()), true, alloc_cu);

  kernel::argmax_selected_rows_cu(logits_tensor, row_indices_tensor, token_ids, nullptr);
  token_ids.to_cpu();

  for (size_t idx = 0; idx < expected.size(); ++idx) {
    EXPECT_EQ(token_ids.index<int32_t>(idx), expected[idx]);
  }
}

TEST(test_sampler_cu, mixed_row_argmax_bf16_matches_cpu) {
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  constexpr int32_t kNumRows = 4;
  constexpr int32_t kVocabSize = 6;
  const std::vector<float> logits = {
      -3.f, -2.f, -1.f, -0.5f, -0.25f, -0.1f,
       0.f,  1.f,  2.f,  2.f,   1.5f,   1.4f,
       4.f,  3.f,  2.f,  1.f,   0.f,   -1.f,
       1.1f, 1.2f, 1.3f, 1.4f,  1.5f,   7.f,
  };
  const std::vector<int32_t> row_indices = {1, 3, 0};
  const std::vector<int32_t> expected =
      cpu_argmax_rows(logits, kNumRows, kVocabSize, row_indices);

  tensor::Tensor logits_tensor(base::DataType::kDataTypeBf16, kNumRows, kVocabSize, true,
                               alloc_cpu);
  for (int32_t idx = 0; idx < kNumRows * kVocabSize; ++idx) {
    logits_tensor.index<uint16_t>(idx) = base::float_to_bf16_bits(logits[idx]);
  }
  logits_tensor.to_cuda();

  tensor::Tensor row_indices_tensor(base::DataType::kDataTypeInt32,
                                    static_cast<int32_t>(row_indices.size()), true, alloc_cpu);
  for (size_t idx = 0; idx < row_indices.size(); ++idx) {
    row_indices_tensor.index<int32_t>(idx) = row_indices[idx];
  }
  row_indices_tensor.to_cuda();

  tensor::Tensor gathered_logits(base::DataType::kDataTypeBf16,
                                 static_cast<int32_t>(row_indices.size()), kVocabSize, true,
                                 alloc_cu);
  tensor::Tensor token_ids(base::DataType::kDataTypeInt32,
                           static_cast<int32_t>(row_indices.size()), true, alloc_cu);

  kernel::sample_argmax_rows_cu(logits_tensor, row_indices_tensor, gathered_logits, token_ids,
                                nullptr);
  token_ids.to_cpu();

  for (size_t idx = 0; idx < expected.size(); ++idx) {
    EXPECT_EQ(token_ids.index<int32_t>(idx), expected[idx]);
  }
}

TEST(test_sampler_cu, topk_topp_selected_rows_top1_matches_argmax) {
  if (!HasCudaDevice()) {
    GTEST_SKIP() << "CUDA device is not available";
  }
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  constexpr int32_t kNumRows = 3;
  constexpr int32_t kVocabSize = 6;
  const std::vector<float> logits = {
      0.1f, 2.0f, 1.5f, -1.0f, 0.0f, 1.9f,
      3.0f, 2.9f, 0.0f,  1.0f, 4.0f, 0.2f,
     -1.0f, 0.5f, 0.4f,  0.3f, 0.2f, 0.1f,
  };
  const std::vector<int32_t> row_indices = {2, 0, 1};
  const std::vector<int32_t> expected =
      cpu_argmax_rows(logits, kNumRows, kVocabSize, row_indices);

  tensor::Tensor logits_tensor(base::DataType::kDataTypeFp32, kNumRows, kVocabSize, true,
                               alloc_cpu);
  for (int32_t idx = 0; idx < kNumRows * kVocabSize; ++idx) {
    logits_tensor.index<float>(idx) = logits[idx];
  }
  logits_tensor.to_cuda();

  const int32_t sample_count = static_cast<int32_t>(row_indices.size());
  tensor::Tensor row_indices_tensor(base::DataType::kDataTypeInt32, sample_count, true,
                                    alloc_cpu);
  tensor::Tensor temperatures(base::DataType::kDataTypeFp32, sample_count, true, alloc_cpu);
  tensor::Tensor top_ps(base::DataType::kDataTypeFp32, sample_count, true, alloc_cpu);
  tensor::Tensor top_ks(base::DataType::kDataTypeInt32, sample_count, true, alloc_cpu);
  tensor::Tensor random_values(base::DataType::kDataTypeFp32, sample_count, true, alloc_cpu);
  for (int32_t idx = 0; idx < sample_count; ++idx) {
    row_indices_tensor.index<int32_t>(idx) = row_indices[idx];
    temperatures.index<float>(idx) = 0.8f;
    top_ps.index<float>(idx) = 1.0f;
    top_ks.index<int32_t>(idx) = 1;
    random_values.index<float>(idx) = 0.5f;
  }
  row_indices_tensor.to_cuda();
  temperatures.to_cuda();
  top_ps.to_cuda();
  top_ks.to_cuda();
  random_values.to_cuda();

  tensor::Tensor token_ids(base::DataType::kDataTypeInt32, sample_count, true, alloc_cu);
  kernel::sample_topk_topp_selected_rows_cu(logits_tensor, row_indices_tensor,
                                            temperatures, top_ps, top_ks,
                                            random_values, token_ids, nullptr);
  token_ids.to_cpu();

  for (int32_t idx = 0; idx < sample_count; ++idx) {
    EXPECT_EQ(token_ids.index<int32_t>(idx), expected[idx]);
  }
}

TEST(test_sampler_cu, topk_ties_use_token_order_across_threads) {
  if (!HasCudaDevice()) {
    GTEST_SKIP() << "CUDA device is not available";
  }
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  constexpr int32_t kNumRows = 3;
  constexpr int32_t kVocabSize = 257;
  std::vector<float> logits(kNumRows * kVocabSize, -100.0f);
  for (int row = 0; row < kNumRows; ++row) {
    logits[row * kVocabSize + 1] = 5.0f;
    logits[row * kVocabSize + 128] = 5.0f;
  }
  const std::vector<int32_t> row_indices = {2, 0, 1};
  const std::vector<int32_t> expected = {1, 128, 1};

  tensor::Tensor logits_tensor(base::DataType::kDataTypeFp32, kNumRows, kVocabSize, true,
                               alloc_cpu);
  for (int32_t idx = 0; idx < kNumRows * kVocabSize; ++idx) {
    logits_tensor.index<float>(idx) = logits[idx];
  }
  logits_tensor.to_cuda();

  const int32_t sample_count = static_cast<int32_t>(row_indices.size());
  tensor::Tensor row_indices_tensor(base::DataType::kDataTypeInt32, sample_count, true,
                                    alloc_cpu);
  tensor::Tensor temperatures(base::DataType::kDataTypeFp32, sample_count, true, alloc_cpu);
  tensor::Tensor top_ps(base::DataType::kDataTypeFp32, sample_count, true, alloc_cpu);
  tensor::Tensor top_ks(base::DataType::kDataTypeInt32, sample_count, true, alloc_cpu);
  tensor::Tensor random_values(base::DataType::kDataTypeFp32, sample_count, true, alloc_cpu);
  for (int32_t idx = 0; idx < sample_count; ++idx) {
    row_indices_tensor.index<int32_t>(idx) = row_indices[idx];
    temperatures.index<float>(idx) = 0.8f;
    top_ps.index<float>(idx) = 1.0f;
    top_ks.index<int32_t>(idx) = 2;
    random_values.index<float>(idx) = idx == 1 ? 0.9f : 0.1f;
  }
  row_indices_tensor.to_cuda();
  temperatures.to_cuda();
  top_ps.to_cuda();
  top_ks.to_cuda();
  random_values.to_cuda();

  tensor::Tensor token_ids(base::DataType::kDataTypeInt32, sample_count, true, alloc_cu);
  kernel::sample_topk_topp_selected_rows_cu(logits_tensor, row_indices_tensor,
                                            temperatures, top_ps, top_ks,
                                            random_values, token_ids, nullptr);
  token_ids.to_cpu();

  for (int32_t idx = 0; idx < sample_count; ++idx) {
    EXPECT_EQ(token_ids.index<int32_t>(idx), expected[idx]);
  }
}
