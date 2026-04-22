// Updated on April 11, 2026
#include <gtest/gtest.h>
#include <vector>
#include "base/alloc.h"
#include "base/bf16.h"
#include "op/kernels/cuda/sampler_kernel.cuh"

namespace {

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
