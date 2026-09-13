// Test paged attention kernel correctness
#include <gtest/gtest.h>
#include <base/alloc.h>
#include <base/block_allocator.h>
#include <base/sequence_kv_manager.h>
#include <base/cuda_config.h>
#include <tensor/tensor.h>
#include "op/kernels/cuda/paged_mha_kernel.cuh"
#include "op/kernels/cuda/fused_mha_kernel.cuh"
#include "op/kernels/cuda/scatter_kv_kernel.cuh"
#include <cmath>

// Test that paged decode attention produces same output as contiguous version
TEST(PagedAttentionTest, SingleSequenceDecodeCorrectness) {
  using namespace base;
  using namespace tensor;
  using namespace kernel;

  // Test parameters
  const int32_t num_layers = 2;
  const int32_t num_heads = 8;
  const int32_t num_kv_heads = 2;  // GQA
  const int32_t head_size = 64;
  const int32_t kv_mul = num_heads / num_kv_heads;
  const int32_t dim = num_heads * head_size;
  const int32_t kv_dim = num_kv_heads * head_size;
  const int32_t block_size = 16;
  const int32_t num_blocks = 128;
  const int32_t seq_len = 512;
  const int32_t test_pos = 35;  // Test at position 35
  const DataType dtype = DataType::kDataTypeFp32;

  auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
  auto alloc_cuda = CUDADeviceAllocatorFactory::get_instance();

  // Create CUDA config
  CudaConfig cuda_config;
  ASSERT_EQ(cudaStreamCreate(&cuda_config.stream), cudaSuccess);

  // 1. Setup contiguous KV cache (baseline)
  Tensor key_cache_contiguous(dtype, num_layers, seq_len, kv_dim, true, alloc_cuda);
  Tensor value_cache_contiguous(dtype, num_layers, seq_len, kv_dim, true, alloc_cuda);

  // Fill with random data
  std::vector<float> key_data(num_layers * seq_len * kv_dim);
  std::vector<float> value_data(num_layers * seq_len * kv_dim);
  for (size_t i = 0; i < key_data.size(); ++i) {
    key_data[i] = (float(rand()) / RAND_MAX) * 2.0f - 1.0f;
    value_data[i] = (float(rand()) / RAND_MAX) * 2.0f - 1.0f;
  }
  alloc_cuda->memcpy(key_data.data(), const_cast<float*>(key_cache_contiguous.ptr<float>()),
                     key_data.size() * sizeof(float), MemcpyKind::kMemcpyCPU2CUDA, nullptr, true);
  alloc_cuda->memcpy(value_data.data(), const_cast<float*>(value_cache_contiguous.ptr<float>()),
                     value_data.size() * sizeof(float), MemcpyKind::kMemcpyCPU2CUDA, nullptr, true);

  // 2. Setup paged KV cache
  std::vector<std::unique_ptr<BlockAllocator>> layer_allocators;
  layer_allocators.reserve(num_layers);
  for (int32_t layer = 0; layer < num_layers; ++layer) {
    layer_allocators.push_back(std::make_unique<BlockAllocator>(
        num_blocks, block_size, num_kv_heads, head_size, dtype, DeviceType::kDeviceCUDA));
  }
  SequenceKVManager kv_manager(block_size, num_layers);

  // Copy contiguous KV to paged blocks
  int32_t layer_idx = 0;  // Test layer 0
  for (int32_t t = 0; t <= test_pos; ++t) {
    ASSERT_TRUE(kv_manager.append_token(layer_allocators));
    auto [block_id, offset] = kv_manager.current_slot(layer_idx);

    // Extract K/V for this token from contiguous cache
    Tensor key_token(dtype, kv_dim, true, alloc_cuda);
    Tensor value_token(dtype, kv_dim, true, alloc_cuda);

    int64_t src_offset = layer_idx * seq_len * kv_dim + t * kv_dim;
    alloc_cuda->memcpy(key_cache_contiguous.ptr<float>(src_offset),
                       const_cast<float*>(key_token.ptr<float>()),
                       kv_dim * sizeof(float), MemcpyKind::kMemcpyCUDA2CUDA, nullptr, true);
    alloc_cuda->memcpy(value_cache_contiguous.ptr<float>(src_offset),
                       const_cast<float*>(value_token.ptr<float>()),
                       kv_dim * sizeof(float), MemcpyKind::kMemcpyCUDA2CUDA, nullptr, true);

    // Scatter to paged cache
    scatter_kv_to_page_cu(key_token, value_token,
                          const_cast<Tensor&>(layer_allocators[layer_idx]->key_pool()),
                          const_cast<Tensor&>(layer_allocators[layer_idx]->value_pool()),
                          block_id, offset, block_size,
                          num_kv_heads, head_size, DeviceType::kDeviceCUDA, &cuda_config);
    // key_token/value_token return their buffers to the caching allocator at
    // the end of this iteration.  Finish the async scatter before those
    // buffers can be reused by the next token.
    ASSERT_EQ(cudaStreamSynchronize(cuda_config.stream), cudaSuccess);
  }

  // 3. Create query
  Tensor query(dtype, dim, true, alloc_cuda);
  std::vector<float> query_data(dim);
  for (int32_t i = 0; i < dim; ++i) {
    query_data[i] = (float(rand()) / RAND_MAX) * 2.0f - 1.0f;
  }
  alloc_cuda->memcpy(query_data.data(), const_cast<float*>(query.ptr<float>()),
                     dim * sizeof(float), MemcpyKind::kMemcpyCPU2CUDA, nullptr, true);

  // 4. Run contiguous MHA
  Tensor output_contiguous(dtype, dim, true, alloc_cuda);
  Tensor score_storage(DataType::kDataTypeFp32, num_heads, seq_len, true, alloc_cuda);
  fused_mha_kernel_cu(test_pos, num_heads, layer_idx, seq_len, kv_dim, kv_mul, head_size,
                      output_contiguous, query, score_storage, key_cache_contiguous,
                      value_cache_contiguous, DeviceType::kDeviceCUDA, &cuda_config);

  // 5. Run paged MHA
  Tensor output_paged(dtype, dim, true, alloc_cuda);

  // Copy block table to GPU
  const auto& block_ids = kv_manager.page_table(layer_idx).block_ids();
  Tensor block_table_gpu(DataType::kDataTypeInt32, block_ids.size(), true, alloc_cuda);
  alloc_cuda->memcpy(block_ids.data(), const_cast<int32_t*>(block_table_gpu.ptr<int32_t>()),
                     block_ids.size() * sizeof(int32_t), MemcpyKind::kMemcpyCPU2CUDA, nullptr, true);

  paged_mha_decode_cu(num_heads, head_size, kv_mul, query, output_paged,
                      layer_allocators[layer_idx]->key_pool(),
                      layer_allocators[layer_idx]->value_pool(),
                      block_table_gpu.ptr<int32_t>(), kv_manager.page_table(layer_idx).num_blocks(),
                      kv_manager.num_tokens_in_last_block(), block_size,
                      num_kv_heads, DeviceType::kDeviceCUDA, &cuda_config);

  ASSERT_EQ(cudaStreamSynchronize(cuda_config.stream), cudaSuccess);

  // 6. Compare outputs
  std::vector<float> output_cont_cpu(dim);
  std::vector<float> output_paged_cpu(dim);
  alloc_cuda->memcpy(output_contiguous.ptr<float>(), output_cont_cpu.data(),
                     dim * sizeof(float), MemcpyKind::kMemcpyCUDA2CPU, nullptr, true);
  alloc_cuda->memcpy(output_paged.ptr<float>(), output_paged_cpu.data(),
                     dim * sizeof(float), MemcpyKind::kMemcpyCUDA2CPU, nullptr, true);

  float max_diff = 0.0f;
  for (int32_t i = 0; i < dim; ++i) {
    float diff = std::abs(output_cont_cpu[i] - output_paged_cpu[i]);
    max_diff = std::max(max_diff, diff);
  }

  std::cout << "Max difference: " << max_diff << std::endl;
  EXPECT_LT(max_diff, 1e-4f) << "Paged and contiguous outputs differ too much";

}
