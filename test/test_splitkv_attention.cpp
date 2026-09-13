// Test Split-KV paged decode attention against serial baseline
#include <gtest/gtest.h>
#include <base/alloc.h>
#include <base/block_allocator.h>
#include <base/sequence_kv_manager.h>
#include <base/cuda_config.h>
#include <tensor/tensor.h>
#include "op/kernels/cuda/paged_mha_kernel.cuh"
#include "op/kernels/cuda/scatter_kv_kernel.cuh"
#include <cmath>
#include <cstdlib>
#include <vector>

namespace {

// Fill a GPU tensor with random float data
void fill_random_gpu(tensor::Tensor& t, const std::shared_ptr<base::DeviceAllocator>& alloc_cu) {
  std::vector<float> data(t.size());
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
  }
  alloc_cu->memcpy(data.data(), const_cast<float*>(t.ptr<float>()),
                   data.size() * sizeof(float),
                   base::MemcpyKind::kMemcpyCPU2CUDA, nullptr, true);
}

}  // namespace

// Test: Split-KV vs serial batched attention, single sequence
TEST(SplitKVAttentionTest, SingleSequenceMatchesBaseline) {
  using namespace base;
  using namespace tensor;
  using namespace kernel;

  srand(42);

  const int32_t num_heads = 8;
  const int32_t num_kv_heads = 2;
  const int32_t head_size = 64;
  const int32_t kv_mul = num_heads / num_kv_heads;
  const int32_t dim = num_heads * head_size;
  const int32_t kv_dim = num_kv_heads * head_size;
  const int32_t block_size = 16;
  const int32_t num_blocks = 128;
  const int32_t context_len = 256;  // 16 KV blocks -> good for split testing
  const int32_t batch_size = 1;
  const DataType dtype = DataType::kDataTypeFp32;

  auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();

  CudaConfig cuda_config;
  ASSERT_EQ(cudaStreamCreate(&cuda_config.stream), cudaSuccess);

  // Create block allocator and fill KV cache
  BlockAllocator allocator(num_blocks, block_size, num_kv_heads, head_size,
                           dtype, DeviceType::kDeviceCUDA);

  SequenceKVManager kv_mgr(block_size, 1);  // 1 layer
  std::vector<std::unique_ptr<BlockAllocator>> allocs;
  allocs.push_back(std::make_unique<BlockAllocator>(
      num_blocks, block_size, num_kv_heads, head_size, dtype, DeviceType::kDeviceCUDA));

  // Append tokens and scatter random KV data
  for (int32_t t = 0; t < context_len; ++t) {
    ASSERT_TRUE(kv_mgr.append_token(allocs));
    auto [blk_id, offset] = kv_mgr.current_slot(0);

    Tensor key_tok(dtype, kv_dim, true, alloc_cu);
    Tensor val_tok(dtype, kv_dim, true, alloc_cu);
    fill_random_gpu(key_tok, alloc_cu);
    fill_random_gpu(val_tok, alloc_cu);

    scatter_kv_to_page_cu(key_tok, val_tok,
                          const_cast<Tensor&>(allocs[0]->key_pool()),
                          const_cast<Tensor&>(allocs[0]->value_pool()),
                          blk_id, offset, block_size,
                          num_kv_heads, head_size,
                          DeviceType::kDeviceCUDA, &cuda_config);
    ASSERT_EQ(cudaStreamSynchronize(cuda_config.stream), cudaSuccess);
  }

  // Query
  Tensor query(dtype, dim, true, alloc_cu);
  fill_random_gpu(query, alloc_cu);

  // Block table -> GPU
  const auto& block_ids = kv_mgr.page_table(0).block_ids();
  int32_t num_kv_blocks = static_cast<int32_t>(block_ids.size());
  Tensor block_table_gpu(DataType::kDataTypeInt32, batch_size * num_kv_blocks, true, alloc_cu);
  alloc_cu->memcpy(block_ids.data(), const_cast<int32_t*>(block_table_gpu.ptr<int32_t>()),
                   num_kv_blocks * sizeof(int32_t),
                   MemcpyKind::kMemcpyCPU2CUDA, nullptr, true);

  // seq_lens -> GPU
  Tensor seq_lens_gpu(DataType::kDataTypeInt32, batch_size, true, alloc_cu);
  int32_t ctx = context_len;
  alloc_cu->memcpy(&ctx, const_cast<int32_t*>(seq_lens_gpu.ptr<int32_t>()),
                   sizeof(int32_t), MemcpyKind::kMemcpyCPU2CUDA, nullptr, true);

  // Output buffers
  Tensor output_baseline(dtype, dim, true, alloc_cu);
  Tensor output_splitkv(dtype, dim, true, alloc_cu);

  // Run baseline (serial batched)
  batched_paged_mha_decode_cu(
      batch_size, num_heads, head_size, kv_mul,
      query, output_baseline,
      allocs[0]->key_pool(), allocs[0]->value_pool(),
      block_table_gpu, seq_lens_gpu,
      num_kv_blocks, block_size, num_kv_heads,
      DeviceType::kDeviceCUDA, &cuda_config);
  ASSERT_EQ(cudaPeekAtLastError(), cudaSuccess);

  // Workspace for split-KV
  constexpr int32_t MAX_PARTS = 32;
  Tensor partial_out(DataType::kDataTypeFp32,
                     batch_size * num_heads * MAX_PARTS * head_size, true, alloc_cu);
  Tensor partial_max(DataType::kDataTypeFp32,
                     batch_size * num_heads * MAX_PARTS, true, alloc_cu);
  Tensor partial_sum(DataType::kDataTypeFp32,
                     batch_size * num_heads * MAX_PARTS, true, alloc_cu);
  partial_out.set_device_type(DeviceType::kDeviceCUDA);
  partial_max.set_device_type(DeviceType::kDeviceCUDA);
  partial_sum.set_device_type(DeviceType::kDeviceCUDA);

  // Run split-KV
  splitkv_batched_paged_mha_decode_cu(
      batch_size, num_heads, head_size, kv_mul,
      query, output_splitkv,
      allocs[0]->key_pool(), allocs[0]->value_pool(),
      block_table_gpu, seq_lens_gpu,
      num_kv_blocks, block_size, num_kv_heads,
      partial_out, partial_max, partial_sum,
      DeviceType::kDeviceCUDA, &cuda_config);
  ASSERT_EQ(cudaPeekAtLastError(), cudaSuccess);

  ASSERT_EQ(cudaStreamSynchronize(cuda_config.stream), cudaSuccess);

  // Compare
  std::vector<float> out_base(dim), out_split(dim);
  alloc_cu->memcpy(output_baseline.ptr<float>(), out_base.data(),
                   dim * sizeof(float), MemcpyKind::kMemcpyCUDA2CPU, nullptr, true);
  alloc_cu->memcpy(output_splitkv.ptr<float>(), out_split.data(),
                   dim * sizeof(float), MemcpyKind::kMemcpyCUDA2CPU, nullptr, true);

  float max_diff = 0.f;
  float max_base = 0.f;
  float max_split = 0.f;
  for (int32_t i = 0; i < dim; ++i) {
    float diff = std::abs(out_base[i] - out_split[i]);
    max_diff = std::max(max_diff, diff);
    max_base = std::max(max_base, std::abs(out_base[i]));
    max_split = std::max(max_split, std::abs(out_split[i]));
  }

  std::cout << "[SplitKV vs Baseline] context_len=" << context_len
            << " max_diff=" << max_diff << " max_base=" << max_base
            << " max_split=" << max_split << std::endl;
  EXPECT_LT(max_diff, 1e-4f) << "Split-KV and baseline outputs differ too much";

}

// Test: Split-KV with short context (num_kv_blocks=1, single partition)
TEST(SplitKVAttentionTest, ShortContextSinglePartition) {
  using namespace base;
  using namespace tensor;
  using namespace kernel;

  srand(123);

  const int32_t num_heads = 4;
  const int32_t num_kv_heads = 2;
  const int32_t head_size = 64;
  const int32_t kv_mul = num_heads / num_kv_heads;
  const int32_t dim = num_heads * head_size;
  const int32_t kv_dim = num_kv_heads * head_size;
  const int32_t block_size = 16;
  const int32_t num_blocks = 32;
  const int32_t context_len = 5;  // Only 1 KV block, 5 tokens
  const int32_t batch_size = 1;
  const DataType dtype = DataType::kDataTypeFp32;

  auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
  CudaConfig cuda_config;
  ASSERT_EQ(cudaStreamCreate(&cuda_config.stream), cudaSuccess);

  std::vector<std::unique_ptr<BlockAllocator>> allocs;
  allocs.push_back(std::make_unique<BlockAllocator>(
      num_blocks, block_size, num_kv_heads, head_size, dtype, DeviceType::kDeviceCUDA));

  SequenceKVManager kv_mgr(block_size, 1);

  for (int32_t t = 0; t < context_len; ++t) {
    ASSERT_TRUE(kv_mgr.append_token(allocs));
    auto [blk_id, offset] = kv_mgr.current_slot(0);
    Tensor k(dtype, kv_dim, true, alloc_cu);
    Tensor v(dtype, kv_dim, true, alloc_cu);
    fill_random_gpu(k, alloc_cu);
    fill_random_gpu(v, alloc_cu);
    scatter_kv_to_page_cu(k, v,
                          const_cast<Tensor&>(allocs[0]->key_pool()),
                          const_cast<Tensor&>(allocs[0]->value_pool()),
                          blk_id, offset, block_size, num_kv_heads, head_size,
                          DeviceType::kDeviceCUDA, &cuda_config);
    ASSERT_EQ(cudaStreamSynchronize(cuda_config.stream), cudaSuccess);
  }

  Tensor query(dtype, dim, true, alloc_cu);
  fill_random_gpu(query, alloc_cu);

  const auto& block_ids = kv_mgr.page_table(0).block_ids();
  int32_t num_kv_blocks = static_cast<int32_t>(block_ids.size());
  Tensor block_table_gpu(DataType::kDataTypeInt32, num_kv_blocks, true, alloc_cu);
  alloc_cu->memcpy(block_ids.data(), const_cast<int32_t*>(block_table_gpu.ptr<int32_t>()),
                   num_kv_blocks * sizeof(int32_t), MemcpyKind::kMemcpyCPU2CUDA, nullptr, true);

  Tensor seq_lens_gpu(DataType::kDataTypeInt32, 1, true, alloc_cu);
  int32_t ctx = context_len;
  alloc_cu->memcpy(&ctx, const_cast<int32_t*>(seq_lens_gpu.ptr<int32_t>()),
                   sizeof(int32_t), MemcpyKind::kMemcpyCPU2CUDA, nullptr, true);

  Tensor output_baseline(dtype, dim, true, alloc_cu);
  Tensor output_splitkv(dtype, dim, true, alloc_cu);

  batched_paged_mha_decode_cu(batch_size, num_heads, head_size, kv_mul,
                              query, output_baseline,
                              allocs[0]->key_pool(), allocs[0]->value_pool(),
                              block_table_gpu, seq_lens_gpu,
                              num_kv_blocks, block_size, num_kv_heads,
                              DeviceType::kDeviceCUDA, &cuda_config);
  ASSERT_EQ(cudaPeekAtLastError(), cudaSuccess);

  constexpr int32_t MAX_PARTS = 32;
  Tensor po(DataType::kDataTypeFp32, num_heads * MAX_PARTS * head_size, true, alloc_cu);
  Tensor pm(DataType::kDataTypeFp32, num_heads * MAX_PARTS, true, alloc_cu);
  Tensor ps(DataType::kDataTypeFp32, num_heads * MAX_PARTS, true, alloc_cu);
  po.set_device_type(DeviceType::kDeviceCUDA);
  pm.set_device_type(DeviceType::kDeviceCUDA);
  ps.set_device_type(DeviceType::kDeviceCUDA);

  splitkv_batched_paged_mha_decode_cu(batch_size, num_heads, head_size, kv_mul,
                                       query, output_splitkv,
                                       allocs[0]->key_pool(), allocs[0]->value_pool(),
                                       block_table_gpu, seq_lens_gpu,
                                       num_kv_blocks, block_size, num_kv_heads,
                                       po, pm, ps,
                                       DeviceType::kDeviceCUDA, &cuda_config);
  ASSERT_EQ(cudaPeekAtLastError(), cudaSuccess);

  ASSERT_EQ(cudaStreamSynchronize(cuda_config.stream), cudaSuccess);

  std::vector<float> out_base(dim), out_split(dim);
  alloc_cu->memcpy(output_baseline.ptr<float>(), out_base.data(),
                   dim * sizeof(float), MemcpyKind::kMemcpyCUDA2CPU, nullptr, true);
  alloc_cu->memcpy(output_splitkv.ptr<float>(), out_split.data(),
                   dim * sizeof(float), MemcpyKind::kMemcpyCUDA2CPU, nullptr, true);

  float max_diff = 0.f;
  float max_base = 0.f;
  float max_split = 0.f;
  for (int32_t i = 0; i < dim; ++i) {
    max_diff = std::max(max_diff, std::abs(out_base[i] - out_split[i]));
    max_base = std::max(max_base, std::abs(out_base[i]));
    max_split = std::max(max_split, std::abs(out_split[i]));
  }

  std::cout << "[SplitKV short] context_len=" << context_len
            << " max_diff=" << max_diff << " max_base=" << max_base
            << " max_split=" << max_split << std::endl;
  EXPECT_LT(max_diff, 1e-4f);

}
