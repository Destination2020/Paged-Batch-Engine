// Tests for fast-path paged decode attention
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <base/alloc.h>
#include <base/base.h>
#include <base/bf16.h>
#include <base/cuda_config.h>
#include <tensor/tensor.h>

#include "op/kernels/cuda/paged_mha_fast_kernel.cuh"
#include "op/kernels/cuda/paged_mha_kernel.cuh"

namespace {

using base::bf16_bits_to_float;
using base::float_to_bf16_bits;

std::vector<float> make_random_floats(size_t count, int seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> data(count);
  for (size_t i = 0; i < count; ++i) {
    data[i] = dist(rng);
  }
  return data;
}

std::vector<uint16_t> make_random_bf16(size_t count, int seed) {
  std::vector<float> tmp = make_random_floats(count, seed);
  std::vector<uint16_t> out(count);
  for (size_t i = 0; i < count; ++i) {
    out[i] = float_to_bf16_bits(tmp[i]);
  }
  return out;
}

void copy_to_gpu_bf16(
    tensor::Tensor& tensor,
    const std::vector<uint16_t>& host,
    const std::shared_ptr<base::DeviceAllocator>& alloc) {
  alloc->memcpy(
      host.data(),
      const_cast<uint16_t*>(tensor.ptr<uint16_t>()),
      host.size() * sizeof(uint16_t),
      base::MemcpyKind::kMemcpyCPU2CUDA,
      nullptr,
      true);
}

void copy_to_gpu_fp32(
    tensor::Tensor& tensor,
    const std::vector<float>& host,
    const std::shared_ptr<base::DeviceAllocator>& alloc) {
  alloc->memcpy(
      host.data(),
      const_cast<float*>(tensor.ptr<float>()),
      host.size() * sizeof(float),
      base::MemcpyKind::kMemcpyCPU2CUDA,
      nullptr,
      true);
}

std::vector<float> copy_from_gpu_bf16(
    const tensor::Tensor& tensor,
    size_t count,
    const std::shared_ptr<base::DeviceAllocator>& alloc) {
  std::vector<uint16_t> host_bits(count);
  alloc->memcpy(
      tensor.ptr<uint16_t>(),
      host_bits.data(),
      count * sizeof(uint16_t),
      base::MemcpyKind::kMemcpyCUDA2CPU,
      nullptr,
      true);
  std::vector<float> out(count);
  for (size_t i = 0; i < count; ++i) {
    out[i] = bf16_bits_to_float(host_bits[i]);
  }
  return out;
}

struct FastPathCase {
  tensor::Tensor queries;
  tensor::Tensor key_pool;
  tensor::Tensor value_pool;
  tensor::Tensor block_tables;
  tensor::Tensor seq_lens;
  tensor::Tensor out_fast;
  tensor::Tensor out_legacy;
  tensor::Tensor partial_out;
  tensor::Tensor partial_max;
  tensor::Tensor partial_sum;
  int32_t batch_size = 0;
  int32_t head_num = 0;
  int32_t num_kv_heads = 0;
  int32_t head_size = 0;
  int32_t kv_mul = 0;
  int32_t block_size = 0;
  int32_t max_blocks_per_seq = 0;
};

FastPathCase build_bf16_case(
    const std::vector<int32_t>& seq_lens_host,
    int32_t head_num,
    int32_t num_kv_heads,
    int32_t head_size,
    int32_t block_size,
    int seed) {
  using namespace base;
  using namespace tensor;

  auto alloc = CUDADeviceAllocatorFactory::get_instance();
  const int32_t batch_size = static_cast<int32_t>(seq_lens_host.size());
  const int32_t kv_mul = head_num / num_kv_heads;
  const int32_t dim = head_num * head_size;
  const int32_t kv_dim = block_size * num_kv_heads * head_size;
  int32_t max_blocks_per_seq = 1;
  int32_t total_blocks = 0;
  std::vector<int32_t> blocks_per_seq(batch_size, 0);
  for (int32_t b = 0; b < batch_size; ++b) {
    blocks_per_seq[b] = (seq_lens_host[b] + block_size - 1) / block_size;
    max_blocks_per_seq = std::max(max_blocks_per_seq, blocks_per_seq[b]);
    total_blocks += blocks_per_seq[b];
  }

  FastPathCase c{
      Tensor(DataType::kDataTypeBf16, batch_size, dim, true, alloc),
      Tensor(DataType::kDataTypeBf16, total_blocks, kv_dim, true, alloc),
      Tensor(DataType::kDataTypeBf16, total_blocks, kv_dim, true, alloc),
      Tensor(DataType::kDataTypeInt32, batch_size, max_blocks_per_seq, true, alloc),
      Tensor(DataType::kDataTypeInt32, batch_size, true, alloc),
      Tensor(DataType::kDataTypeBf16, batch_size, dim, true, alloc),
      Tensor(DataType::kDataTypeBf16, batch_size, dim, true, alloc),
      Tensor(DataType::kDataTypeFp32, batch_size * head_num * 32 * head_size, true, alloc),
      Tensor(DataType::kDataTypeFp32, batch_size * head_num * 32, true, alloc),
      Tensor(DataType::kDataTypeFp32, batch_size * head_num * 32, true, alloc),
      batch_size,
      head_num,
      num_kv_heads,
      head_size,
      kv_mul,
      block_size,
      max_blocks_per_seq,
  };
  c.queries.set_device_type(DeviceType::kDeviceCUDA);
  c.key_pool.set_device_type(DeviceType::kDeviceCUDA);
  c.value_pool.set_device_type(DeviceType::kDeviceCUDA);
  c.block_tables.set_device_type(DeviceType::kDeviceCUDA);
  c.seq_lens.set_device_type(DeviceType::kDeviceCUDA);
  c.out_fast.set_device_type(DeviceType::kDeviceCUDA);
  c.out_legacy.set_device_type(DeviceType::kDeviceCUDA);
  c.partial_out.set_device_type(DeviceType::kDeviceCUDA);
  c.partial_max.set_device_type(DeviceType::kDeviceCUDA);
  c.partial_sum.set_device_type(DeviceType::kDeviceCUDA);

  copy_to_gpu_bf16(c.queries, make_random_bf16(static_cast<size_t>(batch_size) * dim, seed), alloc);
  copy_to_gpu_bf16(
      c.key_pool,
      make_random_bf16(static_cast<size_t>(total_blocks) * kv_dim, seed + 1),
      alloc);
  copy_to_gpu_bf16(
      c.value_pool,
      make_random_bf16(static_cast<size_t>(total_blocks) * kv_dim, seed + 2),
      alloc);

  std::vector<int32_t> block_tables_host(static_cast<size_t>(batch_size) * max_blocks_per_seq, 0);
  int32_t next_block = 0;
  for (int32_t b = 0; b < batch_size; ++b) {
    for (int32_t blk = 0; blk < blocks_per_seq[b]; ++blk) {
      block_tables_host[b * max_blocks_per_seq + blk] = next_block++;
    }
  }

  alloc->memcpy(
      block_tables_host.data(),
      const_cast<int32_t*>(c.block_tables.ptr<int32_t>()),
      block_tables_host.size() * sizeof(int32_t),
      MemcpyKind::kMemcpyCPU2CUDA,
      nullptr,
      true);
  alloc->memcpy(
      seq_lens_host.data(),
      const_cast<int32_t*>(c.seq_lens.ptr<int32_t>()),
      seq_lens_host.size() * sizeof(int32_t),
      MemcpyKind::kMemcpyCPU2CUDA,
      nullptr,
      true);
  alloc->memset_zero(const_cast<uint16_t*>(c.out_fast.ptr<uint16_t>()), c.out_fast.byte_size(), nullptr, true);
  alloc->memset_zero(
      const_cast<uint16_t*>(c.out_legacy.ptr<uint16_t>()), c.out_legacy.byte_size(), nullptr, true);
  alloc->memset_zero(const_cast<float*>(c.partial_out.ptr<float>()), c.partial_out.byte_size(), nullptr, true);
  alloc->memset_zero(const_cast<float*>(c.partial_max.ptr<float>()), c.partial_max.byte_size(), nullptr, true);
  alloc->memset_zero(const_cast<float*>(c.partial_sum.ptr<float>()), c.partial_sum.byte_size(), nullptr, true);
  return c;
}

float run_fast_vs_legacy_diff(FastPathCase& c) {
  using namespace base;
  using namespace kernel;

  CudaConfig cuda_config;
  cudaStreamCreate(&cuda_config.stream);

  const bool launched = splitkv_batched_paged_mha_fast_decode_cu(
      c.batch_size,
      c.head_num,
      c.head_size,
      c.kv_mul,
      c.queries,
      c.out_fast,
      c.key_pool,
      c.value_pool,
      c.block_tables,
      c.seq_lens,
      c.max_blocks_per_seq,
      c.block_size,
      c.num_kv_heads,
      c.partial_out,
      c.partial_max,
      c.partial_sum,
      DeviceType::kDeviceCUDA,
      &cuda_config);
  EXPECT_TRUE(launched);

  splitkv_batched_paged_mha_decode_cu(
      c.batch_size,
      c.head_num,
      c.head_size,
      c.kv_mul,
      c.queries,
      c.out_legacy,
      c.key_pool,
      c.value_pool,
      c.block_tables,
      c.seq_lens,
      c.max_blocks_per_seq,
      c.block_size,
      c.num_kv_heads,
      c.partial_out,
      c.partial_max,
      c.partial_sum,
      DeviceType::kDeviceCUDA,
      &cuda_config);

  cudaStreamSynchronize(cuda_config.stream);

  auto alloc = CUDADeviceAllocatorFactory::get_instance();
  const size_t out_count = static_cast<size_t>(c.batch_size) * c.head_num * c.head_size;
  const std::vector<float> fast = copy_from_gpu_bf16(c.out_fast, out_count, alloc);
  const std::vector<float> legacy = copy_from_gpu_bf16(c.out_legacy, out_count, alloc);

  float max_diff = 0.f;
  for (size_t i = 0; i < out_count; ++i) {
    max_diff = std::max(max_diff, std::abs(fast[i] - legacy[i]));
  }
  return max_diff;
}

}  // namespace

TEST(PagedAttentionFastTest, ReturnsFalseForFp32) {
  using namespace base;
  using namespace tensor;
  using namespace kernel;

  auto alloc = CUDADeviceAllocatorFactory::get_instance();
  constexpr int32_t batch_size = 2;
  constexpr int32_t head_num = 8;
  constexpr int32_t head_size = 64;
  constexpr int32_t num_kv_heads = 2;
  constexpr int32_t kv_mul = head_num / num_kv_heads;
  constexpr int32_t block_size = 16;
  constexpr int32_t max_blocks_per_seq = 4;
  constexpr int32_t dim = head_num * head_size;
  constexpr int32_t kv_dim = block_size * num_kv_heads * head_size;

  Tensor queries(DataType::kDataTypeFp32, batch_size, dim, true, alloc);
  Tensor outputs(DataType::kDataTypeFp32, batch_size, dim, true, alloc);
  Tensor key_pool(DataType::kDataTypeFp32, max_blocks_per_seq, kv_dim, true, alloc);
  Tensor value_pool(DataType::kDataTypeFp32, max_blocks_per_seq, kv_dim, true, alloc);
  Tensor block_tables(DataType::kDataTypeInt32, batch_size, max_blocks_per_seq, true, alloc);
  Tensor seq_lens(DataType::kDataTypeInt32, batch_size, true, alloc);
  Tensor partial_out(DataType::kDataTypeFp32, batch_size * head_num * 32 * head_size, true, alloc);
  Tensor partial_max(DataType::kDataTypeFp32, batch_size * head_num * 32, true, alloc);
  Tensor partial_sum(DataType::kDataTypeFp32, batch_size * head_num * 32, true, alloc);

  std::vector<float> q = make_random_floats(static_cast<size_t>(batch_size) * dim, 11);
  std::vector<float> kv = make_random_floats(static_cast<size_t>(max_blocks_per_seq) * kv_dim, 12);
  std::vector<int32_t> lens = {16, 64};
  std::vector<int32_t> tables(batch_size * max_blocks_per_seq, 0);
  for (int32_t i = 0; i < batch_size * max_blocks_per_seq; ++i) {
    tables[i] = i % max_blocks_per_seq;
  }

  copy_to_gpu_fp32(queries, q, alloc);
  copy_to_gpu_fp32(key_pool, kv, alloc);
  copy_to_gpu_fp32(value_pool, kv, alloc);
  alloc->memcpy(
      tables.data(),
      const_cast<int32_t*>(block_tables.ptr<int32_t>()),
      tables.size() * sizeof(int32_t),
      MemcpyKind::kMemcpyCPU2CUDA,
      nullptr,
      true);
  alloc->memcpy(
      lens.data(),
      const_cast<int32_t*>(seq_lens.ptr<int32_t>()),
      lens.size() * sizeof(int32_t),
      MemcpyKind::kMemcpyCPU2CUDA,
      nullptr,
      true);

  CudaConfig cuda_config;
  cudaStreamCreate(&cuda_config.stream);
  const bool launched = splitkv_batched_paged_mha_fast_decode_cu(
      batch_size,
      head_num,
      head_size,
      kv_mul,
      queries,
      outputs,
      key_pool,
      value_pool,
      block_tables,
      seq_lens,
      max_blocks_per_seq,
      block_size,
      num_kv_heads,
      partial_out,
      partial_max,
      partial_sum,
      DeviceType::kDeviceCUDA,
      &cuda_config);
  EXPECT_FALSE(launched);
}

TEST(PagedAttentionFastTest, FallsBackShortBucket) {
  using namespace base;
  using namespace kernel;

  FastPathCase c = build_bf16_case({32, 48, 64}, 8, 2, 64, 16, 101);
  CudaConfig cuda_config;
  cudaStreamCreate(&cuda_config.stream);
  const bool launched = splitkv_batched_paged_mha_fast_decode_cu(
      c.batch_size,
      c.head_num,
      c.head_size,
      c.kv_mul,
      c.queries,
      c.out_fast,
      c.key_pool,
      c.value_pool,
      c.block_tables,
      c.seq_lens,
      c.max_blocks_per_seq,
      c.block_size,
      c.num_kv_heads,
      c.partial_out,
      c.partial_max,
      c.partial_sum,
      DeviceType::kDeviceCUDA,
      &cuda_config);
  EXPECT_FALSE(launched);
}

TEST(PagedAttentionFastTest, FallsBackMidBucket) {
  using namespace base;
  using namespace kernel;

  FastPathCase c = build_bf16_case({64, 192, 256}, 8, 2, 64, 16, 202);
  CudaConfig cuda_config;
  cudaStreamCreate(&cuda_config.stream);
  const bool launched = splitkv_batched_paged_mha_fast_decode_cu(
      c.batch_size,
      c.head_num,
      c.head_size,
      c.kv_mul,
      c.queries,
      c.out_fast,
      c.key_pool,
      c.value_pool,
      c.block_tables,
      c.seq_lens,
      c.max_blocks_per_seq,
      c.block_size,
      c.num_kv_heads,
      c.partial_out,
      c.partial_max,
      c.partial_sum,
      DeviceType::kDeviceCUDA,
      &cuda_config);
  EXPECT_FALSE(launched);
}

TEST(PagedAttentionFastTest, MatchesLegacyLongBucket) {
  FastPathCase c = build_bf16_case({176, 512, 1024}, 8, 2, 64, 16, 303);
  const float max_diff = run_fast_vs_legacy_diff(c);
  EXPECT_LT(max_diff, 5e-3f);
}

TEST(PagedAttentionFastTest, EmptySequenceProducesZeroOutput) {
  using namespace base;
  using namespace kernel;

  FastPathCase c = build_bf16_case({0, 1024}, 8, 2, 64, 16, 404);
  CudaConfig cuda_config;
  cudaStreamCreate(&cuda_config.stream);
  const bool launched = splitkv_batched_paged_mha_fast_decode_cu(
      c.batch_size,
      c.head_num,
      c.head_size,
      c.kv_mul,
      c.queries,
      c.out_fast,
      c.key_pool,
      c.value_pool,
      c.block_tables,
      c.seq_lens,
      c.max_blocks_per_seq,
      c.block_size,
      c.num_kv_heads,
      c.partial_out,
      c.partial_max,
      c.partial_sum,
      DeviceType::kDeviceCUDA,
      &cuda_config);
  EXPECT_TRUE(launched);
  cudaStreamSynchronize(cuda_config.stream);

  auto alloc = CUDADeviceAllocatorFactory::get_instance();
  const std::vector<float> out = copy_from_gpu_bf16(
      c.out_fast,
      static_cast<size_t>(c.batch_size) * c.head_num * c.head_size,
      alloc);
  float max_abs = 0.f;
  for (int32_t i = 0; i < c.head_num * c.head_size; ++i) {
    max_abs = std::max(max_abs, std::abs(out[i]));
  }
  EXPECT_LT(max_abs, 1e-6f);
}
