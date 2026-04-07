// Standalone benchmark and validation for the paged decode fast path.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
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

constexpr int32_t kMaxPartitions = 32;

struct HeuristicView {
  bool use_fast_path = false;
  int32_t split_k = 1;
  int32_t head_group = 1;
  int32_t threads = 128;
  bool double_buffer = false;
};

struct BenchCase {
  std::string name;
  std::vector<int32_t> seq_lens;
  int32_t iters = 100;
};

struct CaseBuffers {
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
  std::vector<float> data = make_random_floats(count, seed);
  std::vector<uint16_t> out(count);
  for (size_t i = 0; i < count; ++i) {
    out[i] = float_to_bf16_bits(data[i]);
  }
  return out;
}

void copy_to_gpu(
    tensor::Tensor& tensor,
    const void* host,
    size_t byte_size,
    const std::shared_ptr<base::DeviceAllocator>& alloc) {
  alloc->memcpy(
      host,
      tensor.ptr<uint8_t>(),
      byte_size,
      base::MemcpyKind::kMemcpyCPU2CUDA,
      nullptr,
      true);
}

std::vector<float> copy_from_gpu_bf16(
    const tensor::Tensor& tensor,
    size_t count,
    const std::shared_ptr<base::DeviceAllocator>& alloc) {
  std::vector<uint16_t> bits(count);
  alloc->memcpy(
      tensor.ptr<uint16_t>(),
      bits.data(),
      count * sizeof(uint16_t),
      base::MemcpyKind::kMemcpyCUDA2CPU,
      nullptr,
      true);

  std::vector<float> out(count);
  for (size_t i = 0; i < count; ++i) {
    out[i] = bf16_bits_to_float(bits[i]);
  }
  return out;
}

HeuristicView debug_pick_heuristic(int32_t max_blocks_per_seq) {
  HeuristicView cfg;
  if (max_blocks_per_seq <= 16) {
    cfg.use_fast_path = false;
  } else {
    cfg.use_fast_path = true;
    cfg.split_k = 8;
    cfg.head_group = 2;
    cfg.threads = 256;
    cfg.double_buffer = true;
  }
  return cfg;
}

CaseBuffers build_case(
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
  total_blocks = std::max(total_blocks, 1);

  CaseBuffers c{
      Tensor(DataType::kDataTypeBf16, batch_size, dim, true, alloc),
      Tensor(DataType::kDataTypeBf16, total_blocks, kv_dim, true, alloc),
      Tensor(DataType::kDataTypeBf16, total_blocks, kv_dim, true, alloc),
      Tensor(DataType::kDataTypeInt32, batch_size, max_blocks_per_seq, true, alloc),
      Tensor(DataType::kDataTypeInt32, batch_size, true, alloc),
      Tensor(DataType::kDataTypeBf16, batch_size, dim, true, alloc),
      Tensor(DataType::kDataTypeBf16, batch_size, dim, true, alloc),
      Tensor(DataType::kDataTypeFp32, batch_size * head_num * kMaxPartitions * head_size, true, alloc),
      Tensor(DataType::kDataTypeFp32, batch_size * head_num * kMaxPartitions, true, alloc),
      Tensor(DataType::kDataTypeFp32, batch_size * head_num * kMaxPartitions, true, alloc),
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

  const std::vector<uint16_t> queries_host =
      make_random_bf16(static_cast<size_t>(batch_size) * dim, seed);
  const std::vector<uint16_t> key_host =
      make_random_bf16(static_cast<size_t>(total_blocks) * kv_dim, seed + 1);
  const std::vector<uint16_t> value_host =
      make_random_bf16(static_cast<size_t>(total_blocks) * kv_dim, seed + 2);

  copy_to_gpu(c.queries, queries_host.data(), queries_host.size() * sizeof(uint16_t), alloc);
  copy_to_gpu(c.key_pool, key_host.data(), key_host.size() * sizeof(uint16_t), alloc);
  copy_to_gpu(c.value_pool, value_host.data(), value_host.size() * sizeof(uint16_t), alloc);

  std::vector<int32_t> block_table_host(static_cast<size_t>(batch_size) * max_blocks_per_seq, 0);
  int32_t next_block = 0;
  for (int32_t b = 0; b < batch_size; ++b) {
    for (int32_t blk = 0; blk < blocks_per_seq[b]; ++blk) {
      block_table_host[b * max_blocks_per_seq + blk] = next_block++;
    }
  }
  copy_to_gpu(
      c.block_tables,
      block_table_host.data(),
      block_table_host.size() * sizeof(int32_t),
      alloc);
  copy_to_gpu(c.seq_lens, seq_lens_host.data(), seq_lens_host.size() * sizeof(int32_t), alloc);

  alloc->memset_zero(c.out_fast.ptr<uint8_t>(), c.out_fast.byte_size(), nullptr, true);
  alloc->memset_zero(c.out_legacy.ptr<uint8_t>(), c.out_legacy.byte_size(), nullptr, true);
  alloc->memset_zero(c.partial_out.ptr<uint8_t>(), c.partial_out.byte_size(), nullptr, true);
  alloc->memset_zero(c.partial_max.ptr<uint8_t>(), c.partial_max.byte_size(), nullptr, true);
  alloc->memset_zero(c.partial_sum.ptr<uint8_t>(), c.partial_sum.byte_size(), nullptr, true);
  return c;
}

bool launch_routed_once(CaseBuffers& c, kernel::CudaConfig* config) {
  const bool launched = kernel::splitkv_batched_paged_mha_fast_decode_cu(
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
      base::DeviceType::kDeviceCUDA,
      config);
  if (!launched) {
    kernel::splitkv_batched_paged_mha_decode_cu(
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
        base::DeviceType::kDeviceCUDA,
        config);
  }
  return launched;
}

void launch_legacy_once(CaseBuffers& c, kernel::CudaConfig* config) {
  kernel::splitkv_batched_paged_mha_decode_cu(
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
      base::DeviceType::kDeviceCUDA,
      config);
}

float max_abs_diff(CaseBuffers& c) {
  auto alloc = base::CUDADeviceAllocatorFactory::get_instance();
  const size_t out_count = static_cast<size_t>(c.batch_size) * c.head_num * c.head_size;
  const std::vector<float> fast = copy_from_gpu_bf16(c.out_fast, out_count, alloc);
  const std::vector<float> legacy = copy_from_gpu_bf16(c.out_legacy, out_count, alloc);

  float max_diff = 0.f;
  for (size_t i = 0; i < out_count; ++i) {
    max_diff = std::max(max_diff, std::abs(fast[i] - legacy[i]));
  }
  return max_diff;
}

bool first_sequence_is_zero(const CaseBuffers& c) {
  auto alloc = base::CUDADeviceAllocatorFactory::get_instance();
  const std::vector<float> fast = copy_from_gpu_bf16(
      c.out_fast, static_cast<size_t>(c.batch_size) * c.head_num * c.head_size, alloc);
  float max_abs = 0.f;
  for (int32_t i = 0; i < c.head_num * c.head_size; ++i) {
    max_abs = std::max(max_abs, std::abs(fast[i]));
  }
  return max_abs < 1e-6f;
}

float benchmark_ms(
    CaseBuffers& c,
    bool run_routed_path,
    int32_t warmup_iters,
    int32_t iters,
    kernel::CudaConfig* config) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int32_t i = 0; i < warmup_iters; ++i) {
    if (run_routed_path) {
      launch_routed_once(c, config);
    } else {
      launch_legacy_once(c, config);
    }
  }
  cudaStreamSynchronize(config->stream);

  cudaEventRecord(start, config->stream);
  for (int32_t i = 0; i < iters; ++i) {
    if (run_routed_path) {
      launch_routed_once(c, config);
    } else {
      launch_legacy_once(c, config);
    }
  }
  cudaEventRecord(stop, config->stream);
  cudaEventSynchronize(stop);

  float elapsed_ms = 0.f;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return elapsed_ms / static_cast<float>(iters);
}

bool run_case(const BenchCase& bench_case) {
  constexpr int32_t kHeadNum = 8;
  constexpr int32_t kNumKvHeads = 2;
  constexpr int32_t kHeadSize = 64;
  constexpr int32_t kBlockSize = 16;
  constexpr float kDiffThreshold = 5e-3f;

  CaseBuffers c =
      build_case(bench_case.seq_lens, kHeadNum, kNumKvHeads, kHeadSize, kBlockSize, 1000);
  kernel::CudaConfig cuda_config;
  cudaStreamCreate(&cuda_config.stream);

  const bool launched_fast = launch_routed_once(c, &cuda_config);
  launch_legacy_once(c, &cuda_config);
  cudaStreamSynchronize(cuda_config.stream);

  const float diff = max_abs_diff(c);
  const HeuristicView heuristic = debug_pick_heuristic(c.max_blocks_per_seq);
  const float legacy_ms = benchmark_ms(c, false, 20, bench_case.iters, &cuda_config);
  const float fast_ms = benchmark_ms(c, true, 20, bench_case.iters, &cuda_config);

  std::cout << "[case] " << bench_case.name
            << " batch=" << c.batch_size
            << " max_blocks_per_seq=" << c.max_blocks_per_seq
            << " route=" << (launched_fast ? "fast" : "legacy-fallback");
  if (heuristic.use_fast_path) {
    std::cout << " heuristic={split_k=" << heuristic.split_k
              << ", head_group=" << heuristic.head_group
              << ", threads=" << heuristic.threads
              << ", double_buffer=" << (heuristic.double_buffer ? "true" : "false") << "}";
  }
  std::cout
            << " diff=" << diff
            << " legacy_ms=" << legacy_ms
            << " route_ms=" << fast_ms
            << " speedup=" << (legacy_ms / fast_ms)
            << std::endl;

  return diff <= kDiffThreshold;
}

bool run_empty_case() {
  CaseBuffers c = build_case({0, 1024}, 8, 2, 64, 16, 2000);
  kernel::CudaConfig cuda_config;
  cudaStreamCreate(&cuda_config.stream);
  const bool launched_fast = launch_routed_once(c, &cuda_config);
  cudaStreamSynchronize(cuda_config.stream);
  const bool ok = first_sequence_is_zero(c);
  std::cout << "[case] empty-sequence route=" << (launched_fast ? "fast" : "legacy-fallback")
            << " first_output_zero=" << (ok ? "true" : "false")
            << std::endl;
  return ok;
}

}  // namespace

int main() {
  const std::vector<BenchCase> cases = {
      {"short-bucket", {32, 48, 64}, 300},
      {"mid-bucket", {64, 192, 256}, 200},
      {"long-bucket", {176, 512, 1024}, 80},
      {"mixed-lengths", {1, 5, 16, 64, 256, 1024}, 80},
  };

  bool ok = true;
  for (const auto& bench_case : cases) {
    ok &= run_case(bench_case);
  }
  ok &= run_empty_case();

  if (!ok) {
    std::cerr << "paged_attention_fast_bench failed validation." << std::endl;
    return 1;
  }
  std::cout << "paged_attention_fast_bench passed." << std::endl;
  return 0;
}
