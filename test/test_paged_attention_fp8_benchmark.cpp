// Standalone validation and benchmark for the batch-only FP8 KV cache path.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <base/alloc.h>
#include <base/base.h>
#include <base/bf16.h>
#include <base/block_allocator.h>
#include <base/cuda_config.h>
#include <tensor/tensor.h>

#include "op/kernels/cuda/paged_mha_fast_kernel.cuh"
#include "op/kernels/cuda/paged_mha_kernel.cuh"
#include "op/kernels/cuda/scatter_kv_kernel.cuh"

namespace {

using base::bf16_bits_to_float;
using base::float_to_bf16_bits;

constexpr int32_t kBlockSize = 16;
constexpr int32_t kHeadSize = 64;
constexpr int32_t kHeadNum = 14;
constexpr int32_t kNumKvHeads = 2;
constexpr int32_t kKvMul = kHeadNum / kNumKvHeads;
constexpr int32_t kMaxPartitions = 32;
constexpr float kMaxAllowedDiff = 5e-2f;

struct BenchCase {
  std::string name;
  std::vector<int32_t> seq_lens;
  int32_t iters = 100;
};

struct HeuristicView {
  int32_t split_k = 1;
  int32_t head_group = 1;
  int32_t threads = 128;
  bool double_buffer = false;
};

struct CaseBuffers {
  tensor::Tensor queries;
  tensor::Tensor token_keys;
  tensor::Tensor token_values;
  tensor::Tensor slot_mapping;
  tensor::Tensor key_pool_bf16;
  tensor::Tensor value_pool_bf16;
  tensor::Tensor key_pool_fp8;
  tensor::Tensor value_pool_fp8;
  tensor::Tensor key_scale_pool;
  tensor::Tensor value_scale_pool;
  tensor::Tensor block_tables;
  tensor::Tensor seq_lens;
  tensor::Tensor out_bf16;
  tensor::Tensor out_fp8;
  tensor::Tensor partial_out;
  tensor::Tensor partial_max;
  tensor::Tensor partial_sum;
  int32_t batch_size = 0;
  int32_t total_tokens = 0;
  int32_t total_blocks = 0;
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

int32_t clamp_head_group(int32_t preferred_head_group) {
  if (preferred_head_group <= 1) {
    return 1;
  }
  for (int32_t head_group = preferred_head_group; head_group >= 1; head_group /= 2) {
    if (kHeadNum % head_group == 0 && kKvMul % head_group == 0) {
      return head_group;
    }
  }
  return 1;
}

HeuristicView pick_fp8_heuristic(int32_t max_blocks_per_seq) {
  HeuristicView cfg;
  if (max_blocks_per_seq <= 4) {
    cfg.split_k = 1;
    cfg.head_group = clamp_head_group(4);
    cfg.threads = 128;
    cfg.double_buffer = false;
  } else if (max_blocks_per_seq <= 16) {
    cfg.split_k = 1;
    cfg.head_group = clamp_head_group(4);
    cfg.threads = 256;
    cfg.double_buffer = true;
  } else {
    cfg.split_k = 8;
    cfg.head_group = clamp_head_group(2);
    cfg.threads = 256;
    cfg.double_buffer = true;
  }
  return cfg;
}

CaseBuffers build_case(const BenchCase& bench_case, int seed) {
  using base::CUDADeviceAllocatorFactory;
  using base::DataType;
  using base::DeviceType;
  using tensor::Tensor;

  auto alloc = CUDADeviceAllocatorFactory::get_instance();
  const int32_t batch_size = static_cast<int32_t>(bench_case.seq_lens.size());
  const int32_t dim = kHeadNum * kHeadSize;
  const int32_t kv_dim = kNumKvHeads * kHeadSize;

  int32_t total_tokens = 0;
  int32_t total_blocks = 0;
  int32_t max_blocks_per_seq = 1;
  std::vector<int32_t> blocks_per_seq(batch_size, 0);
  for (int32_t b = 0; b < batch_size; ++b) {
    total_tokens += bench_case.seq_lens[b];
    blocks_per_seq[b] = (bench_case.seq_lens[b] + kBlockSize - 1) / kBlockSize;
    total_blocks += blocks_per_seq[b];
    max_blocks_per_seq = std::max(max_blocks_per_seq, blocks_per_seq[b]);
  }
  total_blocks = std::max(total_blocks, 1);
  total_tokens = std::max(total_tokens, 1);

  CaseBuffers c{
      Tensor(DataType::kDataTypeBf16, batch_size, dim, true, alloc),
      Tensor(DataType::kDataTypeBf16, total_tokens, kv_dim, true, alloc),
      Tensor(DataType::kDataTypeBf16, total_tokens, kv_dim, true, alloc),
      Tensor(DataType::kDataTypeInt32, total_tokens, true, alloc),
      Tensor(DataType::kDataTypeBf16, total_blocks, kBlockSize * kv_dim, true, alloc),
      Tensor(DataType::kDataTypeBf16, total_blocks, kBlockSize * kv_dim, true, alloc),
      Tensor(DataType::kDataTypeInt8, total_blocks, kBlockSize * kv_dim, true, alloc),
      Tensor(DataType::kDataTypeInt8, total_blocks, kBlockSize * kv_dim, true, alloc),
      Tensor(DataType::kDataTypeFp32, total_blocks, kBlockSize * kNumKvHeads, true, alloc),
      Tensor(DataType::kDataTypeFp32, total_blocks, kBlockSize * kNumKvHeads, true, alloc),
      Tensor(DataType::kDataTypeInt32, batch_size, max_blocks_per_seq, true, alloc),
      Tensor(DataType::kDataTypeInt32, batch_size, true, alloc),
      Tensor(DataType::kDataTypeBf16, batch_size, dim, true, alloc),
      Tensor(DataType::kDataTypeBf16, batch_size, dim, true, alloc),
      Tensor(DataType::kDataTypeFp32, batch_size * kHeadNum * kMaxPartitions * kHeadSize, true, alloc),
      Tensor(DataType::kDataTypeFp32, batch_size * kHeadNum * kMaxPartitions, true, alloc),
      Tensor(DataType::kDataTypeFp32, batch_size * kHeadNum * kMaxPartitions, true, alloc),
      batch_size,
      total_tokens,
      total_blocks,
      max_blocks_per_seq,
  };

  c.queries.set_device_type(DeviceType::kDeviceCUDA);
  c.token_keys.set_device_type(DeviceType::kDeviceCUDA);
  c.token_values.set_device_type(DeviceType::kDeviceCUDA);
  c.slot_mapping.set_device_type(DeviceType::kDeviceCUDA);
  c.key_pool_bf16.set_device_type(DeviceType::kDeviceCUDA);
  c.value_pool_bf16.set_device_type(DeviceType::kDeviceCUDA);
  c.key_pool_fp8.set_device_type(DeviceType::kDeviceCUDA);
  c.value_pool_fp8.set_device_type(DeviceType::kDeviceCUDA);
  c.key_scale_pool.set_device_type(DeviceType::kDeviceCUDA);
  c.value_scale_pool.set_device_type(DeviceType::kDeviceCUDA);
  c.block_tables.set_device_type(DeviceType::kDeviceCUDA);
  c.seq_lens.set_device_type(DeviceType::kDeviceCUDA);
  c.out_bf16.set_device_type(DeviceType::kDeviceCUDA);
  c.out_fp8.set_device_type(DeviceType::kDeviceCUDA);
  c.partial_out.set_device_type(DeviceType::kDeviceCUDA);
  c.partial_max.set_device_type(DeviceType::kDeviceCUDA);
  c.partial_sum.set_device_type(DeviceType::kDeviceCUDA);

  copy_to_gpu(
      c.queries,
      make_random_bf16(static_cast<size_t>(batch_size) * dim, seed).data(),
      static_cast<size_t>(batch_size) * dim * sizeof(uint16_t),
      alloc);
  copy_to_gpu(
      c.token_keys,
      make_random_bf16(static_cast<size_t>(total_tokens) * kv_dim, seed + 1).data(),
      static_cast<size_t>(total_tokens) * kv_dim * sizeof(uint16_t),
      alloc);
  copy_to_gpu(
      c.token_values,
      make_random_bf16(static_cast<size_t>(total_tokens) * kv_dim, seed + 2).data(),
      static_cast<size_t>(total_tokens) * kv_dim * sizeof(uint16_t),
      alloc);

  std::vector<int32_t> slot_mapping_host(total_tokens, 0);
  std::vector<int32_t> block_tables_host(static_cast<size_t>(batch_size) * max_blocks_per_seq, -1);
  std::vector<int32_t> seq_lens_host = bench_case.seq_lens;
  int32_t token_cursor = 0;
  int32_t next_block_id = 0;
  for (int32_t b = 0; b < batch_size; ++b) {
    const int32_t seq_len = bench_case.seq_lens[b];
    const int32_t num_blocks = blocks_per_seq[b];
    for (int32_t block = 0; block < num_blocks; ++block) {
      block_tables_host[b * max_blocks_per_seq + block] = next_block_id++;
    }
    for (int32_t t = 0; t < seq_len; ++t) {
      const int32_t logical_block = t / kBlockSize;
      const int32_t offset_in_block = t % kBlockSize;
      const int32_t block_id = block_tables_host[b * max_blocks_per_seq + logical_block];
      slot_mapping_host[token_cursor++] = block_id * kBlockSize + offset_in_block;
    }
  }
  if (token_cursor == 0) {
    slot_mapping_host[0] = 0;
  }

  copy_to_gpu(
      c.slot_mapping,
      slot_mapping_host.data(),
      slot_mapping_host.size() * sizeof(int32_t),
      alloc);
  copy_to_gpu(
      c.block_tables,
      block_tables_host.data(),
      block_tables_host.size() * sizeof(int32_t),
      alloc);
  copy_to_gpu(c.seq_lens, seq_lens_host.data(), seq_lens_host.size() * sizeof(int32_t), alloc);

  alloc->memset_zero(c.key_pool_bf16.ptr<uint8_t>(), c.key_pool_bf16.byte_size(), nullptr, true);
  alloc->memset_zero(c.value_pool_bf16.ptr<uint8_t>(), c.value_pool_bf16.byte_size(), nullptr, true);
  alloc->memset_zero(c.key_pool_fp8.ptr<uint8_t>(), c.key_pool_fp8.byte_size(), nullptr, true);
  alloc->memset_zero(c.value_pool_fp8.ptr<uint8_t>(), c.value_pool_fp8.byte_size(), nullptr, true);
  alloc->memset_zero(c.key_scale_pool.ptr<uint8_t>(), c.key_scale_pool.byte_size(), nullptr, true);
  alloc->memset_zero(c.value_scale_pool.ptr<uint8_t>(), c.value_scale_pool.byte_size(), nullptr, true);
  alloc->memset_zero(c.out_bf16.ptr<uint8_t>(), c.out_bf16.byte_size(), nullptr, true);
  alloc->memset_zero(c.out_fp8.ptr<uint8_t>(), c.out_fp8.byte_size(), nullptr, true);
  alloc->memset_zero(c.partial_out.ptr<uint8_t>(), c.partial_out.byte_size(), nullptr, true);
  alloc->memset_zero(c.partial_max.ptr<uint8_t>(), c.partial_max.byte_size(), nullptr, true);
  alloc->memset_zero(c.partial_sum.ptr<uint8_t>(), c.partial_sum.byte_size(), nullptr, true);

  return c;
}

void scatter_bf16_cache(CaseBuffers& c, kernel::CudaConfig* config) {
  kernel::scatter_kv_batch_to_pages_cu(
      c.token_keys,
      c.token_values,
      c.key_pool_bf16,
      c.value_pool_bf16,
      c.slot_mapping,
      kBlockSize,
      kNumKvHeads,
      kHeadSize,
      c.total_tokens,
      base::DeviceType::kDeviceCUDA,
      config);
}

void scatter_fp8_cache(CaseBuffers& c, kernel::CudaConfig* config) {
  kernel::scatter_kv_batch_to_pages_fp8_e4m3_cu(
      c.token_keys,
      c.token_values,
      c.key_pool_fp8,
      c.value_pool_fp8,
      c.key_scale_pool,
      c.value_scale_pool,
      c.slot_mapping,
      kBlockSize,
      kNumKvHeads,
      kHeadSize,
      c.total_tokens,
      base::DeviceType::kDeviceCUDA,
      config);
}

bool decode_bf16(CaseBuffers& c, kernel::CudaConfig* config) {
  const bool launched_fast = kernel::splitkv_batched_paged_mha_fast_decode_cu(
      c.batch_size,
      kHeadNum,
      kHeadSize,
      kKvMul,
      c.queries,
      c.out_bf16,
      c.key_pool_bf16,
      c.value_pool_bf16,
      c.block_tables,
      c.seq_lens,
      c.max_blocks_per_seq,
      kBlockSize,
      kNumKvHeads,
      c.partial_out,
      c.partial_max,
      c.partial_sum,
      base::DeviceType::kDeviceCUDA,
      config);
  if (!launched_fast) {
    kernel::splitkv_batched_paged_mha_decode_cu(
        c.batch_size,
        kHeadNum,
        kHeadSize,
        kKvMul,
        c.queries,
        c.out_bf16,
        c.key_pool_bf16,
        c.value_pool_bf16,
        c.block_tables,
        c.seq_lens,
        c.max_blocks_per_seq,
        kBlockSize,
        kNumKvHeads,
        c.partial_out,
        c.partial_max,
        c.partial_sum,
        base::DeviceType::kDeviceCUDA,
        config);
  }
  return launched_fast;
}

bool decode_fp8(CaseBuffers& c, kernel::CudaConfig* config) {
  return kernel::splitkv_batched_paged_mha_fp8_decode_cu(
      c.batch_size,
      kHeadNum,
      kHeadSize,
      kKvMul,
      c.queries,
      c.out_fp8,
      c.key_pool_fp8,
      c.value_pool_fp8,
      c.key_scale_pool,
      c.value_scale_pool,
      c.block_tables,
      c.seq_lens,
      c.max_blocks_per_seq,
      kBlockSize,
      kNumKvHeads,
      c.partial_out,
      c.partial_max,
      c.partial_sum,
      base::DeviceType::kDeviceCUDA,
      config);
}

template <typename Fn>
float measure_ms(cudaStream_t stream, int32_t iters, Fn&& fn) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int32_t i = 0; i < 10; ++i) {
    fn();
  }
  cudaStreamSynchronize(stream);

  cudaEventRecord(start, stream);
  for (int32_t i = 0; i < iters; ++i) {
    fn();
  }
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);

  float ms = 0.f;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return ms / static_cast<float>(iters);
}

float max_abs_diff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  float max_diff = 0.f;
  for (size_t i = 0; i < lhs.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(lhs[i] - rhs[i]));
  }
  return max_diff;
}

float max_abs_value(const std::vector<float>& values, size_t begin, size_t end) {
  float max_val = 0.f;
  for (size_t i = begin; i < end; ++i) {
    max_val = std::max(max_val, std::fabs(values[i]));
  }
  return max_val;
}

bool all_finite(const std::vector<float>& values) {
  for (float v : values) {
    if (!std::isfinite(v)) {
      return false;
    }
  }
  return true;
}

void run_allocator_smoke_test() {
  auto alloc = base::CUDADeviceAllocatorFactory::get_instance();
  base::BlockAllocator plain(
      4,
      kBlockSize,
      kNumKvHeads,
      kHeadSize,
      base::DataType::kDataTypeBf16,
      base::DeviceType::kDeviceCUDA,
      base::BlockStorageMode::kPlain);
  base::BlockAllocator fp8(
      4,
      kBlockSize,
      kNumKvHeads,
      kHeadSize,
      base::DataType::kDataTypeBf16,
      base::DeviceType::kDeviceCUDA,
      base::BlockStorageMode::kFp8E4M3PerTokenHead);
  (void)alloc;

  if (plain.key_pool().data_type() != base::DataType::kDataTypeBf16 ||
      plain.value_pool().data_type() != base::DataType::kDataTypeBf16 ||
      plain.uses_fp8_storage()) {
    std::cerr << "allocator smoke test failed for plain storage\n";
    std::exit(EXIT_FAILURE);
  }
  if (fp8.key_pool().data_type() != base::DataType::kDataTypeInt8 ||
      fp8.value_pool().data_type() != base::DataType::kDataTypeInt8 ||
      fp8.key_scale_pool().data_type() != base::DataType::kDataTypeFp32 ||
      fp8.value_scale_pool().data_type() != base::DataType::kDataTypeFp32 ||
      !fp8.uses_fp8_storage()) {
    std::cerr << "allocator smoke test failed for fp8 storage\n";
    std::exit(EXIT_FAILURE);
  }
  if (fp8.key_scale_pool().get_dim(1) != kBlockSize * kNumKvHeads ||
      fp8.value_scale_pool().get_dim(1) != kBlockSize * kNumKvHeads) {
    std::cerr << "allocator scale pool shape is wrong\n";
    std::exit(EXIT_FAILURE);
  }
}

void run_correctness_case(
    const BenchCase& bench_case,
    const std::shared_ptr<base::DeviceAllocator>& alloc,
    kernel::CudaConfig* config) {
  CaseBuffers c = build_case(bench_case, 100 + static_cast<int>(bench_case.seq_lens.size()));
  scatter_bf16_cache(c, config);
  scatter_fp8_cache(c, config);

  const bool bf16_fast = decode_bf16(c, config);
  const bool fp8_fast = decode_fp8(c, config);
  cudaStreamSynchronize(config->stream);
  if (!fp8_fast) {
    std::cerr << "FP8 decode launch failed for case " << bench_case.name << "\n";
    std::exit(EXIT_FAILURE);
  }

  const std::vector<float> out_bf16 =
      copy_from_gpu_bf16(c.out_bf16, static_cast<size_t>(c.batch_size) * kHeadNum * kHeadSize, alloc);
  const std::vector<float> out_fp8 =
      copy_from_gpu_bf16(c.out_fp8, static_cast<size_t>(c.batch_size) * kHeadNum * kHeadSize, alloc);
  float diff = 0.f;
  float empty_seq_abs = 0.f;
  bool fp8_finite = true;
  bool bf16_non_empty_finite = true;
  const size_t row_size = static_cast<size_t>(kHeadNum) * kHeadSize;
  for (int32_t b = 0; b < c.batch_size; ++b) {
    const size_t begin = static_cast<size_t>(b) * row_size;
    const size_t end = begin + row_size;
    const std::vector<float> bf16_row(out_bf16.begin() + begin, out_bf16.begin() + end);
    const std::vector<float> fp8_row(out_fp8.begin() + begin, out_fp8.begin() + end);
    fp8_finite = fp8_finite && all_finite(fp8_row);
    if (bench_case.seq_lens[b] == 0) {
      empty_seq_abs = std::max(empty_seq_abs, max_abs_value(out_fp8, begin, end));
      continue;
    }
    bf16_non_empty_finite = bf16_non_empty_finite && all_finite(bf16_row);
    diff = std::max(diff, max_abs_diff(bf16_row, fp8_row));
  }

  const HeuristicView h = pick_fp8_heuristic(c.max_blocks_per_seq);
  std::cout << "[correctness] " << bench_case.name
            << " seq_lens=";
  for (size_t i = 0; i < bench_case.seq_lens.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << bench_case.seq_lens[i];
  }
  std::cout << " bf16_impl=" << (bf16_fast ? "fast" : "legacy")
            << " fp8(split_k=" << h.split_k
            << ", head_group=" << h.head_group
            << ", threads=" << h.threads
            << ", double_buffer=" << (h.double_buffer ? "true" : "false") << ")"
            << " max_abs_diff=" << diff
            << " empty_seq_abs=" << empty_seq_abs << "\n";

  if (!fp8_finite || !bf16_non_empty_finite || diff > kMaxAllowedDiff || empty_seq_abs > 1e-6f) {
    std::cerr << "correctness check failed for " << bench_case.name
              << ", fp8_finite=" << fp8_finite
              << ", bf16_non_empty_finite=" << bf16_non_empty_finite
              << ", diff=" << diff
              << ", empty_seq_abs=" << empty_seq_abs << "\n";
    std::exit(EXIT_FAILURE);
  }
}

void run_perf_case(
    const BenchCase& bench_case,
    kernel::CudaConfig* config) {
  CaseBuffers c = build_case(bench_case, 200 + bench_case.iters);
  scatter_bf16_cache(c, config);
  scatter_fp8_cache(c, config);
  const bool bf16_fast = decode_bf16(c, config);
  const bool fp8_fast = decode_fp8(c, config);
  cudaStreamSynchronize(config->stream);
  if (!fp8_fast) {
    std::cerr << "FP8 decode launch failed for perf case " << bench_case.name << "\n";
    std::exit(EXIT_FAILURE);
  }

  const float bf16_decode_ms = measure_ms(config->stream, bench_case.iters, [&]() {
    decode_bf16(c, config);
  });
  const float fp8_decode_ms = measure_ms(config->stream, bench_case.iters, [&]() {
    decode_fp8(c, config);
  });
  const float bf16_step_ms = measure_ms(config->stream, bench_case.iters, [&]() {
    scatter_bf16_cache(c, config);
    decode_bf16(c, config);
  });
  const float fp8_step_ms = measure_ms(config->stream, bench_case.iters, [&]() {
    scatter_fp8_cache(c, config);
    decode_fp8(c, config);
  });

  const HeuristicView h = pick_fp8_heuristic(c.max_blocks_per_seq);
  std::cout << "[perf] " << bench_case.name
            << " max_blocks=" << c.max_blocks_per_seq
            << " bf16_impl=" << (bf16_fast ? "fast" : "legacy")
            << " fp8(split_k=" << h.split_k
            << ", head_group=" << h.head_group
            << ", threads=" << h.threads
            << ", double_buffer=" << (h.double_buffer ? "true" : "false") << ")"
            << " decode_ms: bf16=" << bf16_decode_ms
            << ", fp8=" << fp8_decode_ms
            << " | step_ms: bf16=" << bf16_step_ms
            << ", fp8=" << fp8_step_ms
            << "\n";
}

}  // namespace

int main() {
  auto alloc = base::CUDADeviceAllocatorFactory::get_instance();

  kernel::CudaConfig config;
  cudaSetDevice(0);
  cudaStreamCreate(&config.stream);

  run_allocator_smoke_test();

  const std::vector<BenchCase> correctness_cases = {
      {"bs1_ctx1", {1}, 1},
      {"bs1_ctx5", {5}, 1},
      {"bs1_ctx16", {16}, 1},
      {"bs1_ctx64", {64}, 1},
      {"bs1_ctx256", {256}, 1},
      {"bs1_ctx1024", {1024}, 1},
      {"bs2_mixed_short", {5, 16}, 1},
      {"bs2_mixed_far", {1, 1024}, 1},
      {"bs8_mixed", {0, 5, 16, 64, 256, 1024, 17, 3}, 1},
  };
  for (const auto& bench_case : correctness_cases) {
    run_correctness_case(bench_case, alloc, &config);
  }

  const std::vector<BenchCase> perf_cases = {
      {"uniform_bs8_ctx64", std::vector<int32_t>(8, 64), 200},
      {"uniform_bs8_ctx256", std::vector<int32_t>(8, 256), 150},
      {"uniform_bs8_ctx1024", std::vector<int32_t>(8, 1024), 80},
      {"mixed_bs8_ctx", {17, 64, 127, 256, 513, 768, 1024, 0}, 120},
  };
  for (const auto& bench_case : perf_cases) {
    run_perf_case(bench_case, &config);
  }

  std::cout << "paged_attention_fp8_bench: all checks passed\n";
  return 0;
}
