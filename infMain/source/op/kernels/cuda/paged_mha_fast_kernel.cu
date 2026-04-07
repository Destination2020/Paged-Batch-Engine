// Fast-path paged MHA decode kernel implementation
#include "op/kernels/cuda/paged_mha_fast_kernel.cuh"

#include <cfloat>
#include <cstdint>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "base/bf16.h"
#include "cuda_type_utils.cuh"

namespace kernel {

namespace {

constexpr int32_t kFastHeadSize = 64;
constexpr int32_t kFastBlockSize = 16;
constexpr int32_t kFastStage2Threads = 128;
constexpr int32_t kFastVec4PerHead = 8;
constexpr int32_t kFastSmemVecStride = kFastVec4PerHead + 1;
constexpr int32_t kFastFp8Vec16PerHead = 4;
constexpr int32_t kFastFp8SmemVecStride = kFastFp8Vec16PerHead + 1;
constexpr int32_t kFastStage2MaxSplitK = 8;
constexpr float kLog2e = 1.4426950408889634f;

struct FastDecodeLaunchConfig {
  bool use_fast_path = false;
  bool use_double_buffer = false;
  int32_t split_k = 1;
  int32_t head_group = 1;
  int32_t threads = 128;
  int32_t num_stages = 1;
};

template <typename T>
__device__ inline T warp_sum(T value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ inline float warp_max(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
  }
  return value;
}

__device__ inline float fp8_e4m3_to_float(uint8_t bits) {
  __nv_fp8_e4m3 value;
  value.__x = bits;
  return static_cast<float>(value);
}

int32_t clamp_head_group(int32_t preferred_head_group, int32_t head_num, int32_t kv_mul) {
  if (preferred_head_group <= 1) {
    return 1;
  }
  for (int32_t head_group = preferred_head_group; head_group >= 1; head_group /= 2) {
    if (head_num % head_group == 0 && kv_mul % head_group == 0) {
      return head_group;
    }
  }
  return 1;
}

template <int BLOCK_SIZE>
__device__ inline void load_kv_tile_vec4_cooperative(
    int32_t buffer_idx,
    int32_t physical_block_id,
    int32_t kv_head,
    int32_t num_kv_heads,
    int32_t thread_idx,
    int32_t thread_count,
    const base::CudaBF16* key_pool,
    const base::CudaBF16* value_pool,
    float4 k_smem[][BLOCK_SIZE][kFastSmemVecStride],
    float4 v_smem[][BLOCK_SIZE][kFastSmemVecStride]) {
  const int32_t token_stride = num_kv_heads * kFastHeadSize;
  const int32_t block_stride = BLOCK_SIZE * token_stride;
  const int32_t tile_vecs = BLOCK_SIZE * kFastVec4PerHead;

  for (int32_t idx = thread_idx; idx < tile_vecs; idx += thread_count) {
    const int32_t token = idx / kFastVec4PerHead;
    const int32_t vec = idx % kFastVec4PerHead;
    const int64_t elem_offset =
        static_cast<int64_t>(physical_block_id) * block_stride +
        static_cast<int64_t>(token) * token_stride +
        static_cast<int64_t>(kv_head) * kFastHeadSize +
        vec * 8;
    k_smem[buffer_idx][token][vec] =
        *reinterpret_cast<const float4*>(key_pool + elem_offset);
    v_smem[buffer_idx][token][vec] =
        *reinterpret_cast<const float4*>(value_pool + elem_offset);
  }
}

template <int BLOCK_SIZE>
__device__ inline void load_fp8_kv_tile_vec16_cooperative(
    int32_t buffer_idx,
    int32_t physical_block_id,
    int32_t kv_head,
    int32_t num_kv_heads,
    int32_t thread_idx,
    int32_t thread_count,
    const int8_t* key_pool,
    const int8_t* value_pool,
    const float* key_scale_pool,
    const float* value_scale_pool,
    int4 k_smem[][BLOCK_SIZE][kFastFp8SmemVecStride],
    int4 v_smem[][BLOCK_SIZE][kFastFp8SmemVecStride],
    float k_scale_smem[][BLOCK_SIZE],
    float v_scale_smem[][BLOCK_SIZE]) {
  const int32_t token_stride_bytes = num_kv_heads * kFastHeadSize;
  const int32_t block_stride_bytes = BLOCK_SIZE * token_stride_bytes;
  const int32_t tile_vecs = BLOCK_SIZE * kFastFp8Vec16PerHead;

  for (int32_t idx = thread_idx; idx < tile_vecs; idx += thread_count) {
    const int32_t token = idx / kFastFp8Vec16PerHead;
    const int32_t vec = idx % kFastFp8Vec16PerHead;
    const int64_t byte_offset =
        static_cast<int64_t>(physical_block_id) * block_stride_bytes +
        static_cast<int64_t>(token) * token_stride_bytes +
        static_cast<int64_t>(kv_head) * kFastHeadSize +
        vec * sizeof(int4);
    k_smem[buffer_idx][token][vec] =
        reinterpret_cast<const int4*>(key_pool)[byte_offset / sizeof(int4)];
    v_smem[buffer_idx][token][vec] =
        reinterpret_cast<const int4*>(value_pool)[byte_offset / sizeof(int4)];
  }

  for (int32_t token = thread_idx; token < BLOCK_SIZE; token += thread_count) {
    const int64_t scale_offset =
        static_cast<int64_t>(physical_block_id) * (BLOCK_SIZE * num_kv_heads) +
        static_cast<int64_t>(token) * num_kv_heads + kv_head;
    k_scale_smem[buffer_idx][token] = key_scale_pool[scale_offset];
    v_scale_smem[buffer_idx][token] = value_scale_pool[scale_offset];
  }
}

template <int HEAD_GROUP>
__device__ inline void load_query_vec4(
    int32_t batch_idx,
    int32_t head_base,
    int32_t head_num,
    const base::CudaBF16* queries,
    float4 q_smem[HEAD_GROUP][kFastVec4PerHead]) {
  const int32_t dim = head_num * kFastHeadSize;
  for (int32_t idx = threadIdx.x; idx < HEAD_GROUP * kFastVec4PerHead; idx += blockDim.x) {
    const int32_t local_head = idx / kFastVec4PerHead;
    const int32_t vec = idx % kFastVec4PerHead;
    const int64_t elem_offset =
        static_cast<int64_t>(batch_idx) * dim +
        static_cast<int64_t>(head_base + local_head) * kFastHeadSize +
        vec * 8;
    q_smem[local_head][vec] = *reinterpret_cast<const float4*>(queries + elem_offset);
  }
}

template <int HEAD_GROUP>
__device__ inline void store_empty_partial(
    int64_t split_group_base,
    float* partial_out,
    float* partial_max,
    float* partial_sum) {
  if (threadIdx.x < HEAD_GROUP * 32) {
    const int32_t local_head = threadIdx.x / 32;
    const int32_t lane = threadIdx.x % 32;
    const int64_t meta_offset = split_group_base + local_head;
    const int64_t base_offset = meta_offset * kFastHeadSize;
    partial_out[base_offset + lane] = 0.f;
    partial_out[base_offset + lane + 32] = 0.f;
    if (lane == 0) {
      partial_max[meta_offset] = -FLT_MAX;
      partial_sum[meta_offset] = 0.f;
    }
  }
}

FastDecodeLaunchConfig choose_fast_decode_config(
    int32_t head_num,
    int32_t head_size,
    int32_t kv_mul,
    int32_t max_blocks_per_seq,
    int32_t block_size,
    int32_t num_kv_heads,
    const tensor::Tensor& queries,
    const tensor::Tensor& outputs,
    const tensor::Tensor& key_pool,
    const tensor::Tensor& value_pool,
    const tensor::Tensor& block_tables,
    const tensor::Tensor& seq_lens,
    const tensor::Tensor& partial_out,
    const tensor::Tensor& partial_max,
    const tensor::Tensor& partial_sum,
    base::DeviceType device_type) {
  FastDecodeLaunchConfig cfg;
  if (device_type != base::DeviceType::kDeviceCUDA) {
    return cfg;
  }
  if (head_size != kFastHeadSize || block_size != kFastBlockSize) {
    return cfg;
  }
  if (queries.data_type() != base::DataType::kDataTypeBf16 ||
      outputs.data_type() != base::DataType::kDataTypeBf16 ||
      key_pool.data_type() != base::DataType::kDataTypeBf16 ||
      value_pool.data_type() != base::DataType::kDataTypeBf16) {
    return cfg;
  }
  if (block_tables.data_type() != base::DataType::kDataTypeInt32 ||
      seq_lens.data_type() != base::DataType::kDataTypeInt32 ||
      partial_out.data_type() != base::DataType::kDataTypeFp32 ||
      partial_max.data_type() != base::DataType::kDataTypeFp32 ||
      partial_sum.data_type() != base::DataType::kDataTypeFp32) {
    return cfg;
  }
  if (num_kv_heads <= 0 || kv_mul <= 0 || head_num <= 0) {
    return cfg;
  }
  if (head_num % num_kv_heads != 0 || kv_mul != head_num / num_kv_heads) {
    return cfg;
  }
  if ((reinterpret_cast<uintptr_t>(queries.ptr<uint16_t>()) & 0xF) != 0 ||
      (reinterpret_cast<uintptr_t>(key_pool.ptr<uint16_t>()) & 0xF) != 0 ||
      (reinterpret_cast<uintptr_t>(value_pool.ptr<uint16_t>()) & 0xF) != 0 ||
      (reinterpret_cast<uintptr_t>(outputs.ptr<uint16_t>()) & 0xF) != 0) {
    return cfg;
  }

  if (max_blocks_per_seq <= 16) {
    return FastDecodeLaunchConfig{};
  }
  cfg.use_fast_path = true;
  cfg.split_k = 8;
  cfg.head_group = 2;
  cfg.threads = 256;
  cfg.num_stages = 3;
  cfg.use_double_buffer = true;

  if (head_num % cfg.head_group != 0 || kv_mul % cfg.head_group != 0) {
    cfg = FastDecodeLaunchConfig{};
  }
  return cfg;
}

FastDecodeLaunchConfig choose_fp8_decode_config(
    int32_t head_num,
    int32_t head_size,
    int32_t kv_mul,
    int32_t max_blocks_per_seq,
    int32_t block_size,
    int32_t num_kv_heads,
    const tensor::Tensor& queries,
    const tensor::Tensor& outputs,
    const tensor::Tensor& key_pool,
    const tensor::Tensor& value_pool,
    const tensor::Tensor& key_scale_pool,
    const tensor::Tensor& value_scale_pool,
    const tensor::Tensor& block_tables,
    const tensor::Tensor& seq_lens,
    const tensor::Tensor& partial_out,
    const tensor::Tensor& partial_max,
    const tensor::Tensor& partial_sum,
    base::DeviceType device_type) {
  FastDecodeLaunchConfig cfg;
  if (device_type != base::DeviceType::kDeviceCUDA) {
    return cfg;
  }
  if (head_size != kFastHeadSize || block_size != kFastBlockSize) {
    return cfg;
  }
  if (queries.data_type() != base::DataType::kDataTypeBf16 ||
      outputs.data_type() != base::DataType::kDataTypeBf16 ||
      key_pool.data_type() != base::DataType::kDataTypeInt8 ||
      value_pool.data_type() != base::DataType::kDataTypeInt8 ||
      key_scale_pool.data_type() != base::DataType::kDataTypeFp32 ||
      value_scale_pool.data_type() != base::DataType::kDataTypeFp32) {
    return cfg;
  }
  if (block_tables.data_type() != base::DataType::kDataTypeInt32 ||
      seq_lens.data_type() != base::DataType::kDataTypeInt32 ||
      partial_out.data_type() != base::DataType::kDataTypeFp32 ||
      partial_max.data_type() != base::DataType::kDataTypeFp32 ||
      partial_sum.data_type() != base::DataType::kDataTypeFp32) {
    return cfg;
  }
  if (num_kv_heads <= 0 || kv_mul <= 0 || head_num <= 0) {
    return cfg;
  }
  if (head_num % num_kv_heads != 0 || kv_mul != head_num / num_kv_heads) {
    return cfg;
  }
  if ((reinterpret_cast<uintptr_t>(queries.ptr<uint16_t>()) & 0xF) != 0 ||
      (reinterpret_cast<uintptr_t>(key_pool.ptr<int8_t>()) & 0xF) != 0 ||
      (reinterpret_cast<uintptr_t>(value_pool.ptr<int8_t>()) & 0xF) != 0 ||
      (reinterpret_cast<uintptr_t>(outputs.ptr<uint16_t>()) & 0xF) != 0) {
    return cfg;
  }

  cfg.use_fast_path = true;
  if (max_blocks_per_seq <= 4) {
    cfg.split_k = 1;
    cfg.head_group = clamp_head_group(4, head_num, kv_mul);
    cfg.threads = 128;
    cfg.num_stages = 2;
    cfg.use_double_buffer = false;
  } else if (max_blocks_per_seq <= 16) {
    cfg.split_k = 1;
    cfg.head_group = clamp_head_group(4, head_num, kv_mul);
    cfg.threads = 256;
    cfg.num_stages = 3;
    cfg.use_double_buffer = true;
  } else {
    cfg.split_k = 8;
    cfg.head_group = clamp_head_group(2, head_num, kv_mul);
    cfg.threads = 256;
    cfg.num_stages = 3;
    cfg.use_double_buffer = true;
  }

  if (head_num % cfg.head_group != 0 || kv_mul % cfg.head_group != 0) {
    cfg = FastDecodeLaunchConfig{};
  }
  return cfg;
}

template <int HEAD_GROUP, int THREADS, bool DOUBLE_BUFFER>
__global__ void splitkv_paged_mha_fast_stage1_kernel(
    const base::CudaBF16* __restrict__ queries,
    const base::CudaBF16* __restrict__ key_pool,
    const base::CudaBF16* __restrict__ value_pool,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ seq_lens,
    float* __restrict__ partial_out,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    int32_t batch_size,
    int32_t head_num,
    int32_t kv_mul,
    int32_t max_blocks_per_seq,
    int32_t num_kv_heads,
    int32_t split_k,
    float scale) {
  constexpr int kBuffers = DOUBLE_BUFFER ? 2 : 1;
  const int32_t head_group_idx = blockIdx.x;
  const int32_t batch_idx = blockIdx.y;
  const int32_t split_idx = blockIdx.z;
  const int32_t num_head_groups = head_num / HEAD_GROUP;

  if (batch_idx >= batch_size || head_group_idx >= num_head_groups) {
    return;
  }

  __shared__ float4 q_smem[HEAD_GROUP][kFastVec4PerHead];
  __shared__ float4 k_smem[kBuffers][kFastBlockSize][kFastSmemVecStride];
  __shared__ float4 v_smem[kBuffers][kFastBlockSize][kFastSmemVecStride];

  const int32_t head_base = head_group_idx * HEAD_GROUP;
  const int32_t kv_head = head_base / kv_mul;
  const int32_t seq_len = seq_lens[batch_idx];
  const int32_t num_kv_blocks = (seq_len + kFastBlockSize - 1) / kFastBlockSize;
  const int32_t blocks_per_split = (num_kv_blocks + split_k - 1) / split_k;
  const int32_t block_start = split_idx * blocks_per_split;
  int32_t block_end = block_start + blocks_per_split;
  if (block_end > num_kv_blocks) {
    block_end = num_kv_blocks;
  }
  const int32_t block_table_base = batch_idx * max_blocks_per_seq;
  const int64_t split_group_base =
      (((static_cast<int64_t>(batch_idx) * num_head_groups + head_group_idx) * split_k +
        split_idx) *
       HEAD_GROUP);

  load_query_vec4<HEAD_GROUP>(batch_idx, head_base, head_num, queries, q_smem);
  __syncthreads();

  if (seq_len <= 0 || block_start >= block_end) {
    store_empty_partial<HEAD_GROUP>(split_group_base, partial_out, partial_max, partial_sum);
    return;
  }

  const int32_t last_block_tokens =
      seq_len - (num_kv_blocks - 1) * kFastBlockSize;
  int32_t current_block = block_start;
  int32_t current_buffer = 0;

  load_kv_tile_vec4_cooperative<kFastBlockSize>(
      current_buffer, block_tables[block_table_base + current_block], kv_head, num_kv_heads,
      threadIdx.x, THREADS, key_pool, value_pool, k_smem, v_smem);
  __syncthreads();

  const int32_t warp_id = threadIdx.x / 32;
  const int32_t lane = threadIdx.x & 31;
  const bool compute_warp = warp_id < HEAD_GROUP;
  const bool loader_thread = DOUBLE_BUFFER && !compute_warp;
  const int32_t local_head = warp_id;
  constexpr int32_t kComputeThreads = HEAD_GROUP * 32;
  constexpr int32_t kLoaderThreads = THREADS - kComputeThreads;

  float q0 = 0.f;
  float q1 = 0.f;
  float acc0 = 0.f;
  float acc1 = 0.f;
  float m = -FLT_MAX;
  float l = 0.f;

  if (compute_warp) {
    const base::CudaBF16* q_head =
        reinterpret_cast<const base::CudaBF16*>(q_smem[local_head]);
    q0 = scalar_to_float(q_head[lane]);
    q1 = scalar_to_float(q_head[lane + 32]);
  }

  while (true) {
    const int32_t valid_tokens =
        (current_block == num_kv_blocks - 1) ? last_block_tokens : kFastBlockSize;
    const int32_t next_block = current_block + 1;
    const bool has_next_block = next_block < block_end;

    if (DOUBLE_BUFFER && has_next_block && loader_thread) {
      const int32_t next_buffer = current_buffer ^ 1;
      load_kv_tile_vec4_cooperative<kFastBlockSize>(
          next_buffer, block_tables[block_table_base + next_block], kv_head, num_kv_heads,
          threadIdx.x - kComputeThreads, kLoaderThreads, key_pool, value_pool, k_smem, v_smem);
    }

    if (compute_warp) {
      #pragma unroll
      for (int32_t token = 0; token < kFastBlockSize; ++token) {
        if (token >= valid_tokens) {
          break;
        }
        const base::CudaBF16* k_row =
            reinterpret_cast<const base::CudaBF16*>(k_smem[current_buffer][token]);
        const base::CudaBF16* v_row =
            reinterpret_cast<const base::CudaBF16*>(v_smem[current_buffer][token]);

        float score = q0 * scalar_to_float(k_row[lane]) +
                      q1 * scalar_to_float(k_row[lane + 32]);
        score = warp_sum(score);
        score *= scale;

        float alpha = 0.f;
        float p = 0.f;
        float m_new = m;
        if (lane == 0) {
          m_new = fmaxf(score, m);
          alpha = exp2f(m - m_new);
          p = exp2f(score - m_new);
          l = l * alpha + p;
          m = m_new;
        }
        m_new = __shfl_sync(0xffffffffu, m_new, 0);
        alpha = __shfl_sync(0xffffffffu, alpha, 0);
        p = __shfl_sync(0xffffffffu, p, 0);
        l = __shfl_sync(0xffffffffu, l, 0);
        m = m_new;

        acc0 = acc0 * alpha + scalar_to_float(v_row[lane]) * p;
        acc1 = acc1 * alpha + scalar_to_float(v_row[lane + 32]) * p;
      }
    }

    __syncthreads();

    ++current_block;
    if (current_block >= block_end) {
      break;
    }

    if (DOUBLE_BUFFER) {
      current_buffer ^= 1;
    } else {
      load_kv_tile_vec4_cooperative<kFastBlockSize>(
          current_buffer, block_tables[block_table_base + current_block], kv_head, num_kv_heads,
          threadIdx.x, THREADS, key_pool, value_pool, k_smem, v_smem);
      __syncthreads();
    }
  }

  if (compute_warp) {
    const int64_t meta_offset = split_group_base + local_head;
    const int64_t out_base = meta_offset * kFastHeadSize;
    partial_out[out_base + lane] = acc0;
    partial_out[out_base + lane + 32] = acc1;

    if (lane == 0) {
      partial_max[meta_offset] = m;
      partial_sum[meta_offset] = l;
    }
  }
}

template <int HEAD_GROUP, int THREADS, bool DOUBLE_BUFFER>
__global__ void splitkv_paged_mha_fp8_stage1_kernel(
    const base::CudaBF16* __restrict__ queries,
    const int8_t* __restrict__ key_pool,
    const int8_t* __restrict__ value_pool,
    const float* __restrict__ key_scale_pool,
    const float* __restrict__ value_scale_pool,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ seq_lens,
    float* __restrict__ partial_out,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    int32_t batch_size,
    int32_t head_num,
    int32_t kv_mul,
    int32_t max_blocks_per_seq,
    int32_t num_kv_heads,
    int32_t split_k,
    float scale) {
  constexpr int kBuffers = DOUBLE_BUFFER ? 2 : 1;
  const int32_t head_group_idx = blockIdx.x;
  const int32_t batch_idx = blockIdx.y;
  const int32_t split_idx = blockIdx.z;
  const int32_t num_head_groups = head_num / HEAD_GROUP;

  if (batch_idx >= batch_size || head_group_idx >= num_head_groups) {
    return;
  }

  __shared__ float4 q_smem[HEAD_GROUP][kFastVec4PerHead];
  __shared__ int4 k_smem[kBuffers][kFastBlockSize][kFastFp8SmemVecStride];
  __shared__ int4 v_smem[kBuffers][kFastBlockSize][kFastFp8SmemVecStride];
  __shared__ float k_scale_smem[kBuffers][kFastBlockSize];
  __shared__ float v_scale_smem[kBuffers][kFastBlockSize];

  const int32_t head_base = head_group_idx * HEAD_GROUP;
  const int32_t kv_head = head_base / kv_mul;
  const int32_t seq_len = seq_lens[batch_idx];
  const int32_t num_kv_blocks = (seq_len + kFastBlockSize - 1) / kFastBlockSize;
  const int32_t blocks_per_split = (num_kv_blocks + split_k - 1) / split_k;
  const int32_t block_start = split_idx * blocks_per_split;
  int32_t block_end = block_start + blocks_per_split;
  if (block_end > num_kv_blocks) {
    block_end = num_kv_blocks;
  }
  const int32_t block_table_base = batch_idx * max_blocks_per_seq;
  const int64_t split_group_base =
      (((static_cast<int64_t>(batch_idx) * num_head_groups + head_group_idx) * split_k +
        split_idx) *
       HEAD_GROUP);

  load_query_vec4<HEAD_GROUP>(batch_idx, head_base, head_num, queries, q_smem);
  __syncthreads();

  if (seq_len <= 0 || block_start >= block_end) {
    store_empty_partial<HEAD_GROUP>(split_group_base, partial_out, partial_max, partial_sum);
    return;
  }

  const int32_t last_block_tokens = seq_len - (num_kv_blocks - 1) * kFastBlockSize;
  int32_t current_block = block_start;
  int32_t current_buffer = 0;

  load_fp8_kv_tile_vec16_cooperative<kFastBlockSize>(
      current_buffer, block_tables[block_table_base + current_block], kv_head, num_kv_heads,
      threadIdx.x, THREADS, key_pool, value_pool, key_scale_pool, value_scale_pool, k_smem,
      v_smem, k_scale_smem, v_scale_smem);
  __syncthreads();

  const int32_t warp_id = threadIdx.x / 32;
  const int32_t lane = threadIdx.x & 31;
  const bool compute_warp = warp_id < HEAD_GROUP;
  const bool loader_thread = DOUBLE_BUFFER && !compute_warp;
  const int32_t local_head = warp_id;
  constexpr int32_t kComputeThreads = HEAD_GROUP * 32;
  constexpr int32_t kLoaderThreads = THREADS - kComputeThreads;

  float q0 = 0.f;
  float q1 = 0.f;
  float acc0 = 0.f;
  float acc1 = 0.f;
  float m = -FLT_MAX;
  float l = 0.f;

  if (compute_warp) {
    const base::CudaBF16* q_head =
        reinterpret_cast<const base::CudaBF16*>(q_smem[local_head]);
    q0 = scalar_to_float(q_head[lane]);
    q1 = scalar_to_float(q_head[lane + 32]);
  }

  while (true) {
    const int32_t valid_tokens =
        (current_block == num_kv_blocks - 1) ? last_block_tokens : kFastBlockSize;
    const int32_t next_block = current_block + 1;
    const bool has_next_block = next_block < block_end;

    if (DOUBLE_BUFFER && has_next_block && loader_thread) {
      const int32_t next_buffer = current_buffer ^ 1;
      load_fp8_kv_tile_vec16_cooperative<kFastBlockSize>(
          next_buffer, block_tables[block_table_base + next_block], kv_head, num_kv_heads,
          threadIdx.x - kComputeThreads, kLoaderThreads, key_pool, value_pool, key_scale_pool,
          value_scale_pool, k_smem, v_smem, k_scale_smem, v_scale_smem);
    }

    if (compute_warp) {
      #pragma unroll
      for (int32_t token = 0; token < kFastBlockSize; ++token) {
        if (token >= valid_tokens) {
          break;
        }
        const int8_t* k_row = reinterpret_cast<const int8_t*>(k_smem[current_buffer][token]);
        const int8_t* v_row = reinterpret_cast<const int8_t*>(v_smem[current_buffer][token]);

        float score = q0 * fp8_e4m3_to_float(static_cast<uint8_t>(k_row[lane])) +
                      q1 * fp8_e4m3_to_float(static_cast<uint8_t>(k_row[lane + 32]));
        score = warp_sum(score);
        score *= scale * k_scale_smem[current_buffer][token];

        float alpha = 0.f;
        float p = 0.f;
        float m_new = m;
        if (lane == 0) {
          m_new = fmaxf(score, m);
          alpha = exp2f(m - m_new);
          p = exp2f(score - m_new);
          l = l * alpha + p;
          m = m_new;
        }
        m_new = __shfl_sync(0xffffffffu, m_new, 0);
        alpha = __shfl_sync(0xffffffffu, alpha, 0);
        p = __shfl_sync(0xffffffffu, p, 0);
        l = __shfl_sync(0xffffffffu, l, 0);
        m = m_new;

        const float pv_scale = p * v_scale_smem[current_buffer][token];
        acc0 = acc0 * alpha + fp8_e4m3_to_float(static_cast<uint8_t>(v_row[lane])) * pv_scale;
        acc1 = acc1 * alpha + fp8_e4m3_to_float(static_cast<uint8_t>(v_row[lane + 32])) * pv_scale;
      }
    }

    __syncthreads();

    ++current_block;
    if (current_block >= block_end) {
      break;
    }

    if (DOUBLE_BUFFER) {
      current_buffer ^= 1;
    } else {
      load_fp8_kv_tile_vec16_cooperative<kFastBlockSize>(
          current_buffer, block_tables[block_table_base + current_block], kv_head, num_kv_heads,
          threadIdx.x, THREADS, key_pool, value_pool, key_scale_pool, value_scale_pool, k_smem,
          v_smem, k_scale_smem, v_scale_smem);
      __syncthreads();
    }
  }

  if (compute_warp) {
    const int64_t meta_offset = split_group_base + local_head;
    const int64_t out_base = meta_offset * kFastHeadSize;
    partial_out[out_base + lane] = acc0;
    partial_out[out_base + lane + 32] = acc1;

    if (lane == 0) {
      partial_max[meta_offset] = m;
      partial_sum[meta_offset] = l;
    }
  }
}

template <int HEAD_GROUP>
__global__ void splitkv_paged_mha_fast_stage2_kernel(
    const float* __restrict__ partial_out,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    base::CudaBF16* __restrict__ output,
    int32_t batch_size,
    int32_t head_num,
    int32_t split_k) {
  const int32_t head_group_idx = blockIdx.x;
  const int32_t batch_idx = blockIdx.y;
  const int32_t num_head_groups = head_num / HEAD_GROUP;
  const int32_t head_base = head_group_idx * HEAD_GROUP;
  const int32_t warp_id = threadIdx.x / 32;
  const int32_t lane = threadIdx.x & 31;

  if (batch_idx >= batch_size || head_group_idx >= num_head_groups || warp_id >= HEAD_GROUP) {
    return;
  }

  const int32_t local_head = warp_id;
  const int32_t head = head_base + local_head;
  const int32_t dim = head_num * kFastHeadSize;
  const int64_t split_group_base =
      (((static_cast<int64_t>(batch_idx) * num_head_groups + head_group_idx) * split_k) *
       HEAD_GROUP) +
      local_head;

  float lane_part_max = -FLT_MAX;
  float lane_part_sum = 0.f;
  if (lane < split_k) {
    const int64_t meta_offset = split_group_base + static_cast<int64_t>(lane) * HEAD_GROUP;
    lane_part_max = partial_max[meta_offset];
    lane_part_sum = partial_sum[meta_offset];
  }

  float m = warp_max(lane_part_max);
  m = __shfl_sync(0xffffffffu, m, 0);
  float lane_denom = 0.f;
  if (lane < split_k && lane_part_sum > 0.f) {
    lane_denom = lane_part_sum * exp2f(lane_part_max - m);
  }
  float denom = warp_sum(lane_denom);
  denom = __shfl_sync(0xffffffffu, denom, 0);
  const float safe_denom = denom > 0.f ? denom : 1.f;

  float out0 = 0.f;
  float out1 = 0.f;
  const int64_t dim_base = static_cast<int64_t>(batch_idx) * dim + head * kFastHeadSize;
  #pragma unroll
  for (int32_t split = 0; split < kFastStage2MaxSplitK; ++split) {
    if (split >= split_k) {
      break;
    }
    const float part_sum = __shfl_sync(0xffffffffu, lane_part_sum, split);
    if (part_sum <= 0.f) {
      continue;
    }
    const float part_max = __shfl_sync(0xffffffffu, lane_part_max, split);
    const float alpha = exp2f(part_max - m);
    const int64_t out_base =
        (split_group_base + static_cast<int64_t>(split) * HEAD_GROUP) * kFastHeadSize;
    out0 += partial_out[out_base + lane] * alpha;
    out1 += partial_out[out_base + lane + 32] * alpha;
  }

  out0 = denom > 0.f ? out0 / safe_denom : 0.f;
  out1 = denom > 0.f ? out1 / safe_denom : 0.f;
  output[dim_base + lane] = float_to_scalar<base::CudaBF16>(out0);
  output[dim_base + lane + 32] = float_to_scalar<base::CudaBF16>(out1);
}

template <int HEAD_GROUP, int THREADS, bool DOUBLE_BUFFER>
void launch_fast_stage1(
    int32_t batch_size,
    int32_t head_num,
    int32_t kv_mul,
    int32_t max_blocks_per_seq,
    int32_t num_kv_heads,
    int32_t split_k,
    const tensor::Tensor& queries,
    const tensor::Tensor& key_pool,
    const tensor::Tensor& value_pool,
    const tensor::Tensor& block_tables,
    const tensor::Tensor& seq_lens,
    const tensor::Tensor& partial_out,
    const tensor::Tensor& partial_max,
    const tensor::Tensor& partial_sum,
    cudaStream_t stream) {
  dim3 grid(head_num / HEAD_GROUP, batch_size, split_k);
  splitkv_paged_mha_fast_stage1_kernel<HEAD_GROUP, THREADS, DOUBLE_BUFFER>
      <<<grid, THREADS, 0, stream>>>(
          reinterpret_cast<const base::CudaBF16*>(queries.ptr<uint16_t>()),
          reinterpret_cast<const base::CudaBF16*>(key_pool.ptr<uint16_t>()),
          reinterpret_cast<const base::CudaBF16*>(value_pool.ptr<uint16_t>()),
          block_tables.ptr<int32_t>(),
          seq_lens.ptr<int32_t>(),
          const_cast<float*>(partial_out.ptr<float>()),
          const_cast<float*>(partial_max.ptr<float>()),
          const_cast<float*>(partial_sum.ptr<float>()),
          batch_size,
          head_num,
          kv_mul,
          max_blocks_per_seq,
          num_kv_heads,
          split_k,
          (1.f / sqrtf(static_cast<float>(kFastHeadSize))) * kLog2e);
}

template <int HEAD_GROUP, int THREADS, bool DOUBLE_BUFFER>
void launch_fp8_stage1(
    int32_t batch_size,
    int32_t head_num,
    int32_t kv_mul,
    int32_t max_blocks_per_seq,
    int32_t num_kv_heads,
    int32_t split_k,
    const tensor::Tensor& queries,
    const tensor::Tensor& key_pool,
    const tensor::Tensor& value_pool,
    const tensor::Tensor& key_scale_pool,
    const tensor::Tensor& value_scale_pool,
    const tensor::Tensor& block_tables,
    const tensor::Tensor& seq_lens,
    const tensor::Tensor& partial_out,
    const tensor::Tensor& partial_max,
    const tensor::Tensor& partial_sum,
    cudaStream_t stream) {
  dim3 grid(head_num / HEAD_GROUP, batch_size, split_k);
  splitkv_paged_mha_fp8_stage1_kernel<HEAD_GROUP, THREADS, DOUBLE_BUFFER>
      <<<grid, THREADS, 0, stream>>>(
          reinterpret_cast<const base::CudaBF16*>(queries.ptr<uint16_t>()),
          key_pool.ptr<int8_t>(),
          value_pool.ptr<int8_t>(),
          key_scale_pool.ptr<float>(),
          value_scale_pool.ptr<float>(),
          block_tables.ptr<int32_t>(),
          seq_lens.ptr<int32_t>(),
          const_cast<float*>(partial_out.ptr<float>()),
          const_cast<float*>(partial_max.ptr<float>()),
          const_cast<float*>(partial_sum.ptr<float>()),
          batch_size,
          head_num,
          kv_mul,
          max_blocks_per_seq,
          num_kv_heads,
          split_k,
          (1.f / sqrtf(static_cast<float>(kFastHeadSize))) * kLog2e);
}

template <int HEAD_GROUP>
void launch_fast_stage2(
    int32_t batch_size,
    int32_t head_num,
    int32_t split_k,
    const tensor::Tensor& partial_out,
    const tensor::Tensor& partial_max,
    const tensor::Tensor& partial_sum,
    const tensor::Tensor& outputs,
    cudaStream_t stream) {
  dim3 grid(head_num / HEAD_GROUP, batch_size);
  splitkv_paged_mha_fast_stage2_kernel<HEAD_GROUP><<<grid, kFastStage2Threads, 0, stream>>>(
      partial_out.ptr<float>(),
      partial_max.ptr<float>(),
      partial_sum.ptr<float>(),
      reinterpret_cast<base::CudaBF16*>(const_cast<uint16_t*>(outputs.ptr<uint16_t>())),
      batch_size,
      head_num,
      split_k);
}

}  // namespace

bool splitkv_batched_paged_mha_fast_decode_cu(
    int32_t batch_size,
    int32_t head_num,
    int32_t head_size,
    int32_t kv_mul,
    const tensor::Tensor& queries,
    const tensor::Tensor& outputs,
    const tensor::Tensor& key_pool,
    const tensor::Tensor& value_pool,
    const tensor::Tensor& block_tables,
    const tensor::Tensor& seq_lens,
    int32_t max_blocks_per_seq,
    int32_t block_size,
    int32_t num_kv_heads,
    const tensor::Tensor& partial_out,
    const tensor::Tensor& partial_max,
    const tensor::Tensor& partial_sum,
    base::DeviceType device_type,
    CudaConfig* config) {
  CHECK_NE(config, nullptr);

  const FastDecodeLaunchConfig cfg = choose_fast_decode_config(
      head_num, head_size, kv_mul, max_blocks_per_seq, block_size, num_kv_heads, queries, outputs,
      key_pool, value_pool, block_tables, seq_lens, partial_out, partial_max, partial_sum,
      device_type);
  if (!cfg.use_fast_path) {
    return false;
  }

  cudaStream_t stream = config->stream;
  if (cfg.head_group == 4 && cfg.threads == 128 && !cfg.use_double_buffer) {
    launch_fast_stage1<4, 128, false>(
        batch_size, head_num, kv_mul, max_blocks_per_seq, num_kv_heads, cfg.split_k, queries,
        key_pool, value_pool, block_tables, seq_lens, partial_out, partial_max, partial_sum,
        stream);
    launch_fast_stage2<4>(
        batch_size, head_num, cfg.split_k, partial_out, partial_max, partial_sum, outputs,
        stream);
    return true;
  }

  if (cfg.head_group == 4 && cfg.threads == 256 && cfg.use_double_buffer) {
    launch_fast_stage1<4, 256, true>(
        batch_size, head_num, kv_mul, max_blocks_per_seq, num_kv_heads, cfg.split_k, queries,
        key_pool, value_pool, block_tables, seq_lens, partial_out, partial_max, partial_sum,
        stream);
    launch_fast_stage2<4>(
        batch_size, head_num, cfg.split_k, partial_out, partial_max, partial_sum, outputs,
        stream);
    return true;
  }

  if (cfg.head_group == 2 && cfg.threads == 256 && cfg.use_double_buffer) {
    launch_fast_stage1<2, 256, true>(
        batch_size, head_num, kv_mul, max_blocks_per_seq, num_kv_heads, cfg.split_k, queries,
        key_pool, value_pool, block_tables, seq_lens, partial_out, partial_max, partial_sum,
        stream);
    launch_fast_stage2<2>(
        batch_size, head_num, cfg.split_k, partial_out, partial_max, partial_sum, outputs,
        stream);
    return true;
  }

  return false;
}

bool splitkv_batched_paged_mha_fp8_decode_cu(
    int32_t batch_size,
    int32_t head_num,
    int32_t head_size,
    int32_t kv_mul,
    const tensor::Tensor& queries,
    const tensor::Tensor& outputs,
    const tensor::Tensor& key_pool,
    const tensor::Tensor& value_pool,
    const tensor::Tensor& key_scale_pool,
    const tensor::Tensor& value_scale_pool,
    const tensor::Tensor& block_tables,
    const tensor::Tensor& seq_lens,
    int32_t max_blocks_per_seq,
    int32_t block_size,
    int32_t num_kv_heads,
    const tensor::Tensor& partial_out,
    const tensor::Tensor& partial_max,
    const tensor::Tensor& partial_sum,
    base::DeviceType device_type,
    CudaConfig* config) {
  CHECK_NE(config, nullptr);

  const FastDecodeLaunchConfig cfg = choose_fp8_decode_config(
      head_num, head_size, kv_mul, max_blocks_per_seq, block_size, num_kv_heads, queries, outputs,
      key_pool, value_pool, key_scale_pool, value_scale_pool, block_tables, seq_lens, partial_out,
      partial_max, partial_sum, device_type);
  if (!cfg.use_fast_path) {
    return false;
  }

  cudaStream_t stream = config->stream;
  if (cfg.head_group == 4 && cfg.threads == 128 && !cfg.use_double_buffer) {
    launch_fp8_stage1<4, 128, false>(
        batch_size, head_num, kv_mul, max_blocks_per_seq, num_kv_heads, cfg.split_k, queries,
        key_pool, value_pool, key_scale_pool, value_scale_pool, block_tables, seq_lens,
        partial_out, partial_max, partial_sum, stream);
    launch_fast_stage2<4>(
        batch_size, head_num, cfg.split_k, partial_out, partial_max, partial_sum, outputs,
        stream);
    return true;
  }

  if (cfg.head_group == 4 && cfg.threads == 256 && cfg.use_double_buffer) {
    launch_fp8_stage1<4, 256, true>(
        batch_size, head_num, kv_mul, max_blocks_per_seq, num_kv_heads, cfg.split_k, queries,
        key_pool, value_pool, key_scale_pool, value_scale_pool, block_tables, seq_lens,
        partial_out, partial_max, partial_sum, stream);
    launch_fast_stage2<4>(
        batch_size, head_num, cfg.split_k, partial_out, partial_max, partial_sum, outputs,
        stream);
    return true;
  }

  if (cfg.head_group == 2 && cfg.threads == 256 && cfg.use_double_buffer) {
    launch_fp8_stage1<2, 256, true>(
        batch_size, head_num, kv_mul, max_blocks_per_seq, num_kv_heads, cfg.split_k, queries,
        key_pool, value_pool, key_scale_pool, value_scale_pool, block_tables, seq_lens,
        partial_out, partial_max, partial_sum, stream);
    launch_fast_stage2<2>(
        batch_size, head_num, cfg.split_k, partial_out, partial_max, partial_sum, outputs,
        stream);
    return true;
  }

  if (cfg.head_group == 1 && cfg.threads == 128 && !cfg.use_double_buffer) {
    launch_fp8_stage1<1, 128, false>(
        batch_size, head_num, kv_mul, max_blocks_per_seq, num_kv_heads, cfg.split_k, queries,
        key_pool, value_pool, key_scale_pool, value_scale_pool, block_tables, seq_lens,
        partial_out, partial_max, partial_sum, stream);
    launch_fast_stage2<1>(
        batch_size, head_num, cfg.split_k, partial_out, partial_max, partial_sum, outputs,
        stream);
    return true;
  }

  if (cfg.head_group == 1 && cfg.threads == 256 && cfg.use_double_buffer) {
    launch_fp8_stage1<1, 256, true>(
        batch_size, head_num, kv_mul, max_blocks_per_seq, num_kv_heads, cfg.split_k, queries,
        key_pool, value_pool, key_scale_pool, value_scale_pool, block_tables, seq_lens,
        partial_out, partial_max, partial_sum, stream);
    launch_fast_stage2<1>(
        batch_size, head_num, cfg.split_k, partial_out, partial_max, partial_sum, outputs,
        stream);
    return true;
  }

  return false;
}

}  // namespace kernel
