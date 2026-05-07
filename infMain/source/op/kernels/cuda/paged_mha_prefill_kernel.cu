// Chunked prefill paged MHA kernel implementation
#include "op/kernels/cuda/paged_mha_prefill_kernel.cuh"

#include <cfloat>
#include <cuda_runtime.h>

#include "cuda_type_utils.cuh"

namespace kernel {

namespace {

constexpr int32_t kWarpSize = 32;
constexpr int32_t kMaxPrefillThreads = 256;
constexpr int32_t kMaxPrefillWarps = kMaxPrefillThreads / kWarpSize;
constexpr int32_t kMaxPrefillSplitKPartitions = 8;

template <typename T>
__device__ inline T warp_sum(T value) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

template <int THREADS>
__device__ inline float reduce_sum_to_thread0(float value, float* warp_sums) {
  static_assert(THREADS % kWarpSize == 0, "THREADS must be warp-aligned.");
  constexpr int32_t kNumWarps = THREADS / kWarpSize;
  const int32_t lane = threadIdx.x & (kWarpSize - 1);
  const int32_t warp = threadIdx.x / kWarpSize;

  value = warp_sum(value);
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();

  if (warp == 0) {
    value = lane < kNumWarps ? warp_sums[lane] : 0.f;
    value = warp_sum(value);
  }
  return value;
}

inline int32_t choose_prefill_threads(int32_t head_size) {
  if (head_size <= 32) {
    return 32;
  }
  if (head_size <= 64) {
    return 64;
  }
  if (head_size <= 128) {
    return 128;
  }
  return 256;
}

inline int32_t choose_prefill_num_partitions(int32_t max_prefix_blocks) {
  if (max_prefix_blocks <= 8) {
    return 1;
  }
  const int32_t preferred = (max_prefix_blocks + 7) / 8;
  if (preferred <= 2) {
    return 2;
  }
  if (preferred <= 4) {
    return 4;
  }
  return kMaxPrefillSplitKPartitions;
}

inline int32_t choose_chunk_tile_tokens(int32_t head_size) {
  if (head_size <= 128) {
    return 16;
  }
  if (head_size <= 256) {
    return 8;
  }
  return 4;
}

template <int TILE_TOKENS>
inline int32_t chunk_tile_smem_size(int32_t head_size, int32_t num_partitions) {
  return (
      2 * head_size +
      2 * TILE_TOKENS * head_size +
      2 * TILE_TOKENS +
      3 +
      num_partitions) * static_cast<int32_t>(sizeof(float));
}

inline int32_t prefix_splitkv_smem_size(int32_t head_size) {
  return (2 * head_size + 4) * static_cast<int32_t>(sizeof(float));
}

template <int THREADS>
__device__ inline void store_empty_partial(
    float* partial_out,
    float* partial_max,
    float* partial_sum,
    int32_t row_idx,
    int32_t head_idx,
    int32_t head_num,
    int32_t num_partitions,
    int32_t part_idx,
    int32_t head_size) {
  const int64_t meta_idx =
      (static_cast<int64_t>(row_idx) * head_num + head_idx) * num_partitions + part_idx;
  float* out_base = partial_out + meta_idx * head_size;
  for (int32_t i = threadIdx.x; i < head_size; i += THREADS) {
    out_base[i] = 0.f;
  }
  if (threadIdx.x == 0) {
    partial_max[meta_idx] = -FLT_MAX;
    partial_sum[meta_idx] = 0.f;
  }
}

template <typename T, int THREADS>
__global__ void splitkv_prefix_prefill_attention_p1_kernel(
    const T* __restrict__ queries,             // [batch_size, head_num * head_size]
    const T* __restrict__ key_pool,
    const T* __restrict__ value_pool,
    const int32_t* __restrict__ block_tables,  // [num_prefill_requests, max_blocks_per_seq]
    const int32_t* __restrict__ request_indices,
    const int32_t* __restrict__ base_context_lens,
    float* __restrict__ partial_out,           // [batch_size, head_num, num_partitions, head_size]
    float* __restrict__ partial_max,           // [batch_size, head_num, num_partitions]
    float* __restrict__ partial_sum,           // [batch_size, head_num, num_partitions]
    int32_t max_blocks_per_seq,
    int32_t block_size,
    int32_t num_kv_heads,
    int32_t head_size,
    int32_t kv_mul,
    int32_t num_partitions) {
  const int32_t head_idx = blockIdx.x;
  const int32_t row_idx = blockIdx.y;
  const int32_t part_idx = blockIdx.z;
  const int32_t head_num = num_kv_heads * kv_mul;
  const int32_t kv_head = head_idx / kv_mul;
  const int32_t request_idx = request_indices[row_idx];
  const int32_t base_context_len = base_context_lens[row_idx];
  const int32_t dim = head_num * head_size;
  const int32_t kv_dim = num_kv_heads * head_size;

  const T* my_query =
      queries + static_cast<int64_t>(row_idx) * dim + static_cast<int64_t>(head_idx) * head_size;
  const int32_t* my_block_table =
      block_tables + static_cast<int64_t>(request_idx) * max_blocks_per_seq;

  extern __shared__ float shared_mem[];
  float* s_query = shared_mem;
  float* s_output = s_query + head_size;
  float* s_m = s_output + head_size;
  float* s_sum = s_m + 1;
  float* s_alpha = s_sum + 1;
  float* s_p = s_alpha + 1;

  __shared__ float warp_sums[kMaxPrefillWarps];

  const float scale = 1.f / sqrtf(static_cast<float>(head_size));

  if (threadIdx.x == 0) {
    s_m[0] = -FLT_MAX;
    s_sum[0] = 0.f;
  }
  for (int32_t idx = threadIdx.x; idx < head_size; idx += blockDim.x) {
    s_query[idx] = scalar_to_float(my_query[idx]);
    s_output[idx] = 0.f;
  }
  __syncthreads();

  const int64_t block_stride = static_cast<int64_t>(block_size) * kv_dim;
  const int64_t token_stride = kv_dim;
  const int64_t kv_head_offset = static_cast<int64_t>(kv_head) * head_size;
  const int32_t num_prefix_blocks = (base_context_len + block_size - 1) / block_size;
  const int32_t blocks_per_part = (num_prefix_blocks + num_partitions - 1) / num_partitions;
  const int32_t kv_block_start = part_idx * blocks_per_part;
  const int32_t kv_block_end = min(kv_block_start + blocks_per_part, num_prefix_blocks);

  if (num_prefix_blocks <= 0 || kv_block_start >= kv_block_end) {
    store_empty_partial<THREADS>(
        partial_out, partial_max, partial_sum, row_idx, head_idx, head_num,
        num_partitions, part_idx, head_size);
    return;
  }

  const int32_t tokens_in_last_prefix_block =
      base_context_len - (num_prefix_blocks - 1) * block_size;
  for (int32_t block_idx = kv_block_start; block_idx < kv_block_end; ++block_idx) {
    const int32_t physical_block_id = my_block_table[block_idx];
    const int32_t valid_tokens =
        block_idx == num_prefix_blocks - 1 ? tokens_in_last_prefix_block : block_size;
    const int64_t block_offset = static_cast<int64_t>(physical_block_id) * block_stride;
    const T* key_base = key_pool + block_offset + kv_head_offset;
    const T* value_base = value_pool + block_offset + kv_head_offset;

    for (int32_t token_offset = 0; token_offset < valid_tokens; ++token_offset) {
      const int64_t token_offset_base = static_cast<int64_t>(token_offset) * token_stride;
      const T* key_ptr = key_base + token_offset_base;
      const T* value_ptr = value_base + token_offset_base;

      float score = 0.f;
      for (int32_t i = threadIdx.x; i < head_size; i += blockDim.x) {
        score += scalar_to_float(key_ptr[i]) * s_query[i];
      }
      score *= scale;
      score = reduce_sum_to_thread0<THREADS>(score, warp_sums);

      if (threadIdx.x == 0) {
        const float m_new = fmaxf(score, s_m[0]);
        s_alpha[0] = expf(s_m[0] - m_new);
        s_p[0] = expf(score - m_new);
        s_sum[0] = s_sum[0] * s_alpha[0] + s_p[0];
        s_m[0] = m_new;
      }
      __syncthreads();

      for (int32_t i = threadIdx.x; i < head_size; i += blockDim.x) {
        s_output[i] = s_output[i] * s_alpha[0] + scalar_to_float(value_ptr[i]) * s_p[0];
      }
      __syncthreads();
    }
  }

  const int64_t meta_idx =
      (static_cast<int64_t>(row_idx) * head_num + head_idx) * num_partitions + part_idx;
  float* out_base = partial_out + meta_idx * head_size;
  for (int32_t i = threadIdx.x; i < head_size; i += blockDim.x) {
    out_base[i] = s_output[i];
  }
  if (threadIdx.x == 0) {
    partial_max[meta_idx] = s_m[0];
    partial_sum[meta_idx] = s_sum[0];
  }
}

template <typename T, int THREADS>
__global__ void splitkv_prefix_prefill_attention_p1_fp8_kernel(
    const T* __restrict__ queries,             // [batch_size, head_num * head_size]
    const int8_t* __restrict__ key_pool,
    const int8_t* __restrict__ value_pool,
    const float* __restrict__ key_scale_pool,
    const float* __restrict__ value_scale_pool,
    const int32_t* __restrict__ block_tables,  // [num_prefill_requests, max_blocks_per_seq]
    const int32_t* __restrict__ request_indices,
    const int32_t* __restrict__ base_context_lens,
    float* __restrict__ partial_out,           // [batch_size, head_num, num_partitions, head_size]
    float* __restrict__ partial_max,           // [batch_size, head_num, num_partitions]
    float* __restrict__ partial_sum,           // [batch_size, head_num, num_partitions]
    int32_t max_blocks_per_seq,
    int32_t block_size,
    int32_t num_kv_heads,
    int32_t head_size,
    int32_t kv_mul,
    int32_t num_partitions) {
  const int32_t head_idx = blockIdx.x;
  const int32_t row_idx = blockIdx.y;
  const int32_t part_idx = blockIdx.z;
  const int32_t head_num = num_kv_heads * kv_mul;
  const int32_t kv_head = head_idx / kv_mul;
  const int32_t request_idx = request_indices[row_idx];
  const int32_t base_context_len = base_context_lens[row_idx];
  const int32_t dim = head_num * head_size;
  const int32_t kv_dim = num_kv_heads * head_size;

  const T* my_query =
      queries + static_cast<int64_t>(row_idx) * dim + static_cast<int64_t>(head_idx) * head_size;
  const int32_t* my_block_table =
      block_tables + static_cast<int64_t>(request_idx) * max_blocks_per_seq;

  extern __shared__ float shared_mem[];
  float* s_query = shared_mem;
  float* s_output = s_query + head_size;
  float* s_m = s_output + head_size;
  float* s_sum = s_m + 1;
  float* s_alpha = s_sum + 1;
  float* s_p = s_alpha + 1;

  __shared__ float warp_sums[kMaxPrefillWarps];

  const float scale = 1.f / sqrtf(static_cast<float>(head_size));

  if (threadIdx.x == 0) {
    s_m[0] = -FLT_MAX;
    s_sum[0] = 0.f;
  }
  for (int32_t idx = threadIdx.x; idx < head_size; idx += blockDim.x) {
    s_query[idx] = scalar_to_float(my_query[idx]);
    s_output[idx] = 0.f;
  }
  __syncthreads();

  const int64_t block_stride = static_cast<int64_t>(block_size) * kv_dim;
  const int64_t token_stride = kv_dim;
  const int64_t kv_head_offset = static_cast<int64_t>(kv_head) * head_size;
  const int32_t num_prefix_blocks = (base_context_len + block_size - 1) / block_size;
  const int32_t blocks_per_part = (num_prefix_blocks + num_partitions - 1) / num_partitions;
  const int32_t kv_block_start = part_idx * blocks_per_part;
  const int32_t kv_block_end = min(kv_block_start + blocks_per_part, num_prefix_blocks);

  if (num_prefix_blocks <= 0 || kv_block_start >= kv_block_end) {
    store_empty_partial<THREADS>(
        partial_out, partial_max, partial_sum, row_idx, head_idx, head_num,
        num_partitions, part_idx, head_size);
    return;
  }

  const int32_t tokens_in_last_prefix_block =
      base_context_len - (num_prefix_blocks - 1) * block_size;
  for (int32_t block_idx = kv_block_start; block_idx < kv_block_end; ++block_idx) {
    const int32_t physical_block_id = my_block_table[block_idx];
    const int32_t valid_tokens =
        block_idx == num_prefix_blocks - 1 ? tokens_in_last_prefix_block : block_size;
    const int64_t block_offset = static_cast<int64_t>(physical_block_id) * block_stride;
    const int64_t scale_offset =
        static_cast<int64_t>(physical_block_id) * (block_size * num_kv_heads) +
        static_cast<int64_t>(kv_head);
    const int8_t* key_base = key_pool + block_offset + kv_head_offset;
    const int8_t* value_base = value_pool + block_offset + kv_head_offset;

    for (int32_t token_offset = 0; token_offset < valid_tokens; ++token_offset) {
      const int64_t token_offset_base = static_cast<int64_t>(token_offset) * token_stride;
      const int64_t token_scale_offset =
          scale_offset + static_cast<int64_t>(token_offset) * num_kv_heads;
      const int8_t* key_ptr = key_base + token_offset_base;
      const int8_t* value_ptr = value_base + token_offset_base;
      const float k_scale = key_scale_pool[token_scale_offset];
      const float v_scale = value_scale_pool[token_scale_offset];

      float score = 0.f;
      for (int32_t i = threadIdx.x; i < head_size; i += blockDim.x) {
        score += scalar_to_float(key_ptr[i]) * k_scale * s_query[i];
      }
      score *= scale;
      score = reduce_sum_to_thread0<THREADS>(score, warp_sums);

      if (threadIdx.x == 0) {
        const float m_new = fmaxf(score, s_m[0]);
        s_alpha[0] = expf(s_m[0] - m_new);
        s_p[0] = expf(score - m_new);
        s_sum[0] = s_sum[0] * s_alpha[0] + s_p[0];
        s_m[0] = m_new;
      }
      __syncthreads();

      for (int32_t i = threadIdx.x; i < head_size; i += blockDim.x) {
        s_output[i] = s_output[i] * s_alpha[0] +
                      scalar_to_float(value_ptr[i]) * v_scale * s_p[0];
      }
      __syncthreads();
    }
  }

  const int64_t meta_idx =
      (static_cast<int64_t>(row_idx) * head_num + head_idx) * num_partitions + part_idx;
  float* out_base = partial_out + meta_idx * head_size;
  for (int32_t i = threadIdx.x; i < head_size; i += blockDim.x) {
    out_base[i] = s_output[i];
  }
  if (threadIdx.x == 0) {
    partial_max[meta_idx] = s_m[0];
    partial_sum[meta_idx] = s_sum[0];
  }
}

template <typename T, int THREADS, int TILE_TOKENS>
__global__ void merge_prefix_splitkv_with_chunk_tile_kernel(
    const T* __restrict__ queries,          // [batch_size, head_num * head_size]
    const T* __restrict__ chunk_keys,       // [batch_size, kv_dim]
    const T* __restrict__ chunk_values,     // [batch_size, kv_dim]
    T* __restrict__ outputs,                // [batch_size, head_num * head_size]
    const int32_t* __restrict__ chunk_row_starts,
    const int32_t* __restrict__ local_token_offsets,
    const float* __restrict__ partial_out,  // [batch_size, head_num, num_partitions, head_size]
    const float* __restrict__ partial_max,  // [batch_size, head_num, num_partitions]
    const float* __restrict__ partial_sum,  // [batch_size, head_num, num_partitions]
    int32_t num_kv_heads,
    int32_t head_size,
    int32_t kv_mul,
    int32_t num_partitions) {
  const int32_t head_idx = blockIdx.x;
  const int32_t row_idx = blockIdx.y;
  const int32_t head_num = num_kv_heads * kv_mul;
  const int32_t kv_head = head_idx / kv_mul;
  const int32_t dim = head_num * head_size;
  const int32_t kv_dim = num_kv_heads * head_size;
  const int64_t kv_head_offset = static_cast<int64_t>(kv_head) * head_size;
  const int64_t part_base =
      (static_cast<int64_t>(row_idx) * head_num + head_idx) * num_partitions;

  const T* query_base =
      queries + static_cast<int64_t>(row_idx) * dim + static_cast<int64_t>(head_idx) * head_size;
  T* output_base =
      outputs + static_cast<int64_t>(row_idx) * dim + static_cast<int64_t>(head_idx) * head_size;

  extern __shared__ float shared_mem[];
  float* s_query = shared_mem;
  float* s_output = s_query + head_size;
  float* s_key_tile = s_output + head_size;
  float* s_value_tile = s_key_tile + TILE_TOKENS * head_size;
  float* s_scores = s_value_tile + TILE_TOKENS * head_size;
  float* s_probs = s_scores + TILE_TOKENS;
  float* s_m = s_probs + TILE_TOKENS;
  float* s_sum = s_m + 1;
  float* s_alpha = s_sum + 1;
  float* s_part_scales = s_alpha + 1;

  __shared__ float warp_sums[kMaxPrefillWarps];

  for (int32_t idx = threadIdx.x; idx < head_size; idx += blockDim.x) {
    s_query[idx] = scalar_to_float(query_base[idx]);
  }

  if (threadIdx.x == 0) {
    float global_max = -FLT_MAX;
    for (int32_t p = 0; p < num_partitions; ++p) {
      global_max = fmaxf(global_max, partial_max[part_base + p]);
    }

    if (global_max == -FLT_MAX) {
      s_m[0] = -FLT_MAX;
      s_sum[0] = 0.f;
      for (int32_t p = 0; p < num_partitions; ++p) {
        s_part_scales[p] = 0.f;
      }
    } else {
      float global_sum = 0.f;
      for (int32_t p = 0; p < num_partitions; ++p) {
        const float pm = partial_max[part_base + p];
        const float part_scale = pm == -FLT_MAX ? 0.f : expf(pm - global_max);
        s_part_scales[p] = part_scale;
        global_sum += partial_sum[part_base + p] * part_scale;
      }
      s_m[0] = global_max;
      s_sum[0] = global_sum;
    }
  }
  __syncthreads();

  for (int32_t idx = threadIdx.x; idx < head_size; idx += blockDim.x) {
    float acc = 0.f;
    for (int32_t p = 0; p < num_partitions; ++p) {
      acc += partial_out[(part_base + p) * head_size + idx] * s_part_scales[p];
    }
    s_output[idx] = acc;
  }
  __syncthreads();

  const int32_t chunk_row_start = chunk_row_starts[row_idx];
  const int32_t total_chunk_tokens = local_token_offsets[row_idx] + 1;
  const T* chunk_key_base =
      chunk_keys + static_cast<int64_t>(chunk_row_start) * kv_dim + kv_head_offset;
  const T* chunk_value_base =
      chunk_values + static_cast<int64_t>(chunk_row_start) * kv_dim + kv_head_offset;
  const float scale = 1.f / sqrtf(static_cast<float>(head_size));

  for (int32_t tile_start = 0; tile_start < total_chunk_tokens; tile_start += TILE_TOKENS) {
    const int32_t valid_tokens = min(TILE_TOKENS, total_chunk_tokens - tile_start);
    const int32_t tile_elems = valid_tokens * head_size;

    for (int32_t idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
      const int32_t token = idx / head_size;
      const int32_t dim_idx = idx % head_size;
      const int64_t token_base = static_cast<int64_t>(tile_start + token) * kv_dim + dim_idx;
      s_key_tile[idx] = scalar_to_float(chunk_key_base[token_base]);
      s_value_tile[idx] = scalar_to_float(chunk_value_base[token_base]);
    }
    __syncthreads();

    for (int32_t token = 0; token < valid_tokens; ++token) {
      const float* key_ptr = s_key_tile + token * head_size;
      float score = 0.f;
      for (int32_t i = threadIdx.x; i < head_size; i += blockDim.x) {
        score += key_ptr[i] * s_query[i];
      }
      score *= scale;
      score = reduce_sum_to_thread0<THREADS>(score, warp_sums);
      if (threadIdx.x == 0) {
        s_scores[token] = score;
      }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      float local_max = -FLT_MAX;
      for (int32_t token = 0; token < valid_tokens; ++token) {
        local_max = fmaxf(local_max, s_scores[token]);
      }
      const float m_new = fmaxf(s_m[0], local_max);
      s_alpha[0] = expf(s_m[0] - m_new);

      float tile_sum = 0.f;
      for (int32_t token = 0; token < valid_tokens; ++token) {
        const float prob = expf(s_scores[token] - m_new);
        s_probs[token] = prob;
        tile_sum += prob;
      }
      s_sum[0] = s_sum[0] * s_alpha[0] + tile_sum;
      s_m[0] = m_new;
    }
    __syncthreads();

    for (int32_t i = threadIdx.x; i < head_size; i += blockDim.x) {
      float value_acc = 0.f;
      for (int32_t token = 0; token < valid_tokens; ++token) {
        value_acc += s_probs[token] * s_value_tile[token * head_size + i];
      }
      s_output[i] = s_output[i] * s_alpha[0] + value_acc;
    }
    __syncthreads();
  }

  for (int32_t idx = threadIdx.x; idx < head_size; idx += blockDim.x) {
    output_base[idx] = s_sum[0] > 0.f
                           ? float_to_scalar<T>(s_output[idx] / s_sum[0])
                           : float_to_scalar<T>(0.f);
  }
}

template <typename T, int THREADS>
void launch_hybrid_prefill_attention(
    dim3 prefix_grid,
    dim3 merge_grid,
    int32_t head_size,
    int32_t chunk_tile_tokens,
    cudaStream_t stream,
    const T* queries,
    const T* chunk_keys,
    const T* chunk_values,
    T* outputs,
    const T* key_pool,
    const T* value_pool,
    const int32_t* block_tables,
    const int32_t* request_indices,
    const int32_t* base_context_lens,
    const int32_t* chunk_row_starts,
    const int32_t* local_token_offsets,
    float* partial_out,
    float* partial_max,
    float* partial_sum,
    int32_t max_blocks_per_seq,
    int32_t block_size,
    int32_t num_kv_heads,
    int32_t kv_mul,
    int32_t num_partitions) {
  const int32_t prefix_smem_size = prefix_splitkv_smem_size(head_size);
  splitkv_prefix_prefill_attention_p1_kernel<T, THREADS>
      <<<prefix_grid, THREADS, prefix_smem_size, stream>>>(
          queries,
          key_pool,
          value_pool,
          block_tables,
          request_indices,
          base_context_lens,
          partial_out,
          partial_max,
          partial_sum,
          max_blocks_per_seq,
          block_size,
          num_kv_heads,
          head_size,
          kv_mul,
          num_partitions);
  const cudaError_t prefix_err = cudaPeekAtLastError();
  CHECK_EQ(prefix_err, cudaSuccess) << "splitkv_prefix_prefill_attention_p1_kernel launch failed: "
                                    << cudaGetErrorString(prefix_err);

  switch (chunk_tile_tokens) {
    case 16:
      merge_prefix_splitkv_with_chunk_tile_kernel<T, THREADS, 16>
          <<<merge_grid, THREADS, chunk_tile_smem_size<16>(head_size, num_partitions), stream>>>(
              queries,
              chunk_keys,
              chunk_values,
              outputs,
              chunk_row_starts,
              local_token_offsets,
              partial_out,
              partial_max,
              partial_sum,
              num_kv_heads,
              head_size,
              kv_mul,
              num_partitions);
      break;
    case 8:
      merge_prefix_splitkv_with_chunk_tile_kernel<T, THREADS, 8>
          <<<merge_grid, THREADS, chunk_tile_smem_size<8>(head_size, num_partitions), stream>>>(
              queries,
              chunk_keys,
              chunk_values,
              outputs,
              chunk_row_starts,
              local_token_offsets,
              partial_out,
              partial_max,
              partial_sum,
              num_kv_heads,
              head_size,
              kv_mul,
              num_partitions);
      break;
    default:
      merge_prefix_splitkv_with_chunk_tile_kernel<T, THREADS, 4>
          <<<merge_grid, THREADS, chunk_tile_smem_size<4>(head_size, num_partitions), stream>>>(
              queries,
              chunk_keys,
              chunk_values,
              outputs,
              chunk_row_starts,
              local_token_offsets,
              partial_out,
              partial_max,
              partial_sum,
              num_kv_heads,
              head_size,
              kv_mul,
              num_partitions);
      break;
  }
  const cudaError_t merge_err = cudaPeekAtLastError();
  CHECK_EQ(merge_err, cudaSuccess)
      << "merge_prefix_splitkv_with_chunk_tile_kernel launch failed: "
      << cudaGetErrorString(merge_err);
}

template <typename T, int THREADS>
void launch_hybrid_prefill_attention_fp8(
    dim3 prefix_grid,
    dim3 merge_grid,
    int32_t head_size,
    int32_t chunk_tile_tokens,
    cudaStream_t stream,
    const T* queries,
    const T* chunk_keys,
    const T* chunk_values,
    T* outputs,
    const int8_t* key_pool,
    const int8_t* value_pool,
    const float* key_scale_pool,
    const float* value_scale_pool,
    const int32_t* block_tables,
    const int32_t* request_indices,
    const int32_t* base_context_lens,
    const int32_t* chunk_row_starts,
    const int32_t* local_token_offsets,
    float* partial_out,
    float* partial_max,
    float* partial_sum,
    int32_t max_blocks_per_seq,
    int32_t block_size,
    int32_t num_kv_heads,
    int32_t kv_mul,
    int32_t num_partitions) {
  const int32_t prefix_smem_size = prefix_splitkv_smem_size(head_size);
  splitkv_prefix_prefill_attention_p1_fp8_kernel<T, THREADS>
      <<<prefix_grid, THREADS, prefix_smem_size, stream>>>(
          queries,
          key_pool,
          value_pool,
          key_scale_pool,
          value_scale_pool,
          block_tables,
          request_indices,
          base_context_lens,
          partial_out,
          partial_max,
          partial_sum,
          max_blocks_per_seq,
          block_size,
          num_kv_heads,
          head_size,
          kv_mul,
          num_partitions);
  const cudaError_t prefix_err = cudaPeekAtLastError();
  CHECK_EQ(prefix_err, cudaSuccess)
      << "splitkv_prefix_prefill_attention_p1_fp8_kernel launch failed: "
      << cudaGetErrorString(prefix_err);

  switch (chunk_tile_tokens) {
    case 16:
      merge_prefix_splitkv_with_chunk_tile_kernel<T, THREADS, 16>
          <<<merge_grid, THREADS, chunk_tile_smem_size<16>(head_size, num_partitions), stream>>>(
              queries,
              chunk_keys,
              chunk_values,
              outputs,
              chunk_row_starts,
              local_token_offsets,
              partial_out,
              partial_max,
              partial_sum,
              num_kv_heads,
              head_size,
              kv_mul,
              num_partitions);
      break;
    case 8:
      merge_prefix_splitkv_with_chunk_tile_kernel<T, THREADS, 8>
          <<<merge_grid, THREADS, chunk_tile_smem_size<8>(head_size, num_partitions), stream>>>(
              queries,
              chunk_keys,
              chunk_values,
              outputs,
              chunk_row_starts,
              local_token_offsets,
              partial_out,
              partial_max,
              partial_sum,
              num_kv_heads,
              head_size,
              kv_mul,
              num_partitions);
      break;
    default:
      merge_prefix_splitkv_with_chunk_tile_kernel<T, THREADS, 4>
          <<<merge_grid, THREADS, chunk_tile_smem_size<4>(head_size, num_partitions), stream>>>(
              queries,
              chunk_keys,
              chunk_values,
              outputs,
              chunk_row_starts,
              local_token_offsets,
              partial_out,
              partial_max,
              partial_sum,
              num_kv_heads,
              head_size,
              kv_mul,
              num_partitions);
      break;
  }
  const cudaError_t merge_err = cudaPeekAtLastError();
  CHECK_EQ(merge_err, cudaSuccess)
      << "merge_prefix_splitkv_with_chunk_tile_kernel launch failed: "
      << cudaGetErrorString(merge_err);
}

}  // namespace

void batched_paged_mha_prefill_cu(
    int32_t batch_size,
    int32_t head_num,
    int32_t head_size,
    int32_t kv_mul,
    const tensor::Tensor& queries,
    const tensor::Tensor& chunk_keys,
    const tensor::Tensor& chunk_values,
    tensor::Tensor& outputs,
    const tensor::Tensor& key_pool,
    const tensor::Tensor& value_pool,
    const tensor::Tensor& block_tables,
    const tensor::Tensor& request_indices,
    const tensor::Tensor& base_context_lens,
    const tensor::Tensor& chunk_row_starts,
    const tensor::Tensor& local_token_offsets,
    tensor::Tensor& partial_out,
    tensor::Tensor& partial_max,
    tensor::Tensor& partial_sum,
    int32_t max_blocks_per_seq,
    int32_t max_prefix_blocks,
    int32_t block_size,
    int32_t num_kv_heads,
    base::DeviceType device_type,
    CudaConfig* config) {
  if (batch_size <= 0 || head_num <= 0) {
    return;
  }
  CHECK_EQ(device_type, base::DeviceType::kDeviceCUDA);
  CHECK(config != nullptr);
  CHECK_GT(head_size, 0);
  CHECK_GT(block_size, 0);
  CHECK_GT(num_kv_heads, 0);
  CHECK_GT(kv_mul, 0);
  CHECK_EQ(head_num % num_kv_heads, 0);
  CHECK_EQ(kv_mul, head_num / num_kv_heads);
  CHECK_EQ(queries.device_type(), base::DeviceType::kDeviceCUDA);
  CHECK_EQ(chunk_keys.device_type(), base::DeviceType::kDeviceCUDA);
  CHECK_EQ(chunk_values.device_type(), base::DeviceType::kDeviceCUDA);
  CHECK_EQ(outputs.device_type(), base::DeviceType::kDeviceCUDA);
  CHECK_EQ(key_pool.device_type(), base::DeviceType::kDeviceCUDA);
  CHECK_EQ(value_pool.device_type(), base::DeviceType::kDeviceCUDA);
  CHECK_EQ(block_tables.device_type(), base::DeviceType::kDeviceCUDA);
  CHECK_EQ(request_indices.device_type(), base::DeviceType::kDeviceCUDA);
  CHECK_EQ(base_context_lens.device_type(), base::DeviceType::kDeviceCUDA);
  CHECK_EQ(chunk_row_starts.device_type(), base::DeviceType::kDeviceCUDA);
  CHECK_EQ(local_token_offsets.device_type(), base::DeviceType::kDeviceCUDA);
  CHECK_EQ(chunk_keys.data_type(), queries.data_type());
  CHECK_EQ(chunk_values.data_type(), queries.data_type());
  CHECK_EQ(outputs.data_type(), queries.data_type());
  CHECK_EQ(key_pool.data_type(), queries.data_type());
  CHECK_EQ(value_pool.data_type(), queries.data_type());
  CHECK_EQ(block_tables.data_type(), base::DataType::kDataTypeInt32);
  CHECK_EQ(request_indices.data_type(), base::DataType::kDataTypeInt32);
  CHECK_EQ(base_context_lens.data_type(), base::DataType::kDataTypeInt32);
  CHECK_EQ(chunk_row_starts.data_type(), base::DataType::kDataTypeInt32);
  CHECK_EQ(local_token_offsets.data_type(), base::DataType::kDataTypeInt32);
  CHECK_EQ(partial_out.device_type(), base::DeviceType::kDeviceCUDA);
  CHECK_EQ(partial_max.device_type(), base::DeviceType::kDeviceCUDA);
  CHECK_EQ(partial_sum.device_type(), base::DeviceType::kDeviceCUDA);
  CHECK_EQ(partial_out.data_type(), base::DataType::kDataTypeFp32);
  CHECK_EQ(partial_max.data_type(), base::DataType::kDataTypeFp32);
  CHECK_EQ(partial_sum.data_type(), base::DataType::kDataTypeFp32);

  cudaStream_t stream = config->stream;
  const dim3 prefix_grid(head_num, batch_size, choose_prefill_num_partitions(max_prefix_blocks));
  const dim3 merge_grid(head_num, batch_size);
  const int32_t threads = choose_prefill_threads(head_size);
  const int32_t chunk_tile_tokens = choose_chunk_tile_tokens(head_size);
  const int32_t num_partitions = prefix_grid.z;

  if (queries.data_type() == base::DataType::kDataTypeFp32) {
    switch (threads) {
      case 32:
        launch_hybrid_prefill_attention<float, 32>(
            prefix_grid,
            merge_grid,
            head_size,
            chunk_tile_tokens,
            stream,
            queries.ptr<float>(),
            chunk_keys.ptr<float>(),
            chunk_values.ptr<float>(),
            outputs.ptr<float>(),
            key_pool.ptr<float>(),
            value_pool.ptr<float>(),
            block_tables.ptr<int32_t>(),
            request_indices.ptr<int32_t>(),
            base_context_lens.ptr<int32_t>(),
            chunk_row_starts.ptr<int32_t>(),
            local_token_offsets.ptr<int32_t>(),
            partial_out.ptr<float>(),
            partial_max.ptr<float>(),
            partial_sum.ptr<float>(),
            max_blocks_per_seq,
            block_size,
            num_kv_heads,
            kv_mul,
            num_partitions);
        break;
      case 64:
        launch_hybrid_prefill_attention<float, 64>(
            prefix_grid,
            merge_grid,
            head_size,
            chunk_tile_tokens,
            stream,
            queries.ptr<float>(),
            chunk_keys.ptr<float>(),
            chunk_values.ptr<float>(),
            outputs.ptr<float>(),
            key_pool.ptr<float>(),
            value_pool.ptr<float>(),
            block_tables.ptr<int32_t>(),
            request_indices.ptr<int32_t>(),
            base_context_lens.ptr<int32_t>(),
            chunk_row_starts.ptr<int32_t>(),
            local_token_offsets.ptr<int32_t>(),
            partial_out.ptr<float>(),
            partial_max.ptr<float>(),
            partial_sum.ptr<float>(),
            max_blocks_per_seq,
            block_size,
            num_kv_heads,
            kv_mul,
            num_partitions);
        break;
      case 128:
        launch_hybrid_prefill_attention<float, 128>(
            prefix_grid,
            merge_grid,
            head_size,
            chunk_tile_tokens,
            stream,
            queries.ptr<float>(),
            chunk_keys.ptr<float>(),
            chunk_values.ptr<float>(),
            outputs.ptr<float>(),
            key_pool.ptr<float>(),
            value_pool.ptr<float>(),
            block_tables.ptr<int32_t>(),
            request_indices.ptr<int32_t>(),
            base_context_lens.ptr<int32_t>(),
            chunk_row_starts.ptr<int32_t>(),
            local_token_offsets.ptr<int32_t>(),
            partial_out.ptr<float>(),
            partial_max.ptr<float>(),
            partial_sum.ptr<float>(),
            max_blocks_per_seq,
            block_size,
            num_kv_heads,
            kv_mul,
            num_partitions);
        break;
      default:
        launch_hybrid_prefill_attention<float, 256>(
            prefix_grid,
            merge_grid,
            head_size,
            chunk_tile_tokens,
            stream,
            queries.ptr<float>(),
            chunk_keys.ptr<float>(),
            chunk_values.ptr<float>(),
            outputs.ptr<float>(),
            key_pool.ptr<float>(),
            value_pool.ptr<float>(),
            block_tables.ptr<int32_t>(),
            request_indices.ptr<int32_t>(),
            base_context_lens.ptr<int32_t>(),
            chunk_row_starts.ptr<int32_t>(),
            local_token_offsets.ptr<int32_t>(),
            partial_out.ptr<float>(),
            partial_max.ptr<float>(),
            partial_sum.ptr<float>(),
            max_blocks_per_seq,
            block_size,
            num_kv_heads,
            kv_mul,
            num_partitions);
        break;
    }
  } else {
    CHECK_EQ(queries.data_type(), base::DataType::kDataTypeBf16);
    switch (threads) {
      case 32:
        launch_hybrid_prefill_attention<base::CudaBF16, 32>(
            prefix_grid,
            merge_grid,
            head_size,
            chunk_tile_tokens,
            stream,
            reinterpret_cast<const base::CudaBF16*>(queries.ptr<uint16_t>()),
            reinterpret_cast<const base::CudaBF16*>(chunk_keys.ptr<uint16_t>()),
            reinterpret_cast<const base::CudaBF16*>(chunk_values.ptr<uint16_t>()),
            reinterpret_cast<base::CudaBF16*>(outputs.ptr<uint16_t>()),
            reinterpret_cast<const base::CudaBF16*>(key_pool.ptr<uint16_t>()),
            reinterpret_cast<const base::CudaBF16*>(value_pool.ptr<uint16_t>()),
            block_tables.ptr<int32_t>(),
            request_indices.ptr<int32_t>(),
            base_context_lens.ptr<int32_t>(),
            chunk_row_starts.ptr<int32_t>(),
            local_token_offsets.ptr<int32_t>(),
            partial_out.ptr<float>(),
            partial_max.ptr<float>(),
            partial_sum.ptr<float>(),
            max_blocks_per_seq,
            block_size,
            num_kv_heads,
            kv_mul,
            num_partitions);
        break;
      case 64:
        launch_hybrid_prefill_attention<base::CudaBF16, 64>(
            prefix_grid,
            merge_grid,
            head_size,
            chunk_tile_tokens,
            stream,
            reinterpret_cast<const base::CudaBF16*>(queries.ptr<uint16_t>()),
            reinterpret_cast<const base::CudaBF16*>(chunk_keys.ptr<uint16_t>()),
            reinterpret_cast<const base::CudaBF16*>(chunk_values.ptr<uint16_t>()),
            reinterpret_cast<base::CudaBF16*>(outputs.ptr<uint16_t>()),
            reinterpret_cast<const base::CudaBF16*>(key_pool.ptr<uint16_t>()),
            reinterpret_cast<const base::CudaBF16*>(value_pool.ptr<uint16_t>()),
            block_tables.ptr<int32_t>(),
            request_indices.ptr<int32_t>(),
            base_context_lens.ptr<int32_t>(),
            chunk_row_starts.ptr<int32_t>(),
            local_token_offsets.ptr<int32_t>(),
            partial_out.ptr<float>(),
            partial_max.ptr<float>(),
            partial_sum.ptr<float>(),
            max_blocks_per_seq,
            block_size,
            num_kv_heads,
            kv_mul,
            num_partitions);
        break;
      case 128:
        launch_hybrid_prefill_attention<base::CudaBF16, 128>(
            prefix_grid,
            merge_grid,
            head_size,
            chunk_tile_tokens,
            stream,
            reinterpret_cast<const base::CudaBF16*>(queries.ptr<uint16_t>()),
            reinterpret_cast<const base::CudaBF16*>(chunk_keys.ptr<uint16_t>()),
            reinterpret_cast<const base::CudaBF16*>(chunk_values.ptr<uint16_t>()),
            reinterpret_cast<base::CudaBF16*>(outputs.ptr<uint16_t>()),
            reinterpret_cast<const base::CudaBF16*>(key_pool.ptr<uint16_t>()),
            reinterpret_cast<const base::CudaBF16*>(value_pool.ptr<uint16_t>()),
            block_tables.ptr<int32_t>(),
            request_indices.ptr<int32_t>(),
            base_context_lens.ptr<int32_t>(),
            chunk_row_starts.ptr<int32_t>(),
            local_token_offsets.ptr<int32_t>(),
            partial_out.ptr<float>(),
            partial_max.ptr<float>(),
            partial_sum.ptr<float>(),
            max_blocks_per_seq,
            block_size,
            num_kv_heads,
            kv_mul,
            num_partitions);
        break;
      default:
        launch_hybrid_prefill_attention<base::CudaBF16, 256>(
            prefix_grid,
            merge_grid,
            head_size,
            chunk_tile_tokens,
            stream,
            reinterpret_cast<const base::CudaBF16*>(queries.ptr<uint16_t>()),
            reinterpret_cast<const base::CudaBF16*>(chunk_keys.ptr<uint16_t>()),
            reinterpret_cast<const base::CudaBF16*>(chunk_values.ptr<uint16_t>()),
            reinterpret_cast<base::CudaBF16*>(outputs.ptr<uint16_t>()),
            reinterpret_cast<const base::CudaBF16*>(key_pool.ptr<uint16_t>()),
            reinterpret_cast<const base::CudaBF16*>(value_pool.ptr<uint16_t>()),
            block_tables.ptr<int32_t>(),
            request_indices.ptr<int32_t>(),
            base_context_lens.ptr<int32_t>(),
            chunk_row_starts.ptr<int32_t>(),
            local_token_offsets.ptr<int32_t>(),
            partial_out.ptr<float>(),
            partial_max.ptr<float>(),
            partial_sum.ptr<float>(),
            max_blocks_per_seq,
            block_size,
            num_kv_heads,
            kv_mul,
            num_partitions);
        break;
    }
  }

}

void batched_paged_mha_prefill_fp8_cu(
    int32_t batch_size,
    int32_t head_num,
    int32_t head_size,
    int32_t kv_mul,
    const tensor::Tensor& queries,
    const tensor::Tensor& chunk_keys,
    const tensor::Tensor& chunk_values,
    tensor::Tensor& outputs,
    const tensor::Tensor& key_pool,
    const tensor::Tensor& value_pool,
    const tensor::Tensor& key_scale_pool,
    const tensor::Tensor& value_scale_pool,
    const tensor::Tensor& block_tables,
    const tensor::Tensor& request_indices,
    const tensor::Tensor& base_context_lens,
    const tensor::Tensor& chunk_row_starts,
    const tensor::Tensor& local_token_offsets,
    tensor::Tensor& partial_out,
    tensor::Tensor& partial_max,
    tensor::Tensor& partial_sum,
    int32_t max_blocks_per_seq,
    int32_t max_prefix_blocks,
    int32_t block_size,
    int32_t num_kv_heads,
    base::DeviceType device_type,
    CudaConfig* config) {
  if (batch_size <= 0 || head_num <= 0) {
    return;
  }
  CHECK_EQ(device_type, base::DeviceType::kDeviceCUDA);
  CHECK(config != nullptr);
  CHECK_EQ(queries.device_type(), base::DeviceType::kDeviceCUDA);
  CHECK_EQ(chunk_keys.device_type(), base::DeviceType::kDeviceCUDA);
  CHECK_EQ(chunk_values.device_type(), base::DeviceType::kDeviceCUDA);
  CHECK_EQ(outputs.device_type(), base::DeviceType::kDeviceCUDA);
  CHECK_EQ(key_pool.device_type(), base::DeviceType::kDeviceCUDA);
  CHECK_EQ(value_pool.device_type(), base::DeviceType::kDeviceCUDA);
  CHECK_EQ(key_scale_pool.device_type(), base::DeviceType::kDeviceCUDA);
  CHECK_EQ(value_scale_pool.device_type(), base::DeviceType::kDeviceCUDA);
  CHECK_EQ(key_pool.data_type(), base::DataType::kDataTypeInt8);
  CHECK_EQ(value_pool.data_type(), base::DataType::kDataTypeInt8);
  CHECK_EQ(key_scale_pool.data_type(), base::DataType::kDataTypeFp32);
  CHECK_EQ(value_scale_pool.data_type(), base::DataType::kDataTypeFp32);
  CHECK_EQ(chunk_keys.data_type(), queries.data_type());
  CHECK_EQ(chunk_values.data_type(), queries.data_type());
  CHECK_EQ(outputs.data_type(), queries.data_type());

  cudaStream_t stream = config->stream;
  const dim3 prefix_grid(head_num, batch_size, choose_prefill_num_partitions(max_prefix_blocks));
  const dim3 merge_grid(head_num, batch_size);
  const int32_t threads = choose_prefill_threads(head_size);
  const int32_t chunk_tile_tokens = choose_chunk_tile_tokens(head_size);
  const int32_t num_partitions = prefix_grid.z;

  if (queries.data_type() == base::DataType::kDataTypeFp32) {
    switch (threads) {
      case 32:
        launch_hybrid_prefill_attention_fp8<float, 32>(
            prefix_grid, merge_grid, head_size, chunk_tile_tokens, stream,
            queries.ptr<float>(), chunk_keys.ptr<float>(), chunk_values.ptr<float>(),
            outputs.ptr<float>(), key_pool.ptr<int8_t>(), value_pool.ptr<int8_t>(),
            key_scale_pool.ptr<float>(), value_scale_pool.ptr<float>(),
            block_tables.ptr<int32_t>(), request_indices.ptr<int32_t>(),
            base_context_lens.ptr<int32_t>(), chunk_row_starts.ptr<int32_t>(),
            local_token_offsets.ptr<int32_t>(), partial_out.ptr<float>(),
            partial_max.ptr<float>(), partial_sum.ptr<float>(),
            max_blocks_per_seq, block_size, num_kv_heads, kv_mul, num_partitions);
        break;
      case 64:
        launch_hybrid_prefill_attention_fp8<float, 64>(
            prefix_grid, merge_grid, head_size, chunk_tile_tokens, stream,
            queries.ptr<float>(), chunk_keys.ptr<float>(), chunk_values.ptr<float>(),
            outputs.ptr<float>(), key_pool.ptr<int8_t>(), value_pool.ptr<int8_t>(),
            key_scale_pool.ptr<float>(), value_scale_pool.ptr<float>(),
            block_tables.ptr<int32_t>(), request_indices.ptr<int32_t>(),
            base_context_lens.ptr<int32_t>(), chunk_row_starts.ptr<int32_t>(),
            local_token_offsets.ptr<int32_t>(), partial_out.ptr<float>(),
            partial_max.ptr<float>(), partial_sum.ptr<float>(),
            max_blocks_per_seq, block_size, num_kv_heads, kv_mul, num_partitions);
        break;
      case 128:
        launch_hybrid_prefill_attention_fp8<float, 128>(
            prefix_grid, merge_grid, head_size, chunk_tile_tokens, stream,
            queries.ptr<float>(), chunk_keys.ptr<float>(), chunk_values.ptr<float>(),
            outputs.ptr<float>(), key_pool.ptr<int8_t>(), value_pool.ptr<int8_t>(),
            key_scale_pool.ptr<float>(), value_scale_pool.ptr<float>(),
            block_tables.ptr<int32_t>(), request_indices.ptr<int32_t>(),
            base_context_lens.ptr<int32_t>(), chunk_row_starts.ptr<int32_t>(),
            local_token_offsets.ptr<int32_t>(), partial_out.ptr<float>(),
            partial_max.ptr<float>(), partial_sum.ptr<float>(),
            max_blocks_per_seq, block_size, num_kv_heads, kv_mul, num_partitions);
        break;
      default:
        launch_hybrid_prefill_attention_fp8<float, 256>(
            prefix_grid, merge_grid, head_size, chunk_tile_tokens, stream,
            queries.ptr<float>(), chunk_keys.ptr<float>(), chunk_values.ptr<float>(),
            outputs.ptr<float>(), key_pool.ptr<int8_t>(), value_pool.ptr<int8_t>(),
            key_scale_pool.ptr<float>(), value_scale_pool.ptr<float>(),
            block_tables.ptr<int32_t>(), request_indices.ptr<int32_t>(),
            base_context_lens.ptr<int32_t>(), chunk_row_starts.ptr<int32_t>(),
            local_token_offsets.ptr<int32_t>(), partial_out.ptr<float>(),
            partial_max.ptr<float>(), partial_sum.ptr<float>(),
            max_blocks_per_seq, block_size, num_kv_heads, kv_mul, num_partitions);
        break;
    }
    return;
  }

  CHECK_EQ(queries.data_type(), base::DataType::kDataTypeBf16);
  switch (threads) {
    case 32:
      launch_hybrid_prefill_attention_fp8<base::CudaBF16, 32>(
          prefix_grid, merge_grid, head_size, chunk_tile_tokens, stream,
          reinterpret_cast<const base::CudaBF16*>(queries.ptr<uint16_t>()),
          reinterpret_cast<const base::CudaBF16*>(chunk_keys.ptr<uint16_t>()),
          reinterpret_cast<const base::CudaBF16*>(chunk_values.ptr<uint16_t>()),
          reinterpret_cast<base::CudaBF16*>(outputs.ptr<uint16_t>()),
          key_pool.ptr<int8_t>(), value_pool.ptr<int8_t>(),
          key_scale_pool.ptr<float>(), value_scale_pool.ptr<float>(),
          block_tables.ptr<int32_t>(), request_indices.ptr<int32_t>(),
          base_context_lens.ptr<int32_t>(), chunk_row_starts.ptr<int32_t>(),
          local_token_offsets.ptr<int32_t>(), partial_out.ptr<float>(),
          partial_max.ptr<float>(), partial_sum.ptr<float>(),
          max_blocks_per_seq, block_size, num_kv_heads, kv_mul, num_partitions);
      break;
    case 64:
      launch_hybrid_prefill_attention_fp8<base::CudaBF16, 64>(
          prefix_grid, merge_grid, head_size, chunk_tile_tokens, stream,
          reinterpret_cast<const base::CudaBF16*>(queries.ptr<uint16_t>()),
          reinterpret_cast<const base::CudaBF16*>(chunk_keys.ptr<uint16_t>()),
          reinterpret_cast<const base::CudaBF16*>(chunk_values.ptr<uint16_t>()),
          reinterpret_cast<base::CudaBF16*>(outputs.ptr<uint16_t>()),
          key_pool.ptr<int8_t>(), value_pool.ptr<int8_t>(),
          key_scale_pool.ptr<float>(), value_scale_pool.ptr<float>(),
          block_tables.ptr<int32_t>(), request_indices.ptr<int32_t>(),
          base_context_lens.ptr<int32_t>(), chunk_row_starts.ptr<int32_t>(),
          local_token_offsets.ptr<int32_t>(), partial_out.ptr<float>(),
          partial_max.ptr<float>(), partial_sum.ptr<float>(),
          max_blocks_per_seq, block_size, num_kv_heads, kv_mul, num_partitions);
      break;
    case 128:
      launch_hybrid_prefill_attention_fp8<base::CudaBF16, 128>(
          prefix_grid, merge_grid, head_size, chunk_tile_tokens, stream,
          reinterpret_cast<const base::CudaBF16*>(queries.ptr<uint16_t>()),
          reinterpret_cast<const base::CudaBF16*>(chunk_keys.ptr<uint16_t>()),
          reinterpret_cast<const base::CudaBF16*>(chunk_values.ptr<uint16_t>()),
          reinterpret_cast<base::CudaBF16*>(outputs.ptr<uint16_t>()),
          key_pool.ptr<int8_t>(), value_pool.ptr<int8_t>(),
          key_scale_pool.ptr<float>(), value_scale_pool.ptr<float>(),
          block_tables.ptr<int32_t>(), request_indices.ptr<int32_t>(),
          base_context_lens.ptr<int32_t>(), chunk_row_starts.ptr<int32_t>(),
          local_token_offsets.ptr<int32_t>(), partial_out.ptr<float>(),
          partial_max.ptr<float>(), partial_sum.ptr<float>(),
          max_blocks_per_seq, block_size, num_kv_heads, kv_mul, num_partitions);
      break;
    default:
      launch_hybrid_prefill_attention_fp8<base::CudaBF16, 256>(
          prefix_grid, merge_grid, head_size, chunk_tile_tokens, stream,
          reinterpret_cast<const base::CudaBF16*>(queries.ptr<uint16_t>()),
          reinterpret_cast<const base::CudaBF16*>(chunk_keys.ptr<uint16_t>()),
          reinterpret_cast<const base::CudaBF16*>(chunk_values.ptr<uint16_t>()),
          reinterpret_cast<base::CudaBF16*>(outputs.ptr<uint16_t>()),
          key_pool.ptr<int8_t>(), value_pool.ptr<int8_t>(),
          key_scale_pool.ptr<float>(), value_scale_pool.ptr<float>(),
          block_tables.ptr<int32_t>(), request_indices.ptr<int32_t>(),
          base_context_lens.ptr<int32_t>(), chunk_row_starts.ptr<int32_t>(),
          local_token_offsets.ptr<int32_t>(), partial_out.ptr<float>(),
          partial_max.ptr<float>(), partial_sum.ptr<float>(),
          max_blocks_per_seq, block_size, num_kv_heads, kv_mul, num_partitions);
      break;
  }
}

}  // namespace kernel
