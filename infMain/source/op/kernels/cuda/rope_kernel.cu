// Updated on March 23, 2026
#include "rope_kernel.cuh"
#include <algorithm>
#include "cuda_type_utils.cuh"

namespace kernel {

template <typename T>
__global__ void rope_kernel_cu_impl(int pos, int dim, int kv_dim, int head_size, T* input_q,
                                    T* input_k, const float* sin_cache, const float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

#if defined(LLAMA3_SUPPORT)
  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  if (idx >= total_pairs) {
    return;
  }

  int head_idx = idx / head_pair_count;
  int head_dim = idx % head_pair_count;
  int i = head_idx * head_size;
  int v0_idx = i + head_dim;
  int v1_idx = i + head_dim + head_size / 2;

  float fci = sin_cache[pos * head_size + head_dim];
  float fcr = cos_cache[pos * head_size + head_dim];
#elif defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT) || defined(QWEN_MOE_SUPPORT)
  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  if (idx >= total_pairs) {
    return;
  }

  int head_idx = idx / head_pair_count;
  int head_dim = idx % head_pair_count;
  int i = head_idx * head_size;
  int v0_idx = i + head_dim;
  int v1_idx = i + head_dim + head_size / 2;

  float fci = sin_cache[pos * head_size + head_dim * 2];
  float fcr = cos_cache[pos * head_size + head_dim * 2];
#else
  idx = idx * 2;
  if (idx >= dim) {
    return;
  }
  int head_dim = idx % head_size;
  int i = idx;
  int v0_idx = i;
  int v1_idx = i + 1;

  float fci = sin_cache[pos * head_size + head_dim];
  float fcr = cos_cache[pos * head_size + head_dim];
#endif

  int rotn = i < kv_dim ? 2 : 1;
  for (int v = 0; v < rotn; ++v) {
    T* vec = v == 0 ? input_q : input_k;
    float v0 = scalar_to_float(vec[v0_idx]);
    float v1 = scalar_to_float(vec[v1_idx]);
    vec[v0_idx] = float_to_scalar<T>(fcr * v0 - fci * v1);
    vec[v1_idx] = float_to_scalar<T>(fcr * v1 + fci * v0);
  }
}

__global__ void sin_cos_calc(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) {
  const int head_dim = threadIdx.x + blockDim.x * blockIdx.x;
  if (head_dim >= head_size) {
    return;
  }

#if defined(LLAMA3_SUPPORT)
  const float freq =
      1.0f / powf(500000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
#elif defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT) || defined(QWEN_MOE_SUPPORT)
  const float freq =
      1.0f / powf(1000000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
#else
  const float freq =
      1.0f / powf(10000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
#endif

  for (int pos = blockIdx.y; pos < max_seq_len; pos += gridDim.y) {
    const float val = static_cast<float>(pos) * freq;
    float fci = 0.f;
    float fcr = 0.f;
    sincosf(val, &fci, &fcr);
    *(sin_cache + pos * head_size + head_dim) = fci;
    *(cos_cache + pos * head_size + head_dim) = fcr;
  }
}

void sin_cos_cache_calc_cu(int head_size, int max_seq_len, const tensor::Tensor& sin_cache,
                           const tensor::Tensor& cos_cache, void* stream) {
  CHECK_EQ(sin_cache.is_empty(), false);
  CHECK_EQ(cos_cache.is_empty(), false);
  constexpr int kThreads = 128;
  constexpr int kMaxPosBlocks = 64;
  const int threads = std::min(head_size, kThreads);
  dim3 block(threads);
  dim3 grid((head_size + threads - 1) / threads, std::min(max_seq_len, kMaxPosBlocks));
  cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
  if (stream_) {
    sin_cos_calc<<<grid, block, 0, stream_>>>(head_size, max_seq_len,
                                             const_cast<float*>(sin_cache.ptr<float>()),
                                             const_cast<float*>(cos_cache.ptr<float>()));
  } else {
    sin_cos_calc<<<grid, block>>>(head_size, max_seq_len, const_cast<float*>(sin_cache.ptr<float>()),
                                  const_cast<float*>(cos_cache.ptr<float>()));
  }
}

void rope_kernel_cu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
                    const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
                    const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache, void* stream) {
  const int32_t pos = *input_pos.ptr<int32_t>(0);
  int threads = 128;
  int blocks = (dim + threads - 1) / threads;
  cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;

  if (input_q.data_type() == base::DataType::kDataTypeFp32) {
    if (stream_) {
      rope_kernel_cu_impl<float><<<blocks, threads, 0, stream_>>>(
          pos, dim, kv_dim, head_size, const_cast<float*>(input_q.ptr<float>()),
          const_cast<float*>(input_k.ptr<float>()), sin_cache.ptr<float>(), cos_cache.ptr<float>());
    } else {
      rope_kernel_cu_impl<float><<<blocks, threads>>>(
          pos, dim, kv_dim, head_size, const_cast<float*>(input_q.ptr<float>()),
          const_cast<float*>(input_k.ptr<float>()), sin_cache.ptr<float>(), cos_cache.ptr<float>());
    }
  } else {
    CHECK_EQ(input_q.data_type(), base::DataType::kDataTypeBf16);
    auto q_ptr = reinterpret_cast<base::CudaBF16*>(const_cast<uint16_t*>(input_q.ptr<uint16_t>()));
    auto k_ptr = reinterpret_cast<base::CudaBF16*>(const_cast<uint16_t*>(input_k.ptr<uint16_t>()));
    if (stream_) {
      rope_kernel_cu_impl<base::CudaBF16><<<blocks, threads, 0, stream_>>>(
          pos, dim, kv_dim, head_size, q_ptr, k_ptr, sin_cache.ptr<float>(), cos_cache.ptr<float>());
    } else {
      rope_kernel_cu_impl<base::CudaBF16><<<blocks, threads>>>(
          pos, dim, kv_dim, head_size, q_ptr, k_ptr, sin_cache.ptr<float>(), cos_cache.ptr<float>());
    }
  }
}

// Batched RoPE: grid = (blocks_per_token, batch_tokens)
template <typename T>
__global__ void rope_kernel_batched_impl(int dim, int kv_dim, int head_size,
                                         T* input_q, T* input_k,
                                         const int32_t* positions,
                                         const float* sin_cache, const float* cos_cache) {
  int token_idx = blockIdx.y;
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int pos = positions[token_idx];

  T* q_ptr = input_q + token_idx * dim;
  T* k_ptr = input_k + token_idx * kv_dim;

#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT) || defined(QWEN_MOE_SUPPORT)
  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  if (idx >= total_pairs) {
    return;
  }

  int head_idx = idx / head_pair_count;
  int head_dim = idx % head_pair_count;
  int i = head_idx * head_size;
  int v0_idx = i + head_dim;
  int v1_idx = i + head_dim + head_size / 2;

  float fci = sin_cache[pos * head_size + head_dim * 2];
  float fcr = cos_cache[pos * head_size + head_dim * 2];
#elif defined(LLAMA3_SUPPORT)
  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  if (idx >= total_pairs) {
    return;
  }

  int head_idx = idx / head_pair_count;
  int head_dim = idx % head_pair_count;
  int i = head_idx * head_size;
  int v0_idx = i + head_dim;
  int v1_idx = i + head_dim + head_size / 2;

  float fci = sin_cache[pos * head_size + head_dim];
  float fcr = cos_cache[pos * head_size + head_dim];
#else
  idx = idx * 2;
  if (idx >= dim) {
    return;
  }
  int head_dim = idx % head_size;
  int i = idx;
  int v0_idx = i;
  int v1_idx = i + 1;

  float fci = sin_cache[pos * head_size + head_dim];
  float fcr = cos_cache[pos * head_size + head_dim];
#endif

  int rotn = i < kv_dim ? 2 : 1;
  for (int v = 0; v < rotn; ++v) {
    T* vec = v == 0 ? q_ptr : k_ptr;
    float v0 = scalar_to_float(vec[v0_idx]);
    float v1 = scalar_to_float(vec[v1_idx]);
    vec[v0_idx] = float_to_scalar<T>(fcr * v0 - fci * v1);
    vec[v1_idx] = float_to_scalar<T>(fcr * v1 + fci * v0);
  }
}

void rope_kernel_batched_cu(int32_t dim, int32_t kv_dim, int32_t head_size,
                            const tensor::Tensor& input_q,
                            const tensor::Tensor& input_k,
                            const tensor::Tensor& positions,
                            const tensor::Tensor& sin_cache,
                            const tensor::Tensor& cos_cache,
                            int32_t batch_tokens, void* stream) {
  int threads = 128;
  int blocks_x = (dim + threads - 1) / threads;
  dim3 grid(blocks_x, batch_tokens);
  cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;

  if (input_q.data_type() == base::DataType::kDataTypeFp32) {
    rope_kernel_batched_impl<float><<<grid, threads, 0, stream_>>>(
        dim, kv_dim, head_size,
        const_cast<float*>(input_q.ptr<float>()),
        const_cast<float*>(input_k.ptr<float>()),
        positions.ptr<int32_t>(),
        sin_cache.ptr<float>(), cos_cache.ptr<float>());
  } else {
    CHECK_EQ(input_q.data_type(), base::DataType::kDataTypeBf16);
    auto q_ptr = reinterpret_cast<base::CudaBF16*>(const_cast<uint16_t*>(input_q.ptr<uint16_t>()));
    auto k_ptr = reinterpret_cast<base::CudaBF16*>(const_cast<uint16_t*>(input_k.ptr<uint16_t>()));
    rope_kernel_batched_impl<base::CudaBF16><<<grid, threads, 0, stream_>>>(
        dim, kv_dim, head_size, q_ptr, k_ptr,
        positions.ptr<int32_t>(),
        sin_cache.ptr<float>(), cos_cache.ptr<float>());
  }
}

template <typename T>
__global__ void mrope_kernel_batched_impl(int dim, int kv_dim, int head_size,
                                          T* input_q, T* input_k,
                                          const int32_t* positions,
                                          const float* sin_cache,
                                          const float* cos_cache,
                                          int temporal_section,
                                          int height_section,
                                          int batch_tokens) {
  const int token_idx = blockIdx.y;
  const int pair_idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int pairs_per_head = head_size / 2;
  const int total_pairs = (dim / head_size) * pairs_per_head;
  if (token_idx >= batch_tokens || pair_idx >= total_pairs) {
    return;
  }
  const int head_idx = pair_idx / pairs_per_head;
  const int pair_in_head = pair_idx % pairs_per_head;
  int axis = 2;
  if (pair_in_head < temporal_section) {
    axis = 0;
  } else if (pair_in_head < temporal_section + height_section) {
    axis = 1;
  }
  const int pos = positions[axis * batch_tokens + token_idx];
  const int cache_offset = pos * head_size + pair_in_head * 2;
  const float fci = sin_cache[cache_offset];
  const float fcr = cos_cache[cache_offset];
  const int first = head_idx * head_size + pair_in_head;
  const int second = first + pairs_per_head;
  T* q = input_q + token_idx * dim;
  T* k = input_k + token_idx * kv_dim;
  const int rotations = first < kv_dim ? 2 : 1;
  for (int rotation = 0; rotation < rotations; ++rotation) {
    T* vector = rotation == 0 ? q : k;
    const float v0 = scalar_to_float(vector[first]);
    const float v1 = scalar_to_float(vector[second]);
    vector[first] = float_to_scalar<T>(fcr * v0 - fci * v1);
    vector[second] = float_to_scalar<T>(fcr * v1 + fci * v0);
  }
}

void mrope_kernel_batched_cu(int32_t dim, int32_t kv_dim, int32_t head_size,
                             const tensor::Tensor& input_q,
                             const tensor::Tensor& input_k,
                             const tensor::Tensor& positions,
                             const tensor::Tensor& sin_cache,
                             const tensor::Tensor& cos_cache,
                             int32_t temporal_section,
                             int32_t height_section,
                             int32_t batch_tokens, void* stream) {
  CHECK_EQ(positions.data_type(), base::DataType::kDataTypeInt32);
  CHECK_EQ(positions.size(), static_cast<size_t>(3 * batch_tokens));
  CHECK_GT(temporal_section, 0);
  CHECK_GT(height_section, 0);
  CHECK_LT(temporal_section + height_section, head_size / 2);
  constexpr int threads = 128;
  const int total_pairs = (dim / head_size) * (head_size / 2);
  dim3 grid((total_pairs + threads - 1) / threads, batch_tokens);
  cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
  if (input_q.data_type() == base::DataType::kDataTypeFp32) {
    mrope_kernel_batched_impl<float><<<grid, threads, 0, stream_>>>(
        dim, kv_dim, head_size, const_cast<float*>(input_q.ptr<float>()),
        const_cast<float*>(input_k.ptr<float>()), positions.ptr<int32_t>(),
        sin_cache.ptr<float>(), cos_cache.ptr<float>(), temporal_section,
        height_section, batch_tokens);
  } else {
    CHECK_EQ(input_q.data_type(), base::DataType::kDataTypeBf16);
    auto* q = reinterpret_cast<base::CudaBF16*>(const_cast<uint16_t*>(input_q.ptr<uint16_t>()));
    auto* k = reinterpret_cast<base::CudaBF16*>(const_cast<uint16_t*>(input_k.ptr<uint16_t>()));
    mrope_kernel_batched_impl<base::CudaBF16><<<grid, threads, 0, stream_>>>(
        dim, kv_dim, head_size, q, k, positions.ptr<int32_t>(),
        sin_cache.ptr<float>(), cos_cache.ptr<float>(), temporal_section,
        height_section, batch_tokens);
  }
}

}  // namespace kernel
