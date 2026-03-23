// Updated on March 23, 2026
#include "rope_kernel.cuh"
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
  if (idx > total_pairs) {
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
  if (idx > total_pairs) {
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
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int head_dim = idx % head_size;
  for (int pos = 0; pos < max_seq_len; ++pos) {
#if defined(LLAMA3_SUPPORT)
    float freq =
        1.0f / powf(500000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
#elif defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT) || defined(QWEN_MOE_SUPPORT)
    float freq =
        1.0f / powf(1000000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
#else
    float freq =
        1.0f / powf(10000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
#endif
    float val = static_cast<float>(pos) * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    *(sin_cache + pos * head_size + head_dim) = fci;
    *(cos_cache + pos * head_size + head_dim) = fcr;
  }
}

void sin_cos_cache_calc_cu(int head_size, int max_seq_len, const tensor::Tensor& sin_cache,
                           const tensor::Tensor& cos_cache, cudaStream_t stream) {
  CHECK_EQ(sin_cache.is_empty(), false);
  CHECK_EQ(cos_cache.is_empty(), false);
  int threads = head_size;
  if (stream) {
    sin_cos_calc<<<1, threads, 0, stream>>>(head_size, max_seq_len,
                                            const_cast<float*>(sin_cache.ptr<float>()),
                                            const_cast<float*>(cos_cache.ptr<float>()));
  } else {
    sin_cos_calc<<<1, threads>>>(head_size, max_seq_len, const_cast<float*>(sin_cache.ptr<float>()),
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

}  // namespace kernel
