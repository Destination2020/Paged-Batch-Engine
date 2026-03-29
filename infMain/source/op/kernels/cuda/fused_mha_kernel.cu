// Updated on March 23, 2026
#include <base/cuda_config.h>
#include <tensor/tensor.h>
#include <cfloat>
#include <cub/cub.cuh>
#include "cuda_type_utils.cuh"
#include "op/kernels/cuda/fused_mha_kernel.cuh"

namespace kernel {
constexpr static int thread_num = 256;

template <typename T>
__global__ void multi_head_attention_kernel(int32_t pos, int32_t seq_len, const T* query, T* output,
                                            const T* key_cache, const T* value_cache,
                                            int32_t kv_dim, int32_t kv_mul, int32_t head_num,
                                            int32_t head_size, int32_t layer_offset) {
  if (pos >= seq_len) {
    return;
  }

  int32_t head = blockIdx.x;
  if (head >= head_num) {
    return;
  }

  extern __shared__ float shared_mem[];
  float* s_query_head = shared_mem;
  float* s_output_head = shared_mem + head_size;

  using BlockReduce = cub::BlockReduce<float, thread_num>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float m;
  __shared__ float alpha;
  __shared__ float p;
  __shared__ float sum;

  float scale = 1.f / sqrtf(static_cast<float>(head_size));
  const T* query_head = query + head * head_size;
  for (size_t i = threadIdx.x; i < head_size; i += blockDim.x) {
    s_query_head[i] = scalar_to_float(query_head[i]);
    s_output_head[i] = 0.f;
  }
  if (threadIdx.x == 0) {
    m = -FLT_MAX;
    sum = 0.f;
  }
  __syncthreads();

  for (int32_t t = 0; t < pos + 1; ++t) {
    const T* key_pos = key_cache + layer_offset + t * kv_dim + (head / kv_mul) * head_size;
    float score = 0.f;
    for (size_t i = threadIdx.x; i < head_size; i += blockDim.x) {
      score += scalar_to_float(key_pos[i]) * s_query_head[i];
    }
    score *= scale;
    score = BlockReduce(temp).Sum(score);
    if (threadIdx.x == 0) {
      const float m_new = max(score, m);
      alpha = expf(m - m_new);
      p = expf(score - m_new);
      sum = sum * alpha + p;
      m = m_new;
    }
    __syncthreads();

    const T* value_pos = value_cache + layer_offset + t * kv_dim + (head / kv_mul) * head_size;
    for (size_t i = threadIdx.x; i < head_size; i += blockDim.x) {
      s_output_head[i] = scalar_to_float(value_pos[i]) * p + s_output_head[i] * alpha;
    }
    __syncthreads();
  }

  T* output_head = output + head * head_size;
  for (size_t i = threadIdx.x; i < head_size; i += blockDim.x) {
    output_head[i] = float_to_scalar<T>(s_output_head[i] / sum);
  }
}

void fused_mha_kernel_cu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                         int32_t kv_dim, int32_t kv_mul, int32_t head_size,
                         const tensor::Tensor& mha_out, const tensor::Tensor& query_tensor,
                         const tensor::Tensor& score_tensor, const tensor::Tensor& key_cache_tensor,
                         const tensor::Tensor& value_cache_tensor, base::DeviceType device_type,
                         CudaConfig* config) {
  UNUSED(score_tensor);
  UNUSED(device_type);
  int32_t layer_offset = layer_index * seq_len * kv_dim;
  cudaStream_t stream = config->stream;

  if (query_tensor.data_type() == base::DataType::kDataTypeFp32) {
    multi_head_attention_kernel<float><<<head_num, thread_num, 2 * head_size * sizeof(float), stream>>>(
        pos, seq_len, query_tensor.ptr<float>(), const_cast<float*>(mha_out.ptr<float>()),
        key_cache_tensor.ptr<float>(), value_cache_tensor.ptr<float>(), kv_dim, kv_mul, head_num,
        head_size, layer_offset);
  } else {
    CHECK_EQ(query_tensor.data_type(), base::DataType::kDataTypeBf16);
    multi_head_attention_kernel<base::CudaBF16>
        <<<head_num, thread_num, 2 * head_size * sizeof(float), stream>>>(
            pos, seq_len,
            reinterpret_cast<const base::CudaBF16*>(query_tensor.ptr<uint16_t>()),
            reinterpret_cast<base::CudaBF16*>(const_cast<uint16_t*>(mha_out.ptr<uint16_t>())),
            reinterpret_cast<const base::CudaBF16*>(key_cache_tensor.ptr<uint16_t>()),
            reinterpret_cast<const base::CudaBF16*>(value_cache_tensor.ptr<uint16_t>()), kv_dim,
            kv_mul, head_num, head_size, layer_offset);
  }
}

}  // namespace kernel
