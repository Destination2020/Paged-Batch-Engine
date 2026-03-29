// Paged Multi-Head Attention Kernel Implementation
#include "op/kernels/cuda/paged_mha_kernel.cuh"
#include <cfloat>
#include <cub/cub.cuh>
#include "cuda_type_utils.cuh"

namespace kernel {

constexpr static int thread_num = 256;

// Paged decode attention kernel
// Each block processes one attention head
// Iterates through paged KV blocks using block_table for indirection
template <typename T>
__global__ void paged_decode_attention_kernel(
    const T* __restrict__ query,       // [head_num, head_size]
    T* __restrict__ output,            // [head_num, head_size]
    const T* __restrict__ key_pool,    // [num_blocks, block_size * num_kv_heads * head_size]
    const T* __restrict__ value_pool,  // same layout
    const int32_t* __restrict__ block_table,  // [num_kv_blocks]
    int32_t num_kv_blocks,
    int32_t num_tokens_in_last_block,
    int32_t block_size,
    int32_t num_kv_heads,
    int32_t head_size,
    int32_t kv_mul) {

  int32_t head = blockIdx.x;
  int32_t kv_head = head / kv_mul;

  extern __shared__ float shared_mem[];
  float* s_query = shared_mem;
  float* s_output = shared_mem + head_size;

  using BlockReduce = cub::BlockReduce<float, thread_num>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float m, sum_val, alpha_s, p_s;

  float scale = 1.f / sqrtf(static_cast<float>(head_size));

  // Load query to shared memory
  const T* query_head = query + head * head_size;
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    s_query[i] = scalar_to_float(query_head[i]);
    s_output[i] = 0.f;
  }
  if (threadIdx.x == 0) {
    m = -FLT_MAX;
    sum_val = 0.f;
  }
  __syncthreads();

  // Stride for one layer in the pool
  int64_t block_stride = block_size * num_kv_heads * head_size;
  int64_t kv_head_stride = head_size;

  // Iterate through all KV blocks
  for (int32_t block_idx = 0; block_idx < num_kv_blocks; ++block_idx) {
    int32_t physical_block_id = block_table[block_idx];

    // Determine valid tokens in this block
    int32_t valid_tokens = (block_idx == num_kv_blocks - 1)
                           ? num_tokens_in_last_block
                           : block_size;

    // Base address for this block's KV
    int64_t block_offset = physical_block_id * block_stride;
    const T* key_base = key_pool + block_offset + kv_head * kv_head_stride;
    const T* value_base = value_pool + block_offset + kv_head * kv_head_stride;

    // Iterate through tokens in this block
    for (int32_t t = 0; t < valid_tokens; ++t) {
      const T* key_ptr = key_base + t * num_kv_heads * head_size;

      // Compute Q·K score
      float score = 0.f;
      for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        score += scalar_to_float(key_ptr[i]) * s_query[i];
      }
      score *= scale;
      score = BlockReduce(temp).Sum(score);

      // Online softmax update
      if (threadIdx.x == 0) {
        float m_new = max(score, m);
        alpha_s = expf(m - m_new);
        p_s = expf(score - m_new);
        sum_val = sum_val * alpha_s + p_s;
        m = m_new;
      }
      __syncthreads();

      // Accumulate value
      const T* value_ptr = value_base + t * num_kv_heads * head_size;
      for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        s_output[i] = s_output[i] * alpha_s + scalar_to_float(value_ptr[i]) * p_s;
      }
      __syncthreads();
    }
  }

  // Normalize and write output
  T* output_head = output + head * head_size;
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    output_head[i] = float_to_scalar<T>(s_output[i] / sum_val);
  }
}

void paged_mha_decode_cu(
    int32_t head_num,
    int32_t head_size,
    int32_t kv_mul,
    const tensor::Tensor& query,
    const tensor::Tensor& output,
    const tensor::Tensor& key_pool,
    const tensor::Tensor& value_pool,
    const int32_t* block_table_gpu,
    int32_t num_kv_blocks,
    int32_t num_tokens_in_last_block,
    int32_t block_size,
    int32_t num_kv_heads,
    base::DeviceType device_type,
    CudaConfig* config) {

  UNUSED(device_type);
  cudaStream_t stream = config->stream;
  int smem_size = 2 * head_size * sizeof(float);

  if (query.data_type() == base::DataType::kDataTypeFp32) {
    paged_decode_attention_kernel<float><<<head_num, thread_num, smem_size, stream>>>(
        query.ptr<float>(),
        const_cast<float*>(output.ptr<float>()),
        key_pool.ptr<float>(),
        value_pool.ptr<float>(),
        block_table_gpu,
        num_kv_blocks,
        num_tokens_in_last_block,
        block_size,
        num_kv_heads,
        head_size,
        kv_mul);
  } else {
    CHECK_EQ(query.data_type(), base::DataType::kDataTypeBf16);
    paged_decode_attention_kernel<base::CudaBF16><<<head_num, thread_num, smem_size, stream>>>(
        reinterpret_cast<const base::CudaBF16*>(query.ptr<uint16_t>()),
        reinterpret_cast<base::CudaBF16*>(const_cast<uint16_t*>(output.ptr<uint16_t>())),
        reinterpret_cast<const base::CudaBF16*>(key_pool.ptr<uint16_t>()),
        reinterpret_cast<const base::CudaBF16*>(value_pool.ptr<uint16_t>()),
        block_table_gpu,
        num_kv_blocks,
        num_tokens_in_last_block,
        block_size,
        num_kv_heads,
        head_size,
        kv_mul);
  }
}

}  // namespace kernel
