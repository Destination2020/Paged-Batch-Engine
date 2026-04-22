// Paged Multi-Head Attention Kernel Implementation
#include "op/kernels/cuda/paged_mha_kernel.cuh"
#include <cfloat>
#include <cstdlib>
#include <sstream>
#include <string>
#include <unordered_set>
#include <cub/cub.cuh>
#include <glog/logging.h>
#include "cuda_type_utils.cuh"

namespace kernel {

constexpr static int thread_num = 256;

namespace {

bool decode_attn_diag_enabled() {
  const char* env = std::getenv("KUIPER_DECODE_ATTN_DIAG");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

void maybe_log_splitkv_decode_choice(int32_t batch_size,
                                     int32_t max_blocks_per_seq,
                                     int32_t num_partitions,
                                     int32_t head_num,
                                     int32_t head_size,
                                     int32_t kv_mul,
                                     int32_t num_kv_heads,
                                     base::DataType data_type) {
  if (!decode_attn_diag_enabled()) {
    return;
  }

  const char* dtype_name = "unknown";
  switch (data_type) {
    case base::DataType::kDataTypeFp32:
      dtype_name = "fp32";
      break;
    case base::DataType::kDataTypeBf16:
      dtype_name = "bf16";
      break;
    default:
      break;
  }

  static std::unordered_set<std::string> seen;
  std::ostringstream os;
  os << "splitkv_decode"
     << ":bs=" << batch_size
     << ":max_blocks=" << max_blocks_per_seq
     << ":num_partitions=" << num_partitions
     << ":head_num=" << head_num
     << ":head_size=" << head_size
     << ":kv_mul=" << kv_mul
     << ":kv_heads=" << num_kv_heads
     << ":dtype=" << dtype_name;
  const std::string key = os.str();
  if (seen.insert(key).second) {
    LOG(WARNING) << "[decode-attn-diag] " << key;
  }
}

}  // namespace

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

// Batched paged decode attention kernel
// grid: dim3(head_num, batch_size), block: (thread_num)
template <typename T>
__global__ void batched_paged_decode_attention_kernel(
    const T* __restrict__ queries,         // [batch_size, head_num * head_size]
    T* __restrict__ outputs,               // same
    const T* __restrict__ key_pool,
    const T* __restrict__ value_pool,
    const int32_t* __restrict__ block_tables,  // [batch_size, max_blocks_per_seq]
    const int32_t* __restrict__ seq_lens,      // [batch_size]
    int32_t max_blocks_per_seq,
    int32_t block_size,
    int32_t num_kv_heads,
    int32_t head_size,
    int32_t kv_mul) {

  int32_t head = blockIdx.x;
  int32_t batch_idx = blockIdx.y;
  int32_t kv_head = head / kv_mul;
  int32_t context_len = seq_lens[batch_idx];

  if (context_len <= 0) return;

  int32_t dim = gridDim.x * head_size;  // head_num * head_size
  const T* my_query = queries + batch_idx * dim;
  T* my_output = outputs + batch_idx * dim;
  const int32_t* my_block_table = block_tables + batch_idx * max_blocks_per_seq;

  extern __shared__ float shared_mem[];
  float* s_query = shared_mem;
  float* s_output = shared_mem + head_size;

  using BlockReduce = cub::BlockReduce<float, thread_num>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float m, sum_val, alpha_s, p_s;

  float scale = 1.f / sqrtf(static_cast<float>(head_size));

  // Load query to shared memory
  const T* query_head = my_query + head * head_size;
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    s_query[i] = scalar_to_float(query_head[i]);
    s_output[i] = 0.f;
  }
  if (threadIdx.x == 0) {
    m = -FLT_MAX;
    sum_val = 0.f;
  }
  __syncthreads();

  int64_t block_stride = block_size * num_kv_heads * head_size;
  int32_t num_kv_blocks = (context_len + block_size - 1) / block_size;
  int32_t tokens_in_last = context_len - (num_kv_blocks - 1) * block_size;

  for (int32_t block_idx = 0; block_idx < num_kv_blocks; ++block_idx) {
    int32_t physical_block_id = my_block_table[block_idx];
    int32_t valid_tokens = (block_idx == num_kv_blocks - 1) ? tokens_in_last : block_size;

    int64_t block_offset = physical_block_id * block_stride;
    const T* key_base = key_pool + block_offset + kv_head * head_size;
    const T* value_base = value_pool + block_offset + kv_head * head_size;

    for (int32_t t = 0; t < valid_tokens; ++t) {
      const T* key_ptr = key_base + t * num_kv_heads * head_size;

      float score = 0.f;
      for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        score += scalar_to_float(key_ptr[i]) * s_query[i];
      }
      score *= scale;
      score = BlockReduce(temp).Sum(score);

      if (threadIdx.x == 0) {
        float m_new = max(score, m);
        alpha_s = expf(m - m_new);
        p_s = expf(score - m_new);
        sum_val = sum_val * alpha_s + p_s;
        m = m_new;
      }
      __syncthreads();

      const T* value_ptr = value_base + t * num_kv_heads * head_size;
      for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        s_output[i] = s_output[i] * alpha_s + scalar_to_float(value_ptr[i]) * p_s;
      }
      __syncthreads();
    }
  }

  T* output_head = my_output + head * head_size;
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    output_head[i] = float_to_scalar<T>(s_output[i] / sum_val);
  }
}

void batched_paged_mha_decode_cu(
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
    base::DeviceType device_type,
    CudaConfig* config) {

  UNUSED(device_type);
  cudaStream_t stream = config->stream;
  int smem_size = 2 * head_size * sizeof(float);
  dim3 grid(head_num, batch_size);

  if (queries.data_type() == base::DataType::kDataTypeFp32) {
    batched_paged_decode_attention_kernel<float><<<grid, thread_num, smem_size, stream>>>(
        queries.ptr<float>(),
        const_cast<float*>(outputs.ptr<float>()),
        key_pool.ptr<float>(),
        value_pool.ptr<float>(),
        block_tables.ptr<int32_t>(),
        seq_lens.ptr<int32_t>(),
        max_blocks_per_seq, block_size, num_kv_heads, head_size, kv_mul);
  } else {
    CHECK_EQ(queries.data_type(), base::DataType::kDataTypeBf16);
    batched_paged_decode_attention_kernel<base::CudaBF16><<<grid, thread_num, smem_size, stream>>>(
        reinterpret_cast<const base::CudaBF16*>(queries.ptr<uint16_t>()),
        reinterpret_cast<base::CudaBF16*>(const_cast<uint16_t*>(outputs.ptr<uint16_t>())),
        reinterpret_cast<const base::CudaBF16*>(key_pool.ptr<uint16_t>()),
        reinterpret_cast<const base::CudaBF16*>(value_pool.ptr<uint16_t>()),
        block_tables.ptr<int32_t>(),
        seq_lens.ptr<int32_t>(),
        max_blocks_per_seq, block_size, num_kv_heads, head_size, kv_mul);
  }
}

// ============================================================================
// Split-KV Paged Decode Attention (FlashDecoding style)
// Two-pass: partition KV blocks across multiple CUDA blocks, then reduce
// ============================================================================

constexpr static int SPLITKV_THREADS = 256;
constexpr static int MAX_PARTITIONS = 32;

// Pass 1: Each CUDA block handles a partition of KV blocks for one (head, batch)
// grid: dim3(head_num, batch_size, num_partitions)
template <typename T>
__global__ void splitkv_paged_decode_p1_kernel(
    const T* __restrict__ queries,         // [batch, head_num * head_size]
    const T* __restrict__ key_pool,
    const T* __restrict__ value_pool,
    const int32_t* __restrict__ block_tables,  // [batch, max_blocks_per_seq]
    const int32_t* __restrict__ seq_lens,      // [batch]
    float* __restrict__ partial_out,       // [batch, head_num, num_partitions, head_size]
    float* __restrict__ partial_max,       // [batch, head_num, num_partitions]
    float* __restrict__ partial_sum,       // [batch, head_num, num_partitions]
    int32_t max_blocks_per_seq,
    int32_t block_size,
    int32_t num_kv_heads,
    int32_t head_size,
    int32_t kv_mul,
    int32_t num_partitions) {

  const int32_t head = blockIdx.x;
  const int32_t batch_idx = blockIdx.y;
  const int32_t part_idx = blockIdx.z;
  const int32_t kv_head = head / kv_mul;
  const int32_t context_len = seq_lens[batch_idx];

  if (context_len <= 0) {
    // Write identity values for empty partitions
    if (threadIdx.x == 0) {
      int32_t head_num = gridDim.x;
      int64_t meta_idx = (static_cast<int64_t>(batch_idx) * head_num + head) * num_partitions + part_idx;
      partial_max[meta_idx] = -FLT_MAX;
      partial_sum[meta_idx] = 0.f;
    }
    return;
  }

  const int32_t head_num = gridDim.x;
  const int32_t dim = head_num * head_size;
  const int32_t num_kv_blocks = (context_len + block_size - 1) / block_size;
  const int32_t tokens_in_last = context_len - (num_kv_blocks - 1) * block_size;

  // Determine this partition's KV block range
  const int32_t blocks_per_part = (num_kv_blocks + num_partitions - 1) / num_partitions;
  const int32_t kv_block_start = part_idx * blocks_per_part;
  const int32_t kv_block_end = min(kv_block_start + blocks_per_part, num_kv_blocks);

  if (kv_block_start >= num_kv_blocks) {
    // This partition has no work
    if (threadIdx.x == 0) {
      int64_t meta_idx = (static_cast<int64_t>(batch_idx) * head_num + head) * num_partitions + part_idx;
      partial_max[meta_idx] = -FLT_MAX;
      partial_sum[meta_idx] = 0.f;
    }
    return;
  }

  // Shared memory: s_query[head_size] + s_output[head_size]
  extern __shared__ float shared_mem[];
  float* s_query = shared_mem;
  float* s_output = shared_mem + head_size;

  using BlockReduce = cub::BlockReduce<float, SPLITKV_THREADS>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float m, sum_val, alpha_s, p_s;

  const float scale = 1.f / sqrtf(static_cast<float>(head_size));

  // Load query to shared memory
  const T* my_query = queries + batch_idx * dim + head * head_size;
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    s_query[i] = scalar_to_float(my_query[i]);
    s_output[i] = 0.f;
  }
  if (threadIdx.x == 0) {
    m = -FLT_MAX;
    sum_val = 0.f;
  }
  __syncthreads();

  const int64_t pool_block_stride = static_cast<int64_t>(block_size) * num_kv_heads * head_size;
  const int32_t* my_block_table = block_tables + batch_idx * max_blocks_per_seq;

  // Process assigned KV blocks
  for (int32_t blk = kv_block_start; blk < kv_block_end; ++blk) {
    const int32_t physical_block_id = my_block_table[blk];
    const int32_t valid_tokens = (blk == num_kv_blocks - 1) ? tokens_in_last : block_size;

    const int64_t block_offset = physical_block_id * pool_block_stride;
    const T* key_base = key_pool + block_offset + kv_head * head_size;
    const T* value_base = value_pool + block_offset + kv_head * head_size;

    for (int32_t t = 0; t < valid_tokens; ++t) {
      const T* key_ptr = key_base + t * num_kv_heads * head_size;

      // Q · K
      float score = 0.f;
      for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        score += scalar_to_float(key_ptr[i]) * s_query[i];
      }
      score *= scale;
      score = BlockReduce(temp).Sum(score);

      // Online softmax
      if (threadIdx.x == 0) {
        float m_new = fmaxf(score, m);
        alpha_s = expf(m - m_new);
        p_s = expf(score - m_new);
        sum_val = sum_val * alpha_s + p_s;
        m = m_new;
      }
      __syncthreads();

      // V accumulate
      const T* val_ptr = value_base + t * num_kv_heads * head_size;
      for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        s_output[i] = s_output[i] * alpha_s + scalar_to_float(val_ptr[i]) * p_s;
      }
      __syncthreads();
    }
  }

  // Write partial results
  // Note: s_output is unnormalized (= sum_val * weighted_avg), which is what we need for merging
  int64_t meta_idx = (static_cast<int64_t>(batch_idx) * head_num + head) * num_partitions + part_idx;
  float* out_base = partial_out +
      (static_cast<int64_t>(batch_idx) * head_num + head) * num_partitions * head_size
      + part_idx * head_size;

  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    out_base[i] = s_output[i];  // unnormalized weighted sum
  }
  if (threadIdx.x == 0) {
    partial_max[meta_idx] = m;
    partial_sum[meta_idx] = sum_val;
  }
}

// Pass 2: Merge partitions into final output
// grid: dim3(head_num, batch_size), block: (SPLITKV_THREADS)
template <typename T>
__global__ void splitkv_paged_decode_p2_kernel(
    const float* __restrict__ partial_out,  // [batch, head_num, num_partitions, head_size]
    const float* __restrict__ partial_max,  // [batch, head_num, num_partitions]
    const float* __restrict__ partial_sum,  // [batch, head_num, num_partitions]
    T* __restrict__ output,                 // [batch, head_num * head_size]
    int32_t head_size,
    int32_t num_partitions) {

  const int32_t head = blockIdx.x;
  const int32_t batch_idx = blockIdx.y;
  const int32_t head_num = gridDim.x;
  const int32_t dim = head_num * head_size;

  const int64_t base = (static_cast<int64_t>(batch_idx) * head_num + head) * num_partitions;

  // Find global max across all partitions
  float global_max = -FLT_MAX;
  for (int32_t p = 0; p < num_partitions; ++p) {
    float pm = partial_max[base + p];
    if (pm > global_max) global_max = pm;
  }

  // Compute global sum with rescaling
  float global_sum = 0.f;
  for (int32_t p = 0; p < num_partitions; ++p) {
    float pm = partial_max[base + p];
    float ps = partial_sum[base + p];
    if (ps > 0.f) {
      global_sum += ps * expf(pm - global_max);
    }
  }

  // Merge partial outputs
  T* out_head = output + batch_idx * dim + head * head_size;
  const float* partial_base = partial_out +
      (static_cast<int64_t>(batch_idx) * head_num + head) * num_partitions * head_size;

  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    float acc = 0.f;
    for (int32_t p = 0; p < num_partitions; ++p) {
      float pm = partial_max[base + p];
      float ps = partial_sum[base + p];
      if (ps > 0.f) {
        float rescale = expf(pm - global_max);
        acc += partial_base[p * head_size + i] * rescale;
      }
    }
    out_head[i] = float_to_scalar<T>(acc / global_sum);
  }
}

void splitkv_batched_paged_mha_decode_cu(
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

  UNUSED(device_type);
  cudaStream_t stream = config->stream;

  // Adaptive number of partitions
  int32_t num_partitions = min(MAX_PARTITIONS, max_blocks_per_seq);
  if (num_partitions < 1) num_partitions = 1;
  maybe_log_splitkv_decode_choice(
      batch_size,
      max_blocks_per_seq,
      num_partitions,
      head_num,
      head_size,
      kv_mul,
      num_kv_heads,
      queries.data_type());

  int smem_size = 2 * head_size * sizeof(float);

  // Pass 1: partition-level attention
  dim3 grid_p1(head_num, batch_size, num_partitions);

  if (queries.data_type() == base::DataType::kDataTypeFp32) {
    splitkv_paged_decode_p1_kernel<float><<<grid_p1, SPLITKV_THREADS, smem_size, stream>>>(
        queries.ptr<float>(),
        key_pool.ptr<float>(),
        value_pool.ptr<float>(),
        block_tables.ptr<int32_t>(),
        seq_lens.ptr<int32_t>(),
        const_cast<float*>(partial_out.ptr<float>()),
        const_cast<float*>(partial_max.ptr<float>()),
        const_cast<float*>(partial_sum.ptr<float>()),
        max_blocks_per_seq, block_size, num_kv_heads, head_size, kv_mul,
        num_partitions);
  } else {
    CHECK_EQ(queries.data_type(), base::DataType::kDataTypeBf16);
    splitkv_paged_decode_p1_kernel<base::CudaBF16><<<grid_p1, SPLITKV_THREADS, smem_size, stream>>>(
        reinterpret_cast<const base::CudaBF16*>(queries.ptr<uint16_t>()),
        reinterpret_cast<const base::CudaBF16*>(key_pool.ptr<uint16_t>()),
        reinterpret_cast<const base::CudaBF16*>(value_pool.ptr<uint16_t>()),
        block_tables.ptr<int32_t>(),
        seq_lens.ptr<int32_t>(),
        const_cast<float*>(partial_out.ptr<float>()),
        const_cast<float*>(partial_max.ptr<float>()),
        const_cast<float*>(partial_sum.ptr<float>()),
        max_blocks_per_seq, block_size, num_kv_heads, head_size, kv_mul,
        num_partitions);
  }

  // Pass 2: reduce partitions
  dim3 grid_p2(head_num, batch_size);

  if (queries.data_type() == base::DataType::kDataTypeFp32) {
    splitkv_paged_decode_p2_kernel<float><<<grid_p2, SPLITKV_THREADS, 0, stream>>>(
        partial_out.ptr<float>(),
        partial_max.ptr<float>(),
        partial_sum.ptr<float>(),
        const_cast<float*>(outputs.ptr<float>()),
        head_size, num_partitions);
  } else {
    splitkv_paged_decode_p2_kernel<base::CudaBF16><<<grid_p2, SPLITKV_THREADS, 0, stream>>>(
        partial_out.ptr<float>(),
        partial_max.ptr<float>(),
        partial_sum.ptr<float>(),
        reinterpret_cast<base::CudaBF16*>(const_cast<uint16_t*>(outputs.ptr<uint16_t>())),
        head_size, num_partitions);
  }
}

}  // namespace kernel
