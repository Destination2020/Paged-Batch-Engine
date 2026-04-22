// Chunked prefill paged MHA kernel
#ifndef KUIPER_INCLUDE_OP_KERNELS_CUDA_PAGED_MHA_PREFILL_KERNEL_CUH_
#define KUIPER_INCLUDE_OP_KERNELS_CUDA_PAGED_MHA_PREFILL_KERNEL_CUH_

#include "base/base.h"
#include "base/cuda_config.h"
#include "tensor/tensor.h"

namespace kernel {

// Batched chunked-prefill attention.
// Each row is one token inside a prefill chunk. A row attends to:
// 1. prefix KV already materialized in paged cache
// 2. chunk-local K/V up to and including the current token
void batched_paged_mha_prefill_cu(
    int32_t batch_size,
    int32_t head_num,
    int32_t head_size,
    int32_t kv_mul,
    const tensor::Tensor& queries,             // [batch_size, head_num * head_size]
    const tensor::Tensor& chunk_keys,          // [batch_size, num_kv_heads * head_size]
    const tensor::Tensor& chunk_values,        // [batch_size, num_kv_heads * head_size]
    tensor::Tensor& outputs,                   // [batch_size, head_num * head_size]
    const tensor::Tensor& key_pool,
    const tensor::Tensor& value_pool,
    const tensor::Tensor& block_tables,        // [num_prefill_requests, max_blocks_per_seq] int32 GPU
    const tensor::Tensor& request_indices,     // [batch_size] int32 GPU, local prefill request idx
    const tensor::Tensor& base_context_lens,   // [batch_size] int32 GPU
    const tensor::Tensor& chunk_row_starts,    // [batch_size] int32 GPU
    const tensor::Tensor& local_token_offsets, // [batch_size] int32 GPU
    tensor::Tensor& partial_out,               // fp32 workspace
    tensor::Tensor& partial_max,               // fp32 workspace
    tensor::Tensor& partial_sum,               // fp32 workspace
    int32_t max_blocks_per_seq,
    int32_t max_prefix_blocks,
    int32_t block_size,
    int32_t num_kv_heads,
    base::DeviceType device_type,
    CudaConfig* config);

void batched_paged_mha_prefill_fp8_cu(
    int32_t batch_size,
    int32_t head_num,
    int32_t head_size,
    int32_t kv_mul,
    const tensor::Tensor& queries,             // [batch_size, head_num * head_size]
    const tensor::Tensor& chunk_keys,          // [batch_size, num_kv_heads * head_size]
    const tensor::Tensor& chunk_values,        // [batch_size, num_kv_heads * head_size]
    tensor::Tensor& outputs,                   // [batch_size, head_num * head_size]
    const tensor::Tensor& key_pool,            // fp8/int8
    const tensor::Tensor& value_pool,          // fp8/int8
    const tensor::Tensor& key_scale_pool,      // fp32
    const tensor::Tensor& value_scale_pool,    // fp32
    const tensor::Tensor& block_tables,        // [num_prefill_requests, max_blocks_per_seq] int32 GPU
    const tensor::Tensor& request_indices,     // [batch_size] int32 GPU
    const tensor::Tensor& base_context_lens,   // [batch_size] int32 GPU
    const tensor::Tensor& chunk_row_starts,    // [batch_size] int32 GPU
    const tensor::Tensor& local_token_offsets, // [batch_size] int32 GPU
    tensor::Tensor& partial_out,               // fp32 workspace
    tensor::Tensor& partial_max,               // fp32 workspace
    tensor::Tensor& partial_sum,               // fp32 workspace
    int32_t max_blocks_per_seq,
    int32_t max_prefix_blocks,
    int32_t block_size,
    int32_t num_kv_heads,
    base::DeviceType device_type,
    CudaConfig* config);

}  // namespace kernel

#endif  // KUIPER_INCLUDE_OP_KERNELS_CUDA_PAGED_MHA_PREFILL_KERNEL_CUH_
