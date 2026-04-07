// Paged Multi-Head Attention Kernel for PagedAttention
#ifndef KUIPER_INCLUDE_OP_KERNELS_CUDA_PAGED_MHA_KERNEL_CUH_
#define KUIPER_INCLUDE_OP_KERNELS_CUDA_PAGED_MHA_KERNEL_CUH_

#include "base/base.h"
#include "base/cuda_config.h"
#include "tensor/tensor.h"

namespace kernel {

// Single-sequence paged decode attention
// Each query attends to KV stored in paged blocks
void paged_mha_decode_cu(
    int32_t head_num,
    int32_t head_size,
    int32_t kv_mul,
    const tensor::Tensor& query,           // [head_num * head_size]
    const tensor::Tensor& output,          // [head_num * head_size]
    const tensor::Tensor& key_pool,        // [num_blocks, num_layers, block_size * num_kv_heads * head_size]
    const tensor::Tensor& value_pool,      // same layout
    const int32_t* block_table_gpu,        // [num_kv_blocks] on GPU
    int32_t num_kv_blocks,
    int32_t num_tokens_in_last_block,
    int32_t block_size,
    int32_t num_kv_heads,
    base::DeviceType device_type,
    CudaConfig* config);

// Batched paged decode attention: multiple sequences in parallel
void batched_paged_mha_decode_cu(
    int32_t batch_size,
    int32_t head_num,
    int32_t head_size,
    int32_t kv_mul,
    const tensor::Tensor& queries,       // [batch_size, head_num * head_size]
    const tensor::Tensor& outputs,       // [batch_size, head_num * head_size]
    const tensor::Tensor& key_pool,      // [num_blocks, block_size * num_kv_heads * head_size]
    const tensor::Tensor& value_pool,
    const tensor::Tensor& block_tables,  // [batch_size, max_blocks_per_seq] int32 GPU
    const tensor::Tensor& seq_lens,      // [batch_size] int32 GPU
    int32_t max_blocks_per_seq,
    int32_t block_size,
    int32_t num_kv_heads,
    base::DeviceType device_type,
    CudaConfig* config);

// Split-KV batched paged decode attention (FlashDecoding style)
// Partitions KV blocks across multiple CUDA blocks for parallel processing
// Requires pre-allocated workspace buffers for partial results
void splitkv_batched_paged_mha_decode_cu(
    int32_t batch_size,
    int32_t head_num,
    int32_t head_size,
    int32_t kv_mul,
    const tensor::Tensor& queries,       // [batch_size, head_num * head_size]
    const tensor::Tensor& outputs,       // [batch_size, head_num * head_size]
    const tensor::Tensor& key_pool,
    const tensor::Tensor& value_pool,
    const tensor::Tensor& block_tables,  // [batch_size, max_blocks_per_seq] int32 GPU
    const tensor::Tensor& seq_lens,      // [batch_size] int32 GPU
    int32_t max_blocks_per_seq,
    int32_t block_size,
    int32_t num_kv_heads,
    // Workspace (all fp32, pre-allocated)
    const tensor::Tensor& partial_out,   // [batch, head_num, MAX_PARTITIONS, head_size]
    const tensor::Tensor& partial_max,   // [batch, head_num, MAX_PARTITIONS]
    const tensor::Tensor& partial_sum,   // [batch, head_num, MAX_PARTITIONS]
    base::DeviceType device_type,
    CudaConfig* config);

}  // namespace kernel

#endif  // KUIPER_INCLUDE_OP_KERNELS_CUDA_PAGED_MHA_KERNEL_CUH_
