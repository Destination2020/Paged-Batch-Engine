// Scatter KV to paged blocks kernel
#ifndef KUIPER_INCLUDE_OP_KERNELS_CUDA_SCATTER_KV_KERNEL_CUH_
#define KUIPER_INCLUDE_OP_KERNELS_CUDA_SCATTER_KV_KERNEL_CUH_

#include "base/base.h"
#include "base/cuda_config.h"
#include "tensor/tensor.h"

namespace kernel {

// Scatter a single token's K/V to paged cache
// Used after computing Wk(x) and Wv(x) in forward pass
void scatter_kv_to_page_cu(
    const tensor::Tensor& key_tensor,      // [num_kv_heads * head_size]
    const tensor::Tensor& value_tensor,    // [num_kv_heads * head_size]
    tensor::Tensor& key_pool,              // [num_blocks, num_layers, block_size * num_kv_heads * head_size]
    tensor::Tensor& value_pool,
    int32_t physical_block_id,
    int32_t offset_in_block,
    int32_t block_size,
    int32_t num_kv_heads,
    int32_t head_size,
    base::DeviceType device_type,
    CudaConfig* config);

// Batched scatter: write [batch_tokens, kv_dim] K/V to paged pool via slot_mapping
void scatter_kv_batch_to_pages_cu(
    const tensor::Tensor& key_tensor,     // [batch_tokens, kv_dim]
    const tensor::Tensor& value_tensor,
    tensor::Tensor& key_pool,
    tensor::Tensor& value_pool,
    const tensor::Tensor& slot_mapping,   // [batch_tokens] int32 GPU: slot=block_id*block_size+offset
    int32_t block_size,
    int32_t num_kv_heads,
    int32_t head_size,
    int32_t batch_tokens,
    base::DeviceType device_type,
    CudaConfig* config);

// Batched scatter with per-(token, kv_head) FP8 E4M3 quantization.
void scatter_kv_batch_to_pages_fp8_e4m3_cu(
    const tensor::Tensor& key_tensor,     // [batch_tokens, kv_dim], bf16/fp32
    const tensor::Tensor& value_tensor,
    tensor::Tensor& key_pool,             // raw fp8 bit-patterns stored as int8
    tensor::Tensor& value_pool,
    tensor::Tensor& key_scale_pool,       // [num_blocks, block_size * num_kv_heads], fp32
    tensor::Tensor& value_scale_pool,
    const tensor::Tensor& slot_mapping,   // [batch_tokens] int32 GPU: slot=block_id*block_size+offset
    int32_t block_size,
    int32_t num_kv_heads,
    int32_t head_size,
    int32_t batch_tokens,
    base::DeviceType device_type,
    CudaConfig* config);

}  // namespace kernel

#endif  // KUIPER_INCLUDE_OP_KERNELS_CUDA_SCATTER_KV_KERNEL_CUH_


