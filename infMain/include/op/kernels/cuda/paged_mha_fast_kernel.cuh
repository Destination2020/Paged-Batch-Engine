// Fast-path paged MHA decode kernel
#ifndef KUIPER_INCLUDE_OP_KERNELS_CUDA_PAGED_MHA_FAST_KERNEL_CUH_
#define KUIPER_INCLUDE_OP_KERNELS_CUDA_PAGED_MHA_FAST_KERNEL_CUH_

#include "base/base.h"
#include "base/cuda_config.h"
#include "tensor/tensor.h"

namespace kernel {

// Returns true when the fast path is launched, false when the caller should
// fall back to the legacy split-KV implementation.
bool splitkv_batched_paged_mha_fast_decode_cu(
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
    const tensor::Tensor& partial_out,   // Opaque scratch workspace
    const tensor::Tensor& partial_max,   // Opaque scratch workspace
    const tensor::Tensor& partial_sum,   // Opaque scratch workspace
    base::DeviceType device_type,
    CudaConfig* config);

// Batch-only FP8 KV cache decode path.
bool splitkv_batched_paged_mha_fp8_decode_cu(
    int32_t batch_size,
    int32_t head_num,
    int32_t head_size,
    int32_t kv_mul,
    const tensor::Tensor& queries,       // [batch_size, head_num * head_size], bf16
    const tensor::Tensor& outputs,       // [batch_size, head_num * head_size], bf16
    const tensor::Tensor& key_pool,      // raw fp8 bytes stored in int8 tensor
    const tensor::Tensor& value_pool,
    const tensor::Tensor& key_scale_pool,
    const tensor::Tensor& value_scale_pool,
    const tensor::Tensor& block_tables,  // [batch_size, max_blocks_per_seq] int32 GPU
    const tensor::Tensor& seq_lens,      // [batch_size] int32 GPU
    int32_t max_blocks_per_seq,
    int32_t block_size,
    int32_t num_kv_heads,
    const tensor::Tensor& partial_out,   // Opaque scratch workspace
    const tensor::Tensor& partial_max,   // Opaque scratch workspace
    const tensor::Tensor& partial_sum,   // Opaque scratch workspace
    base::DeviceType device_type,
    CudaConfig* config);

}  // namespace kernel

#endif  // KUIPER_INCLUDE_OP_KERNELS_CUDA_PAGED_MHA_FAST_KERNEL_CUH_
