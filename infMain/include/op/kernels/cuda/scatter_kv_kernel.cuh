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

}  // namespace kernel

#endif  // KUIPER_INCLUDE_OP_KERNELS_CUDA_SCATTER_KV_KERNEL_CUH_



