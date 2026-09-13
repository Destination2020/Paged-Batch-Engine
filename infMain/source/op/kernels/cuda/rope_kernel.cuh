// Updated on March 15, 2026
#ifndef ROPE_KERNEL_CU_CUH
#define ROPE_KERNEL_CU_CUH
#include "tensor/tensor.h"
namespace kernel {
void rope_kernel_cu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
                    const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
                    const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache, void* stream);

void sin_cos_cache_calc_cu(int head_size, int max_seq_len, const tensor::Tensor& sin_cache,
                           const tensor::Tensor& cos_cache, void* stream);

// Batched RoPE: processes batch_tokens tokens with individual positions
void rope_kernel_batched_cu(int32_t dim, int32_t kv_dim, int32_t head_size,
                            const tensor::Tensor& input_q,     // [batch_tokens, dim]
                            const tensor::Tensor& input_k,     // [batch_tokens, kv_dim]
                            const tensor::Tensor& positions,   // [batch_tokens] int32 on GPU
                            const tensor::Tensor& sin_cache,
                            const tensor::Tensor& cos_cache,
                            int32_t batch_tokens, void* stream);

void mrope_kernel_batched_cu(int32_t dim, int32_t kv_dim, int32_t head_size,
                             const tensor::Tensor& input_q,
                             const tensor::Tensor& input_k,
                             const tensor::Tensor& positions,
                             const tensor::Tensor& sin_cache,
                             const tensor::Tensor& cos_cache,
                             int32_t temporal_section,
                             int32_t height_section,
                             int32_t batch_tokens, void* stream);

}  // namespace kernel
#endif  // ROPE_KERNEL_CU_CUH
