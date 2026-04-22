// CUDA kernels for mixed-batch logits row gathering and greedy sampling.
#ifndef KUIPER_INCLUDE_OP_KERNELS_CUDA_SAMPLER_KERNEL_CUH_
#define KUIPER_INCLUDE_OP_KERNELS_CUDA_SAMPLER_KERNEL_CUH_

#include "tensor/tensor.h"

namespace kernel {

void gather_logits_rows_cu(const tensor::Tensor& logits,
                           const tensor::Tensor& row_indices,
                           tensor::Tensor& gathered_logits,
                           void* stream);

void argmax_rows_cu(const tensor::Tensor& logits,
                    tensor::Tensor& token_ids,
                    void* stream);

void argmax_selected_rows_cu(const tensor::Tensor& logits,
                             const tensor::Tensor& row_indices,
                             tensor::Tensor& token_ids,
                             void* stream);

void sample_argmax_rows_cu(const tensor::Tensor& logits,
                           const tensor::Tensor& row_indices,
                           tensor::Tensor& gathered_logits,
                           tensor::Tensor& token_ids,
                           void* stream);

}  // namespace kernel

#endif  // KUIPER_INCLUDE_OP_KERNELS_CUDA_SAMPLER_KERNEL_CUH_
