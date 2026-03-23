#ifndef MOE_KERNEL_CUH
#define MOE_KERNEL_CUH
#include "tensor/tensor.h"
#include <base/cuda_config.h>

namespace kernel {
void moe_router_softmax_topk_cu(tensor::Tensor& router_logits, int32_t num_experts, int32_t topk,
                                tensor::Tensor& topk_values, tensor::Tensor& topk_indices,
                                bool norm_topk_prob, CudaConfig* config = nullptr);

void moe_scale_add_cu(tensor::Tensor& input_tensor, const tensor::Tensor& expert_output, float scale,
                      CudaConfig* config = nullptr);
}

#endif  // MOE_KERNEL_CUH
