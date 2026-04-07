// Broadcast bias add kernel for batched matmul output
#ifndef ADD_BIAS_KERNEL_CUH
#define ADD_BIAS_KERNEL_CUH
#include <base/cuda_config.h>
#include <tensor/tensor.h>

namespace kernel {

// output[b, i] += bias[i] for b in [0, batch_tokens)
void add_bias_kernel_cu(const tensor::Tensor& output,  // [batch_tokens, dim]
                        const tensor::Tensor& bias,     // [dim]
                        int32_t batch_tokens, int32_t dim, void* stream);

}  // namespace kernel

#endif  // ADD_BIAS_KERNEL_CUH
