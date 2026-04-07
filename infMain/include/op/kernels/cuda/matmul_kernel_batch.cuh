// Batched matmul kernel using cuBLAS
#ifndef MATMUL_KERNEL_BATCH_CUH
#define MATMUL_KERNEL_BATCH_CUH
#include <base/cuda_config.h>
#include <tensor/tensor.h>

namespace kernel {

// Y[M, N] = X[M, K] × W^T[K, N]  (row-major)
// input: [batch_tokens, in_dim], weight: [out_dim, in_dim], output: [batch_tokens, out_dim]
void matmul_batch_kernel_cu(const tensor::Tensor& input,
                            const tensor::Tensor& weight,
                            const tensor::Tensor& output,
                            int32_t batch_tokens,
                            const CudaConfig* config);

}  // namespace kernel

#endif  // MATMUL_KERNEL_BATCH_CUH
