// Updated on March 15, 2026
#ifndef MATMUL_KERNEL_CU_CUH
#define MATMUL_KERNEL_CU_CUH
#include "tensor/tensor.h"
namespace kernel {
struct CudaConfig;

void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, float scale = 1.f,
                      const CudaConfig* config = nullptr);

void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, int32_t group_size,
                            const tensor::Tensor& scale, const CudaConfig* config = nullptr);
}  // namespace kernel

#endif  // MATMUL_KERNEL_CU_CUH
