// Updated on March 15, 2026
#ifndef LLAMA_INFER_MATMUL_KERNEL_H
#define LLAMA_INFER_MATMUL_KERNEL_H
#include "tensor/tensor.h"
namespace base {
struct DeviceContext;
}

namespace kernel {
void matmul_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, float scale = 1.f,
                       const base::DeviceContext* context = nullptr);
}  // namespace kernel
#endif  // LLAMA_INFER_MATMUL_KERNEL_H
