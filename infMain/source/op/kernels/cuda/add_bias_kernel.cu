// Broadcast bias add kernel implementation
#include "op/kernels/cuda/add_bias_kernel.cuh"
#include "cuda_type_utils.cuh"

namespace kernel {

template <typename T>
__global__ void add_bias_kernel_impl(T* output, const T* bias,
                                     int32_t batch_tokens, int32_t dim) {
  int32_t b = blockIdx.x;
  if (b >= batch_tokens) return;

  T* row = output + b * dim;
  for (int32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    float val = scalar_to_float(row[i]) + scalar_to_float(bias[i]);
    row[i] = float_to_scalar<T>(val);
  }
}

void add_bias_kernel_cu(const tensor::Tensor& output,
                        const tensor::Tensor& bias,
                        int32_t batch_tokens, int32_t dim, void* stream) {
  cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
  int32_t threads = 256;

  if (output.data_type() == base::DataType::kDataTypeFp32) {
    add_bias_kernel_impl<float><<<batch_tokens, threads, 0, stream_>>>(
        const_cast<float*>(output.ptr<float>()),
        bias.ptr<float>(),
        batch_tokens, dim);
  } else {
    CHECK_EQ(output.data_type(), base::DataType::kDataTypeBf16);
    add_bias_kernel_impl<base::CudaBF16><<<batch_tokens, threads, 0, stream_>>>(
        reinterpret_cast<base::CudaBF16*>(const_cast<uint16_t*>(output.ptr<uint16_t>())),
        reinterpret_cast<const base::CudaBF16*>(bias.ptr<uint16_t>()),
        batch_tokens, dim);
  }
}

}  // namespace kernel
