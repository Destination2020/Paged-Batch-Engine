// Updated on March 15, 2026
#include "add_kernel.cuh"
#include "cuda_type_utils.cuh"

namespace kernel {
template <typename T>
__global__ void add_kernel_cu_impl(int32_t size, const T* in1, const T* in2, T* out) {
  int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size) {
    return;
  }
  float in_val1 = scalar_to_float(in1[tid]);
  float in_val2 = scalar_to_float(in2[tid]);
  out[tid] = float_to_scalar<T>(in_val1 + in_val2);
}

void add_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                   const tensor::Tensor& output, void* stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK_EQ(input2.is_empty(), false);
  CHECK_EQ(output.is_empty(), false);
  int32_t size = static_cast<int32_t>(input1.size());
  CHECK_EQ(size, input2.size());
  CHECK_EQ(size, output.size());
  int32_t thread_num = 512;
  int32_t block_num = (size + thread_num - 1) / thread_num;
  cudaStream_t stream_ = stream ? static_cast<CUstream_st*>(stream) : nullptr;
  if (stream) {
    if (input1.data_type() == base::DataType::kDataTypeFp32) {
      add_kernel_cu_impl<float><<<block_num, thread_num, 0, stream_>>>(
          size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
    } else {
      add_kernel_cu_impl<base::CudaBF16><<<block_num, thread_num, 0, stream_>>>(
          size, reinterpret_cast<const base::CudaBF16*>(input1.ptr<uint16_t>()),
          reinterpret_cast<const base::CudaBF16*>(input2.ptr<uint16_t>()),
          reinterpret_cast<base::CudaBF16*>(const_cast<uint16_t*>(output.ptr<uint16_t>())));
    }
  } else if (input1.data_type() == base::DataType::kDataTypeFp32) {
    add_kernel_cu_impl<float><<<block_num, thread_num>>>(
        size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
  } else {
    add_kernel_cu_impl<base::CudaBF16><<<block_num, thread_num>>>(
        size, reinterpret_cast<const base::CudaBF16*>(input1.ptr<uint16_t>()),
        reinterpret_cast<const base::CudaBF16*>(input2.ptr<uint16_t>()),
        reinterpret_cast<base::CudaBF16*>(const_cast<uint16_t*>(output.ptr<uint16_t>())));
  }
}
}  // namespace kernel
