// Updated on March 15, 2026
#include <tensor/tensor.h>
#include "swiglu_kernel.cuh"
#include "cuda_type_utils.cuh"
namespace kernel {
template <typename T>
__global__ void swiglu_kernel_cu_impl(int size, const T* in1, const T* in2, T* out) {
  int tid = threadIdx.x;
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= size) {
    return;
  }
  extern __shared__ float shared_mem[];
  float* smem1 = shared_mem;
  float* smem2 = shared_mem + blockDim.x;

  smem1[tid] = scalar_to_float(in1[idx]);
  smem2[tid] = scalar_to_float(in2[idx]);
  __syncthreads();

  float value = 1.0f / (1.0f + expf(-smem1[tid]));
  smem1[tid] = smem1[tid] * value;

  out[idx] = float_to_scalar<T>(smem1[tid] * smem2[tid]);
}

void swiglu_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                      const tensor::Tensor& output, void* stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK(input1.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK_EQ(input2.is_empty(), false);
  CHECK(input2.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK_EQ(output.is_empty(), false);
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

  int size = static_cast<int32_t>(input1.size());
  int threads = 128;
  int blocks = (size + threads - 1) / threads;
  const size_t shmem = threads * sizeof(float) * 2;
  const bool is_fp32 = input1.data_type() == base::DataType::kDataTypeFp32;
  if (!stream) {
    if (is_fp32) {
      swiglu_kernel_cu_impl<float><<<blocks, threads, shmem>>>(
          size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
    } else {
      swiglu_kernel_cu_impl<base::CudaBF16><<<blocks, threads, shmem>>>(
          size, reinterpret_cast<const base::CudaBF16*>(input1.ptr<uint16_t>()),
          reinterpret_cast<const base::CudaBF16*>(input2.ptr<uint16_t>()),
          reinterpret_cast<base::CudaBF16*>(const_cast<uint16_t*>(output.ptr<uint16_t>())));
    }
  } else {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    if (is_fp32) {
      swiglu_kernel_cu_impl<float><<<blocks, threads, shmem, stream_>>>(
          size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
    } else {
      swiglu_kernel_cu_impl<base::CudaBF16><<<blocks, threads, shmem, stream_>>>(
          size, reinterpret_cast<const base::CudaBF16*>(input1.ptr<uint16_t>()),
          reinterpret_cast<const base::CudaBF16*>(input2.ptr<uint16_t>()),
          reinterpret_cast<base::CudaBF16*>(const_cast<uint16_t*>(output.ptr<uint16_t>())));
    }
  }
}
}  // namespace kernel
