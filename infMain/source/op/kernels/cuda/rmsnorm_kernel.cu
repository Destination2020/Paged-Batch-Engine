// Updated on March 23, 2026
#include <device_launch_parameters.h>
#include <cub/block/block_reduce.cuh>
#include "cuda_type_utils.cuh"
#include "rmsnorm_kernel.cuh"

namespace kernel {

template <typename T>
static __global__ void row_rmsnorm_dim_impl(const T* in, const T* wei, T* out, int dim_size,
                                            int size, float eps) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  if (bid >= dim_size) {
    return;
  }

  const T* block_in = in + bid * size;
  T* block_out = out + bid * size;

  float sum = 0.0f;
  for (int i = tid; i < size; i += blockDim.x) {
    const float value = scalar_to_float(block_in[i]);
    sum += value * value;
  }

  using BlockReduce = cub::BlockReduce<float, 128>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  const float scale = rsqrtf(shared_val / static_cast<float>(size) + eps);

  for (int i = tid; i < size; i += blockDim.x) {
    const float value = scalar_to_float(wei[i]) * scalar_to_float(block_in[i]) * scale;
    block_out[i] = float_to_scalar<T>(value);
  }
}

template <typename T, int32_t BLOCK_DIM>
static __global__ void row_rmsnorm_impl(const T* in, const T* wei, T* out, int size, float eps) {
  const int tid = threadIdx.x;

  float sum = 0.0f;
  for (int i = tid; i < size; i += blockDim.x) {
    const float value = scalar_to_float(in[i]);
    sum += value * value;
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  const float scale = rsqrtf(shared_val / static_cast<float>(size) + eps);

  for (int i = tid; i < size; i += blockDim.x) {
    const float value = scalar_to_float(wei[i]) * scalar_to_float(in[i]) * scale;
    out[i] = float_to_scalar<T>(value);
  }
}

void rmsnorm_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
        weight.device_type() == base::DeviceType::kDeviceCUDA &&
        output.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK_EQ(input.data_type(), weight.data_type());
  CHECK_EQ(input.data_type(), output.data_type());

#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT) || defined(QWEN_MOE_SUPPORT)
  const float eps = 1e-6f;
#else
  const float eps = 1e-5f;
#endif
  const int32_t size = static_cast<int32_t>(input.size());
  constexpr int threads_num = 128;
  cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;

  if (input.data_type() == base::DataType::kDataTypeFp32) {
    if (stream_) {
      row_rmsnorm_impl<float, 128><<<1, threads_num, 0, stream_>>>(
          input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), size, eps);
    } else {
      row_rmsnorm_impl<float, 128><<<1, threads_num>>>(
          input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), size, eps);
    }
  } else {
    CHECK_EQ(input.data_type(), base::DataType::kDataTypeBf16);
    if (stream_) {
      row_rmsnorm_impl<base::CudaBF16, 128><<<1, threads_num, 0, stream_>>>(
          reinterpret_cast<const base::CudaBF16*>(input.ptr<uint16_t>()),
          reinterpret_cast<const base::CudaBF16*>(weight.ptr<uint16_t>()),
          reinterpret_cast<base::CudaBF16*>(const_cast<uint16_t*>(output.ptr<uint16_t>())), size,
          eps);
    } else {
      row_rmsnorm_impl<base::CudaBF16, 128><<<1, threads_num>>>(
          reinterpret_cast<const base::CudaBF16*>(input.ptr<uint16_t>()),
          reinterpret_cast<const base::CudaBF16*>(weight.ptr<uint16_t>()),
          reinterpret_cast<base::CudaBF16*>(const_cast<uint16_t*>(output.ptr<uint16_t>())), size,
          eps);
    }
  }
}

void rmsnorm_kernel_cu_dim(const tensor::Tensor& input, const tensor::Tensor& weight,
                           const tensor::Tensor& output, int32_t dim, void* stream) {
  UNUSED(dim);
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
        weight.device_type() == base::DeviceType::kDeviceCUDA &&
        output.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK_EQ(input.data_type(), weight.data_type());
  CHECK_EQ(input.data_type(), output.data_type());

  const float eps = 1e-6f;
  const int32_t total_size = static_cast<int32_t>(input.size());
  const int32_t size = input.get_dim(input.dims_size() - 1);
  const int32_t dim_size = total_size / size;
  constexpr int threads_num = 128;
  cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;

  if (input.data_type() == base::DataType::kDataTypeFp32) {
    if (stream_) {
      row_rmsnorm_dim_impl<float><<<dim_size, threads_num, 0, stream_>>>(
          input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), dim_size,
          size, eps);
    } else {
      row_rmsnorm_dim_impl<float><<<dim_size, threads_num>>>(
          input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), dim_size,
          size, eps);
    }
  } else {
    CHECK_EQ(input.data_type(), base::DataType::kDataTypeBf16);
    if (stream_) {
      row_rmsnorm_dim_impl<base::CudaBF16><<<dim_size, threads_num, 0, stream_>>>(
          reinterpret_cast<const base::CudaBF16*>(input.ptr<uint16_t>()),
          reinterpret_cast<const base::CudaBF16*>(weight.ptr<uint16_t>()),
          reinterpret_cast<base::CudaBF16*>(const_cast<uint16_t*>(output.ptr<uint16_t>())),
          dim_size, size, eps);
    } else {
      row_rmsnorm_dim_impl<base::CudaBF16><<<dim_size, threads_num>>>(
          reinterpret_cast<const base::CudaBF16*>(input.ptr<uint16_t>()),
          reinterpret_cast<const base::CudaBF16*>(weight.ptr<uint16_t>()),
          reinterpret_cast<base::CudaBF16*>(const_cast<uint16_t*>(output.ptr<uint16_t>())),
          dim_size, size, eps);
    }
  }
}

}  // namespace kernel
