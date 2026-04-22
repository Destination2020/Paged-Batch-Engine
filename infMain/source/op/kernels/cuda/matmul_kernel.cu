// Updated on March 23, 2026
#include <base/cuda_config.h>
#include <tensor/tensor.h>
#include <cub/block/block_reduce.cuh>
#include "../kernels_interface.h"
#include "cuda_type_utils.cuh"
#include "matmul_kernel.cuh"

namespace kernel {

template <typename InputT, typename WeightT, typename OutputT, int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_impl(const InputT* input, const WeightT* weight, OutputT* output,
                                      int M, int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  const unsigned int tid = threadIdx.x;

  const int start_row = blockIdx.x * ROW_PER_BLOCK;
  const int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }

#pragma unroll
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0.f;
    const int row_offset = p * M;

    for (int i = tid; i < M; i += THREAD_PER_BLOCK) {
      sdata[tid] += scalar_to_float(input[i]) * scalar_to_float(weight[row_offset + i]);
    }
    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    const float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = float_to_scalar<OutputT>(part_sum);
    }
    __syncthreads();
  }
}

template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32int8(const float* input, const int8_t* weight,
                                          const float* scales, const int32_t group_size,
                                          float* output, int M, int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0;
    for (int i = tid; i < M; i += THREAD_PER_BLOCK) {
      const int weight_idx = p * M + i;
      const int group_idx = weight_idx / group_size;
      sdata[tid] += input[i] * scales[group_idx] * static_cast<float>(weight[weight_idx]);
    }
    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, const float scale, const CudaConfig* config) {
  UNUSED(scale);
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK_EQ(input.data_type(), weight.data_type());
  CHECK_EQ(input.data_type(), output.data_type());

  const int32_t K = weight.get_dim(0);
  const int32_t M = weight.get_dim(1);
  CHECK_EQ(M, input.get_dim(0));

  cudaStream_t stream = config ? config->stream : nullptr;
  if (input.data_type() == base::DataType::kDataTypeFp32) {
    if (stream) {
      matmul_kernel_cu_impl<float, float, float, 128, 1><<<K, 128, 0, stream>>>(
          input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), M, K);
    } else {
      matmul_kernel_cu_impl<float, float, float, 128, 1><<<K, 128>>>(
          input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), M, K);
    }
  } else {
    CHECK_EQ(input.data_type(), base::DataType::kDataTypeBf16);
    auto input_ptr = reinterpret_cast<const base::CudaBF16*>(input.ptr<uint16_t>());
    auto weight_ptr = reinterpret_cast<const base::CudaBF16*>(weight.ptr<uint16_t>());
    auto output_ptr = reinterpret_cast<base::CudaBF16*>(const_cast<uint16_t*>(output.ptr<uint16_t>()));
    if (stream) {
      matmul_kernel_cu_impl<base::CudaBF16, base::CudaBF16, base::CudaBF16, 128, 1>
          <<<K, 128, 0, stream>>>(input_ptr, weight_ptr, output_ptr, M, K);
    } else {
      matmul_kernel_cu_impl<base::CudaBF16, base::CudaBF16, base::CudaBF16, 128, 1>
          <<<K, 128>>>(input_ptr, weight_ptr, output_ptr, M, K);
    }
  }
}

void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, int32_t group_size,
                            const tensor::Tensor& scale, const CudaConfig* config) {
  CHECK(config != nullptr);
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t K = weight.get_dim(0);
  const int32_t M = weight.get_dim(1);
  CHECK_EQ(M, input.get_dim(0));
  CHECK_EQ(input.data_type(), base::DataType::kDataTypeFp32);
  CHECK_EQ(output.data_type(), base::DataType::kDataTypeFp32);

  if (config->stream) {
    matmul_kernel_cu_fp32int8<128, 1><<<K, 128, 0, config->stream>>>(
        input.ptr<float>(), weight.ptr<int8_t>(), scale.ptr<float>(), group_size,
        const_cast<float*>(output.ptr<float>()), M, K);
  } else {
    matmul_kernel_cu_fp32int8<128, 1><<<K, 128>>>(
        input.ptr<float>(), weight.ptr<int8_t>(), scale.ptr<float>(), group_size,
        const_cast<float*>(output.ptr<float>()), M, K);
  }
}

}  // namespace kernel
