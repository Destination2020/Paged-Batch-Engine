// Updated on March 15, 2026
#include "emb_kernel.cuh"
#include "cuda_type_utils.cuh"
namespace kernel {
template <typename T>
__global__ void emb_kernel_cu_impl(int32_t vocab_size, int32_t token_num, int32_t weight_dim,
                                   const int32_t* input_ptr, const T* weight_ptr, T* output_ptr) {
  int32_t token_idx = blockIdx.x;
  if (token_idx >= token_num) {
    return;
  }
  int32_t token = input_ptr[token_idx];
  if (token >= vocab_size) {
    return;
  }

  T* output_ptr_start = output_ptr + token_idx * weight_dim;
  const T* weight_ptr_start = weight_ptr + token * weight_dim;

  for (int32_t i = threadIdx.x; i < weight_dim; i += blockDim.x) {
    output_ptr_start[i] = weight_ptr_start[i];
  }
}

void emb_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                   const tensor::Tensor& output, int32_t vocab_size, void* stream) {
  tensor::Tensor input_cu;
  if (input.device_type() != base::DeviceType::kDeviceCUDA) {
    input_cu = input.clone();
    input_cu.to_cuda();
  }
  const int32_t input_num = static_cast<int32_t>(input.size());
  const int32_t weight_dim = weight.get_dim(1);
  CHECK(weight.device_type() == output.device_type());
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

  constexpr int32_t max_seq_len = 512;
  constexpr int32_t thread_num = 128;
  int32_t* in_ptr = input_cu.ptr<int32_t>();
  cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
  if (stream) {
    if (weight.data_type() == base::DataType::kDataTypeFp32) {
      emb_kernel_cu_impl<float><<<max_seq_len, thread_num, 0, stream_>>>(
          vocab_size, input_num, weight_dim, in_ptr, weight.ptr<float>(),
          const_cast<float*>(output.ptr<float>()));
    } else {
      emb_kernel_cu_impl<base::CudaBF16><<<max_seq_len, thread_num, 0, stream_>>>(
          vocab_size, input_num, weight_dim, in_ptr,
          reinterpret_cast<const base::CudaBF16*>(weight.ptr<uint16_t>()),
          reinterpret_cast<base::CudaBF16*>(const_cast<uint16_t*>(output.ptr<uint16_t>())));
    }
  } else if (weight.data_type() == base::DataType::kDataTypeFp32) {
    emb_kernel_cu_impl<float><<<max_seq_len, thread_num>>>(
        vocab_size, input_num, weight_dim, in_ptr, weight.ptr<float>(),
        const_cast<float*>(output.ptr<float>()));
  } else {
    emb_kernel_cu_impl<base::CudaBF16><<<max_seq_len, thread_num>>>(
        vocab_size, input_num, weight_dim, in_ptr,
        reinterpret_cast<const base::CudaBF16*>(weight.ptr<uint16_t>()),
        reinterpret_cast<base::CudaBF16*>(const_cast<uint16_t*>(output.ptr<uint16_t>())));
  }
}
}  // namespace kernel
