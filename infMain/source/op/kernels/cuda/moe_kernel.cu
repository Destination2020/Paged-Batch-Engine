// Updated on March 23, 2026
#include "moe_kernel.cuh"
#include <base/cuda_config.h>
#include <tensor/tensor.h>
#include <cfloat>
#include "cuda_type_utils.cuh"

namespace kernel {
namespace {
constexpr int32_t kWarpSize = 32;
constexpr int32_t kScaleAddThreads = 256;

__device__ inline float warp_reduce_max(float val) {
  unsigned mask = 0xffffffffu;
  for (int32_t offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_down_sync(mask, val, offset));
  }
  return val;
}

__device__ inline float warp_reduce_sum(float val) {
  unsigned mask = 0xffffffffu;
  for (int32_t offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(mask, val, offset);
  }
  return val;
}

__device__ inline void warp_reduce_argmax(float& val, int32_t& idx) {
  unsigned mask = 0xffffffffu;
  for (int32_t offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    float other_val = __shfl_down_sync(mask, val, offset);
    int32_t other_idx = __shfl_down_sync(mask, idx, offset);
    if (other_val > val || (other_val == val && other_idx >= 0 &&
                            (idx < 0 || other_idx < idx))) {
      val = other_val;
      idx = other_idx;
    }
  }
}
}  // namespace

template <typename T>
__global__ void moe_router_softmax_topk_kernel(const T* router_logits, int32_t num_experts,
                                               int32_t topk, float* topk_values,
                                               int32_t* topk_indices, bool norm_topk_prob) {
  const int32_t lane = threadIdx.x;
  if (lane >= kWarpSize) {
    return;
  }

  extern __shared__ float s_prob[];
  for (int32_t i = lane; i < num_experts; i += kWarpSize) {
    s_prob[i] = scalar_to_float(router_logits[i]);
  }
  __syncwarp();

  float local_max = -FLT_MAX;
  for (int32_t i = lane; i < num_experts; i += kWarpSize) {
    local_max = fmaxf(local_max, s_prob[i]);
  }
  float max_val = warp_reduce_max(local_max);
  max_val = __shfl_sync(0xffffffffu, max_val, 0);

  float local_sum = 0.f;
  for (int32_t i = lane; i < num_experts; i += kWarpSize) {
    s_prob[i] = expf(s_prob[i] - max_val);
    local_sum += s_prob[i];
  }
  float sum_val = warp_reduce_sum(local_sum);
  sum_val = __shfl_sync(0xffffffffu, sum_val, 0);

  for (int32_t i = lane; i < num_experts; i += kWarpSize) {
    s_prob[i] /= sum_val;
  }
  __syncwarp();

  for (int32_t k = 0; k < topk; ++k) {
    float local_best = -1.f;
    int32_t local_idx = -1;
    for (int32_t i = lane; i < num_experts; i += kWarpSize) {
      const float value = s_prob[i];
      if (value > local_best || (value == local_best && value >= 0.f &&
                                 (local_idx < 0 || i < local_idx))) {
        local_best = value;
        local_idx = i;
      }
    }

    warp_reduce_argmax(local_best, local_idx);
    local_best = __shfl_sync(0xffffffffu, local_best, 0);
    local_idx = __shfl_sync(0xffffffffu, local_idx, 0);

    if (lane == 0) {
      topk_values[k] = local_best;
      topk_indices[k] = local_idx;
      if (local_idx >= 0) {
        s_prob[local_idx] = -1.f;
      }
    }
    __syncwarp();
  }

  if (norm_topk_prob && lane == 0) {
    float topk_sum = 0.f;
    for (int32_t i = 0; i < topk; ++i) {
      topk_sum += topk_values[i];
    }
    if (topk_sum > 0.f) {
      for (int32_t i = 0; i < topk; ++i) {
        topk_values[i] /= topk_sum;
      }
    }
  }
}

template <typename T>
__global__ void moe_scale_add_kernel(T* input, const T* expert_output, float scale, int32_t size) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    const float value = scalar_to_float(input[idx]) + scale * scalar_to_float(expert_output[idx]);
    input[idx] = float_to_scalar<T>(value);
  }
}

void moe_router_softmax_topk_cu(tensor::Tensor& router_logits, int32_t num_experts, int32_t topk,
                                tensor::Tensor& topk_values, tensor::Tensor& topk_indices,
                                bool norm_topk_prob, CudaConfig* config) {
  CHECK_NE(config, nullptr);
  CHECK(router_logits.is_empty() == false);
  CHECK(topk_values.is_empty() == false);
  CHECK(topk_indices.is_empty() == false);
  CHECK(router_logits.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(topk_values.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(topk_indices.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(router_logits.data_type() == base::DataType::kDataTypeFp32 ||
        router_logits.data_type() == base::DataType::kDataTypeBf16);
  CHECK(topk_values.data_type() == base::DataType::kDataTypeFp32);
  CHECK(topk_indices.data_type() == base::DataType::kDataTypeInt32);
  CHECK(static_cast<int32_t>(router_logits.size()) == num_experts);
  CHECK(static_cast<int32_t>(topk_values.size()) == topk);
  CHECK(static_cast<int32_t>(topk_indices.size()) == topk);
  CHECK_GT(topk, 0);
  CHECK_LE(topk, num_experts);

  size_t shared_mem_size = static_cast<size_t>(num_experts) * sizeof(float);
  if (router_logits.data_type() == base::DataType::kDataTypeFp32) {
    moe_router_softmax_topk_kernel<float><<<1, kWarpSize, shared_mem_size, config->stream>>>(
        router_logits.ptr<float>(), num_experts, topk, topk_values.ptr<float>(),
        topk_indices.ptr<int32_t>(), norm_topk_prob);
  } else {
    moe_router_softmax_topk_kernel<base::CudaBF16>
        <<<1, kWarpSize, shared_mem_size, config->stream>>>(
            reinterpret_cast<const base::CudaBF16*>(router_logits.ptr<uint16_t>()), num_experts,
            topk, topk_values.ptr<float>(), topk_indices.ptr<int32_t>(), norm_topk_prob);
  }
}

void moe_scale_add_cu(tensor::Tensor& input_tensor, const tensor::Tensor& expert_output, float scale,
                      CudaConfig* config) {
  CHECK_NE(config, nullptr);
  CHECK(input_tensor.is_empty() == false);
  CHECK(expert_output.is_empty() == false);
  CHECK(input_tensor.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(expert_output.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(input_tensor.data_type() == expert_output.data_type());
  CHECK(input_tensor.size() == expert_output.size());

  int32_t size = static_cast<int32_t>(input_tensor.size());
  int32_t blocks = (size + kScaleAddThreads - 1) / kScaleAddThreads;
  if (input_tensor.data_type() == base::DataType::kDataTypeFp32) {
    moe_scale_add_kernel<float><<<blocks, kScaleAddThreads, 0, config->stream>>>(
        input_tensor.ptr<float>(), expert_output.ptr<float>(), scale, size);
  } else {
    CHECK_EQ(input_tensor.data_type(), base::DataType::kDataTypeBf16);
    moe_scale_add_kernel<base::CudaBF16><<<blocks, kScaleAddThreads, 0, config->stream>>>(
        reinterpret_cast<base::CudaBF16*>(input_tensor.ptr<uint16_t>()),
        reinterpret_cast<const base::CudaBF16*>(expert_output.ptr<uint16_t>()), scale, size);
  }
}

}  // namespace kernel
