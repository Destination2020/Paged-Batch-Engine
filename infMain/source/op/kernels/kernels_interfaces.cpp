// Updated on March 15, 2026
#include <algorithm>
#include <base/base.h>
#include "base/bf16.h"
#include "base/backend_type.h"
#include "base/cuda_config.h"
#include "base/device_context.h"
#include "cpu/add_kernel.h"
#include "cpu/emb_kernel.h"
#include "cpu/matmul_kernel.h"
#include "cpu/mha_kernel.h"
#include "cpu/rmsnorm_kernel.h"
#include "cpu/rope_kernel.h"
#include "cpu/scale_kernel.h"
#include "cpu/scale_sum_kernel.h"
#include "cpu/softmax_kernel.h"
#include "cpu/swiglu_kernel.h"
#include "cuda/add_kernel.cuh"
#include "op/kernels/cuda/add_bias_kernel.cuh"
#include "cuda/argmax_kernel.cuh"
#include "cuda/emb_kernel.cuh"
#include "cuda/matmul_kernel.cuh"
#include "op/kernels/cuda/matmul_kernel_batch.cuh"
#include "cuda/mha_kernel.cuh"
#include "cuda/moe_kernel.cuh"
#include "cuda/rmsnorm_kernel.cuh"
#include "cuda/rope_kernel.cuh"
#include "op/kernels/cuda/sampler_kernel.cuh"
#include "cuda/swiglu_kernel.cuh"
#include "kernels_interface.h"
#include "op/kernels/cuda/fused_mha_kernel.cuh"
namespace kernel {
namespace {
const CudaConfig* cuda_config_from_context(const base::DeviceContext* context) {
  if (context == nullptr || context->backend != base::BackendType::kCUDA ||
      context->native_context == nullptr) {
    return nullptr;
  }
  return static_cast<const CudaConfig*>(context->native_context.get());
}

CudaConfig* mutable_cuda_config_from_context(const base::DeviceContext* context) {
  return const_cast<CudaConfig*>(cuda_config_from_context(context));
}

void matmul_kernel_cu_from_context(const tensor::Tensor& input, const tensor::Tensor& weight,
                                   const tensor::Tensor& output, float scale,
                                   const base::DeviceContext* context) {
  matmul_kernel_cu(input, weight, output, scale, cuda_config_from_context(context));
}

void matmul_kernel_cu_qint8_from_context(const tensor::Tensor& input,
                                         const tensor::Tensor& weight,
                                         const tensor::Tensor& output, int32_t group_size,
                                         const tensor::Tensor& scale,
                                         const base::DeviceContext* context) {
  matmul_kernel_cu_qint8(input, weight, output, group_size, scale,
                         cuda_config_from_context(context));
}

void matmul_batch_kernel_cu_from_context(const tensor::Tensor& input,
                                         const tensor::Tensor& weight,
                                         const tensor::Tensor& output,
                                         int32_t batch_tokens,
                                         const base::DeviceContext* context) {
  matmul_batch_kernel_cu(input, weight, output, batch_tokens,
                         cuda_config_from_context(context));
}

void matmul_batch_kernel_cpu_from_context(const tensor::Tensor& input,
                                          const tensor::Tensor& weight,
                                          const tensor::Tensor& output,
                                          int32_t batch_tokens,
                                          const base::DeviceContext* context) {
  UNUSED(batch_tokens);
  matmul_kernel_cpu(input, weight, output, 1.f, context);
}

void fused_mha_kernel_cu_from_context(
    int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
    int32_t kv_dim, int32_t kv_mul, int32_t head_size,
    const tensor::Tensor& mha_out, const tensor::Tensor& query_tensor,
    const tensor::Tensor& score_tensor, const tensor::Tensor& key_cache_tensor,
    const tensor::Tensor& value_cache_tensor, base::DeviceType device_type,
    const base::DeviceContext* context) {
  fused_mha_kernel_cu(pos, head_num, layer_index, seq_len, kv_dim, kv_mul, head_size,
                      mha_out, query_tensor, score_tensor, key_cache_tensor,
                      value_cache_tensor, device_type,
                      mutable_cuda_config_from_context(context));
}

void moe_router_softmax_topk_cu_from_context(tensor::Tensor& router_logits,
                                             int32_t num_experts, int32_t topk,
                                             tensor::Tensor& topk_values,
                                             tensor::Tensor& topk_indices,
                                             bool norm_topk_prob,
                                             const base::DeviceContext* context) {
  moe_router_softmax_topk_cu(router_logits, num_experts, topk, topk_values,
                             topk_indices, norm_topk_prob,
                             mutable_cuda_config_from_context(context));
}

void moe_scale_add_cu_from_context(tensor::Tensor& input_tensor,
                                   const tensor::Tensor& expert_output, float scale,
                                   const base::DeviceContext* context) {
  moe_scale_add_cu(input_tensor, expert_output, scale,
                   mutable_cuda_config_from_context(context));
}

void sin_cos_cache_calc_cpu_from_tensors(int32_t head_size,
                                         int32_t max_seq_len,
                                         const tensor::Tensor& sin_cache,
                                         const tensor::Tensor& cos_cache,
                                         void* stream) {
  UNUSED(stream);
  sin_cos_cache_calc_cpu(head_size, max_seq_len,
                         const_cast<float*>(sin_cache.ptr<float>()),
                         const_cast<float*>(cos_cache.ptr<float>()));
}

void argmax_selected_rows_cpu_not_supported(const tensor::Tensor& logits,
                                            const tensor::Tensor& row_indices,
                                            tensor::Tensor& token_ids,
                                            void* stream) {
  UNUSED(logits);
  UNUSED(row_indices);
  UNUSED(token_ids);
  UNUSED(stream);
  LOG(FATAL) << "argmax_selected_rows kernel is only implemented for CUDA.";
}

void add_bias_cpu_not_supported(const tensor::Tensor& output,
                                const tensor::Tensor& bias,
                                int32_t batch_tokens,
                                int32_t dim,
                                void* stream) {
  UNUSED(output);
  UNUSED(bias);
  UNUSED(batch_tokens);
  UNUSED(dim);
  UNUSED(stream);
  LOG(FATAL) << "add_bias kernel is only implemented for CUDA.";
}

void rope_batch_cpu_not_supported(int32_t dim,
                                  int32_t kv_dim,
                                  int32_t head_size,
                                  const tensor::Tensor& input_q,
                                  const tensor::Tensor& input_k,
                                  const tensor::Tensor& positions,
                                  const tensor::Tensor& sin_cache,
                                  const tensor::Tensor& cos_cache,
                                  int32_t batch_tokens,
                                  void* stream) {
  UNUSED(dim);
  UNUSED(kv_dim);
  UNUSED(head_size);
  UNUSED(input_q);
  UNUSED(input_k);
  UNUSED(positions);
  UNUSED(sin_cache);
  UNUSED(cos_cache);
  UNUSED(batch_tokens);
  UNUSED(stream);
  LOG(FATAL) << "batched rope kernel is only implemented for CUDA.";
}

void mrope_batch_cpu_not_supported(int32_t dim, int32_t kv_dim, int32_t head_size,
                                   const tensor::Tensor& input_q,
                                   const tensor::Tensor& input_k,
                                   const tensor::Tensor& positions,
                                   const tensor::Tensor& sin_cache,
                                   const tensor::Tensor& cos_cache,
                                   int32_t temporal_section,
                                   int32_t height_section,
                                   int32_t batch_tokens, void* stream) {
  UNUSED(dim);
  UNUSED(kv_dim);
  UNUSED(head_size);
  UNUSED(input_q);
  UNUSED(input_k);
  UNUSED(positions);
  UNUSED(sin_cache);
  UNUSED(cos_cache);
  UNUSED(temporal_section);
  UNUSED(height_section);
  UNUSED(batch_tokens);
  UNUSED(stream);
  LOG(FATAL) << "batched mrope kernel is only implemented for CUDA.";
}

size_t argmax_logits_cpu(const void* logits,
                         size_t size,
                         base::DataType data_type,
                         void* stream) {
  UNUSED(stream);
  CHECK(logits != nullptr);
  CHECK_GT(size, 0);
  if (data_type == base::DataType::kDataTypeFp32) {
    const float* fp32_logits = static_cast<const float*>(logits);
    return std::distance(fp32_logits, std::max_element(fp32_logits, fp32_logits + size));
  }

  CHECK_EQ(data_type, base::DataType::kDataTypeBf16);
  const uint16_t* bf16_logits = static_cast<const uint16_t*>(logits);
  size_t best_index = 0;
  float best_value = base::bf16_bits_to_float(bf16_logits[0]);
  for (size_t i = 1; i < size; ++i) {
    const float value = base::bf16_bits_to_float(bf16_logits[i]);
    if (value > best_value) {
      best_value = value;
      best_index = i;
    }
  }
  return best_index;
}

size_t argmax_logits_cuda(const void* logits,
                          size_t size,
                          base::DataType data_type,
                          void* stream) {
  CHECK(logits != nullptr);
  CHECK_GT(size, 0);
  if (data_type == base::DataType::kDataTypeFp32) {
    return argmax_kernel_cu(static_cast<const float*>(logits), size, stream);
  }

  CHECK_EQ(data_type, base::DataType::kDataTypeBf16);
  return argmax_kernel_cu_bf16(static_cast<const uint16_t*>(logits), size, stream);
}
}  // namespace

AddKernel get_add_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return add_kernel_cpu;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return add_kernel_cu;
  } else {
    LOG(FATAL) << "Unknown device type for get a add kernel.";
    return nullptr;
  }
}

EmbeddingKernel get_emb_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return emb_kernel_normal;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return emb_kernel_cu;
  } else {
    LOG(FATAL) << "Unknown device type for get an embedding kernel.";
    return nullptr;
  }
}

MatmulKernel get_matmul_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return matmul_kernel_cpu;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return matmul_kernel_cu_from_context;
  } else {
    LOG(FATAL) << "Unknown device type for get an matmul kernel.";
    return nullptr;
  }
}

MatmulBatchKernel get_matmul_batch_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return matmul_batch_kernel_cpu_from_context;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return matmul_batch_kernel_cu_from_context;
  } else {
    LOG(FATAL) << "Unknown device type for get a batched matmul kernel.";
    return nullptr;
  }
}

MatmulKernelQuant get_matmul_kernel_quant8(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCUDA) {
    return matmul_kernel_cu_qint8_from_context;
  } else {
    LOG(FATAL) << "Unknown device type for get an matmul kernel.";
    return nullptr;
  }
}

MHAKernel get_mha_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return mha_kernel;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return fused_mha_kernel_cu_from_context;
  } else {
    LOG(FATAL) << "Unknown device type for get an mha kernel.";
    return nullptr;
  }
}

RoPEKernel get_rope_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return rope_kernel_cpu;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return rope_kernel_cu;
  } else {
    LOG(FATAL) << "Unknown device type for get a rope kernel.";
    return nullptr;
  }
}

RoPEBatchKernel get_rope_batch_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCUDA) {
    return rope_kernel_batched_cu;
  } else if (device_type == base::DeviceType::kDeviceCPU) {
    return rope_batch_cpu_not_supported;
  } else {
    LOG(FATAL) << "Unknown device type for get a batched rope kernel.";
    return nullptr;
  }
}

MRoPEBatchKernel get_mrope_batch_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCUDA) {
    return mrope_kernel_batched_cu;
  } else if (device_type == base::DeviceType::kDeviceCPU) {
    return mrope_batch_cpu_not_supported;
  } else {
    LOG(FATAL) << "Unknown device type for get a batched mrope kernel.";
    return nullptr;
  }
}

SinCosCacheKernel get_sin_cos_cache_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return sin_cos_cache_calc_cpu_from_tensors;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return sin_cos_cache_calc_cu;
  } else {
    LOG(FATAL) << "Unknown device type for get a sin/cos cache kernel.";
    return nullptr;
  }
}

AddBiasKernel get_add_bias_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCUDA) {
    return add_bias_kernel_cu;
  } else if (device_type == base::DeviceType::kDeviceCPU) {
    return add_bias_cpu_not_supported;
  } else {
    LOG(FATAL) << "Unknown device type for get an add bias kernel.";
    return nullptr;
  }
}

ScaleKernel get_scale_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return scale_inplace_cpu;
  } else {
    LOG(FATAL) << "Unknown device type for get a rope kernel.";
    return nullptr;
  }
}

SoftmaxInplaceKernel get_softmax_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return softmax_inplace_cpu;
  } else {
    LOG(FATAL) << "Unknown device type for get an softmax kernel.";
    return nullptr;
  }
}

SwigluKernel get_swiglu_kernel(base::DeviceType device_type, void* stream) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return swiglu_kernel_cpu;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return swiglu_kernel_cu;
  } else {
    LOG(FATAL) << "Unknown device type for get a swiglu kernel.";
    return nullptr;
  }
}

RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return rmsnorm_kernel_cpu;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return rmsnorm_kernel_cu;
  } else {
    LOG(FATAL) << "Unknown device type for get a rmsnorm kernel.";
    return nullptr;
  }
}

RMSNormKernelDim get_rmsnorm_dim_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCUDA) {
    return rmsnorm_kernel_cu_dim;
  } else {
    LOG(FATAL) << "Unknown device type for get a rmsnorm dim kernel.";
    return nullptr;
  }
}

ScaleSumKernel get_scale_sum_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return scale_sum_kernel_cpu;
  } else {
    LOG(FATAL) << "Unknown device type for get a scale and reduce kernel.";
    return nullptr;
  }
}

MoeSoftmaxTopKKernel get_moe_softmax_topk_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCUDA) {
    return moe_router_softmax_topk_cu_from_context;
  } else {
    LOG(FATAL) << "Unknown device type for get a moe softmax topk kernel.";
    return nullptr;
  }
}

MoeScaleAddKernel get_moe_scale_add_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCUDA) {
    return moe_scale_add_cu_from_context;
  } else {
    LOG(FATAL) << "Unknown device type for get a moe scale add kernel.";
    return nullptr;
  }
}

ArgmaxSelectedRowsKernel get_argmax_selected_rows_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCUDA) {
    return argmax_selected_rows_cu;
  } else if (device_type == base::DeviceType::kDeviceCPU) {
    return argmax_selected_rows_cpu_not_supported;
  } else {
    LOG(FATAL) << "Unknown device type for get an argmax selected rows kernel.";
    return nullptr;
  }
}

ArgmaxLogitsKernel get_argmax_logits_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return argmax_logits_cpu;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return argmax_logits_cuda;
  } else {
    LOG(FATAL) << "Unknown device type for get an argmax logits kernel.";
    return nullptr;
  }
}

}  // namespace kernel
