// Updated on March 15, 2026
#ifndef KERNELS_INTERFACE_H
#define KERNELS_INTERFACE_H
#include "tensor/tensor.h"
namespace base {
struct DeviceContext;
}

namespace kernel {

typedef void (*AddKernel)(const tensor::Tensor& input1, const tensor::Tensor& input2,
                          const tensor::Tensor& output, void* stream);

typedef void (*MatmulKernel)(const tensor::Tensor& input, const tensor::Tensor& weight,
                             const tensor::Tensor& output, float scale,
                             const base::DeviceContext* context);

typedef void (*MatmulBatchKernel)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                  const tensor::Tensor& output, int32_t batch_tokens,
                                  const base::DeviceContext* context);

typedef void (*MatmulKernelQuant)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                  const tensor::Tensor& output, int32_t group_size,
                                  const tensor::Tensor& scale,
                                  const base::DeviceContext* context);

typedef void (*EmbeddingKernel)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                const tensor::Tensor& output, int32_t vocab_size, void* stream);

typedef void (*SwigluKernel)(const tensor::Tensor& input1, const tensor::Tensor& input2,
                             const tensor::Tensor& output, void* stream);

typedef void (*MHAKernel)(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                          int32_t kv_dim, int32_t kv_mul, int32_t head_size,
                          const tensor::Tensor& mha_out, const tensor::Tensor& query_tensor,
                          const tensor::Tensor& score_tensor,
                          const tensor::Tensor& key_cache_tensor,
                          const tensor::Tensor& value_cache_tensor, base::DeviceType device_type,
                          const base::DeviceContext* context);

typedef void (*RMSNormKernel)(const tensor::Tensor& input, const tensor::Tensor& weight,
                              const tensor::Tensor& output, void* stream);

typedef void (*RMSNormKernelDim)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                 const tensor::Tensor& output, int32_t dim, void* stream);

typedef void (*RoPEKernel)(int32_t dim, int32_t kv_dim, int32_t head_size,
                           const tensor::Tensor& input_q, const tensor::Tensor& input_k,
                           const tensor::Tensor& input_pos, const tensor::Tensor& sin_cache,
                           const tensor::Tensor& cos_cache, void* stream);

typedef void (*RoPEBatchKernel)(int32_t dim, int32_t kv_dim, int32_t head_size,
                                const tensor::Tensor& input_q,
                                const tensor::Tensor& input_k,
                                const tensor::Tensor& positions,
                                const tensor::Tensor& sin_cache,
                                const tensor::Tensor& cos_cache,
                                int32_t batch_tokens,
                                void* stream);

typedef void (*SinCosCacheKernel)(int32_t head_size,
                                  int32_t max_seq_len,
                                  const tensor::Tensor& sin_cache,
                                  const tensor::Tensor& cos_cache,
                                  void* stream);

typedef void (*AddBiasKernel)(const tensor::Tensor& output,
                              const tensor::Tensor& bias,
                              int32_t batch_tokens,
                              int32_t dim,
                              void* stream);

typedef void (*ScaleKernel)(float scale, const tensor::Tensor& input, void* stream);

typedef void (*SoftmaxInplaceKernel)(const tensor::Tensor& input, void* stream);

typedef void (*ScaleSumKernel)(const tensor::Tensor& value, const tensor::Tensor& scale,
                               const tensor::Tensor& output, int t, int size, int stride,
                               void* stream);

typedef void (*MoeSoftmaxTopKKernel)(tensor::Tensor& router_logits, int32_t num_experts, int32_t topk, 
                                     tensor::Tensor& topk_values, tensor::Tensor& topk_indices, bool norm_topk_prob, 
                                     const base::DeviceContext* context);

typedef void (*MoeScaleAddKernel)(tensor::Tensor& input_tensor,
                                  const tensor::Tensor& expert_output, float scale,
                                  const base::DeviceContext* context);

typedef void (*ArgmaxSelectedRowsKernel)(const tensor::Tensor& logits,
                                         const tensor::Tensor& row_indices,
                                         tensor::Tensor& token_ids,
                                         void* stream);

typedef size_t (*ArgmaxLogitsKernel)(const void* logits,
                                     size_t size,
                                     base::DataType data_type,
                                     void* stream);

void softmax_inplace_cpu(const float* input_ptr, size_t size);

AddKernel get_add_kernel(base::DeviceType device_type);

EmbeddingKernel get_emb_kernel(base::DeviceType device_type);

MatmulKernel get_matmul_kernel(base::DeviceType device_type);

MatmulBatchKernel get_matmul_batch_kernel(base::DeviceType device_type);

MatmulKernelQuant get_matmul_kernel_quant8(base::DeviceType device_type);

MHAKernel get_mha_kernel(base::DeviceType device_type);

RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type);

RoPEKernel get_rope_kernel(base::DeviceType device_type);

RoPEBatchKernel get_rope_batch_kernel(base::DeviceType device_type);

SinCosCacheKernel get_sin_cos_cache_kernel(base::DeviceType device_type);

AddBiasKernel get_add_bias_kernel(base::DeviceType device_type);

ScaleKernel get_scale_kernel(base::DeviceType device_type);

SoftmaxInplaceKernel get_softmax_kernel(base::DeviceType device_type);

SwigluKernel get_swiglu_kernel(base::DeviceType device_type, void* stream = nullptr);

ScaleSumKernel get_scale_sum_kernel(base::DeviceType device_type);

RMSNormKernelDim get_rmsnorm_dim_kernel(base::DeviceType device_type);

MoeSoftmaxTopKKernel get_moe_softmax_topk_kernel(base::DeviceType device_type);

MoeScaleAddKernel get_moe_scale_add_kernel(base::DeviceType device_type);

ArgmaxSelectedRowsKernel get_argmax_selected_rows_kernel(base::DeviceType device_type);

ArgmaxLogitsKernel get_argmax_logits_kernel(base::DeviceType device_type);
}  // namespace kernel
#endif  // KERNELS_INTERFACE_H
