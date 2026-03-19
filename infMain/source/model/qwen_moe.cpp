// Updated on March 15, 2026
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <op/matmul.h>
#include <op/mha.h>
#include <op/rmsnorm.h>
#include <sentencepiece_processor.h>
#include <utility>
#include "../op/kernels/cpu/rope_kernel.h"
#include "../op/kernels/cuda/rope_kernel.cuh"
#include <base/tick.h>
#include <model/qwen_moe.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstring>

namespace {
  void cpu_softmax_inplace(float* data, int n) {
    float max_value = data[0];
    for (int i = 1; i < n; ++i) {
      max_value = std::max(max_value, data[i]);
    }
    float sum = 0.f;
    for (int i = 0; i < n; ++i) {
      data[i] = std::exp(data[i] - max_value);
      sum += data[i];
    }
    for (int i = 0; i < n; ++i) {
      data[i] /= sum;
    }
  }

  void cpu_topk(const float* data, int n, int k, float* out_values, int32_t* out_indices) {
    std::vector<std::pair<float, int32_t>> vec(n);
    for (int i = 0; i < n; ++i) {
      vec[i] = {data[i], i};
    }
    std::partial_sort(vec.begin(), vec.begin() + k, vec.end(), 
                      [](const auto& a, const auto& b)->bool {return a.first > b.first; });
    for (int i = 0; i < k; ++i) {
      out_values[i] = vec[i].first;
      out_indices[i] = vec[i].second;
    }
  }

  void scale_add(float* out_values, float scale, const float* added_values, int32_t dim_) {
    for (int i = 0; i < dim_; ++i) {
      out_values[i] += added_values[i] * scale;
    }
  }
}



namespace model {

void QwenMoeLayers::to_cuda(std::shared_ptr<kernel::CudaConfig> config) {
  if (add_layer_) {
    add_layer_->set_cuda_config(config);
    add_layer_->to_cuda();
  }

  if (rope_layer_) {
    rope_layer_->set_cuda_config(config);
    rope_layer_->to_cuda();
  }

  if (swiglu_layer_) {
    swiglu_layer_->set_cuda_config(config);
    swiglu_layer_->to_cuda();
  }

  if (shared_swiglu_layer_) {
    shared_swiglu_layer_->set_cuda_config(config);
    shared_swiglu_layer_->to_cuda();
  }

  if (cls_layer_) {
    cls_layer_->set_cuda_config(config);
    cls_layer_->to_cuda();
  }

  if (embedding_layer_) {
    embedding_layer_->set_cuda_config(config);
    embedding_layer_->to_cuda();
  }

  if (mha_layer_) {
    mha_layer_->set_cuda_config(config);
    mha_layer_->to_cuda();
  }

  for (auto& weight_layer : wq_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : wk_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : wv_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : wo_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : w1_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : w2_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : w3_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& router_layer : router_layers_) {
    if (router_layer) {
      router_layer->set_cuda_config(config);
      router_layer->to_cuda();
    }
  }

  for (auto& expert_w1_layer : expert_w1_layers_) {
    for (auto& layer : expert_w1_layer) {
      if (layer) {
        layer->set_cuda_config(config);
        layer->to_cuda();
      }
    }
  }

  for (auto& expert_w2_layer : expert_w2_layers_) {
    for (auto& layer : expert_w2_layer) {
      if (layer) {
        layer->set_cuda_config(config);
        layer->to_cuda();
      }
    }
  }

  for (auto& expert_w3_layer : expert_w3_layers_) {
    for (auto& layer : expert_w3_layer) {
      if (layer) {
        layer->set_cuda_config(config);
        layer->to_cuda();
      }
    }
  }

  for (auto& rms_norm_layer : rmsnorm_layers_) {
    if (rms_norm_layer) {
      rms_norm_layer->set_cuda_config(config);
      rms_norm_layer->to_cuda();
    }
  }
  for (auto& shared_w1_layer : shared_w1_layers_) {
    if (shared_w1_layer) {
      shared_w1_layer->set_cuda_config(config);
      shared_w1_layer->to_cuda();
    }
  }
  for (auto& shared_w2_layer : shared_w2_layers_) {
    if (shared_w2_layer) {
      shared_w2_layer->set_cuda_config(config);
      shared_w2_layer->to_cuda();
    }
  }
  for (auto& shared_w3_layer : shared_w3_layers_) {
    if (shared_w3_layer) {
      shared_w3_layer->set_cuda_config(config);
      shared_w3_layer->to_cuda();
    }
  }
  for (auto& shared_gate_layer : shared_gate_layers_) {
    if (shared_gate_layer) {
      shared_gate_layer->set_cuda_config(config);
      shared_gate_layer->to_cuda();
    }
  }

}

QwenMoeModel::QwenMoeModel(base::TokenizerType tokenizer_type, std::string token_path,
                       std::string model_path, bool is_quant_model)
    : Model(tokenizer_type, base::ModelType::kModelTypeLLama2, std::move(token_path),
            std::move(model_path), is_quant_model) {}

base::Status QwenMoeModel::init(base::DeviceType device_type) {
  using namespace base;
  if (token_path_.empty()) {
    return error::PathNotValid(token_path_);
  }
  if (device_type == base::DeviceType::kDeviceCPU && is_quant_model_) {
    return error::InternalError("The cpu device do not support int8 quant model.");
  }
  if (is_quant_model_) {
    return error::InternalError("Quantized model is not supported for QwenMoeModel yet.");
  }
  device_type_ = device_type;
  if (device_type == DeviceType::kDeviceCUDA) {
    cudaSetDevice(0);
    cuda_config_ = std::make_shared<kernel::CudaConfig>();
    cudaStreamCreate(&cuda_config_->stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      return error::InternalError("The cuda hanle create failed.");
    }
  }

  Status read_status = gen_model_from_file();
  if (!read_status) {
    return read_status;
  }
  init_mem();
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    kernel::sin_cos_cache_calc_cpu(config_->head_size_, config_->seq_len_,
                                   get_buffer(ModelBufferType::kSinCache).ptr<float>(),
                                   get_buffer(ModelBufferType::kCosCache).ptr<float>());
  } else {
    CHECK_NE(cuda_config_, nullptr);
    kernel::sin_cos_cache_calc_cu(config_->head_size_, config_->seq_len_,
                                  get_buffer(ModelBufferType::kSinCache),
                                  get_buffer(ModelBufferType::kCosCache), cuda_config_->stream);
  }

  sampler_ = std::make_unique<sampler::ArgmaxSampler>(device_type_);
  return error::Success();
}

base::Status QwenMoeModel::forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                                 int& next) const {
  if (input.is_empty()) {
    return base::error::InvalidArgument("The input tensor is empty.");
  }
  if (device_type_ == base::DeviceType::kDeviceCPU && is_quant_model_) {
    return base::error::InternalError("Unsupported int8 quant in the cpu device");
  }

  for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    attention_rms(layer_idx, input);
    // attention (wq wk wv @ input)
    attention_qkv(layer_idx, pos_tensor);
    // multi-head attention
    attention_mha(layer_idx, pos_tensor);
    // feed forward
    feed_forward(layer_idx, input);
  }
  cls_logits(input);
  return base::error::Success();
}


// Layout for QwenMoeModel fp32 .bin (after ModelConfig [+ optional MoeHeader]):
//
// - All offsets below are counted in float elements relative to weight_data,
//   i.e. RawModelData::weight(pos) returns (float*)weight_data + pos.
//
// 0) Embedding
//    embedding.weight: [vocab_size, dim]
//    size = vocab_size * dim
//
// 1) Per-layer blocks, for layer l in [0, layer_num):
//    a) attention_norm.weight           : [dim]
//
//    b) attention.wq
//       - wq.weight                     : [dim, dim]       // out_dim = dim, in_dim = dim
//       - wq.bias                       : [dim]
//
//    c) attention.wk
//       - wk.weight                     : [kv_dim, dim]    // out_dim = kv_dim, in_dim = dim
//       - wk.bias                       : [kv_dim]
//
//    d) attention.wv
//       - wv.weight                     : [kv_dim, dim]
//       - wv.bias                       : [kv_dim]
//
//    e) attention.wo
//       - wo.weight                     : [dim, dim]
//
//    f) ffn_norm.weight                 : [dim]
//
//    g) MoE router (per-token gate)
//       - router.weight                 : [moe_expert_num, dim]   // out = num_expert
//       - router.bias                   : [moe_expert_num]
//
//    h) MoE experts (per-layer, per-expert FFN)
//       For expert e in [0, moe_expert_num):
//         - expert_w1[e].weight         : [moe_hidden_dim, dim]
//         - expert_w2[e].weight         : [dim, moe_hidden_dim]
//         - expert_w3[e].weight         : [moe_hidden_dim, dim]
//
// 2) Final layer norm before LM head
//    final_norm.weight                  : [dim]
//
// 3) Classifier / lm_head
//    - If config_->is_shared_weight_ == true:
//        * reuse embedding.weight as lm_head.weight (tied weights),
//          no extra tensor stored in the file.
//    - Else:
//        * lm_head.weight               : [vocab_size, dim]
//
// NOTE:
//   - This layout is ONLY for QwenMoeModel.
//   - A dedicated exporter (e.g. tools/export_qwen_moe.py) will write weights
//     in exactly this order.
//   - create_param_layers() will be updated to parse according to this layout.





void QwenMoeModel::create_nonparam_layers() {
  CHECK(qwen_layers_ != nullptr);
  qwen_layers_->rope_layer_ = std::make_shared<op::RoPELayer>(
      device_type_, config_->dim_, config_->kv_dim_, config_->head_size_);

  qwen_layers_->mha_layer_ = std::make_shared<op::MultiHeadAttention>(
      device_type_, 0, config_->kv_mul_, config_->kv_dim_, config_->seq_len_, config_->head_num_,
      config_->head_size_);

  qwen_layers_->add_layer_ = std::make_shared<op::VecAddLayer>(device_type_);

  qwen_layers_->swiglu_layer_ =
      std::make_shared<op::SwiGLULayer>(device_type_, config_->hidden_dim_);

  qwen_layers_->shared_swiglu_layer_ = 
      std::make_shared<op::SwiGLULayer>(device_type_, config_->moe_shared_hidden_dim_);
}

void QwenMoeModel::create_param_quant_layers() {
  CHECK(is_quant_model_);
  CHECK(qwen_layers_ != nullptr);

  size_t pos = 0;
  int32_t dim = config_->dim_;
  auto cpu_device_type = base::DeviceType::kDeviceCPU;

  // query
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, true);
    wq->set_group_size(group_size_);
    wq->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->wq_layers_.push_back(wq);
    pos = pos + dim * dim + wq->get_scale_num() * sizeof(float);
  }

  // key
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wk = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim, true);
    wk->set_group_size(group_size_);
    wk->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->wk_layers_.push_back(wk);
    pos = pos + config_->kv_dim_ * dim + wk->get_scale_num() * sizeof(float);
  }

  // value
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wv = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim, true);
    wv->set_group_size(group_size_);
    wv->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->wv_layers_.push_back(wv);
    pos += config_->kv_dim_ * dim + wv->get_scale_num() * sizeof(float);
  }

  // output
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, true);
    wo->set_group_size(group_size_);
    wo->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->wo_layers_.push_back(wo);
    pos = pos + dim * dim + wo->get_scale_num() * sizeof(float);
  }

  // w1 layers
  int32_t hidden_dim = config_->hidden_dim_;
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w1 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim, true);
    w1->set_group_size(group_size_);
    w1->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->w1_layers_.push_back(w1);
    pos = pos + dim * hidden_dim + w1->get_scale_num() * sizeof(float);
  }

  // w2 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w2 = std::make_shared<op::MatmulLayer>(device_type_, dim, hidden_dim, true);
    w2->set_group_size(group_size_);
    w2->set_weight(0, {dim, hidden_dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->w2_layers_.push_back(w2);
    pos = pos + dim * hidden_dim + w2->get_scale_num() * sizeof(float);
  }

  // w3 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w3 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim, true);
    w3->set_group_size(group_size_);
    w3->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->w3_layers_.push_back(w3);
    pos = pos + dim * hidden_dim + w3->get_scale_num() * sizeof(float);
  }

  // wcls layer
  auto cls_layer = std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim, true);
  cls_layer->set_group_size(group_size_);
  if (config_->is_shared_weight_) {
    // using token embedding weight
    cls_layer->set_weight(0, {config_->vocab_size_, dim}, this->raw_model_data_->weight(pos),
                          cpu_device_type);
  } else {
    // no shared
    cls_layer->set_weight(0, {config_->vocab_size_, dim}, this->raw_model_data_->weight(pos),
                          cpu_device_type);
    pos = pos + config_->vocab_size_ * dim + cls_layer->get_scale_num() * sizeof(float);
  }
  qwen_layers_->cls_layer_ = cls_layer;

  // embedding layer
  float* weight_ptr = (float*)raw_model_data_->weight(pos);
  qwen_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
      device_type_, config_->dim_, config_->seq_len_, std::abs(config_->vocab_size_));
  qwen_layers_->embedding_layer_->set_weight(0, {std::abs(config_->vocab_size_), dim}, weight_ptr,
                                             cpu_device_type);
  weight_ptr += config_->vocab_size_ * dim;

  // rmsnorm attention attention,ffn,final
  for (int32_t i = 0; i < 2 * config_->layer_num_ + 1; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, dim);

    rms_norm_layer->set_weight(0, {dim}, weight_ptr, cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    weight_ptr += dim;
  }
}

/*
weight_data (float*)  --->  pos = 0

[0] Embedding
  embedding.weight: [vocab, dim]
  pos += vocab * dim

for layer l in [0..L-1]:
  [1] input_layernorm
    attn_norm[l]: [dim]
    pos += dim

  [2] Self-Attn QKV/O
    wq.weight: [dim, dim]           pos += dim*dim
    wq.bias:   [dim]                pos += dim

    wk.weight: [kv_dim, dim]        pos += kv_dim*dim
    wk.bias:   [kv_dim]             pos += kv_dim

    wv.weight: [kv_dim, dim]        pos += kv_dim*dim
    wv.bias:   [kv_dim]             pos += kv_dim

    wo.weight: [dim, dim]           pos += dim*dim

  [3] post_attention_layernorm
    ffn_norm[l]: [dim]
    pos += dim

  [4] Router (MoE gate)
    router.weight: [num_experts, dim]   # no bias
    pos += num_experts * dim

  [5] Routed Experts (for e in [0..E-1])
    expert_w1[l][e]: [moe_hidden, dim]  pos += moe_hidden*dim
    expert_w2[l][e]: [dim, moe_hidden]  pos += dim*moe_hidden
    expert_w3[l][e]: [moe_hidden, dim]  pos += moe_hidden*dim

  [6] Shared Expert (if enabled)
    shared_w1[l]: [shared_hidden, dim]  pos += shared_hidden*dim
    shared_w2[l]: [dim, shared_hidden]  pos += dim*shared_hidden
    shared_w3[l]: [shared_hidden, dim]  pos += shared_hidden*dim
    shared_gate[l]: [1, dim]            pos += dim

[7] Final norm
  final_norm: [dim]
  pos += dim

[8] LM Head
  if tie_word_embeddings == false:
    lm_head.weight: [vocab, dim]
    pos += vocab * dim
  else:
    lm_head 复用 embedding.weight，不新增 pos
*/
void QwenMoeModel::create_param_layers() {
  CHECK(!is_quant_model_);
  CHECK(qwen_layers_ != nullptr);
  // The embedding layer
  auto cpu_device_type = base::DeviceType::kDeviceCPU;
  const int32_t dim = config_->dim_;
  const int32_t kv_dim = config_->kv_dim_;
  const int32_t layer_num = config_->layer_num_;
  const int32_t vocab = config_->vocab_size_;
  const int32_t moe_experts = config_->moe_expert_num_;
  const int32_t moe_hidden = config_->moe_hidden_dim_;
  const int32_t shared_hidden = config_->moe_shared_hidden_dim_;
  const int32_t has_shared = (config_->moe_shared_expert_num_ > 0);

  CHECK_EQ(config_->moe_sparse_step_, 1) << "Current implementation only supports decoder_sparse_step = 1;";
  CHECK_GT(moe_experts, 0);
  CHECK_GT(moe_hidden, 0);
  if (has_shared) {
    CHECK_GT(shared_hidden, 0);
  }
  
  qwen_layers_->wq_layers_.clear();
  qwen_layers_->wk_layers_.clear();
  qwen_layers_->wv_layers_.clear();
  qwen_layers_->wo_layers_.clear();
  qwen_layers_->w1_layers_.clear();
  qwen_layers_->w2_layers_.clear();
  qwen_layers_->w3_layers_.clear();
  qwen_layers_->rmsnorm_layers_.clear();
  qwen_layers_->router_layers_.assign(layer_num, nullptr);
  qwen_layers_->expert_w1_layers_.assign(layer_num, {});
  qwen_layers_->expert_w2_layers_.assign(layer_num, {});
  qwen_layers_->expert_w3_layers_.assign(layer_num, {});
  qwen_layers_->shared_gate_layers_.assign(layer_num, nullptr);
  qwen_layers_->shared_w1_layers_.assign(layer_num, nullptr);
  qwen_layers_->shared_w2_layers_.assign(layer_num, nullptr);
  qwen_layers_->shared_w3_layers_.assign(layer_num, nullptr);

  size_t pos = 0;
  // 0) embedding
  qwen_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
      device_type_, dim, config_->seq_len_, std::abs(config_->vocab_size_));
  qwen_layers_->embedding_layer_->set_weight(0, {std::abs(config_->vocab_size_), dim}, raw_model_data_->weight(pos), cpu_device_type);
  pos += static_cast<size_t>(std::abs(config_->vocab_size_)) * dim;

  std::vector<std::shared_ptr<op::RmsNormLayer>> attn_norms;
  std::vector<std::shared_ptr<op::RmsNormLayer>> ffn_norms;
  attn_norms.reserve(layer_num);
  ffn_norms.reserve(layer_num);

  for (int32_t l = 0; l < layer_num; ++l) {
    // input_layernorm
    auto attn_norm = std::make_shared<op::RmsNormLayer>(device_type_, dim);
    attn_norm->set_weight(0, {dim}, raw_model_data_->weight(pos), cpu_device_type);
    pos += dim;
    attn_norms.push_back(attn_norm);

    //q
    auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, false, true);
    wq->set_weight(0, {dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
    pos += static_cast<size_t>(dim) * dim;
    int32_t q_bias_dim = dim;
    wq->set_bias(0, q_bias_dim, raw_model_data_->weight(pos), cpu_device_type);
    pos += dim;
    qwen_layers_->wq_layers_.push_back(wq);

    // k
    auto wk = std::make_shared<op::MatmulLayer>(device_type_, kv_dim, dim, false, true);
    wk->set_weight(0, {kv_dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
    pos += static_cast<size_t>(kv_dim) * dim;
    int32_t k_bias_dim = kv_dim;
    wk->set_bias(0, k_bias_dim, raw_model_data_->weight(pos), cpu_device_type);
    pos += kv_dim;
    qwen_layers_->wk_layers_.push_back(wk);

    //v
    auto wv = std::make_shared<op::MatmulLayer>(device_type_, kv_dim, dim, false, true);
    wv->set_weight(0, {kv_dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
    pos += static_cast<size_t>(kv_dim) * dim;
    int32_t v_bias_dim = kv_dim;
    wv->set_bias(0, v_bias_dim, raw_model_data_->weight(pos), cpu_device_type);
    pos += kv_dim;
    qwen_layers_->wv_layers_.push_back(wv);

    // o
    auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, dim);
    wo->set_weight(0, {dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
    pos += static_cast<size_t>(dim) * dim;
    qwen_layers_->wo_layers_.push_back(wo);

    //post_attention_rmsnorm
    auto ffn_norm = std::make_shared<op::RmsNormLayer>(device_type_, dim);
    ffn_norm->set_weight(0, {dim}, raw_model_data_->weight(pos), cpu_device_type);
    pos += dim;
    ffn_norms.push_back(ffn_norm);

    //router: [num_experts, dim], no bias
    auto router = std::make_shared<op::MatmulLayer>(device_type_, moe_experts, dim, false, false);
    router->set_weight(0, {moe_experts, dim}, raw_model_data_->weight(pos), cpu_device_type);
    pos += static_cast<size_t>(moe_experts) * dim;
    qwen_layers_->router_layers_[l] = router;

    //experts
    qwen_layers_->expert_w1_layers_[l].reserve(moe_experts);
    qwen_layers_->expert_w2_layers_[l].reserve(moe_experts);
    qwen_layers_->expert_w3_layers_[l].reserve(moe_experts);

    for (int32_t e = 0; e < moe_experts; ++e) {
      auto ew1 = std::make_shared<op::MatmulLayer>(device_type_, moe_hidden, dim);
      ew1->set_weight(0, {moe_hidden, dim}, raw_model_data_->weight(pos), cpu_device_type);
      pos += static_cast<size_t>(moe_hidden) * dim;
      qwen_layers_->expert_w1_layers_[l].push_back(ew1);

      auto ew2 = std::make_shared<op::MatmulLayer>(device_type_, dim, moe_hidden);
      ew2->set_weight(0, {dim, moe_hidden}, raw_model_data_->weight(pos), cpu_device_type);
      pos += static_cast<size_t>(dim) * moe_hidden;
      qwen_layers_->expert_w2_layers_[l].push_back(ew2);

      auto ew3 = std::make_shared<op::MatmulLayer>(device_type_, moe_hidden, dim);
      ew3->set_weight(0, {moe_hidden, dim}, raw_model_data_->weight(pos), cpu_device_type);
      pos += static_cast<size_t>(moe_hidden) * dim;
      qwen_layers_->expert_w3_layers_[l].push_back(ew3);

      if (e == 0) {
        qwen_layers_->w1_layers_.push_back(ew1);
        qwen_layers_->w2_layers_.push_back(ew2);
        qwen_layers_->w3_layers_.push_back(ew3);
      }
    }

    if (has_shared) {
      auto sw1 = std::make_shared<op::MatmulLayer>(device_type_, shared_hidden, dim);
      sw1->set_weight(0, {shared_hidden, dim}, raw_model_data_->weight(pos), cpu_device_type);
      pos += static_cast<size_t>(shared_hidden) * dim;
      qwen_layers_->shared_w1_layers_[l] = sw1;

      auto sw2 = std::make_shared<op::MatmulLayer>(device_type_, dim, shared_hidden);
      sw2->set_weight(0, {dim, shared_hidden}, raw_model_data_->weight(pos), cpu_device_type);
      pos += static_cast<size_t>(dim) * shared_hidden;
      qwen_layers_->shared_w2_layers_[l] = sw2;

      auto sw3 = std::make_shared<op::MatmulLayer>(device_type_, shared_hidden, dim);
      sw3->set_weight(0, {shared_hidden, dim}, raw_model_data_->weight(pos), cpu_device_type);
      pos += static_cast<size_t>(shared_hidden) * dim;
      qwen_layers_->shared_w3_layers_[l] = sw3;

      auto sgate = std::make_shared<op::MatmulLayer>(device_type_, 1, dim, false, false);
      sgate->set_weight(0, {1, dim}, raw_model_data_->weight(pos), cpu_device_type);
      pos += dim;
      qwen_layers_->shared_gate_layers_[l] = sgate;
    }
  }
  // final norm
  auto final_norm = std::make_shared<op::RmsNormLayer>(device_type_, dim);
  final_norm->set_weight(0, {dim}, raw_model_data_->weight(pos), cpu_device_type);
  pos += dim;
  // 按旧索引约定拼装 rmsnorm_layers_
  for (auto& n : attn_norms) qwen_layers_->rmsnorm_layers_.push_back(n);
  for (auto& n : ffn_norms) qwen_layers_->rmsnorm_layers_.push_back(n);
  qwen_layers_->rmsnorm_layers_.push_back(final_norm);

  // lm_head
  auto cls = std::make_shared<op::MatmulLayer>(device_type_, vocab, dim);
  if (config_->is_shared_weight_) {
    cls->set_weight(0, {vocab, dim}, raw_model_data_->weight(0), cpu_device_type);
  } else {
    cls->set_weight(0, {vocab, dim}, raw_model_data_->weight(pos), cpu_device_type);
    pos += static_cast<size_t>(vocab) * dim;
  }
  qwen_layers_->cls_layer_ = cls;
}


void QwenMoeModel::init_mem() {
  std::shared_ptr<base::DeviceAllocator> alloc;
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    alloc = base::CPUDeviceAllocatorFactory::get_instance();
  } else {
    alloc = base::CUDADeviceAllocatorFactory::get_instance();
  }

  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK_NE(cuda_config_, nullptr);
    qwen_layers_->to_cuda(cuda_config_);
  }

  std::shared_ptr<base::DeviceAllocator> alloc_cpu =
      base::CPUDeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::DeviceAllocator> alloc_cu =
      base::CUDADeviceAllocatorFactory::get_instance();

  tensor::Tensor input_tokens(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  tensor::Tensor input_embeddings(base::DataType::kDataTypeFp32, 1, config_->dim_, true, alloc);
  tensor::Tensor sin_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,
                           true, alloc);
  tensor::Tensor cos_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,
                           true, alloc);

  CHECK(insert_buffer(ModelBufferType::kSinCache, sin_cache));
  CHECK(insert_buffer(ModelBufferType::kCosCache, cos_cache));

  CHECK(insert_buffer(ModelBufferType::kInputTokens, input_tokens));
  CHECK(insert_buffer(ModelBufferType::kInputEmbeddings, input_embeddings));

  tensor::Tensor rms_output(base::DataType::kDataTypeFp32, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm, rms_output));
  CHECK(insert_buffer(ModelBufferType::kOutputMHA, rms_output));
  CHECK(insert_buffer(ModelBufferType::kW2Output, rms_output));
  CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, rms_output));

  tensor::Tensor w1_output(base::DataType::kDataTypeFp32, config_->hidden_dim_, true, alloc);
  tensor::Tensor w3_output(base::DataType::kDataTypeFp32, config_->hidden_dim_, true, alloc);

  CHECK(insert_buffer(ModelBufferType::kW1Output, w1_output));
  CHECK(insert_buffer(ModelBufferType::kW3Output, w3_output));

  // kv cache
  tensor::Tensor key_cache(base::DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_,
                           config_->kv_dim_, true, alloc);
  tensor::Tensor value_cache(base::DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_,
                             config_->kv_dim_, true, alloc);

  CHECK(insert_buffer(ModelBufferType::kKeyCache, key_cache));
  CHECK(insert_buffer(ModelBufferType::kValueCache, value_cache));

  // Wq query output
  tensor::Tensor query(base::DataType::kDataTypeFp32, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kQuery, query));

  // Pos tensor
  tensor::Tensor pos_tensor(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  CHECK(insert_buffer(ModelBufferType::kInputPos, pos_tensor));

  // Attention output
  tensor::Tensor attn(base::DataType::kDataTypeFp32, config_->head_num_, config_->seq_len_, true,
                      alloc);
  CHECK(insert_buffer(ModelBufferType::kScoreStorage, attn));
  CHECK(insert_buffer(ModelBufferType::kAttnOutput, query));

  //router_logits
  tensor::Tensor router_logits(base::DataType::kDataTypeFp32, config_->moe_expert_num_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kRouterLogits, router_logits));

  //topk output
  tensor::Tensor topk_value(base::DataType::kDataTypeFp32, config_->moe_topk_, true, alloc_cpu);
  tensor::Tensor topk_index(base::DataType::kDataTypeInt32, config_->moe_topk_, true, alloc_cpu);
  CHECK(insert_buffer(ModelBufferType::kTopKValue, topk_value));
  CHECK(insert_buffer(ModelBufferType::kTopKIndex, topk_index));
  
  //MoE sum
  tensor::Tensor moe_sum(base::DataType::kDataTypeFp32, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kMoeAccum, moe_sum));

  //MoE single buffer
  int32_t max_hidden = std::max(config_->moe_hidden_dim_, config_->moe_shared_hidden_dim_);
  tensor::Tensor expert_h1(base::DataType::kDataTypeFp32, max_hidden, true, alloc);
  tensor::Tensor expert_h2(base::DataType::kDataTypeFp32, max_hidden, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kExpertH1, expert_h1));
  CHECK(insert_buffer(ModelBufferType::kExpertH2, expert_h2));
  
  //output of single expert
  tensor::Tensor expert_output(base::DataType::kDataTypeFp32, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kExpertOutput, expert_output));

  //shared expert output
  tensor::Tensor shared_gate_out(base::DataType::kDataTypeFp32, 1, true, alloc_cpu);
  CHECK(insert_buffer(ModelBufferType::kSharedGateOutput, shared_gate_out));

  // final forward output
  tensor::Tensor forward_output(base::DataType::kDataTypeFp32, config_->vocab_size_, true, alloc);
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    tensor::Tensor forward_output_cpu(base::DataType::kDataTypeFp32, config_->vocab_size_, true,
                                      alloc_cpu);
    CHECK(insert_buffer(ModelBufferType::kForwardOutputCPU, forward_output_cpu));
  }

  CHECK(insert_buffer(ModelBufferType::kForwardOutput, forward_output));
}

base::Status QwenMoeModel::create_layers() {
  using namespace base;
  if (!qwen_layers_) {
    qwen_layers_ = std::make_unique<QwenMoeLayers>();
  }

  if (!is_quant_model_) {
    create_param_layers();
  } else {
    return error::InternalError("Quantized model is not supported for QwenMoeModel yet.");
    create_param_quant_layers();
  }
  create_nonparam_layers();

  if (!qwen_layers_->embedding_layer_) {
    return error::InternalError("Create the embedding layer for the llama model failed!");
  }

  if (qwen_layers_->rmsnorm_layers_.size() != 2 * config_->layer_num_ + 1) {
    return error::InternalError("Create the rmsnorm layers for the llama model failed!");
  }

  if (qwen_layers_->wq_layers_.size() != config_->layer_num_ ||
      qwen_layers_->wk_layers_.size() != config_->layer_num_ ||
      qwen_layers_->wv_layers_.size() != config_->layer_num_ ||
      qwen_layers_->wo_layers_.size() != config_->layer_num_) {
    return error::InternalError(
        "Create the matmul layer in the attention and ffn attention layers for "
        "the llama model "
        "failed.");
  }

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    if (!qwen_layers_->wq_layers_.at(i) || !qwen_layers_->wk_layers_.at(i) ||
        !qwen_layers_->wv_layers_.at(i) || !qwen_layers_->wo_layers_.at(i)) {
      return error::InternalError(
          "Create the matmul layer in the attention and ffn attention layers for "
          "the llama model "
          "failed.");
    }
  }

  if (qwen_layers_->w1_layers_.size() != config_->layer_num_ ||
      qwen_layers_->w2_layers_.size() != config_->layer_num_ ||
      qwen_layers_->w3_layers_.size() != config_->layer_num_) {
    return error::InternalError(
        "Create the matmul layer in the feedforward layers for the llama model "
        "failed.");
  }

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    if (!qwen_layers_->w1_layers_.at(i) || !qwen_layers_->w2_layers_.at(i) ||
        !qwen_layers_->w3_layers_.at(i)) {
      return error::InternalError(
          "Create the matmul layer in the feedforward layers for the llama model "
          "failed.");
    }
  }

  if (qwen_layers_->router_layers_.size() != config_->layer_num_) {
    return error::InternalError(
        "Create the matmul layer in the feedforward layers for the qwen model"
        "failded."
    );
  }
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    if (!qwen_layers_->router_layers_.at(i)) {
      return error::InternalError(
          "Create the matmul layer in the feedforward layers for the qwen model"
          "failded."
      );
    }
  }
  if (config_->moe_shared_expert_num_ > 0) {
    if (qwen_layers_->shared_gate_layers_.size() != config_->layer_num_ ||
        qwen_layers_->shared_w1_layers_.size() != config_->layer_num_ ||
        qwen_layers_->shared_w2_layers_.size() != config_->layer_num_ ||
        qwen_layers_->shared_w3_layers_.size() != config_->layer_num_) {
      return error::InternalError(
          "Create the matmul layer in the feedforward layers for the qwen model"
          "failded."
      );
    }
  }
  
  if (config_->moe_shared_expert_num_ > 0) {
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
      if (!qwen_layers_->shared_gate_layers_.at(i) ||
          !qwen_layers_->shared_w1_layers_.at(i) ||
          !qwen_layers_->shared_w2_layers_.at(i) ||
          !qwen_layers_->shared_w3_layers_.at(i)) {
            return error::InternalError(
              "Create the matmul layer in the feedforward layers for the qwen model"
              "failded."
            );
          }
    }
  }

  if (qwen_layers_->expert_w1_layers_.size() != config_->layer_num_ ||
      qwen_layers_->expert_w2_layers_.size() != config_->layer_num_ ||
      qwen_layers_->expert_w3_layers_.size() != config_->layer_num_) {
    return error::InternalError(
        "Create the matmul layer in the feedforward layers for the qwen model"
        "failded."
    );
  }
  for (int i = 0; i < config_->layer_num_; ++i) {
    if (qwen_layers_->expert_w1_layers_.at(i).size() != config_->moe_expert_num_ ||
        qwen_layers_->expert_w2_layers_.at(i).size() != config_->moe_expert_num_ ||
        qwen_layers_->expert_w3_layers_.at(i).size() != config_->moe_expert_num_) {
      return error::InternalError(
          "Create the matmul layer in the feedforward layers for the qwen model"
          "failded."
      );
    }
    for (int e = 0; e < config_->moe_expert_num_; ++e) {
      if (!qwen_layers_->expert_w1_layers_.at(i).at(e) ||
          !qwen_layers_->expert_w2_layers_.at(i).at(e) ||
          !qwen_layers_->expert_w3_layers_.at(i).at(e)) {
        return error::InternalError(
            "Create the matmul layer in the feedforward layers for the qwen model"
            "failded."
        );
      }
    }
  }

  if (!qwen_layers_->rope_layer_) {
    return error::InternalError("Create the rope layer for the llama model failed!");
  }

  if (!qwen_layers_->add_layer_) {
    return error::InternalError("Create the add layer for the llama model failed!");
  }

  if (!qwen_layers_->mha_layer_) {
    return error::InternalError("Create the mha layer for the llama model failed!");
  }

  if (!qwen_layers_->swiglu_layer_) {
    return error::InternalError("Create the SwiGLU layer for the llama model failed!");
  }
  if (!qwen_layers_->cls_layer_) {
    return error::InternalError("Create the cls layer for the llama model failed!");
  }
  return error::Success();
}

void QwenMoeModel::attention_rms(int32_t layer_idx, const tensor::Tensor& input) const {
  CHECK(qwen_layers_ != nullptr);
  // attn rmsnorm
  tensor::Tensor rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  std::shared_ptr<op::Layer> rmsnorm_layer = qwen_layers_->rmsnorm_layers_.at(layer_idx);
  if (!rmsnorm_layer) {
    LOG(FATAL) << "The attention rmsnorm layer is a null pointer in the llama2 model";
  }
  STATUS_CHECK(rmsnorm_layer->forward(input, rmsnorm_output));
}

void QwenMoeModel::attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(qwen_layers_ != nullptr);
  // kv cache
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
  int32_t pos = pos_tensor.index<int32_t>(0);
  // wq wk wv @ input
  const auto& [key, val] = slice_kv_cache(layer_idx, pos);
  // query
  const auto& query_layer = qwen_layers_->wq_layers_.at(layer_idx);
  CHECK_NE(query_layer, nullptr) << "The query layer in the attention block is null pointer.";

  auto rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  STATUS_CHECK(query_layer->forward(rmsnorm_output, query));

  // key
  const auto& key_layer = qwen_layers_->wk_layers_.at(layer_idx);
  CHECK_NE(key_layer, nullptr) << "The key layer in the attention block is null pointer.";
  STATUS_CHECK(key_layer->forward(rmsnorm_output, key));
  // value
  const auto& value_layer = qwen_layers_->wv_layers_.at(layer_idx);
  CHECK_NE(value_layer, nullptr) << "The value layer in the attention block is null pointer.";
  STATUS_CHECK(value_layer->forward(rmsnorm_output, val));

  // rope
  CHECK_NE(qwen_layers_->rope_layer_, nullptr)
      << "The RoPE layer in the attention block is null pointer.";
  STATUS_CHECK(qwen_layers_->rope_layer_->forward(
      query, key, pos_tensor, get_buffer(ModelBufferType::kSinCache),
      get_buffer(ModelBufferType::kCosCache), tensor::Tensor{}));
}

base::Status QwenMoeModel::predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                                 bool is_prompt, int& next) const {
  auto status = forward(input, pos_tensor, next);
  if (!status) {
    return status;
  }
  next = post_processing(pos_tensor, is_prompt);
  return base::error::Success();
}

void QwenMoeModel::attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(qwen_layers_ != nullptr);
  // mha
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  // VAL = [val1,val2,...val t]
  // output @ VAL = 最终的结果
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);

  tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
  tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);

  const auto& mha_layer = qwen_layers_->mha_layer_;
  CHECK_NE(mha_layer, nullptr) << "The multi head attention layer is null pointer.";
  int pos = pos_tensor.index<int32_t>(0);
  std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_pos(pos);
  std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_layer_idx(layer_idx);
  STATUS_CHECK(mha_layer->forward(query, score_storage, key_cache, val_cache, mha_output));

  // wo @ attention output
  tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
  const auto& wo_layer = qwen_layers_->wo_layers_.at(layer_idx);
  CHECK_NE(wo_layer, nullptr) << "The weight output layer is null pointer.";
  STATUS_CHECK(wo_layer->forward(mha_output, attn_output));
}

void QwenMoeModel::feed_forward(int32_t layer_idx, const tensor::Tensor& input) const {
  CHECK(qwen_layers_ != nullptr);
  // residual add
  CHECK_NE(qwen_layers_->add_layer_, nullptr)
      << "The add layer in the feedforward block is null pointer";
  STATUS_CHECK(
      qwen_layers_->add_layer_->forward(input, get_buffer(ModelBufferType::kAttnOutput), input));

  // ffn rmsnorm
  tensor::Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
  const auto& ffn_rmsnorm = qwen_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
  CHECK_NE(ffn_rmsnorm, nullptr)
      << "The final rmsnorm layer in the feedforward block is null pointer";
  STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_output));
  //MoE
  moe_accum_zero();
  moe_router_topk(layer_idx, ffn_norm_output);
  moe_routed_experts(layer_idx, ffn_norm_output);
  moe_shared_expert(layer_idx, ffn_norm_output);

  //residual scale_add
  moe_residual_add(input);

}

op::EmbeddingOutput QwenMoeModel::embedding(const std::vector<int>& tokens) const {
  auto input_tokens = get_buffer(ModelBufferType::kInputTokens);
  auto input_embeddings = get_buffer(ModelBufferType::kInputEmbeddings);
  if (input_tokens.size() != tokens.size()) {
    input_tokens.reshape({static_cast<int32_t>(tokens.size())});
    input_embeddings.reshape({static_cast<int32_t>(tokens.size()), config_->dim_});
  }
  for (int32_t i = 0; i < tokens.size(); ++i) {
    input_tokens.index<int32_t>(i) = tokens.at(i);
  }

  auto input_token_num =
      tensor::Tensor(base::DataType::kDataTypeInt32, static_cast<int32_t>(tokens.size()));
  LOG_IF(FATAL, !qwen_layers_->embedding_layer_)
      << "The embedding layer in the llama2 model is null pointer.";
  STATUS_CHECK(
      qwen_layers_->embedding_layer_->forward(input_tokens, input_token_num, input_embeddings));

  op::EmbeddingOutput output(input_tokens, input_embeddings, input_token_num);
  return output;
}


void QwenMoeModel::cls_logits(const tensor::Tensor& input) const {
  CHECK(qwen_layers_ != nullptr);
  const auto& norm = qwen_layers_->rmsnorm_layers_.at(2 * config_->layer_num_);
  CHECK_NE(norm, nullptr);
  STATUS_CHECK(norm->forward(input, input));

  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  CHECK_NE(qwen_layers_->cls_layer_, nullptr);
  STATUS_CHECK(qwen_layers_->cls_layer_->forward(input, forward_output));
}

int32_t QwenMoeModel::post_processing(const tensor::Tensor& pos, bool is_prompt) const {
  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  const float* forward_logits = forward_output.ptr<float>();

  int32_t next = 0;
  if (is_prompt) {
    next = -1;
  } else {
    next = static_cast<int32_t>(sampler_->sample(forward_logits, forward_output.size(),
                                                 cuda_config_ ? cuda_config_->stream : nullptr));
  }
  return next;
}

base::Status QwenMoeModel::generate_model_infos(const ModelConfig& config) const {
  auto status = Model::generate_model_infos(config);
  if (!status) {
    return status;
  }

  if (config_->moe_expert_num_ <= 0 || config_->moe_hidden_dim_ <= 0) {
    return base::error::ModelParseError(
        "QwenMoeModel requires valid MoE config (moe_expert_num_ > 0 and moe_hidden_dim_ > 0). "
        "Make sure the model file contains a MoeHeader, or use Qwen2Model for dense models.");
  }

  if (config_->moe_topk_ <= 0 || config_->moe_topk_ > config_->moe_expert_num_) {
    return base::error::ModelParseError(
        "QwenMoeModel requires valid MoE top-k config (0 < moe_topk_ <= moe_expert_num_).");
  }

  if (config_->moe_shared_expert_num_ < 0 ||
      config_->moe_shared_expert_num_ > config_->moe_expert_num_) {
    return base::error::ModelParseError(
        "Invalid MoE shared_expert_num_: must be in [0, moe_expert_num_].");
  }

  if (config_->moe_sparse_step_ <= 0) {
    return base::error::ModelParseError("Invalid MoE sparse step: moe_sparse_step_ must be > 0.");
  }
  if (config_->moe_shared_expert_num_ > 0 && config_->moe_shared_hidden_dim_ <= 0) {
    return base::error::ModelParseError(
      "Invalid MoE shared expert config: moe_shared_hidden_dim_ must be > 0 when shared experts are enabled.");
  }

  if (config_->moe_hidden_dim_ > 0) {
    config_->hidden_dim_ = config_->moe_hidden_dim_;
  }

  if (config_->moe_shared_expert_num_ == 0 && config_->moe_shared_hidden_dim_ > 0) {
    config_->moe_shared_expert_num_ = 1;
  }
  if (config_->moe_shared_expert_num_ > 1) {
    return base::error::ModelParseError("Current QwenMoeModel supports at most 1 shared expert.");
  }
  if (config_->moe_sparse_step_ != 1) {
    return base::error::ModelParseError("Current QwenMoeModel only supports decoder_sparse_step=1.");
  }
  
  return base::error::Success();
    
}

void QwenMoeModel::moe_router_topk(int32_t layer_idx, const tensor::Tensor& ffn_norm_output) const {
    tensor::Tensor router_logits = get_buffer(ModelBufferType::kRouterLogits);
    tensor::Tensor topk_value = get_buffer(ModelBufferType::kTopKValue);
    tensor::Tensor topk_index = get_buffer(ModelBufferType::kTopKIndex);
    const auto& router_layer = qwen_layers_->router_layers_.at(layer_idx);
    CHECK_NE(router_layer, nullptr);
    STATUS_CHECK(router_layer->forward(ffn_norm_output, router_logits));
    // top-k
    cpu_softmax_inplace(router_logits.ptr<float>(), router_logits.size());

    cpu_topk(router_logits.ptr<float>(), router_logits.size(), config_->moe_topk_, topk_value.ptr<float>(),
             topk_index.ptr<int32_t>());
    if(config_->moe_norm_topk_prob_) {
      float sum = 0.f;
      for (int i = 0; i < config_->moe_topk_; ++i) {
        sum += topk_value.index<float>(i);
      }
      for (int i = 0; i < config_->moe_topk_; ++i) {
        topk_value.index<float>(i) /= sum;
      }
    }
  }

  void QwenMoeModel::moe_accum_zero() const {
    tensor::Tensor moe_accum = get_buffer(ModelBufferType::kMoeAccum);
    CHECK(moe_accum.device_type() == device_type_);
    if (device_type_ == base::DeviceType::kDeviceCUDA) {
      CHECK_NE(cuda_config_, nullptr);
      cudaMemsetAsync(moe_accum.ptr<float>(), 0, moe_accum.byte_size(), cuda_config_->stream);
    } else {
      std::memset(moe_accum.ptr<float>(), 0, moe_accum.byte_size());
    }
  }

  void QwenMoeModel::moe_routed_experts(int32_t layer_idx, const tensor::Tensor& ffn_norm_output) const {
    auto topk_index = get_buffer(ModelBufferType::kTopKIndex);
    auto topk_value = get_buffer(ModelBufferType::kTopKValue);
    auto moe_accum = get_buffer(ModelBufferType::kMoeAccum);
    auto expert_h1 = get_buffer(ModelBufferType::kExpertH1);
    auto expert_h2 = get_buffer(ModelBufferType::kExpertH2);
    auto expert_output = get_buffer(ModelBufferType::kExpertOutput);
    for (int k = 0; k < config_->moe_topk_; ++k) {
      int32_t expert_idx = topk_index.index<int32_t>(k);
      float expert_weight = topk_value.index<float>(k);
      const auto& ew1 = qwen_layers_->expert_w1_layers_.at(layer_idx).at(expert_idx);
      const auto& ew2 = qwen_layers_->expert_w2_layers_.at(layer_idx).at(expert_idx);
      const auto& ew3 = qwen_layers_->expert_w3_layers_.at(layer_idx).at(expert_idx);

      // forward through the expert
      STATUS_CHECK(ew1->forward(ffn_norm_output, expert_h1));
      STATUS_CHECK(ew3->forward(ffn_norm_output, expert_h2));
      STATUS_CHECK(qwen_layers_->swiglu_layer_->forward(expert_h1, expert_h2, expert_h1));
      STATUS_CHECK(ew2->forward(expert_h1, expert_output));

      //CUDA kernel scale_add
      scale_add(moe_accum.ptr<float>(), expert_weight, expert_output.ptr<float>(), moe_accum.size());

    }
  }

  void QwenMoeModel::moe_shared_expert(int32_t layer_idx, const tensor::Tensor& ffn_norm_output) const {
    if (config_->moe_shared_expert_num_ <= 0) {
      return;
    }
    auto shared_gate_layer = qwen_layers_->shared_gate_layers_.at(layer_idx);
    auto shared_w1 = qwen_layers_->shared_w1_layers_.at(layer_idx);
    auto shared_w2 = qwen_layers_->shared_w2_layers_.at(layer_idx);
    auto shared_w3 = qwen_layers_->shared_w3_layers_.at(layer_idx);
    auto shared_gate_out = get_buffer(ModelBufferType::kSharedGateOutput);
    auto shared_h1 = get_buffer(ModelBufferType::kExpertH1);
    auto shared_h2 = get_buffer(ModelBufferType::kExpertH2);
    auto shared_output = get_buffer(ModelBufferType::kExpertOutput);

    // forward through the shared expert
    STATUS_CHECK(shared_w1->forward(ffn_norm_output, shared_h1));
    STATUS_CHECK(shared_w3->forward(ffn_norm_output, shared_h2));
    STATUS_CHECK(qwen_layers_->shared_swiglu_layer_->forward(shared_h1, shared_h2, shared_h1));
    STATUS_CHECK(shared_w2->forward(shared_h1, shared_output));
    STATUS_CHECK(shared_gate_layer->forward(ffn_norm_output, shared_gate_out));

    //sigmoid
    float gate_value = shared_gate_out.index<float>(0);
    gate_value = 1.f / (1.f + std::exp(-gate_value));

    //CUDA kernel scale_add
    auto moe_accum = get_buffer(ModelBufferType::kMoeAccum);
    scale_add(moe_accum.ptr<float>(), gate_value, shared_output.ptr<float>(), moe_accum.size());
  }
  
  void QwenMoeModel::moe_residual_add(const tensor::Tensor& input) const {
    auto moe_accum = get_buffer(ModelBufferType::kMoeAccum);
    // residual add
    CHECK_NE(qwen_layers_->add_layer_, nullptr)
        << "The add layer in the feedforward block is null pointer";
    STATUS_CHECK(
        qwen_layers_->add_layer_->forward(input, moe_accum, input));
  }

}  // namespace model


