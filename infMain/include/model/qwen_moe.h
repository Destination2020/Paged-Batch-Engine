// Updated on March 15, 2026
#ifndef HighPer_INCLUDE_MODEL_QWEN_MOE_H_
#define HighPer_INCLUDE_MODEL_QWEN_MOE_H_
#include "base/device_context.h"
#include "model.h"
#include "op/add.h"
#include "op/embedding.h"
#include "op/rope.h"
#include "op/swiglu.h"

namespace kernel {
struct CudaConfig;
}

namespace model {

struct QwenMoeLayers {
  std::shared_ptr<op::Layer> add_layer_;
  std::shared_ptr<op::Layer> rope_layer_;
  std::shared_ptr<op::Layer> swiglu_layer_;
  std::shared_ptr<op::Layer> shared_swiglu_layer_;
  std::shared_ptr<op::Layer> mha_layer_;

  std::vector<std::shared_ptr<op::Layer>> wq_layers_;
  std::vector<std::shared_ptr<op::Layer>> wk_layers_;
  std::vector<std::shared_ptr<op::Layer>> wv_layers_;
  std::vector<std::shared_ptr<op::Layer>> wo_layers_;

  std::vector<std::shared_ptr<op::Layer>> w1_layers_;
  std::vector<std::shared_ptr<op::Layer>> w2_layers_;
  std::vector<std::shared_ptr<op::Layer>> rmsnorm_layers_;
  std::vector<std::shared_ptr<op::Layer>> w3_layers_;

  std::vector<std::shared_ptr<op::Layer>> router_layers_;
  std::vector<std::vector<std::shared_ptr<op::Layer>>> expert_w1_layers_;
  std::vector<std::vector<std::shared_ptr<op::Layer>>> expert_w2_layers_;
  std::vector<std::vector<std::shared_ptr<op::Layer>>> expert_w3_layers_;

  std::vector<std::shared_ptr<op::Layer>>  shared_gate_layers_;
  std::vector<std::shared_ptr<op::Layer>> shared_w1_layers_;
  std::vector<std::shared_ptr<op::Layer>> shared_w2_layers_;
  std::vector<std::shared_ptr<op::Layer>> shared_w3_layers_;

  std::shared_ptr<op::Layer> cls_layer_;

  std::shared_ptr<op::Layer> embedding_layer_;

  void materialize(std::shared_ptr<base::DeviceContext> context,
                   base::DataType runtime_data_type);

  // Legacy compatibility helper. Prefer materialize() in new code.
  void to_cuda(std::shared_ptr<base::DeviceContext> context, base::DataType runtime_data_type);
};

class QwenMoeModel : public Model {
 public:
  explicit QwenMoeModel(base::TokenizerType tokenizer_type, std::string token_path,
                        std::string model_path, bool is_quant_model);

  base::Status init(base::DeviceType device_type) override;

  base::Status predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       bool is_prompt, int& next) const override;

  base::Status forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       int& next) const override;

  op::EmbeddingOutput embedding(const std::vector<int>& tokens) const override;

  base::Status generate_model_infos(const ModelConfig& config) const override;

 private:
  void init_mem() override;

  base::Status create_layers() override;

  void create_param_layers() override;

  void create_nonparam_layers() override;

  void create_param_quant_layers() override;

  void attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

  void attention_rms(int32_t layer_idx, const tensor::Tensor& input) const;

  void feed_forward(int32_t layer_idx, const tensor::Tensor& input) const;

  void attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

  void cls_logits(const tensor::Tensor& input) const;

  int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) const override;

  void moe_router_topk(int32_t layer_idx, const tensor::Tensor& ffn_norm_output) const;

  void moe_routed_experts(int32_t layer_idx, const tensor::Tensor& ffn_norm_output) const;

  void moe_shared_expert(int32_t layer_idx, const tensor::Tensor& ffn_norm_output) const;

  void moe_accum_zero() const;
  
  void moe_residual_add(const tensor::Tensor& input) const;

 private:
  std::shared_ptr<kernel::CudaConfig> cuda_config_;
  std::unique_ptr<QwenMoeLayers> qwen_layers_;
};
}  // namespace model

#endif
