// Updated on March 31, 2026
#ifndef KUIPER_INCLUDE_MODEL_LLAMA_H_
#define KUIPER_INCLUDE_MODEL_LLAMA_H_
#include <base/cuda_config.h>
#include "base/kv_cache_manager.h"
#include "serving/decode_batch.h"
#include "model.h"
#include "op/add.h"
#include "op/embedding.h"
#include "op/rope.h"
#include "op/swiglu.h"
namespace model {
constexpr int32_t model_block_size = 16;
constexpr int32_t model_num_blocks = 30;
constexpr int32_t model_max_batch_size = 8;

struct Qwen2Layers {
  std::shared_ptr<op::Layer> add_layer_;
  std::shared_ptr<op::Layer> rope_layer_;
  std::shared_ptr<op::Layer> swiglu_layer_;
  std::shared_ptr<op::Layer> mha_layer_;

  std::vector<std::shared_ptr<op::Layer>> wq_layers_;
  std::vector<std::shared_ptr<op::Layer>> wk_layers_;
  std::vector<std::shared_ptr<op::Layer>> wv_layers_;
  std::vector<std::shared_ptr<op::Layer>> wo_layers_;

  std::vector<std::shared_ptr<op::Layer>> w1_layers_;
  std::vector<std::shared_ptr<op::Layer>> w2_layers_;
  std::vector<std::shared_ptr<op::Layer>> rmsnorm_layers_;
  std::vector<std::shared_ptr<op::Layer>> w3_layers_;
  std::shared_ptr<op::Layer> cls_layer_;

  std::shared_ptr<op::Layer> embedding_layer_;

  void to_cuda(std::shared_ptr<kernel::CudaConfig> config, base::DataType runtime_data_type);
};

class Qwen2Model : public Model {
 public:
  explicit Qwen2Model(base::TokenizerType tokenizer_type, std::string token_path,
                      std::string model_path, bool is_quant_model);

  base::Status init(base::DeviceType device_type) override;

  base::Status predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       bool is_prompt, int& next) const override;

  base::Status forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       int& next) const override;

  // Forward with explicit request_id (for scheduler prefill)
  base::Status forward_with_request(const tensor::Tensor& input,
                                    const tensor::Tensor& pos_tensor,
                                    base::RequestId request_id,
                                    int& next) const;

  // Predict with explicit request_id
  base::Status predict_with_request(const tensor::Tensor& input,
                                    const tensor::Tensor& pos_tensor,
                                    base::RequestId request_id,
                                    bool is_prompt, int& next) const;

  op::EmbeddingOutput embedding(const std::vector<int>& tokens) const override;

  // Legacy setters kept for backward compatibility with demos.
  // Paged KV cache is now always enabled (the only supported path).
  void set_use_paged_kv(bool /*enable*/) {}
  void set_use_fp8_kv_cache(bool enable) { use_fp8_kv_cache_ = enable; }

  // Batched decode forward for multi-sequence inference
  base::Status forward_decode_batch(const serving::DecodeBatchMetadata& batch) const;

  // Batch sample: argmax over [batch_size, vocab_size] forward output
  std::vector<int32_t> batch_sample(int32_t batch_size) const;

  // Get KVCacheManager for external use (scheduler, prefill, etc.)
  base::KVCacheManager* kv_cache_manager() const { return kv_cache_manager_.get(); }

  // Access cuda config (for scheduler prefill)
  const std::shared_ptr<kernel::CudaConfig>& cuda_config() const { return cuda_config_; }

 private:
  void init_mem() override;

  base::Status create_layers() override;

  void create_param_layers() override;

  void create_nonparam_layers() override;

  void create_param_quant_layers() override;

  // Single-sequence forward building blocks (use kv_cache_manager_ internally)
  void attention_rms(int32_t layer_idx, const tensor::Tensor& input) const;

  void feed_forward(int32_t layer_idx, const tensor::Tensor& input) const;

  void cls_logits(const tensor::Tensor& input) const;

  int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) const override;

 private:
  std::shared_ptr<kernel::CudaConfig> cuda_config_;
  std::unique_ptr<Qwen2Layers> qwen_layers_;
  bool use_fp8_kv_cache_ = false;
  mutable std::unique_ptr<base::KVCacheManager> kv_cache_manager_;

  // Single-sequence request ID for the simple forward() path
  // Registered once, reused across calls
  mutable base::RequestId single_seq_request_id_ = -1;
};
}  // namespace model

#endif
