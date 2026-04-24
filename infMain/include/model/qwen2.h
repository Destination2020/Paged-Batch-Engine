// Updated on March 31, 2026
#ifndef KUIPER_INCLUDE_MODEL_LLAMA_H_
#define KUIPER_INCLUDE_MODEL_LLAMA_H_
#include "base/backend_runtime.h"
#include "base/device_context.h"
#include "base/kv_cache_manager.h"
#include "serving/mixed_batch.h"
#include "serving/serving_capacity.h"
#include "model.h"
#include "op/add.h"
#include "op/embedding.h"
#include "op/rope.h"
#include "op/swiglu.h"

namespace model {
constexpr int32_t model_block_size = 16;
constexpr int32_t model_num_blocks = 30;
constexpr int32_t model_max_batch_size = 8;

class PagedKVRuntime;

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

  void materialize(std::shared_ptr<base::DeviceContext> context,
                   base::DataType runtime_data_type);
};

struct Qwen2HostDeviceInt32TensorPair {
  // Host side is normally pinned CPU memory; device side is backend-local
  // memory for the active execution device.
  // The pair is reused across steps to avoid allocating metadata buffers on
  // every scheduler/model call.
  tensor::Tensor host;
  tensor::Tensor device;
};

struct Qwen2SingleSeqForwardWorkspace {
  // Single-token forward_with_request() still uses the paged decode kernel.
  // These tensors hold the one-request block table and seq length passed to
  // attention, and are allocated once during init_mem().
  tensor::Tensor block_table_device;  // [max_blocks_per_seq] int32
  tensor::Tensor seq_lens_device;     // [1] int32
};

struct Qwen2BatchSamplerWorkspace {
  // Row indices select which logits rows should be sampled. token_ids stores
  // the sampled token results before they are exposed as SampledTokenView.
  Qwen2HostDeviceInt32TensorPair row_indices;
  Qwen2HostDeviceInt32TensorPair token_ids;

  // Recorded after the device-to-host token copy so batch_sample_device() can wait
  // only for the sampler stream work instead of synchronizing the whole device.
  std::shared_ptr<base::RuntimeEvent> copy_done_event;
};

struct Qwen2RowAttentionMetadataWorkspace {
  // forward_mixed_batch() expands request-level scheduling metadata to
  // per-token attention metadata. These buffers are resized up to the largest
  // observed batch and reused for both decode rows and chunked-prefill rows.
  Qwen2HostDeviceInt32TensorPair slot_mapping;
  Qwen2HostDeviceInt32TensorPair seq_lens;
  Qwen2HostDeviceInt32TensorPair block_tables;

  // Prefill-only metadata. These fields describe each token's chunk-local
  // offset and prefix context so the prefill attention kernel can combine
  // existing KV cache blocks with causal attention inside the current chunk.
  Qwen2HostDeviceInt32TensorPair prefill_base_context_lens;
  Qwen2HostDeviceInt32TensorPair prefill_chunk_row_starts;
  Qwen2HostDeviceInt32TensorPair prefill_local_token_offsets;
  Qwen2HostDeviceInt32TensorPair prefill_request_indices;
};

class Qwen2Model : public Model {
 public:
  explicit Qwen2Model(base::TokenizerType tokenizer_type, std::string token_path,
                      std::string model_path, bool is_quant_model);
  ~Qwen2Model();

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
  ServingWorkspaceProfile serving_workspace_profile(
      base::DataType runtime_data_type) const override;

  // Legacy setters kept for backward compatibility with demos.
  // Paged KV cache is now always enabled (the only supported path).
  void set_use_paged_kv(bool /*enable*/) {}
  void set_use_fp8_kv_cache(bool enable) {
    set_kv_cache_storage_mode(enable ? base::BlockStorageMode::kFp8E4M3PerTokenHead
                                     : base::BlockStorageMode::kPlain);
  }
  void set_radix_cache_enabled(bool enable);
  void set_kv_cache_memory_utilization(double utilization);
  void set_kv_cache_gpu_memory_utilization(double utilization);
  void set_serving_workspace_token_capacity(int32_t token_capacity);

  // Unified batch entry. Current implementation still executes decode rows only.
  base::Status forward_mixed_batch(const serving::MixedBatchMetadata& batch) const;

  // Dedicated decode-only fast path.
  base::Status forward_decode_batch(const serving::MixedBatchMetadata& batch) const;

  // Batch sample: argmax over [batch_size, vocab_size] forward output
  serving::SampledTokenView batch_sample(int32_t batch_size) const;
  serving::SampledTokenView batch_sample(const serving::MixedBatchMetadata& batch) const;

  // Prefill a chunk of prompt tokens for a request.
  // Returns next_token prediction from the last token in the chunk.
  int32_t prefill_chunk(base::RequestId request_id,
                        const std::vector<int32_t>& prompt_tokens,
                        int32_t start_pos, int32_t chunk_size);

  // Get KVCacheManager for external use (scheduler, prefill, etc.)
  base::KVCacheManager* kv_cache_manager() const { return kv_cache_manager_.get(); }

  serving::ServingCapacityInfo serving_capacity_info() const;

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

  void ensure_batch_sampler_index_capacity(int32_t index_count) const;
  void ensure_batch_sampler_workspace(int32_t sample_count) const;
  void ensure_batch_sampler_sync_event() const;
  void ensure_row_attention_metadata_workspace(int32_t token_capacity,
                                               int32_t request_capacity,
                                               int32_t block_table_capacity) const;
  serving::SampledTokenView batch_sample_device(const tensor::Tensor& logits,
                                                const tensor::Tensor& row_indices,
                                                int32_t sample_count) const;
  serving::SampledTokenView batch_sample_cpu(const tensor::Tensor& logits,
                                             int32_t num_rows,
                                             const std::vector<int32_t>& row_indices) const;

  int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) const override;

 private:
  std::unique_ptr<Qwen2Layers> qwen_layers_;
  std::shared_ptr<PagedKVRuntime> paged_kv_runtime_;
  mutable std::unique_ptr<base::KVCacheManager> kv_cache_manager_;
  double kv_cache_memory_utilization_ = 0.80;
  int32_t serving_workspace_token_capacity_ = model_max_batch_size;
  size_t serving_workspace_reserved_bytes_ = 0;
  bool radix_cache_enabled_override_set_ = false;
  bool radix_cache_enabled_override_ = true;

  // Single-sequence request ID for the simple forward() path
  mutable base::RequestId single_seq_request_id_ = -1;

  // These workspaces are mutable because the public forward/sample methods are
  // logically read-only with respect to model weights, but they update reusable
  // runtime scratch buffers. Qwen2Model is therefore not thread-safe for
  // concurrent forward/sample calls that share the same instance.
  mutable Qwen2SingleSeqForwardWorkspace single_seq_workspace_;
  mutable Qwen2BatchSamplerWorkspace batch_sampler_workspace_;
  mutable Qwen2RowAttentionMetadataWorkspace row_attn_workspace_;
};
}  // namespace model

#endif
