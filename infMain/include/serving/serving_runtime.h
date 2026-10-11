#pragma once
#include <chrono>
#include <functional>
#include <mutex>
#include <unordered_map>
#include "base/kv_cache_manager.h"
#include "serving/mixed_batch.h"
#include "serving/serving_capacity.h"
#include "serving/serving_config.h"
#include "serving/pd_handoff.h"
#include "serving/serving_zmq_rpc.h"

namespace serving {
// Model execution contract shared by online serving and offline benchmarking.
// Online components depend on this interface, not command-line or metrics code.
class ServingRuntime {
 public:
  virtual ~ServingRuntime() = default;
  virtual const BenchConfig& bench_config() const = 0;
  virtual std::string postprocess_decoded_text(std::string text) const { return text; }
  struct PDGenerationResult {
    std::vector<int32_t> output_tokens;
    bool failed = false;
    std::string error;
  };

  virtual int32_t max_model_batch_size() const = 0;
  virtual base::KVCacheManager* kv_cache_manager() const = 0;
  virtual ServingCapacityInfo serving_capacity_info() const = 0;
  virtual void* model_stream() const = 0;
  virtual std::vector<int32_t> encode_prompt(const std::string& user_prompt) const = 0;
  virtual std::string decode_tokens(const std::vector<int32_t>& token_ids) const = 0;
  virtual bool is_sentence_ending(int32_t token) const = 0;
  virtual base::Status forward_mixed_batch(const MixedBatchMetadata& batch) const = 0;
  virtual base::Status forward_decode_batch(const MixedBatchMetadata& batch) const = 0;
  virtual SampledTokenView batch_sample(const MixedBatchMetadata& batch,
                                      const SchedulerOutput& sched_out) const = 0;

  virtual bool pd_dual_gpu_supported() const { return false; }
  virtual base::KVCacheManager* pd_prefill_kv_cache_manager() const {
    return kv_cache_manager();
  }
  virtual base::KVCacheManager* pd_decode_kv_cache_manager() const {
    return kv_cache_manager();
  }
  virtual ServingCapacityInfo pd_prefill_serving_capacity_info() const {
    return serving_capacity_info();
  }
  virtual ServingCapacityInfo pd_decode_serving_capacity_info() const {
    return serving_capacity_info();
  }
  virtual KVPoolDescriptor pd_prefill_kv_pool() const;
  virtual KVPoolDescriptor pd_decode_kv_pool() const;
  virtual void* pd_prefill_stream() const { return model_stream(); }
  virtual void* pd_decode_stream() const { return model_stream(); }
  virtual void* pd_transfer_stream() const { return pd_decode_stream(); }
  virtual void set_pd_prefill_layer_kv_connector(
      LayerKVTransferConnector* connector,
      LayerKVConnectorRole role) const {
    UNUSED(connector);
    UNUSED(role);
  }
  virtual void set_pd_decode_layer_kv_connector(
      LayerKVTransferConnector* connector,
      LayerKVConnectorRole role) const {
    UNUSED(connector);
    UNUSED(role);
  }
  virtual base::Status pd_forward_prefill_batch(
      const MixedBatchMetadata& batch) const {
    return forward_mixed_batch(batch);
  }
  virtual base::Status pd_forward_decode_batch(
      const MixedBatchMetadata& batch) const {
    return forward_decode_batch(batch);
  }
  virtual SampledTokenView pd_batch_sample_prefill(
      const MixedBatchMetadata& batch,
      const SchedulerOutput& sched_out) const {
    return batch_sample(batch, sched_out);
  }
  virtual SampledTokenView pd_batch_sample_decode(
      const MixedBatchMetadata& batch,
      const SchedulerOutput& sched_out) const {
    return batch_sample(batch, sched_out);
  }
  PDGenerationResult run_dual_gpu_pd_generation(
      std::vector<int32_t> prompt_tokens,
      GenerationConfig generation_config,
      const std::function<void(int32_t)>& on_token = nullptr) const;
  PDGenerationResult run_dual_gpu_p2p_generation(
      std::vector<int32_t> prompt_tokens,
      GenerationConfig generation_config,
      const std::function<void(int32_t)>& on_token = nullptr) const {
    return run_dual_gpu_pd_generation_with_mode(
        "dual-gpu-p2p", std::move(prompt_tokens), std::move(generation_config),
        on_token);
  }
  PDGenerationResult run_dual_gpu_nccl_generation(
      std::vector<int32_t> prompt_tokens,
      GenerationConfig generation_config,
      const std::function<void(int32_t)>& on_token = nullptr) const {
    return run_dual_gpu_pd_generation_with_mode(
        "dual-gpu-nccl", std::move(prompt_tokens), std::move(generation_config),
        on_token);
  }
  PDGenerationResult run_dual_gpu_nccl_layer_generation(
      std::vector<int32_t> prompt_tokens,
      GenerationConfig generation_config,
      const std::function<void(int32_t)>& on_token = nullptr) const {
    return run_dual_gpu_pd_generation_with_mode(
        "dual-gpu-nccl-layer", std::move(prompt_tokens),
        std::move(generation_config), on_token);
  }
  RemotePrefillResult run_remote_prefill_generation(
      std::vector<int32_t> prompt_tokens,
      GenerationConfig generation_config) const;
  base::Status run_remote_nccl_kv_send(const KVBlockManifest& manifest,
                                       const std::string& nccl_unique_id) const;
  void retain_remote_prefill(HandoffId handoff_id,
                             base::RequestId request_id) const;
  void release_remote_prefill(HandoffId handoff_id) const;
  PDGenerationResult run_remote_zmq_cpu_pd_generation(
      std::vector<int32_t> prompt_tokens,
      GenerationConfig generation_config,
      const std::function<void(int32_t)>& on_token = nullptr) const;
  PDGenerationResult run_remote_zmq_nccl_pd_generation(
      std::vector<int32_t> prompt_tokens,
      GenerationConfig generation_config,
      const std::function<void(int32_t)>& on_token = nullptr) const;


 private:
  using Clock = std::chrono::steady_clock;
  PDGenerationResult run_dual_gpu_pd_generation_with_mode(
      const std::string& pd_mode,
      std::vector<int32_t> prompt_tokens,
      GenerationConfig generation_config,
      const std::function<void(int32_t)>& on_token) const;
  struct PendingRemotePrefill {
    base::RequestId request_id = -1;
  };
  mutable std::mutex pending_remote_prefills_mu_;
  mutable std::unordered_map<uint64_t, PendingRemotePrefill>
      pending_remote_prefills_;
};
} // namespace serving
