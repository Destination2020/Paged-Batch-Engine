#pragma once
#include <chrono>
#include <functional>
#include <mutex>
#include <unordered_map>
#include "serving/serving_zmq_rpc.h"

namespace serving {
class ServingRuntime;
// Owns PD orchestration and outstanding source leases. Runtime must outlive it.
class PDExecutor {
 public:
  explicit PDExecutor(ServingRuntime& runtime) : runtime_(runtime) {}
  struct PDGenerationResult {
    std::vector<int32_t> output_tokens;
    bool failed = false;
    std::string error;
  };

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
  ServingRuntime& runtime_;
};
} // namespace serving
