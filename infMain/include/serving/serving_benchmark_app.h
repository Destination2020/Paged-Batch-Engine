// Generic continuous-batching benchmark app scaffold.
#ifndef KUIPER_INCLUDE_SERVING_SERVING_BENCHMARK_APP_H_
#define KUIPER_INCLUDE_SERVING_SERVING_BENCHMARK_APP_H_

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "base/base.h"
#include "base/kv_cache_manager.h"
#include "serving/mixed_batch.h"
#include "serving/pd_handoff.h"
#include "serving/scheduler.h"
#include "serving/serving_capacity.h"
#include "serving/serving_config.h"
#include "serving/serving_metrics.h"
#include "serving/serving_zmq_rpc.h"

namespace serving {

// ServingBenchmarkApp owns the common offline serving lifecycle:
// argument parsing, prompt submission, warmup, scheduler loop, profiling,
// request metrics, and final summary. Model-specific demos only need to
// implement the virtual hooks below.
class ServingBenchmarkApp {
 public:
  virtual ~ServingBenchmarkApp() = default;

  struct PDGenerationResult {
    std::vector<int32_t> output_tokens;
    bool failed = false;
    std::string error;
  };

  int run(int argc, char* argv[]);
  const BenchConfig& bench_config() const { return bench_config_; }
  virtual const char* usage_name() const = 0;
  virtual bool initialize_model(const std::string& model_path,
                                const std::string& tokenizer_path,
                                const BenchConfig& bench_config) = 0;
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
  RemotePrefillResult run_remote_prefill_layer_generation(
      std::vector<int32_t> prompt_tokens,
      GenerationConfig generation_config,
      const LayerKVTransferRequest& layer_request,
      const std::string& nccl_unique_id) const;
  base::Status run_remote_nccl_kv_send(const KVBlockManifest& manifest,
                                       const std::string& nccl_unique_id) const;
  void release_remote_prefill(HandoffId handoff_id) const;
  PDGenerationResult run_remote_zmq_cpu_pd_generation(
      std::vector<int32_t> prompt_tokens,
      GenerationConfig generation_config,
      const std::function<void(int32_t)>& on_token = nullptr) const;
  PDGenerationResult run_remote_zmq_nccl_pd_generation(
      std::vector<int32_t> prompt_tokens,
      GenerationConfig generation_config,
      const std::function<void(int32_t)>& on_token = nullptr) const;

  virtual std::vector<std::string> default_prompts() const;
  virtual std::string postprocess_decoded_text(std::string text) const;

 private:
  using Clock = std::chrono::steady_clock;

  bool parse_args(int argc, char* argv[]);
  void collect_prompts(int argc, char* argv[]);
  void prepare_benchmark_config();
  void run_warmup();
  void create_scheduler();
  void submit_all_requests();
  void submit_requests_to(Scheduler& scheduler,
                          int32_t max_new_tokens,
                          bool quiet) const;
  void run_serving_loop();
  int run_dual_gpu_pd_offline();
  PDGenerationResult run_dual_gpu_pd_generation_with_mode(
      const std::string& pd_mode,
      std::vector<int32_t> prompt_tokens,
      GenerationConfig generation_config,
      const std::function<void(int32_t)>& on_token) const;
  void run_serving_step(void* stream);
  StepProfile build_step_profile(const SchedulerOutput& sched_out,
                                 const MixedBatchMetadata& batch,
                                 size_t sampled_token_count,
                                 Clock::time_point schedule_start,
                                 Clock::time_point schedule_end,
                                 Clock::time_point build_start,
                                 Clock::time_point build_end,
                                 Clock::time_point forward_start,
                                 Clock::time_point forward_end,
                                 Clock::time_point sample_start,
                                 Clock::time_point sample_end,
                                 Clock::time_point process_start,
                                 Clock::time_point process_end,
                                 Clock::time_point step_start,
                                 Clock::time_point step_end) const;
  void process_finished(const std::vector<SequenceState>& finished);
  void record_no_progress_step(const SchedulerOutput& sched_out);
  void record_step_profile(const StepProfile& profile,
                           bool decode_only_step,
                           int32_t finished_count);
  void maybe_print_step_profile(const StepProfile& profile,
                                const MixedBatchMetadata& batch,
                                size_t sampled_token_count) const;
  void print_done(double duration, double throughput) const;

  std::string model_path_;
  std::string tokenizer_path_;
  BenchConfig bench_config_;
  std::vector<std::string> prompts_;
  std::unique_ptr<Scheduler> scheduler_;
  SummaryStats summary_;
  int32_t total_decode_steps_ = 0;
  int32_t step_ = 0;
  struct PendingRemotePrefill {
    base::RequestId request_id = -1;
  };
  mutable std::mutex pending_remote_prefills_mu_;
  mutable std::unordered_map<uint64_t, PendingRemotePrefill>
      pending_remote_prefills_;
};

}  // namespace serving

#endif  // KUIPER_INCLUDE_SERVING_SERVING_BENCHMARK_APP_H_
