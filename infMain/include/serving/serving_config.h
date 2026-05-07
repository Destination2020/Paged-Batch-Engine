// Common benchmark/serving launch configuration shared by serving demos.
#ifndef KUIPER_INCLUDE_SERVING_SERVING_CONFIG_H_
#define KUIPER_INCLUDE_SERVING_SERVING_CONFIG_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include "serving/scheduler.h"
#include "serving/serving_capacity.h"

namespace serving {

inline constexpr int32_t kAutoMaxBatchedTokensSafetyCap = 4096;
inline constexpr int32_t kAutoPrefillChunkCapSafetyCap = 1024;
inline constexpr double kDefaultKVCacheMemoryUtilization = 0.80;

struct AutoScheduleEstimate {
  int32_t usable_free_kv_blocks = 0;
  int32_t kv_token_budget = 0;
  size_t device_workspace_bytes_budget = 0;
  int32_t workspace_token_budget = 0;
  int32_t capacity_token_budget = 0;
  int32_t scheduler_seq_window = 0;
  int32_t target_decode_concurrency = 0;
  int32_t prefill_reserved_seqs = 0;
  int32_t target_decode_token_window = 0;
  int32_t prompt_prefill_chunk_target = 0;
  int32_t scheduler_token_budget = 0;
  int32_t safety_token_cap = 0;
  int32_t prefill_safety_token_cap = 0;
  int32_t raw_max_batched_tokens = 0;
  int32_t raw_prefill_chunk_cap = 0;
};

struct PromptTokenStats {
  int32_t prompt_count = 0;
  int32_t min_prompt_tokens = 0;
  int32_t p50_prompt_tokens = 0;
  int32_t p95_prompt_tokens = 0;
  int32_t max_prompt_tokens = 0;
  int32_t total_prompt_tokens = 0;
  double mean_prompt_tokens = 0.0;
};

struct BenchConfig {
  int32_t max_new_tokens = 256;
  int32_t max_num_batched_tokens = kAutoMaxBatchedTokensSafetyCap;
  int32_t prefill_chunk_cap = kAutoPrefillChunkCapSafetyCap;
  SchedulingPolicy scheduling_policy = SchedulingPolicy::kFCFS;
  int32_t long_prefill_token_threshold = 0;
  int32_t max_partial_prefills = 0;
  int32_t max_long_partial_prefills = 0;
  int32_t warmup_rounds = 0;
  double kv_cache_memory_utilization = kDefaultKVCacheMemoryUtilization;
  std::string max_num_batched_tokens_request = "auto";
  std::string prefill_chunk_cap_request = "auto";
  bool auto_max_num_batched_tokens = true;
  bool auto_prefill_chunk_cap = true;
  bool radix_cache_config_explicit = false;
  bool radix_cache_enabled = true;
  PromptTokenStats prompt_token_stats;
  AutoScheduleEstimate auto_estimate;
  ServingCapacityInfo capacity_info;
  bool quiet = false;
  bool print_step_profile = false;
  bool print_step_trace = false;
  bool print_final_summary = true;
  bool online_server = false;
  std::string listen_host = "127.0.0.1";
  int32_t listen_port = 8080;
  int32_t http_worker_threads = 0;   // 0 auto-selects a bounded worker pool
  int32_t http_listen_backlog = 1024;
  int32_t max_queue_size = 128;
  int32_t request_timeout_ms = 0;  // 0 disables request timeout
  int32_t max_prompt_tokens = 0;   // 0 disables prompt length check
  int32_t device_id = 0;
  std::string pd_mode = "off";
  int32_t prefill_device_id = 0;
  int32_t decode_device_id = 1;
  std::string online_process_role = "inproc";
  std::string engine_zmq_endpoint = "tcp://127.0.0.1:19090";
  std::string prefill_zmq_endpoint = "tcp://127.0.0.1:19091";
  int32_t engine_zmq_timeout_ms = 30000;
};

inline bool is_dual_gpu_pd_mode(const std::string& pd_mode) {
  return pd_mode == "dual-gpu-p2p" || pd_mode == "dual-gpu-nccl" ||
         pd_mode == "dual-gpu-nccl-layer";
}

inline bool is_remote_pd_mode(const std::string& pd_mode) {
  return pd_mode == "remote-zmq-cpu" || pd_mode == "remote-zmq-nccl" ||
         pd_mode == "remote-zmq-nccl-layer";
}

inline bool is_pd_mode(const std::string& pd_mode) {
  return is_dual_gpu_pd_mode(pd_mode) || is_remote_pd_mode(pd_mode);
}

inline const char* pd_transfer_backend(const std::string& pd_mode) {
  if (pd_mode == "remote-zmq-cpu") {
    return "zmq-cpu";
  }
  if (pd_mode == "remote-zmq-nccl") {
    return "zmq-nccl";
  }
  if (pd_mode == "remote-zmq-nccl-layer") {
    return "zmq-nccl-layer";
  }
  if (pd_mode == "dual-gpu-nccl" || pd_mode == "dual-gpu-nccl-layer") {
    return "nccl";
  }
  if (pd_mode == "dual-gpu-p2p") {
    return "p2p";
  }
  return "none";
}

}  // namespace serving

#endif  // KUIPER_INCLUDE_SERVING_SERVING_CONFIG_H_
