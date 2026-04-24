// Common benchmark/serving launch configuration shared by serving demos.
#ifndef KUIPER_INCLUDE_SERVING_SERVING_CONFIG_H_
#define KUIPER_INCLUDE_SERVING_SERVING_CONFIG_H_

#include <cstddef>
#include <cstdint>
#include <string>
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
};

}  // namespace serving

#endif  // KUIPER_INCLUDE_SERVING_SERVING_CONFIG_H_
