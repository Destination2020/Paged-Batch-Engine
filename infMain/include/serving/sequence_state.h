// Per-request sequence state for continuous batching
#ifndef KUIPER_INCLUDE_SERVING_SEQUENCE_STATE_H_
#define KUIPER_INCLUDE_SERVING_SEQUENCE_STATE_H_

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>
#include <glog/logging.h>
#include "base/kv_cache_manager.h"

namespace serving {

enum class SequenceStatus {
  kWaiting,
  kRunning,
  kPreempted,
  kFinished,
  kFailed,
};

struct SequenceState {
  base::RequestId request_id = -1;
  int64_t client_request_id = -1;
  std::vector<int32_t> prompt_tokens;
  std::vector<int32_t> output_tokens;
  int32_t max_new_tokens = 0;
  int32_t generated_tokens = 0;
  int32_t next_token = -1;
  bool finished = false;
  bool failed = false;
  bool first_token_recorded = false;
  bool finished_time_recorded = false;
  bool recompute_pending = false;
  bool radix_cache_published = false;
  SequenceStatus status = SequenceStatus::kWaiting;
  std::string finish_reason;
  std::string last_preempt_reason;
  int32_t preemption_count = 0;
  int64_t ready_step = 0;

  std::chrono::steady_clock::time_point arrival_time;
  std::chrono::steady_clock::time_point first_token_time;
  std::chrono::steady_clock::time_point last_token_time;
  std::chrono::steady_clock::time_point finished_time;

  // Chunked prefill tracking
  int32_t num_prompt_tokens_computed = 0;  // how many prompt tokens written to KV cache
  int32_t scheduled_tokens = 0;           // tokens allocated this step (transient)

  // Derived state
  int32_t prefill_target_tokens() const {
    return static_cast<int32_t>(prompt_tokens.size()) +
           (recompute_pending ? static_cast<int32_t>(output_tokens.size()) : 0);
  }

  bool is_prefill() const {
    return num_prompt_tokens_computed < prefill_target_tokens();
  }

  int32_t remaining_prompt_tokens() const {
    return prefill_target_tokens() - num_prompt_tokens_computed;
  }

  int32_t prefill_token_at(int32_t index) const {
    const int32_t prompt_size = static_cast<int32_t>(prompt_tokens.size());
    CHECK_GE(index, 0);
    CHECK_LT(index, prefill_target_tokens());
    if (index < prompt_size) {
      return prompt_tokens[index];
    }
    return output_tokens[index - prompt_size];
  }

  int32_t context_len() const {
    return static_cast<int32_t>(prompt_tokens.size()) +
           static_cast<int32_t>(output_tokens.size());
  }

  void record_generated_token(std::chrono::steady_clock::time_point now) {
    if (!first_token_recorded) {
      first_token_time = now;
      first_token_recorded = true;
    }
    last_token_time = now;
    ++generated_tokens;
  }

  void mark_finished(std::chrono::steady_clock::time_point now) {
    finished = true;
    status = SequenceStatus::kFinished;
    finished_time = now;
    finished_time_recorded = true;
  }

  void mark_failed(std::chrono::steady_clock::time_point now,
                   const std::string& reason) {
    failed = true;
    finish_reason = reason;
    mark_finished(now);
    status = SequenceStatus::kFailed;
  }

  double ttft_ms() const {
    if (!first_token_recorded) return 0.0;
    return duration_ms(arrival_time, first_token_time);
  }

  double itl_ms() const {
    if (generated_tokens <= 1 || !first_token_recorded) return 0.0;
    return duration_ms(first_token_time, last_token_time) /
           static_cast<double>(generated_tokens - 1);
  }

  double latency_ms() const {
    if (!finished_time_recorded) return 0.0;
    return duration_ms(arrival_time, finished_time);
  }

 private:
  static double duration_ms(std::chrono::steady_clock::time_point start,
                            std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
  }
};

}  // namespace serving

#endif  // KUIPER_INCLUDE_SERVING_SEQUENCE_STATE_H_
