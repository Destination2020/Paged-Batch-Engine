// Common serving metrics structures shared by serving demos.
#ifndef KUIPER_INCLUDE_SERVING_SERVING_METRICS_H_
#define KUIPER_INCLUDE_SERVING_SERVING_METRICS_H_

#include <cstdint>
#include <vector>

namespace serving {

struct StepProfile {
  int32_t step = 0;
  int32_t num_requests = 0;
  int32_t num_tokens = 0;
  int32_t num_decode_tokens = 0;
  int32_t num_prefill_tokens = 0;
  int32_t num_sampled_tokens = 0;
  double schedule_ms = 0.0;
  double build_metadata_ms = 0.0;
  double forward_ms = 0.0;
  double sample_ms = 0.0;
  double process_outputs_ms = 0.0;
  double step_ms = 0.0;
};

struct SummaryStats {
  int32_t total_steps = 0;
  int32_t active_steps = 0;
  int32_t decode_only_steps = 0;
  int32_t total_decode_tokens = 0;
  int32_t total_prefill_tokens = 0;
  int32_t total_batched_tokens = 0;
  int32_t completed_requests = 0;
  int32_t failed_requests = 0;
  std::vector<double> request_ttft_ms;
  std::vector<double> request_itl_ms;
  std::vector<double> request_latency_ms;
  double total_schedule_ms = 0.0;
  double total_build_metadata_ms = 0.0;
  double total_forward_ms = 0.0;
  double total_sample_ms = 0.0;
  double total_process_outputs_ms = 0.0;
  double total_step_ms = 0.0;
};

}  // namespace serving

#endif  // KUIPER_INCLUDE_SERVING_SERVING_METRICS_H_
