#ifndef KUIPER_INCLUDE_SERVING_ROLE_PLACEMENT_H_
#define KUIPER_INCLUDE_SERVING_ROLE_PLACEMENT_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace serving {

struct RoleCostSnapshot {
  std::string worker;
  std::string incarnation;
  std::string model_revision;
  std::string representation;
  int64_t observed_monotonic_ms = 0;
  int64_t queued_tokens = 0;
  int64_t inflight_placements = 0;
  size_t admissible_bytes = 0;
  size_t resident_prefix_bytes = 0;
  double bandwidth_bytes_per_ms = 1.0;
  double prefill_ms_per_token = 0.0;
  double decode_ms_per_token = 0.0;
  double rpc_layout_ms = 0.0;
};

struct PlacementRequest {
  std::string model_revision;
  std::string representation;
  size_t required_bytes = 0;
  size_t prefix_bytes = 0;
  int64_t prefill_tokens = 0;
  int64_t decode_tokens = 0;
  int64_t now_monotonic_ms = 0;
  int64_t stale_after_ms = 1000;
  int64_t conservative_queue_tokens = 1024;
  int64_t max_inflight_per_worker = 8;
  double hysteresis_ms = 0.0;
  std::string incumbent;
  std::vector<std::string> excluded_workers;
};

struct CandidateCost {
  std::string worker;
  bool eligible = false;
  std::string reason;
  int64_t stats_age_ms = 0;
  bool stale = false;
  size_t missing_bytes = 0;
  double queue_wait_ms = 0.0;
  double transfer_ms = 0.0;
  double compute_ms = 0.0;
  double rpc_layout_ms = 0.0;
  double total_ms = 0.0;
};

struct PlacementDecision {
  bool found = false;
  std::string worker;
  std::string incarnation;
  double score_ms = 0.0;
  std::vector<CandidateCost> candidates;
};

PlacementDecision ChooseRole(const PlacementRequest& request,
                             const std::vector<RoleCostSnapshot>& snapshots);

}  // namespace serving
#endif
