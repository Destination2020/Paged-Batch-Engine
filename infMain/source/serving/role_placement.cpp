#include "serving/role_placement.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace serving {

PlacementDecision ChooseRole(const PlacementRequest& request,
                             const std::vector<RoleCostSnapshot>& snapshots) {
  PlacementDecision result;
  double best = std::numeric_limits<double>::infinity();
  size_t best_index = snapshots.size();
  size_t incumbent_index = snapshots.size();
  for (size_t i = 0; i < snapshots.size(); ++i) {
    const auto& role = snapshots[i];
    CandidateCost cost;
    cost.worker = role.worker;
    cost.stats_age_ms = std::max<int64_t>(0, request.now_monotonic_ms -
                                                role.observed_monotonic_ms);
    cost.stale = cost.stats_age_ms > request.stale_after_ms;
    if (std::find(request.excluded_workers.begin(), request.excluded_workers.end(),
                  role.worker) != request.excluded_workers.end()) {
      cost.reason = "excluded_after_reserve_failure";
    } else if (role.model_revision != request.model_revision ||
               role.representation != request.representation) {
      cost.reason = "incompatible_model_or_representation";
    } else if (role.admissible_bytes < request.required_bytes) {
      cost.reason = "insufficient_admission_capacity";
    } else if (role.inflight_placements >= request.max_inflight_per_worker) {
      cost.reason = "hotspot_inflight_limit";
    } else if (role.bandwidth_bytes_per_ms <= 0.0 ||
               role.prefill_ms_per_token < 0.0 || role.decode_ms_per_token < 0.0) {
      cost.reason = "invalid_calibration";
    } else {
      cost.eligible = true;
      cost.reason = cost.stale ? "eligible_with_stale_conservative_queue"
                               : "eligible";
      const int64_t queued = cost.stale
          ? std::max(role.queued_tokens, request.conservative_queue_tokens)
          : role.queued_tokens;
      cost.queue_wait_ms = queued * role.decode_ms_per_token;
      cost.missing_bytes = request.prefix_bytes > role.resident_prefix_bytes
          ? request.prefix_bytes - role.resident_prefix_bytes : 0;
      cost.transfer_ms = cost.missing_bytes / role.bandwidth_bytes_per_ms;
      cost.compute_ms = request.prefill_tokens * role.prefill_ms_per_token +
                        request.decode_tokens * role.decode_ms_per_token;
      cost.rpc_layout_ms = role.rpc_layout_ms;
      cost.total_ms = cost.queue_wait_ms + cost.transfer_ms +
                      cost.compute_ms + cost.rpc_layout_ms;
      if (cost.total_ms < best) {
        best = cost.total_ms;
        best_index = i;
      }
      if (role.worker == request.incumbent) incumbent_index = i;
    }
    result.candidates.push_back(std::move(cost));
  }
  if (best_index == snapshots.size()) return result;
  if (incumbent_index != snapshots.size()) {
    const auto& incumbent_cost = result.candidates[incumbent_index];
    if (incumbent_cost.eligible &&
        incumbent_cost.total_ms <= best + request.hysteresis_ms) {
      best_index = incumbent_index;
      best = incumbent_cost.total_ms;
    }
  }
  result.found = true;
  result.worker = snapshots[best_index].worker;
  result.incarnation = snapshots[best_index].incarnation;
  result.score_ms = best;
  return result;
}

}  // namespace serving
