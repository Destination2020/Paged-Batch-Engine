#include "serving/prefix_builder.h"

namespace serving {

bool PrefixBuilder::build(const std::vector<int32_t>& prefix,
                          const Compute& compute) {
  if (!manager_ || prefix.empty() || !compute) return false;
  ++stats_.tasks_started;
  const auto request = manager_->register_request_with_radix_cache(prefix);
  const int32_t reused = manager_->get_context_len(request);
  const int32_t remaining = static_cast<int32_t>(prefix.size()) - reused;
  bool ok = remaining >= 0 &&
      (remaining == 0 || manager_->append_slots(request, remaining)) &&
      compute(request) &&
      manager_->commit_kv(request, static_cast<int32_t>(prefix.size()));
  if (ok) {
    manager_->publish_radix_cache(request, prefix);
    ++stats_.tasks_succeeded;
    stats_.computed_tokens += remaining;
  } else {
    ++stats_.tasks_failed;
  }
  manager_->free_request(request);
  return ok;
}

}  // namespace serving
