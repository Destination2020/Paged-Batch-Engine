#ifndef KUIPER_INCLUDE_SERVING_PREFIX_BUILDER_H_
#define KUIPER_INCLUDE_SERVING_PREFIX_BUILDER_H_

#include <cstdint>
#include <functional>
#include <vector>

#include "base/kv_cache_manager.h"

namespace serving {

struct PrefixBuilderStats {
  int64_t tasks_started = 0;
  int64_t tasks_succeeded = 0;
  int64_t tasks_failed = 0;
  int64_t computed_tokens = 0;
};

// An explicit synchronous prefill task. It owns a temporary request only for
// computation; successful full pages remain discoverable in the radix cache
// after that request exits and are bounded by the KV allocator budget.
class PrefixBuilder {
 public:
  using Compute = std::function<bool(base::RequestId)>;
  explicit PrefixBuilder(base::KVCacheManager* manager) : manager_(manager) {}

  bool build(const std::vector<int32_t>& multimodal_prefix, const Compute& compute);
  const PrefixBuilderStats& stats() const { return stats_; }

 private:
  base::KVCacheManager* manager_ = nullptr;
  PrefixBuilderStats stats_;
};

}  // namespace serving
#endif
