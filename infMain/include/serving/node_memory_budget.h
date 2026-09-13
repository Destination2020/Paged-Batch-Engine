#ifndef KUIPER_INCLUDE_SERVING_NODE_MEMORY_BUDGET_H_
#define KUIPER_INCLUDE_SERVING_NODE_MEMORY_BUDGET_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace serving {

enum class MemoryPool : uint8_t {
  kWeights, kWorkspaces, kKV, kBundles, kStaging, kQuarantine, kHostCache, kCount
};

struct MemoryDemand { MemoryPool pool; size_t bytes; };
struct MemoryPoolState { size_t capacity = 0; size_t used = 0; size_t peak = 0; };

class NodeMemoryBudget {
 public:
  bool set_capacity(MemoryPool pool, size_t bytes);
  void freeze() { frozen_ = true; }
  bool frozen() const { return frozen_; }
  // All-or-nothing across pools. Failure changes no counter.
  bool reserve(const std::vector<MemoryDemand>& demands);
  void release(const std::vector<MemoryDemand>& demands);
  MemoryPoolState state(MemoryPool pool) const;
  bool invariant_holds() const;

 private:
  static size_t index(MemoryPool pool) { return static_cast<size_t>(pool); }
  std::array<MemoryPoolState, static_cast<size_t>(MemoryPool::kCount)> pools_{};
  bool frozen_ = false;
};

}  // namespace serving
#endif
