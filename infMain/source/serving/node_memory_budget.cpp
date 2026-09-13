#include "serving/node_memory_budget.h"

#include <algorithm>
#include <limits>

namespace serving {

bool NodeMemoryBudget::set_capacity(MemoryPool pool, size_t bytes) {
  const auto i = index(pool);
  if (frozen_ || i >= pools_.size() || bytes < pools_[i].used) return false;
  pools_[i].capacity = bytes;
  return true;
}

bool NodeMemoryBudget::reserve(const std::vector<MemoryDemand>& demands) {
  std::array<size_t, static_cast<size_t>(MemoryPool::kCount)> delta{};
  for (const auto& demand : demands) {
    const auto i = index(demand.pool);
    if (i >= delta.size() || demand.bytes > std::numeric_limits<size_t>::max() - delta[i])
      return false;
    delta[i] += demand.bytes;
  }
  for (size_t i = 0; i < pools_.size(); ++i) {
    if (delta[i] > pools_[i].capacity - pools_[i].used) return false;
  }
  for (size_t i = 0; i < pools_.size(); ++i) {
    pools_[i].used += delta[i];
    pools_[i].peak = std::max(pools_[i].peak, pools_[i].used);
  }
  return true;
}

void NodeMemoryBudget::release(const std::vector<MemoryDemand>& demands) {
  std::array<size_t, static_cast<size_t>(MemoryPool::kCount)> delta{};
  for (const auto& demand : demands) {
    const auto i = index(demand.pool);
    if (i >= delta.size() || demand.bytes > std::numeric_limits<size_t>::max() - delta[i])
      std::terminate();
    delta[i] += demand.bytes;
  }
  for (size_t i = 0; i < pools_.size(); ++i) {
    if (delta[i] > pools_[i].used) std::terminate();
    pools_[i].used -= delta[i];
  }
}

MemoryPoolState NodeMemoryBudget::state(MemoryPool pool) const {
  const auto i = index(pool);
  return i < pools_.size() ? pools_[i] : MemoryPoolState{};
}

bool NodeMemoryBudget::invariant_holds() const {
  for (const auto& pool : pools_) if (pool.used > pool.capacity || pool.peak > pool.capacity)
    return false;
  return true;
}

}  // namespace serving
