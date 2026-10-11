#include "base/global_chunk_allocator.h"
#include <limits>
#include <stdexcept>

namespace base {
GlobalChunkAllocator::GlobalChunkAllocator(uint64_t bytes, uint64_t epoch)
    : bytes_(bytes), epoch_(epoch) {
  if (!bytes || !epoch) throw std::invalid_argument("invalid arena size/epoch");
  free_[0] = bytes;
}
void GlobalChunkAllocator::register_model(const std::string& model, uint64_t limit) {
  if (model.empty() || !limit || limit > bytes_ || limits_.count(model))
    throw std::invalid_argument("invalid/duplicate model or KV hard limit");
  limits_[model] = limit;
  used_[model] = 0;
}
uint64_t GlobalChunkAllocator::leased_bytes(const std::string& model) const {
  return used_.at(model);
}
uint64_t GlobalChunkAllocator::free_bytes() const {
  uint64_t bytes = 0;
  for (const auto& interval : free_) bytes += interval.second;
  return bytes;
}
ChunkLease GlobalChunkAllocator::acquire(const std::string& model,
    const std::string& layout, uint64_t bytes, uint64_t alignment) {
  if (!limits_.count(model) || layout.empty() || !bytes || !alignment || bytes % alignment)
    throw std::invalid_argument("invalid chunk request");
  if (bytes > limits_.at(model) - used_.at(model))
    throw std::runtime_error("model_kv_quota_exhausted");
  auto best = free_.end();
  uint64_t best_offset = 0, best_waste = std::numeric_limits<uint64_t>::max();
  for (auto it = free_.begin(); it != free_.end(); ++it) {
    const uint64_t rem = it->first % alignment;
    const uint64_t padding = rem ? alignment - rem : 0;
    if (padding > it->second || bytes > it->second - padding) continue;
    if (it->second - bytes < best_waste) {
      best = it;
      best_offset = it->first + padding;
      best_waste = it->second - bytes;
    }
  }
  if (best == free_.end()) throw std::runtime_error(
      free_bytes() >= bytes ? "arena_fragmented" : "arena_exhausted");
  const uint64_t begin = best->first, end = begin + best->second;
  free_.erase(best);
  if (best_offset > begin) free_[begin] = best_offset - begin;
  if (best_offset + bytes < end) free_[best_offset + bytes] = end - best_offset - bytes;
  ChunkLease lease{epoch_, next_generation_, next_generation_, best_offset, bytes, model, layout};
  ++next_generation_;
  leases_[lease.id] = lease;
  used_[model] += bytes;
  return lease;
}
void GlobalChunkAllocator::release(const ChunkLease& lease) {
  const auto it = leases_.find(lease.id);
  if (lease.epoch != epoch_ || it == leases_.end())
    throw std::invalid_argument("stale_or_duplicate_chunk_release");
  const auto& live = it->second;
  if (live.generation != lease.generation || live.model != lease.model ||
      live.layout != lease.layout || live.offset != lease.offset || live.bytes != lease.bytes)
    throw std::invalid_argument("chunk_identity_mismatch");
  uint64_t begin = live.offset, bytes = live.bytes;
  used_.at(live.model) -= bytes;
  leases_.erase(it);
  auto next = free_.lower_bound(begin);
  if (next != free_.begin()) {
    auto prev = std::prev(next);
    if (prev->first + prev->second == begin) {
      begin = prev->first;
      bytes += prev->second;
      free_.erase(prev);
    }
  }
  next = free_.lower_bound(begin);
  if (next != free_.end() && begin + bytes == next->first) {
    bytes += next->second;
    free_.erase(next);
  }
  free_[begin] = bytes;
}
}  // namespace base
