#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <stdexcept>
#include <unordered_map>

namespace base {
enum class ChunkAllocationFailure { kQuota = 1, kFragmented = 2, kExhausted = 3 };
class ChunkAllocationError : public std::runtime_error {
 public:
  explicit ChunkAllocationError(ChunkAllocationFailure code)
      : std::runtime_error(code == ChunkAllocationFailure::kQuota ? "model_kv_quota_exhausted" :
          code == ChunkAllocationFailure::kFragmented ? "arena_fragmented" : "arena_exhausted"),
        code_(code) {}
  ChunkAllocationFailure code() const { return code_; }
 private:
  ChunkAllocationFailure code_;
};
struct ChunkLease {
  uint64_t epoch = 0, id = 0, generation = 0;
  uint64_t offset = 0, bytes = 0;
  std::string model, layout;
};

// CPU metadata only. The memory owner serializes all calls to this object.
class GlobalChunkAllocator {
 public:
  GlobalChunkAllocator(uint64_t bytes, uint64_t epoch);
  void register_model(const std::string& model, uint64_t hard_limit);
  ChunkLease acquire(const std::string& model, const std::string& layout,
                     uint64_t bytes, uint64_t alignment);
  void release(const ChunkLease& lease);
  uint64_t free_bytes() const;
  uint64_t leased_bytes(const std::string& model) const;
  uint64_t epoch() const { return epoch_; }
  uint64_t bytes() const { return bytes_; }
  size_t lease_count() const { return leases_.size(); }
 private:
  uint64_t bytes_, epoch_, next_generation_ = 1;
  std::map<uint64_t, uint64_t> free_;
  std::unordered_map<uint64_t, ChunkLease> leases_;
  std::unordered_map<std::string, uint64_t> limits_, used_;
};
}  // namespace base
