#ifndef KUIPER_INCLUDE_CACHE_HOST_STORE_H_
#define KUIPER_INCLUDE_CACHE_HOST_STORE_H_
#include <functional>
#include <map>
#include <memory>
#include "cache/layout_codec.h"

namespace cache {
using HostTicket = uint64_t;
enum class HostPageState { kReserved, kCopying, kReady, kQuarantined };
class HostStore;
class HostReadLease {
 public:
  HostReadLease() = default;
  ~HostReadLease();
  HostReadLease(const HostReadLease&) = delete;
  HostReadLease& operator=(const HostReadLease&) = delete;
  HostReadLease(HostReadLease&& other) noexcept;
  HostReadLease& operator=(HostReadLease&& other) noexcept;
  const uint8_t* data() const { return data_; }
  HostTicket ticket() const { return ticket_; }
  const HostStore* store() const { return owner_; }
  explicit operator bool() const { return owner_ != nullptr; }
  void reset();
 private:
  friend class HostStore;
  HostStore* owner_ = nullptr;
  HostTicket ticket_ = 0;
  const uint8_t* data_ = nullptr;
};

// Owner-thread metadata; destruction requires all DMA drained and leases released.
// An injected allocator may provide pinned CUDA memory. Default is a CPU oracle.
class HostStore {
 public:
  using Allocate = std::function<std::shared_ptr<uint8_t>(size_t)>;
  HostStore(size_t byte_capacity, size_t page_capacity, Allocate allocate = {});
  ~HostStore();
  HostStore(const HostStore&) = delete;
  HostStore& operator=(const HostStore&) = delete;
  base::Status reserve(const PageSchema& schema, const LayoutDescriptor& layout, HostTicket* ticket);
  uint8_t* begin_copy(HostTicket ticket);
  bool cancel(HostTicket ticket);
  // Call only after backend fence establishes no remaining writes.
  bool complete(HostTicket ticket, bool valid);
  bool quarantine(HostTicket ticket);
  // Caller must first establish backend quiescence, e.g. successful stream drain.
  bool reclaim_after_quiescence(HostTicket ticket);
  bool acquire(HostTicket ticket, HostReadLease* lease);
  bool evict(HostTicket ticket);
  // Remove discoverability now; retain storage until existing readers finish.
  bool retire(HostTicket ticket);
  bool matches_schema(HostTicket ticket, const PageSchema& schema) const;
  bool state(HostTicket ticket, HostPageState* result) const;
  const LayoutDescriptor* layout(HostTicket ticket) const;
  size_t bytes_used() const { return used_; }
  size_t pages_used() const { return pages_.size(); }
 private:
  friend class HostReadLease;
  void release(HostTicket ticket);
  struct Entry {
    LayoutDescriptor layout;
    std::shared_ptr<uint8_t> buffer;
    HostPageState state = HostPageState::kReserved;
    size_t readers = 0;
    bool cancelled = false;
    bool retired = false;
    std::string schema_fingerprint;
  };
  void remove(HostTicket ticket);
  size_t byte_capacity_, page_capacity_, used_ = 0;
  HostTicket next_ = 1;
  Allocate allocate_;
  std::map<HostTicket, Entry> pages_;
};
}  // namespace cache
#endif
