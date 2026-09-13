#include "cache/host_store.h"
#include <exception>
#include <limits>
#include <utility>

namespace cache {
HostStore::HostStore(size_t bytes, size_t pages, Allocate allocate)
    : byte_capacity_(bytes), page_capacity_(pages), allocate_(std::move(allocate)) {
  if (!allocate_) allocate_ = [](size_t n) {
    return std::shared_ptr<uint8_t>(new uint8_t[n](), std::default_delete<uint8_t[]>());
  };
}
HostStore::~HostStore() {
  for (const auto& item : pages_) {
    const auto& e = item.second;
    if (e.readers || e.state == HostPageState::kCopying || e.state == HostPageState::kQuarantined)
      std::terminate(); // Never free memory whose physical use is unproven.
  }
}
base::Status HostStore::reserve(const PageSchema& schema, const LayoutDescriptor& layout,
                               HostTicket* ticket) {
  if (!ticket || !schema.validate() || !layout.validate(schema) || layout.total_bytes == 0)
    return base::error::InvalidArgument("invalid host reservation layout");
  if (pages_.size() >= page_capacity_ || layout.total_bytes > byte_capacity_ - used_ ||
      next_ == std::numeric_limits<HostTicket>::max())
    return base::error::InvalidArgument("host store capacity exhausted");
  std::shared_ptr<uint8_t> buffer;
  try { buffer = allocate_(layout.total_bytes); }
  catch (const std::bad_alloc&) { return base::error::InvalidArgument("host allocation failed"); }
  if (!buffer) return base::error::InvalidArgument("host allocation failed");
  const auto id = next_;
  Entry entry;
  entry.layout = layout;
  entry.buffer = std::move(buffer);
  entry.schema_fingerprint = schema.fingerprint();
  pages_.emplace(id, std::move(entry));
  ++next_;
  used_ += layout.total_bytes;
  *ticket = id;
  return base::error::Success();
}
uint8_t* HostStore::begin_copy(HostTicket ticket) {
  auto it = pages_.find(ticket);
  if (it == pages_.end() || it->second.state != HostPageState::kReserved) return nullptr;
  it->second.state = HostPageState::kCopying;
  return it->second.buffer.get();
}
void HostStore::remove(HostTicket ticket) {
  const auto it = pages_.find(ticket);
  used_ -= it->second.layout.total_bytes;
  pages_.erase(it);
}
bool HostStore::cancel(HostTicket ticket) {
  auto it = pages_.find(ticket);
  if (it == pages_.end()) return false;
  if (it->second.state == HostPageState::kReserved) { remove(ticket); return true; }
  if (it->second.state != HostPageState::kCopying) return false;
  it->second.cancelled = true;
  return true;
}
bool HostStore::complete(HostTicket ticket, bool valid) {
  auto it = pages_.find(ticket);
  if (it == pages_.end() || it->second.state != HostPageState::kCopying) return false;
  if (!valid || it->second.cancelled) { remove(ticket); return true; }
  it->second.state = HostPageState::kReady;
  return true;
}
bool HostStore::quarantine(HostTicket ticket) {
  auto it = pages_.find(ticket);
  if (it == pages_.end() || it->second.state != HostPageState::kCopying) return false;
  it->second.state = HostPageState::kQuarantined;
  return true;
}
bool HostStore::reclaim_after_quiescence(HostTicket ticket) {
  auto it = pages_.find(ticket);
  if (it == pages_.end() || it->second.state != HostPageState::kQuarantined) return false;
  remove(ticket);
  return true;
}
bool HostStore::acquire(HostTicket ticket, HostReadLease* lease) {
  auto it = pages_.find(ticket);
  if (!lease || *lease || it == pages_.end() || it->second.state != HostPageState::kReady ||
      it->second.retired) return false;
  ++it->second.readers;
  lease->owner_ = this;
  lease->ticket_ = ticket;
  lease->data_ = it->second.buffer.get();
  return true;
}
void HostStore::release(HostTicket ticket) {
  auto it = pages_.find(ticket);
  if (it == pages_.end() || !it->second.readers) std::terminate();
  --it->second.readers;
  if (!it->second.readers && it->second.retired) remove(ticket);
}
bool HostStore::evict(HostTicket ticket) {
  auto it = pages_.find(ticket);
  if (it == pages_.end() || it->second.state != HostPageState::kReady || it->second.readers) return false;
  remove(ticket);
  return true;
}
bool HostStore::retire(HostTicket ticket) {
  auto it = pages_.find(ticket);
  if (it == pages_.end() || it->second.state != HostPageState::kReady) return false;
  it->second.retired = true;
  if (!it->second.readers) remove(ticket);
  return true;
}
bool HostStore::matches_schema(HostTicket ticket, const PageSchema& schema) const {
  const auto it = pages_.find(ticket);
  return it != pages_.end() && it->second.schema_fingerprint == schema.fingerprint();
}
bool HostStore::state(HostTicket ticket, HostPageState* result) const {
  const auto it = pages_.find(ticket);
  if (!result || it == pages_.end()) return false;
  *result = it->second.state;
  return true;
}
const LayoutDescriptor* HostStore::layout(HostTicket ticket) const {
  const auto it = pages_.find(ticket);
  return it == pages_.end() ? nullptr : &it->second.layout;
}
HostReadLease::~HostReadLease() { reset(); }
HostReadLease::HostReadLease(HostReadLease&& other) noexcept { *this = std::move(other); }
HostReadLease& HostReadLease::operator=(HostReadLease&& other) noexcept {
  if (this != &other) {
    reset();
    owner_ = std::exchange(other.owner_, nullptr);
    data_ = std::exchange(other.data_, nullptr);
    ticket_ = other.ticket_;
  }
  return *this;
}
void HostReadLease::reset() {
  if (owner_) { auto* owner = std::exchange(owner_, nullptr); owner->release(ticket_); }
  data_ = nullptr;
}
}  // namespace cache
