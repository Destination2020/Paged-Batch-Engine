#include "data/content_registry.h"

#include <algorithm>
#include <limits>

namespace data {
namespace {
std::string ChecksumString(const Digest256& digest) {
  return std::string(reinterpret_cast<const char*>(digest.data()), digest.size());
}
}

ContentRegistry::ContentRegistry(uint64_t incarnation, uint64_t capacity_bytes,
                                 size_t capacity_objects, size_t capacity_leases)
    : runtime_(incarnation, capacity_bytes, capacity_objects),
      capacity_leases_(capacity_leases) {}

DataError ContentRegistry::reserve(const DataReservation& r, AllocationHandle* h) {
  std::lock_guard<std::mutex> lock(mutex_);
  return runtime_.reserve(r, h);
}
DataError ContentRegistry::seal(const AllocationHandle& h,
                                const std::vector<uint8_t>& bytes,
                                const Digest256& checksum, DataRef* canonical) {
  std::lock_guard<std::mutex> lock(mutex_);
  uint8_t* target = nullptr; size_t size = 0;
  auto status = runtime_.begin_write(h, &target, &size);
  if (status == DataError::kAlreadySealed)
    return runtime_.seal(h, bytes.size(), ChecksumString(checksum), canonical);
  if (status != DataError::kOk) return status;
  if (size != bytes.size()) return DataError::kCoverageMismatch;
  std::copy(bytes.begin(), bytes.end(), target);
  return runtime_.seal(h, bytes.size(), ChecksumString(checksum), canonical);
}
DataError ContentRegistry::acquire(DataKind kind, const ContentId& content,
                                   const RepresentationId& representation,
                                   DataLeaseKind lease_kind,
                                   const OperationId& operation,
                                   RemoteDataLease* out) {
  if (!out || operation.owner_incarnation == 0 || operation.sequence == 0)
    return DataError::kInvalidArgument;
  std::lock_guard<std::mutex> lock(mutex_);
  auto replay = acquire_operations_.find(operation);
  if (replay != acquire_operations_.end()) {
    auto existing = leases_.find(replay->second);
    if (existing == leases_.end()) return DataError::kInvalidArgument;
    const auto& entry = existing->second;
    if (entry.kind != kind || !(entry.content == content) ||
        !(entry.representation == representation) || entry.lease_kind != lease_kind)
      return DataError::kInvalidArgument;
    out->token = {runtime_.owner_incarnation(), operation.owner_incarnation,
                  existing->first};
    out->ref = entry.lease.ref();
    out->bytes.assign(entry.lease.data(), entry.lease.data() + entry.lease.size());
    return DataError::kOk;
  }
  if (leases_.size() >= capacity_leases_ || next_lease_id_ == 0)
    return DataError::kCapacityExhausted;
  DataLease lease;
  auto status = runtime_.acquire(kind, content, representation, lease_kind, &lease);
  if (status != DataError::kOk) return status;
  const uint64_t id = next_lease_id_++;
  out->token = {runtime_.owner_incarnation(), operation.owner_incarnation, id};
  out->ref = lease.ref();
  out->bytes.assign(lease.data(), lease.data() + lease.size());
  LeaseEntry entry;
  entry.operation = operation; entry.kind = kind; entry.content = content;
  entry.representation = representation; entry.lease_kind = lease_kind;
  entry.lease = std::move(lease);
  leases_.emplace(id, std::move(entry));
  acquire_operations_.emplace(operation, id);
  return DataError::kOk;
}
DataError ContentRegistry::release(const LeaseToken& token) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (token.service_incarnation != runtime_.owner_incarnation())
    return DataError::kOwnerRestarted;
  auto it = leases_.find(token.lease_id);
  if (it == leases_.end()) return DataError::kOk;
  if (it->second.operation.owner_incarnation != token.consumer_incarnation)
    return DataError::kInvalidArgument;
  acquire_operations_.erase(it->second.operation);
  leases_.erase(it);
  return DataError::kOk;
}
DataError ContentRegistry::withdraw(const DataRef& ref) {
  std::lock_guard<std::mutex> lock(mutex_); return runtime_.withdraw(ref);
}
DataError ContentRegistry::release_producer(const AllocationHandle& h) {
  std::lock_guard<std::mutex> lock(mutex_); return runtime_.release_producer(h);
}
DataServiceStats ContentRegistry::stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return {runtime_.owner_incarnation(), runtime_.bytes_used(), runtime_.object_count(),
          leases_.size(), bytes_received_, bytes_sent_, requests_};
}
void ContentRegistry::add_transport_bytes(uint64_t received, uint64_t sent) {
  std::lock_guard<std::mutex> lock(mutex_); bytes_received_ += received; bytes_sent_ += sent;
}
void ContentRegistry::add_request() {
  std::lock_guard<std::mutex> lock(mutex_); ++requests_;
}

}  // namespace data
