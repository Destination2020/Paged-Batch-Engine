#include "data/data_runtime.h"

#include <limits>
#include <tuple>
#include <utility>

namespace data {

bool LocalDataRuntime::Identity::operator<(const Identity& other) const {
  return std::tie(kind, content, representation) <
         std::tie(other.kind, other.content, other.representation);
}

DataLease::~DataLease() { reset(); }
DataLease::DataLease(DataLease&& other) noexcept { *this = std::move(other); }
DataLease& DataLease::operator=(DataLease&& other) noexcept {
  if (this == &other) return *this;
  reset();
  runtime_ = other.runtime_;
  ref_ = other.ref_;
  data_ = other.data_;
  size_ = other.size_;
  kind_ = other.kind_;
  other.runtime_ = nullptr;
  other.data_ = nullptr;
  other.size_ = 0;
  return *this;
}
void DataLease::reset() {
  if (runtime_ != nullptr) runtime_->release_consumer(ref_.allocation, kind_);
  runtime_ = nullptr;
  data_ = nullptr;
  size_ = 0;
}

LocalDataRuntime::LocalDataRuntime(uint64_t owner_incarnation, uint64_t capacity_bytes,
                                   size_t capacity_objects)
    : owner_incarnation_(owner_incarnation), capacity_bytes_(capacity_bytes),
      capacity_objects_(capacity_objects) {}
LocalDataRuntime::~LocalDataRuntime() {
  // Leases must not outlive their authoritative runtime owner.
  for (const auto& item : objects_) {
    if (total_consumer_refs(*item.second) != 0) std::terminate();
  }
}

DataError LocalDataRuntime::find_current(const AllocationHandle& handle, Object** object) {
  if (handle.owner_incarnation != owner_incarnation_) return DataError::kOwnerRestarted;
  auto it = objects_.find(handle.allocation_id);
  if (it == objects_.end()) return DataError::kUnknownObject;
  if (it->second->handle.generation != handle.generation) return DataError::kStaleGeneration;
  *object = it->second.get();
  return DataError::kOk;
}
DataError LocalDataRuntime::find_current(const AllocationHandle& handle,
                                         const Object** object) const {
  return const_cast<LocalDataRuntime*>(this)->find_current(
      handle, const_cast<Object**>(object));
}

DataError LocalDataRuntime::reserve(const DataReservation& reservation,
                                    AllocationHandle* handle) {
  if (handle == nullptr || reservation.operation.owner_incarnation != owner_incarnation_ ||
      reservation.operation.sequence == 0 || reservation.kind == DataKind::kUnknown ||
      reservation.logical_bytes == 0) return DataError::kInvalidArgument;
  auto prior = operations_.find(reservation.operation);
  if (prior != operations_.end()) {
    const auto& object = *objects_.at(prior->second);
    if (object.reservation.kind != reservation.kind ||
        !(object.reservation.content == reservation.content) ||
        !(object.reservation.representation == reservation.representation) ||
        object.reservation.logical_bytes != reservation.logical_bytes) {
      return DataError::kInvalidArgument;
    }
    *handle = object.handle;
    return DataError::kOk;
  }
  if (objects_.size() >= capacity_objects_ ||
      reservation.logical_bytes > capacity_bytes_ - bytes_used_ ||
      next_allocation_id_ == std::numeric_limits<uint64_t>::max()) {
    return DataError::kCapacityExhausted;
  }
  auto object = std::make_unique<Object>();
  object->reservation = reservation;
  object->handle = {owner_incarnation_, next_allocation_id_++, 1};
  object->storage.resize(static_cast<size_t>(reservation.logical_bytes));
  const uint64_t id = object->handle.allocation_id;
  bytes_used_ += reservation.logical_bytes;
  *handle = object->handle;
  objects_.emplace(id, std::move(object));
  operations_.emplace(reservation.operation, id);
  return DataError::kOk;
}

DataError LocalDataRuntime::begin_write(const AllocationHandle& handle, uint8_t** data,
                                        size_t* size) {
  if (data == nullptr || size == nullptr) return DataError::kInvalidArgument;
  Object* object = nullptr;
  const auto status = find_current(handle, &object);
  if (status != DataError::kOk) return status;
  if (object->state != State::kReserved) return DataError::kAlreadySealed;
  *data = object->storage.data();
  *size = object->storage.size();
  return DataError::kOk;
}

DataRef LocalDataRuntime::make_ref(const Object& object) const {
  DataRef ref;
  ref.kind = object.reservation.kind;
  ref.content = object.reservation.content;
  ref.representation = object.reservation.representation;
  ref.allocation = object.handle;
  ref.logical_bytes = object.reservation.logical_bytes;
  return ref;
}

DataError LocalDataRuntime::seal(const AllocationHandle& handle, uint64_t bytes_written,
                                 std::string checksum, DataRef* canonical) {
  if (canonical == nullptr || checksum.empty()) return DataError::kInvalidArgument;
  Object* object = nullptr;
  auto status = find_current(handle, &object);
  if (status != DataError::kOk) return status;
  if (object->state == State::kSealed) {
    *canonical = make_ref(*object);
    return DataError::kOk;
  }
  if (object->state == State::kRetired && object->canonical_id != 0) {
    *canonical = make_ref(*objects_.at(object->canonical_id));
    return DataError::kOk;
  }
  if (object->state != State::kReserved) return DataError::kQuarantined;
  if (bytes_written != object->reservation.logical_bytes) return DataError::kCoverageMismatch;
  Identity identity{object->reservation.kind, object->reservation.content,
                    object->reservation.representation};
  auto existing = canonical_.find(identity);
  if (existing == canonical_.end()) {
    object->state = State::kSealed;
    object->checksum = std::move(checksum);
    canonical_.emplace(identity, object->handle.allocation_id);
    *canonical = make_ref(*object);
    return DataError::kOk;
  }
  Object& winner = *objects_.at(existing->second);
  if (winner.checksum != checksum) {
    object->state = State::kQuarantined;
    object->checksum = std::move(checksum);
    quarantined_bytes_ += object->storage.size();
    return DataError::kChecksumMismatch;
  }
  object->state = State::kRetired;
  object->canonical_id = winner.handle.allocation_id;
  bytes_used_ -= object->storage.size();
  object->storage.clear();
  object->storage.shrink_to_fit();
  *canonical = make_ref(winner);
  collect_if_unused(object->handle.allocation_id);
  return DataError::kOk;
}

DataError LocalDataRuntime::acquire(DataKind kind, const ContentId& content,
                                    const RepresentationId& representation,
                                    DataLease* lease) {
  return acquire(kind, content, representation, DataLeaseKind::kRead, lease);
}

DataError LocalDataRuntime::acquire(DataKind kind, const ContentId& content,
                                    const RepresentationId& representation,
                                    DataLeaseKind lease_kind, DataLease* lease) {
  if (lease == nullptr || *lease) return DataError::kInvalidArgument;
  const size_t lease_index = static_cast<size_t>(lease_kind);
  if (lease_index >= 3) return DataError::kInvalidArgument;
  auto found = canonical_.find({kind, content, representation});
  if (found == canonical_.end()) return DataError::kNotReady;
  Object& object = *objects_.at(found->second);
  if (object.state != State::kSealed) return DataError::kNotReady;
  ++object.consumer_refs[lease_index];
  lease->runtime_ = this;
  lease->ref_ = make_ref(object);
  lease->data_ = object.storage.data();
  lease->size_ = object.storage.size();
  lease->kind_ = lease_kind;
  return DataError::kOk;
}

DataError LocalDataRuntime::ensure_local(DataKind kind, const ContentId& content,
                                         const RepresentationId& representation,
                                         DataLeaseKind lease_kind, DataLease* lease) {
  return acquire(kind, content, representation, lease_kind, lease);
}

DataError LocalDataRuntime::withdraw(const DataRef& ref) {
  Object* object = nullptr;
  auto status = find_current(ref.allocation, &object);
  if (status != DataError::kOk) return status;
  Identity identity{ref.kind, ref.content, ref.representation};
  auto found = canonical_.find(identity);
  if (found == canonical_.end() || found->second != object->handle.allocation_id)
    return object->state == State::kRetired ? DataError::kOk
                                            : DataError::kUnknownObject;
  canonical_.erase(found);
  object->state = State::kRetired;
  collect_if_unused(object->handle.allocation_id);
  return DataError::kOk;
}

DataError LocalDataRuntime::release_producer(const AllocationHandle& handle) {
  Object* object = nullptr;
  auto status = find_current(handle, &object);
  if (status != DataError::kOk) return status;
  object->producer_ref = false;
  if (object->state == State::kReserved) object->state = State::kRetired;
  collect_if_unused(handle.allocation_id);
  return DataError::kOk;
}

DataError LocalDataRuntime::reclaim_quarantine(const AllocationHandle& handle) {
  Object* object = nullptr;
  auto status = find_current(handle, &object);
  if (status != DataError::kOk) return status;
  if (object->state != State::kQuarantined || object->producer_ref ||
      total_consumer_refs(*object) != 0) return DataError::kNotReady;
  if (object->storage.size() > quarantined_bytes_) std::terminate();
  quarantined_bytes_ -= object->storage.size();
  object->state = State::kRetired;
  collect_if_unused(handle.allocation_id);
  return DataError::kOk;
}

uint32_t LocalDataRuntime::total_consumer_refs(const Object& object) {
  return object.consumer_refs[0] + object.consumer_refs[1] + object.consumer_refs[2];
}

void LocalDataRuntime::release_consumer(const AllocationHandle& handle,
                                        DataLeaseKind kind) {
  Object* object = nullptr;
  const size_t lease_index = static_cast<size_t>(kind);
  if (lease_index >= 3 || find_current(handle, &object) != DataError::kOk ||
      object->consumer_refs[lease_index] == 0) {
    std::terminate();
  }
  --object->consumer_refs[lease_index];
  collect_if_unused(handle.allocation_id);
}

void LocalDataRuntime::collect_if_unused(uint64_t allocation_id) {
  auto found = objects_.find(allocation_id);
  if (found == objects_.end()) return;
  Object& object = *found->second;
  if (total_consumer_refs(object) != 0 || object.producer_ref || object.state == State::kSealed ||
      object.state == State::kQuarantined) return;
  bytes_used_ -= object.storage.size();
  operations_.erase(object.reservation.operation);
  objects_.erase(found);
}

}  // namespace data
