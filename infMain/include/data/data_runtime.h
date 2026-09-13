#ifndef KUIPER_INCLUDE_DATA_DATA_RUNTIME_H_
#define KUIPER_INCLUDE_DATA_DATA_RUNTIME_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "data/data_ref.h"

namespace data {

enum class DataLeaseKind : uint8_t { kRead = 0, kCompute = 1, kIO = 2 };

struct DataReservation {
  OperationId operation;
  DataKind kind = DataKind::kUnknown;
  ContentId content;
  RepresentationId representation;
  uint64_t logical_bytes = 0;
};

class LocalDataRuntime;

class DataLease {
 public:
  DataLease() = default;
  ~DataLease();
  DataLease(DataLease&& other) noexcept;
  DataLease& operator=(DataLease&& other) noexcept;
  DataLease(const DataLease&) = delete;
  DataLease& operator=(const DataLease&) = delete;
  const uint8_t* data() const { return data_; }
  size_t size() const { return size_; }
  const DataRef& ref() const { return ref_; }
  explicit operator bool() const { return runtime_ != nullptr; }
  void reset();

 private:
  friend class LocalDataRuntime;
  LocalDataRuntime* runtime_ = nullptr;
  DataRef ref_;
  const uint8_t* data_ = nullptr;
  size_t size_ = 0;
  DataLeaseKind kind_ = DataLeaseKind::kRead;
};

class LocalDataRuntime {
 public:
  LocalDataRuntime(uint64_t owner_incarnation, uint64_t capacity_bytes,
                   size_t capacity_objects);
  ~LocalDataRuntime();
  LocalDataRuntime(const LocalDataRuntime&) = delete;
  LocalDataRuntime& operator=(const LocalDataRuntime&) = delete;

  DataError reserve(const DataReservation& reservation, AllocationHandle* handle);
  DataError begin_write(const AllocationHandle& handle, uint8_t** data, size_t* size);
  DataError seal(const AllocationHandle& handle, uint64_t bytes_written,
                 std::string checksum, DataRef* canonical);
  DataError acquire(DataKind kind, const ContentId& content,
                    const RepresentationId& representation, DataLease* lease);
  DataError acquire(DataKind kind, const ContentId& content,
                    const RepresentationId& representation,
                    DataLeaseKind lease_kind, DataLease* lease);
  // Local objects are already resident; this keeps the same contract as the
  // M3 client, where ensure_local may schedule a transfer before acquiring.
  DataError ensure_local(DataKind kind, const ContentId& content,
                         const RepresentationId& representation,
                         DataLeaseKind lease_kind, DataLease* lease);
  DataError withdraw(const DataRef& ref);
  DataError release_producer(const AllocationHandle& handle);
  // Only the owner may call this after it has established quiescence.
  DataError reclaim_quarantine(const AllocationHandle& handle);

  uint64_t bytes_used() const { return bytes_used_; }
  uint64_t quarantined_bytes() const { return quarantined_bytes_; }
  size_t object_count() const { return objects_.size(); }
  uint64_t owner_incarnation() const { return owner_incarnation_; }

 private:
  friend class DataLease;
  struct Identity {
    DataKind kind;
    ContentId content;
    RepresentationId representation;
    bool operator<(const Identity& other) const;
  };
  enum class State : uint8_t { kReserved, kSealed, kRetired, kQuarantined };
  struct Object {
    DataReservation reservation;
    AllocationHandle handle;
    State state = State::kReserved;
    std::vector<uint8_t> storage;
    std::string checksum;
    uint64_t canonical_id = 0;
    std::array<uint32_t, 3> consumer_refs{};
    bool producer_ref = true;
  };

  DataError find_current(const AllocationHandle& handle, Object** object);
  DataError find_current(const AllocationHandle& handle, const Object** object) const;
  DataRef make_ref(const Object& object) const;
  void release_consumer(const AllocationHandle& handle, DataLeaseKind kind);
  static uint32_t total_consumer_refs(const Object& object);
  void collect_if_unused(uint64_t allocation_id);

  uint64_t owner_incarnation_;
  uint64_t capacity_bytes_;
  size_t capacity_objects_;
  uint64_t next_allocation_id_ = 1;
  uint64_t bytes_used_ = 0;
  uint64_t quarantined_bytes_ = 0;
  std::map<uint64_t, std::unique_ptr<Object>> objects_;
  std::map<OperationId, uint64_t> operations_;
  std::map<Identity, uint64_t> canonical_;
};

}  // namespace data

#endif  // KUIPER_INCLUDE_DATA_DATA_RUNTIME_H_
