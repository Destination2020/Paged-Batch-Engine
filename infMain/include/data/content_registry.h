#ifndef KUIPER_INCLUDE_DATA_CONTENT_REGISTRY_H_
#define KUIPER_INCLUDE_DATA_CONTENT_REGISTRY_H_

#include <cstdint>
#include <map>
#include <mutex>
#include <vector>

#include "data/data_client.h"

namespace data {

// Thread-safe authoritative metadata owner. Socket and device copies happen
// outside its lock; live leases remain here until an explicit release.
class ContentRegistry {
 public:
  ContentRegistry(uint64_t incarnation, uint64_t capacity_bytes,
                  size_t capacity_objects, size_t capacity_leases);

  DataError reserve(const DataReservation& reservation, AllocationHandle* handle);
  DataError seal(const AllocationHandle& handle, const std::vector<uint8_t>& bytes,
                 const Digest256& checksum, DataRef* canonical);
  DataError acquire(DataKind kind, const ContentId& content,
                    const RepresentationId& representation,
                    DataLeaseKind lease_kind, const OperationId& operation,
                    RemoteDataLease* lease);
  DataError release(const LeaseToken& token);
  DataError withdraw(const DataRef& ref);
  DataError release_producer(const AllocationHandle& handle);
  DataServiceStats stats() const;
  void add_transport_bytes(uint64_t received, uint64_t sent);
  void add_request();

 private:
  mutable std::mutex mutex_;
  LocalDataRuntime runtime_;
  size_t capacity_leases_;
  uint64_t next_lease_id_ = 1;
  struct LeaseEntry {
    OperationId operation;
    DataKind kind = DataKind::kUnknown;
    ContentId content;
    RepresentationId representation;
    DataLeaseKind lease_kind = DataLeaseKind::kRead;
    DataLease lease;
  };
  std::map<uint64_t, LeaseEntry> leases_;
  std::map<OperationId, uint64_t> acquire_operations_;
  uint64_t bytes_received_ = 0;
  uint64_t bytes_sent_ = 0;
  uint64_t requests_ = 0;
};

}  // namespace data
#endif
