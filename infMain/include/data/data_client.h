#ifndef KUIPER_INCLUDE_DATA_DATA_CLIENT_H_
#define KUIPER_INCLUDE_DATA_DATA_CLIENT_H_

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "data/data_runtime.h"
#include "data/ipc_pool.h"
#include "data/shared_weight.h"

namespace data {

inline constexpr const char* kDefaultDataEndpoint = "/tmp/pbe-data-service.sock";
inline constexpr uint32_t kDataServiceWireMagic = 0x53534250;  // "PBSS"
inline constexpr uint16_t kDataServiceProtocolVersion = 3;
inline constexpr uint64_t kMaxDataServicePayloadBytes = 1ULL << 30;

enum class DataServiceOp : uint16_t {
  kPing = 1,
  kReserve = 2,
  kSeal = 3,
  kAcquire = 4,
  kRelease = 5,
  kWithdraw = 6,
  kReleaseProducer = 7,
  kStats = 8,
  kShutdown = 9,
  kGetIpcPool = 10,
  kReserveIpcSlots = 11,
  kReleaseIpcSlots = 12,
  kIpcPoolStats = 13,
  kAcquireSharedWeight = 14,
  kReleaseSharedWeight = 15,
  kSharedWeightStats = 16,
  kAcquireIpcAttach = 17,
  kReleaseIpcAttach = 18,
};

struct DataServiceStats {
  uint64_t owner_incarnation = 0;
  uint64_t bytes_used = 0;
  uint64_t object_count = 0;
  uint64_t active_leases = 0;
  uint64_t bytes_received = 0;
  uint64_t bytes_sent = 0;
  uint64_t requests = 0;
};

struct LeaseToken {
  uint64_t service_incarnation = 0;
  uint64_t consumer_incarnation = 0;
  uint64_t lease_id = 0;
  bool operator==(const LeaseToken& other) const {
    return service_incarnation == other.service_incarnation &&
           consumer_incarnation == other.consumer_incarnation &&
           lease_id == other.lease_id;
  }
};

struct RemoteDataLease {
  LeaseToken token;
  DataRef ref;
  std::vector<uint8_t> bytes;
};

struct IpcSlotGrant {
  LeaseToken token;
  std::vector<int32_t> slots;
};

// Request-scoped permission to install immutable pages produced under a
// different worker's slot grant.  The data service, rather than the consumer,
// validates the metadata object, source generation and exact page subset.
struct IpcAttachGrant {
  LeaseToken token;
  LeaseToken source_grant;
  AllocationHandle metadata_allocation;
  uint64_t provider_incarnation = 0;
  uint64_t target_incarnation = 0;
  uint32_t valid_tokens = 0;
  std::vector<int32_t> slots;
};

struct IpcPoolStats {
  uint64_t total_slots = 0;
  uint64_t free_slots = 0;
  uint64_t active_grants = 0;
};

struct SharedWeightLease {
  LeaseToken token;
  SharedWeightDescriptor descriptor;
};

class DataClient {
 public:
  explicit DataClient(std::string endpoint = kDefaultDataEndpoint);

  DataError ping(uint64_t* owner_incarnation) const;
  DataError reserve(const DataReservation& reservation, AllocationHandle* handle) const;
  DataError seal(const AllocationHandle& handle, const std::vector<uint8_t>& bytes,
                 const Digest256& checksum, DataRef* canonical) const;
  DataError acquire(DataKind kind, const ContentId& content,
                    const RepresentationId& representation,
                    DataLeaseKind lease_kind, RemoteDataLease* lease,
                    OperationId operation = {}) const;
  DataError release(const LeaseToken& token) const;
  DataError withdraw(const DataRef& ref) const;
  DataError release_producer(const AllocationHandle& handle) const;
  DataError stats(DataServiceStats* stats) const;
  DataError shutdown() const;
  DataError get_ipc_pool(IpcPoolDescriptor* descriptor) const;
  DataError reserve_ipc_slots(uint32_t count,IpcSlotGrant* grant,
                              OperationId operation = {}) const;
  DataError release_ipc_slots(const LeaseToken& token) const;
  DataError ipc_pool_stats(IpcPoolStats* stats) const;
  DataError acquire_ipc_attach(const RemoteDataLease& metadata,
                               const LeaseToken& source_grant,
                               uint64_t provider_incarnation,
                               uint64_t target_incarnation,
                               uint32_t valid_tokens,
                               const std::vector<int32_t>& slots,
                               IpcAttachGrant* grant,
                               OperationId operation = {}) const;
  DataError release_ipc_attach(const LeaseToken& token) const;
  DataError acquire_shared_weight(const Digest256& model_content,
                                  const Digest256& layout_identity,
                                  base::DataType dtype, uint64_t expected_bytes,
                                  SharedWeightLease* lease,
                                  OperationId operation = {}) const;
  DataError release_shared_weight(const LeaseToken& token) const;
  DataError shared_weight_stats(SharedWeightStats* stats) const;

 private:
  DataError transact(DataServiceOp op, const std::vector<uint8_t>& request,
                     std::vector<uint8_t>* response) const;
  std::string endpoint_;
  uint64_t consumer_incarnation_ = 0;
  mutable std::atomic<uint64_t> next_operation_{1};
};

Digest256 DataChecksum(const uint8_t* bytes, size_t size);
ContentId ContentIdFromBytes(const uint8_t* bytes, size_t size);
RepresentationId RepresentationIdFromString(const std::string& value);

}  // namespace data

#endif
