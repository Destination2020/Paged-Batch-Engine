#ifndef KUIPER_INCLUDE_DATA_DATA_REF_H_
#define KUIPER_INCLUDE_DATA_DATA_REF_H_

#include <array>
#include <cstdint>
#include <tuple>

namespace data {

using Digest256 = std::array<uint8_t, 32>;

struct ContentId {
  Digest256 digest{};
  bool operator==(const ContentId& other) const { return digest == other.digest; }
  bool operator<(const ContentId& other) const { return digest < other.digest; }
};

struct RepresentationId {
  Digest256 digest{};
  bool operator==(const RepresentationId& other) const { return digest == other.digest; }
  bool operator<(const RepresentationId& other) const { return digest < other.digest; }
};

enum class DataKind : uint8_t {
  kUnknown = 0,
  kKVPage = 1,
  kTensorBundle = 2,
  kCheckpoint = 3,
};

struct OperationId {
  uint64_t owner_incarnation = 0;
  uint64_t sequence = 0;
  bool operator==(const OperationId& other) const {
    return owner_incarnation == other.owner_incarnation && sequence == other.sequence;
  }
  bool operator<(const OperationId& other) const {
    return std::tie(owner_incarnation, sequence) <
           std::tie(other.owner_incarnation, other.sequence);
  }
};

struct AllocationHandle {
  uint64_t owner_incarnation = 0;
  uint64_t allocation_id = 0;
  uint32_t generation = 0;
  bool operator==(const AllocationHandle& other) const {
    return owner_incarnation == other.owner_incarnation &&
           allocation_id == other.allocation_id && generation == other.generation;
  }
  bool operator<(const AllocationHandle& other) const {
    return std::tie(owner_incarnation, allocation_id, generation) <
           std::tie(other.owner_incarnation, other.allocation_id, other.generation);
  }
};

struct DataRef {
  uint16_t protocol_version = 1;
  DataKind kind = DataKind::kUnknown;
  ContentId content;
  RepresentationId representation;
  AllocationHandle allocation;
  uint64_t logical_bytes = 0;
  bool operator==(const DataRef& other) const {
    return protocol_version == other.protocol_version && kind == other.kind &&
           content == other.content && representation == other.representation &&
           allocation == other.allocation && logical_bytes == other.logical_bytes;
  }
};

enum class DataError : uint16_t {
  kOk = 0,
  kInvalidArgument = 1,
  kUnsupportedVersion = 2,
  kUnknownObject = 3,
  kStaleGeneration = 4,
  kNotReady = 5,
  kAlreadySealed = 6,
  kCapacityExhausted = 7,
  kCoverageMismatch = 8,
  kChecksumMismatch = 9,
  kOwnerRestarted = 10,
  kCancelled = 11,
  kQuarantined = 12,
};

}  // namespace data

#endif  // KUIPER_INCLUDE_DATA_DATA_REF_H_
