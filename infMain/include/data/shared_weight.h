#ifndef KUIPER_INCLUDE_DATA_SHARED_WEIGHT_H_
#define KUIPER_INCLUDE_DATA_SHARED_WEIGHT_H_

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "base/base.h"
#include "data/data_ref.h"

namespace data {

inline constexpr uint32_t kSharedWeightMagic = 0x57534250;  // "PBSW"
inline constexpr uint16_t kSharedWeightVersion = 1;
inline constexpr const char* kQwen2Bf16SharedWeightLayout = "pbe-qwen2-bf16-v1";

enum class SharedWeightState : uint8_t {
  kAbsent = 0,
  kLoading = 1,
  kReady = 2,
  kFailed = 3,
  kDraining = 4,
};

struct SharedWeightDescriptor {
  uint16_t version = kSharedWeightVersion;
  SharedWeightState state = SharedWeightState::kAbsent;
  uint64_t owner_incarnation = 0;
  uint64_t allocation_id = 0;
  uint32_t generation = 0;
  int32_t device = -1;
  uint64_t bytes = 0;
  base::DataType dtype = base::DataType::kDataTypeUnknown;
  Digest256 model_content{};
  Digest256 layout_identity{};
  cudaUUID_t device_uuid{};
  cudaIpcMemHandle_t memory{};
  cudaIpcEventHandle_t ready{};
};

struct SharedWeightStats {
  SharedWeightState state = SharedWeightState::kAbsent;
  uint64_t owner_incarnation = 0;
  uint64_t allocation_id = 0;
  uint32_t generation = 0;
  uint64_t physical_bytes = 0;
  uint64_t upload_bytes = 0;
  uint64_t upload_count = 0;
  uint64_t active_leases = 0;
  uint64_t allocation_count = 0;
  uint64_t staging_peak_bytes = 0;
  double upload_ms = 0.0;
};

bool EncodeSharedWeightDescriptor(const SharedWeightDescriptor& descriptor,
                                  std::vector<uint8_t>* output);
bool DecodeSharedWeightDescriptor(const uint8_t* bytes, size_t size,
                                  SharedWeightDescriptor* output);

class SharedWeightOwner {
 public:
  ~SharedWeightOwner();
  bool load_file(const std::string& path, int32_t device,
                 uint64_t owner_incarnation, const Digest256& model_content,
                 const Digest256& layout_identity, base::DataType dtype,
                 uint64_t staging_bytes, std::string* error);
  bool descriptor(SharedWeightDescriptor* output, std::string* error) const;
  const SharedWeightStats& stats() const { return stats_; }
  void set_active_leases(uint64_t value) { stats_.active_leases = value; }

 private:
  void reset();
  SharedWeightDescriptor descriptor_{};
  SharedWeightStats stats_{};
  void* memory_ = nullptr;
  cudaEvent_t ready_ = nullptr;
};

class SharedWeightImport {
 public:
  ~SharedWeightImport();
  bool open(const SharedWeightDescriptor& descriptor, int32_t device,
            std::string* error);
  bool wait_ready(void* stream, std::string* error);
  void* mapped() const { return mapped_; }
  const SharedWeightDescriptor& descriptor() const { return descriptor_; }

 private:
  SharedWeightDescriptor descriptor_{};
  int32_t device_ = -1;
  void* mapped_ = nullptr;
  cudaEvent_t ready_ = nullptr;
};

}  // namespace data

#endif  // KUIPER_INCLUDE_DATA_SHARED_WEIGHT_H_
