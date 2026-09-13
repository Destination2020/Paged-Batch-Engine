#ifndef KUIPER_INCLUDE_DATA_TENSOR_BUNDLE_H_
#define KUIPER_INCLUDE_DATA_TENSOR_BUNDLE_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "data/data_ref.h"

namespace data {

inline constexpr uint32_t kTensorBundleSchemaVersion = 1;

// Wire values are fixed. They deliberately do not reuse a compiler ABI enum.
enum class TensorDType : uint16_t {
  kUnknown = 0,
  kFloat32 = 1,
  kFloat16 = 2,
  kBFloat16 = 3,
  kInt32 = 4,
  kInt64 = 5,
  kUint8 = 6,
};

struct TensorComponent {
  std::string name;
  TensorDType dtype = TensorDType::kUnknown;
  std::vector<uint64_t> shape;
  uint64_t byte_offset = 0;
  uint64_t byte_length = 0;
  Digest256 checksum{};
};

struct TensorBundleSchema {
  uint32_t version = kTensorBundleSchemaVersion;
  uint32_t alignment = 1;
  uint64_t total_bytes = 0;
  std::vector<TensorComponent> components;
};

// Validates the complete allocation before reserve or the first copy. Padding
// between components is allowed, but components must be ordered and disjoint.
DataError ValidateTensorBundle(const TensorBundleSchema& schema,
                               std::string* reason = nullptr);
// Decodes and verifies the portable bundle emitted by C++ or Python clients.
DataError DecodeTensorBundlePayload(const uint8_t* bytes, size_t size,
                                    TensorBundleSchema* schema,
                                    std::string* reason = nullptr);

}  // namespace data

#endif  // KUIPER_INCLUDE_DATA_TENSOR_BUNDLE_H_
