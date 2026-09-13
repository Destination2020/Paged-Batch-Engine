#include "data/protocol.h"

#include <algorithm>

namespace data {
namespace {

template <typename T>
void PutLittleEndian(T value, std::vector<uint8_t>* output) {
  for (size_t i = 0; i < sizeof(T); ++i) {
    output->push_back(static_cast<uint8_t>((value >> (8 * i)) & 0xff));
  }
}

template <typename T>
bool GetLittleEndian(const uint8_t** cursor, const uint8_t* end, T* value) {
  if (static_cast<size_t>(end - *cursor) < sizeof(T)) return false;
  *value = 0;
  for (size_t i = 0; i < sizeof(T); ++i) {
    *value |= static_cast<T>((*cursor)[i]) << (8 * i);
  }
  *cursor += sizeof(T);
  return true;
}

bool ValidKind(DataKind kind) {
  return kind == DataKind::kKVPage || kind == DataKind::kTensorBundle ||
         kind == DataKind::kCheckpoint;
}

}  // namespace

DataError EncodeDataRef(const DataRef& ref, std::vector<uint8_t>* output) {
  if (output == nullptr || !output->empty() ||
      ref.protocol_version != kDataProtocolVersion || !ValidKind(ref.kind) ||
      ref.allocation.owner_incarnation == 0 || ref.allocation.allocation_id == 0 ||
      ref.allocation.generation == 0 || ref.logical_bytes == 0) {
    return DataError::kInvalidArgument;
  }
  output->reserve(kDataRefWireBytes);
  PutLittleEndian<uint32_t>(kDataRefWireMagic, output);
  PutLittleEndian<uint16_t>(ref.protocol_version, output);
  output->push_back(static_cast<uint8_t>(ref.kind));
  output->push_back(0);
  output->insert(output->end(), ref.content.digest.begin(), ref.content.digest.end());
  output->insert(output->end(), ref.representation.digest.begin(), ref.representation.digest.end());
  PutLittleEndian<uint64_t>(ref.allocation.owner_incarnation, output);
  PutLittleEndian<uint64_t>(ref.allocation.allocation_id, output);
  PutLittleEndian<uint32_t>(ref.allocation.generation, output);
  PutLittleEndian<uint64_t>(ref.logical_bytes, output);
  return DataError::kOk;
}

DataError DecodeDataRef(const uint8_t* bytes, size_t size, DataRef* output) {
  if (bytes == nullptr || output == nullptr || size != kDataRefWireBytes) {
    return DataError::kInvalidArgument;
  }
  const uint8_t* cursor = bytes;
  const uint8_t* end = bytes + size;
  uint32_t magic = 0;
  uint16_t version = 0;
  if (!GetLittleEndian(&cursor, end, &magic) || !GetLittleEndian(&cursor, end, &version)) {
    return DataError::kInvalidArgument;
  }
  if (magic != kDataRefWireMagic || version != kDataProtocolVersion) {
    return DataError::kUnsupportedVersion;
  }
  DataRef decoded;
  decoded.protocol_version = version;
  decoded.kind = static_cast<DataKind>(*cursor++);
  const uint8_t reserved = *cursor++;
  if (!ValidKind(decoded.kind) || reserved != 0) return DataError::kInvalidArgument;
  std::copy(cursor, cursor + decoded.content.digest.size(), decoded.content.digest.begin());
  cursor += decoded.content.digest.size();
  std::copy(cursor, cursor + decoded.representation.digest.size(),
            decoded.representation.digest.begin());
  cursor += decoded.representation.digest.size();
  if (!GetLittleEndian(&cursor, end, &decoded.allocation.owner_incarnation) ||
      !GetLittleEndian(&cursor, end, &decoded.allocation.allocation_id) ||
      !GetLittleEndian(&cursor, end, &decoded.allocation.generation) ||
      !GetLittleEndian(&cursor, end, &decoded.logical_bytes) || cursor != end ||
      decoded.allocation.owner_incarnation == 0 || decoded.allocation.allocation_id == 0 ||
      decoded.allocation.generation == 0 || decoded.logical_bytes == 0) {
    return DataError::kInvalidArgument;
  }
  *output = decoded;
  return DataError::kOk;
}

}  // namespace data
