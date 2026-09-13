#ifndef KUIPER_INCLUDE_DATA_PROTOCOL_H_
#define KUIPER_INCLUDE_DATA_PROTOCOL_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "data/data_ref.h"

namespace data {

inline constexpr uint32_t kDataRefWireMagic = 0x52444250;  // "PBDR" little endian
inline constexpr uint16_t kDataProtocolVersion = 1;
inline constexpr size_t kDataRefWireBytes = 4 + 2 + 1 + 1 + 32 + 32 + 8 + 8 + 4 + 8;

DataError EncodeDataRef(const DataRef& ref, std::vector<uint8_t>* output);
DataError DecodeDataRef(const uint8_t* bytes, size_t size, DataRef* output);

}  // namespace data

#endif  // KUIPER_INCLUDE_DATA_PROTOCOL_H_
