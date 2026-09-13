#ifndef KUIPER_INCLUDE_DATA_KV_SNAPSHOT_CODEC_H_
#define KUIPER_INCLUDE_DATA_KV_SNAPSHOT_CODEC_H_

#include <cstddef>
#include <cstdint>
#include <vector>
#include "base/kv_cache_manager.h"
#include "data/data_ref.h"

namespace data {
inline constexpr uint32_t kKVSnapshotMagic = 0x564b4250;  // "PBKV"
inline constexpr uint16_t kKVSnapshotWireVersion = 1;
DataError EncodeKVSnapshot(const base::KVRequestSnapshot& snapshot,
                           std::vector<uint8_t>* output);
DataError DecodeKVSnapshot(const uint8_t* bytes, size_t size, uint64_t max_payload_bytes,
                           base::KVRequestSnapshot* output);
}  // namespace data
#endif
