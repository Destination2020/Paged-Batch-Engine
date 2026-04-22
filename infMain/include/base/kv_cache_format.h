#ifndef KUIPER_INCLUDE_BASE_KV_CACHE_FORMAT_H_
#define KUIPER_INCLUDE_BASE_KV_CACHE_FORMAT_H_

#include <cstddef>
#include <cstdint>
#include "base/base.h"

namespace base {

enum class BlockStorageMode : uint8_t {
  kPlain = 0,
  kFp8E4M3PerTokenHead = 1,
};

inline const char* BlockStorageModeName(BlockStorageMode storage_mode) {
  switch (storage_mode) {
    case BlockStorageMode::kPlain:
      return "plain";
    case BlockStorageMode::kFp8E4M3PerTokenHead:
      return "fp8_e4m3_per_token_head";
    default:
      return "unknown";
  }
}

struct KVCacheStorageSpec {
  BlockStorageMode storage_mode = BlockStorageMode::kPlain;
  DataType logical_dtype = DataType::kDataTypeUnknown;
  DataType storage_dtype = DataType::kDataTypeUnknown;
  DataType scale_dtype = DataType::kDataTypeUnknown;

  bool has_scales() const { return scale_dtype != DataType::kDataTypeUnknown; }

  bool uses_fp8_storage() const {
    return storage_mode == BlockStorageMode::kFp8E4M3PerTokenHead;
  }
};

inline KVCacheStorageSpec MakeKVCacheStorageSpec(DataType logical_dtype,
                                                 BlockStorageMode storage_mode) {
  KVCacheStorageSpec spec;
  spec.storage_mode = storage_mode;
  spec.logical_dtype = logical_dtype;
  if (storage_mode == BlockStorageMode::kFp8E4M3PerTokenHead) {
    spec.storage_dtype = DataType::kDataTypeInt8;
    spec.scale_dtype = DataType::kDataTypeFp32;
  } else {
    spec.storage_dtype = logical_dtype;
    spec.scale_dtype = DataType::kDataTypeUnknown;
  }
  return spec;
}

inline Status ValidateKVCacheStorageSpec(const KVCacheStorageSpec& spec,
                                         DeviceType device_type,
                                         DataType runtime_data_type) {
  if (spec.logical_dtype == DataType::kDataTypeUnknown ||
      spec.storage_dtype == DataType::kDataTypeUnknown) {
    return error::InternalError("KV cache storage spec is incomplete.");
  }

  if (spec.storage_mode == BlockStorageMode::kPlain) {
    if (spec.storage_dtype != spec.logical_dtype || spec.has_scales()) {
      return error::InternalError("Plain KV cache spec must use runtime dtype without scales.");
    }
    return error::Success();
  }

  if (spec.storage_mode == BlockStorageMode::kFp8E4M3PerTokenHead) {
    if (device_type != DeviceType::kDeviceCUDA) {
      return error::InternalError("FP8 KV cache is only supported on CUDA.");
    }
    if (runtime_data_type != DataType::kDataTypeBf16 ||
        spec.logical_dtype != DataType::kDataTypeBf16) {
      return error::InternalError("FP8 KV cache requires BF16 runtime.");
    }
    if (spec.storage_dtype != DataType::kDataTypeInt8 ||
        spec.scale_dtype != DataType::kDataTypeFp32) {
      return error::InternalError("FP8 KV cache spec must use int8 storage and fp32 scales.");
    }
    return error::Success();
  }

  return error::InternalError("Unsupported KV cache storage mode.");
}

inline size_t KVCacheBytesPerToken(const KVCacheStorageSpec& spec,
                                   int32_t layer_num,
                                   int32_t kv_head_num,
                                   int32_t head_size) {
  const size_t layer_count = static_cast<size_t>(layer_num);
  const size_t kv_heads = static_cast<size_t>(kv_head_num);
  const size_t head_dim = static_cast<size_t>(head_size);
  size_t bytes = layer_count * 2u * kv_heads * head_dim * DataTypeSize(spec.storage_dtype);
  if (spec.has_scales()) {
    bytes += layer_count * 2u * kv_heads * DataTypeSize(spec.scale_dtype);
  }
  return bytes;
}

}  // namespace base

#endif  // KUIPER_INCLUDE_BASE_KV_CACHE_FORMAT_H_
