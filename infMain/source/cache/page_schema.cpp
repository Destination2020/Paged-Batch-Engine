#include "cache/page_schema.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <unordered_set>

namespace cache {
namespace {

bool IsScale(ComponentKind kind) {
  return kind == ComponentKind::kKeyScale || kind == ComponentKind::kValueScale;
}

bool IsValidKind(ComponentKind kind) {
  return kind == ComponentKind::kKey || kind == ComponentKind::kValue ||
         kind == ComponentKind::kKeyScale || kind == ComponentKind::kValueScale;
}

base::Status ValidateDescriptor(const ComponentDescriptor& component) {
  if (component.layer < 0 || component.token_count <= 0 || component.head_begin < 0 ||
      component.head_count <= 0 || component.elements_per_head <= 0 ||
      component.dtype == base::DataType::kDataTypeUnknown || !IsValidKind(component.kind)) {
    return base::error::InvalidArgument("Component descriptor has an invalid shape or dtype.");
  }
  if (component.byte_size() == 0) {
    return base::error::InvalidArgument("Component byte size is zero or overflows size_t.");
  }
  return base::error::Success();
}

}  // namespace

const char* ComponentKindName(ComponentKind kind) {
  switch (kind) {
    case ComponentKind::kKey:
      return "key";
    case ComponentKind::kValue:
      return "value";
    case ComponentKind::kKeyScale:
      return "key_scale";
    case ComponentKind::kValueScale:
      return "value_scale";
  }
  return "unknown";
}

size_t ComponentDescriptor::element_count() const {
  if (token_count <= 0 || head_count <= 0 || elements_per_head <= 0) {
    return 0;
  }
  const size_t tokens = static_cast<size_t>(token_count);
  const size_t heads = static_cast<size_t>(head_count);
  const size_t width = static_cast<size_t>(elements_per_head);
  if (tokens > std::numeric_limits<size_t>::max() / heads ||
      tokens * heads > std::numeric_limits<size_t>::max() / width) {
    return 0;
  }
  return tokens * heads * width;
}

size_t ComponentDescriptor::byte_size() const {
  const size_t elements = element_count();
  const size_t dtype_size = base::DataTypeSize(dtype);
  if (dtype_size == 0 || elements > std::numeric_limits<size_t>::max() / dtype_size) {
    return 0;
  }
  return elements * dtype_size;
}

std::string ComponentDescriptor::semantic_key() const {
  std::ostringstream stream;
  stream << layer << ':' << static_cast<int>(kind) << ':' << token_count << ':' << head_begin
         << ':' << head_count << ':' << elements_per_head << ':' << static_cast<int>(dtype);
  return stream.str();
}

bool ComponentDescriptor::operator==(const ComponentDescriptor& other) const {
  return layer == other.layer && kind == other.kind && token_count == other.token_count &&
         head_begin == other.head_begin && head_count == other.head_count &&
         elements_per_head == other.elements_per_head && dtype == other.dtype;
}

base::Status PageSchema::validate() const {
  if (version == 0 || layer_count <= 0 || page_size <= 0 || kv_head_count <= 0 ||
      head_size <= 0) {
    return base::error::InvalidArgument("Page schema dimensions and version must be positive.");
  }
  if (storage.logical_dtype == base::DataType::kDataTypeUnknown ||
      storage.storage_dtype == base::DataType::kDataTypeUnknown) {
    return base::error::InvalidArgument("Page schema storage spec is incomplete.");
  }
  if (storage.storage_mode == base::BlockStorageMode::kPlain) {
    if (storage.storage_dtype != storage.logical_dtype || storage.has_scales()) {
      return base::error::InvalidArgument("Plain page storage must use logical dtype without scales.");
    }
  } else if (storage.storage_mode == base::BlockStorageMode::kFp8E4M3PerTokenHead) {
    if (storage.logical_dtype != base::DataType::kDataTypeBf16 ||
        storage.storage_dtype != base::DataType::kDataTypeInt8 ||
        storage.scale_dtype != base::DataType::kDataTypeFp32) {
      return base::error::InvalidArgument(
          "FP8 page storage requires BF16 logical, int8 payload, and fp32 scales.");
    }
  } else {
    return base::error::InvalidArgument("Page schema storage mode is unsupported.");
  }

  std::unordered_set<std::string> seen;
  using Coverage = std::vector<std::pair<int32_t, int32_t>>;
  std::vector<std::vector<Coverage>> coverage(
      static_cast<size_t>(layer_count), std::vector<Coverage>(4));
  for (const auto& component : components) {
    base::Status status = ValidateDescriptor(component);
    if (!status) {
      return status;
    }
    if (component.layer >= layer_count || component.token_count != page_size ||
        component.head_begin > kv_head_count ||
        component.head_count > kv_head_count - component.head_begin) {
      return base::error::InvalidArgument(
          "Page component lies outside the schema head coverage.");
    }
    const bool is_scale = IsScale(component.kind);
    const base::DataType expected_dtype = is_scale ? storage.scale_dtype : storage.storage_dtype;
    const int32_t expected_width = is_scale ? 1 : head_size;
    if (expected_dtype == base::DataType::kDataTypeUnknown || component.dtype != expected_dtype ||
        component.elements_per_head != expected_width) {
      return base::error::InvalidArgument("Page component shape or dtype disagrees with storage.");
    }
    if (!storage.has_scales() && is_scale) {
      return base::error::InvalidArgument("Plain storage must not contain scale components.");
    }

    const std::string key = component.semantic_key();
    if (!seen.insert(key).second) {
      return base::error::InvalidArgument("Page schema contains a duplicate component.");
    }
    coverage[static_cast<size_t>(component.layer)]
            [static_cast<size_t>(component.kind)]
                .emplace_back(component.head_begin,
                              component.head_begin + component.head_count);
  }

  const size_t required_kinds = storage.has_scales() ? 4u : 2u;
  for (int32_t layer = 0; layer < layer_count; ++layer) {
    for (size_t kind = 0; kind < required_kinds; ++kind) {
      auto& ranges = coverage[static_cast<size_t>(layer)][kind];
      std::sort(ranges.begin(), ranges.end());
      int32_t cursor = 0;
      for (const auto& range : ranges) {
        if (range.first != cursor) {
          return base::error::InvalidArgument(
              "Page schema head coverage is missing or overlapping.");
        }
        cursor = range.second;
      }
      if (cursor != kv_head_count) {
        return base::error::InvalidArgument(
            "Page schema is missing required K/V/scale head coverage.");
      }
    }
    for (size_t kind = required_kinds; kind < 4; ++kind) {
      if (!coverage[static_cast<size_t>(layer)][kind].empty()) {
        return base::error::InvalidArgument(
            "Page schema contains scales for a representation without scales.");
      }
    }
  }
  return base::error::Success();
}

std::string PageSchema::logical_fingerprint() const {
  std::ostringstream stream;
  stream << version << '|' << layer_count << '|' << page_size << '|'
         << kv_head_count << '|' << head_size << '|'
         << static_cast<int>(storage.logical_dtype);
  return stream.str();
}

std::string PageSchema::fingerprint() const {
  std::vector<std::string> keys;
  keys.reserve(components.size());
  for (const auto& component : components) {
    keys.push_back(component.semantic_key());
  }
  std::sort(keys.begin(), keys.end());

  std::ostringstream stream;
  stream << version << '|' << layer_count << '|' << page_size << '|' << kv_head_count << '|'
         << head_size << '|' << static_cast<int>(storage.storage_mode) << '|'
         << static_cast<int>(storage.logical_dtype) << '|'
         << static_cast<int>(storage.storage_dtype) << '|'
         << static_cast<int>(storage.scale_dtype);
  for (const auto& key : keys) {
    stream << '|' << key;
  }
  return stream.str();
}

base::Status MakeKVPageSchema(uint32_t version,
                              int32_t layer_count,
                              int32_t page_size,
                              int32_t kv_head_count,
                              int32_t head_size,
                              const base::KVCacheStorageSpec& storage,
                              PageSchema* schema) {
  if (schema == nullptr) {
    return base::error::InvalidArgument("Page schema output is null.");
  }
  PageSchema candidate;
  candidate.version = version;
  candidate.layer_count = layer_count;
  candidate.page_size = page_size;
  candidate.kv_head_count = kv_head_count;
  candidate.head_size = head_size;
  candidate.storage = storage;
  for (int32_t layer = 0; layer < layer_count; ++layer) {
    candidate.components.push_back({layer, ComponentKind::kKey, page_size, 0, kv_head_count,
                                    head_size, storage.storage_dtype});
    candidate.components.push_back({layer, ComponentKind::kValue, page_size, 0, kv_head_count,
                                    head_size, storage.storage_dtype});
    if (storage.has_scales()) {
      candidate.components.push_back({layer, ComponentKind::kKeyScale, page_size, 0,
                                      kv_head_count, 1, storage.scale_dtype});
      candidate.components.push_back({layer, ComponentKind::kValueScale, page_size, 0,
                                      kv_head_count, 1, storage.scale_dtype});
    }
  }
  base::Status status = candidate.validate();
  if (!status) {
    return status;
  }
  *schema = std::move(candidate);
  return base::error::Success();
}

}  // namespace cache
