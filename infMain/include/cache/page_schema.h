#ifndef KUIPER_INCLUDE_CACHE_PAGE_SCHEMA_H_
#define KUIPER_INCLUDE_CACHE_PAGE_SCHEMA_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "base/base.h"
#include "base/kv_cache_format.h"

namespace cache {

enum class ComponentKind : uint8_t {
  kKey = 0,
  kValue = 1,
  kKeyScale = 2,
  kValueScale = 3,
};

const char* ComponentKindName(ComponentKind kind);

struct ComponentDescriptor {
  int32_t layer = -1;
  ComponentKind kind = ComponentKind::kKey;
  int32_t token_count = 0;
  int32_t head_begin = 0;
  int32_t head_count = 0;
  int32_t elements_per_head = 0;
  base::DataType dtype = base::DataType::kDataTypeUnknown;

  size_t element_count() const;
  size_t byte_size() const;
  std::string semantic_key() const;
  bool operator==(const ComponentDescriptor& other) const;
};

struct PageSchema {
  uint32_t version = 1;
  int32_t layer_count = 0;
  int32_t page_size = 0;
  int32_t kv_head_count = 0;
  int32_t head_size = 0;
  base::KVCacheStorageSpec storage;
  std::vector<ComponentDescriptor> components;

  base::Status validate() const;
  // Stable model-compute identity. Physical head partitioning and storage
  // layout live in fingerprint(), not in this value.
  std::string logical_fingerprint() const;
  std::string fingerprint() const;
};

base::Status MakeKVPageSchema(uint32_t version,
                              int32_t layer_count,
                              int32_t page_size,
                              int32_t kv_head_count,
                              int32_t head_size,
                              const base::KVCacheStorageSpec& storage,
                              PageSchema* schema);

}  // namespace cache

#endif  // KUIPER_INCLUDE_CACHE_PAGE_SCHEMA_H_
