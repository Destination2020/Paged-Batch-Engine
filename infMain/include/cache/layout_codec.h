#ifndef KUIPER_INCLUDE_CACHE_LAYOUT_CODEC_H_
#define KUIPER_INCLUDE_CACHE_LAYOUT_CODEC_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "base/base.h"
#include "cache/page_schema.h"

namespace cache {

struct ComponentLayout {
  ComponentDescriptor component;
  size_t offset_bytes = 0;
  size_t size_bytes = 0;
  int64_t virtual_block_id = -1;
};

struct LayoutDescriptor {
  std::string layout_id;
  uint64_t epoch = 0;
  size_t total_bytes = 0;
  std::vector<ComponentLayout> components;

  base::Status validate(const PageSchema& schema) const;
};

base::Status MakePackedLayout(const PageSchema& schema,
                              const std::string& layout_id,
                              uint64_t epoch,
                              const std::vector<size_t>& component_order,
                              const std::vector<int64_t>& virtual_block_ids,
                              LayoutDescriptor* layout);

}  // namespace cache

#endif  // KUIPER_INCLUDE_CACHE_LAYOUT_CODEC_H_
