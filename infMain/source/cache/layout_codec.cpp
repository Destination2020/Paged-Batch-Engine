#include "cache/layout_codec.h"

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace cache {

base::Status LayoutDescriptor::validate(const PageSchema& schema) const {
  base::Status schema_status = schema.validate();
  if (!schema_status) {
    return schema_status;
  }
  if (layout_id.empty()) {
    return base::error::InvalidArgument("Layout id must not be empty.");
  }
  if (components.size() != schema.components.size()) {
    return base::error::InvalidArgument("Layout does not cover every schema component.");
  }

  std::unordered_map<std::string, ComponentDescriptor> expected;
  for (const auto& component : schema.components) {
    expected.emplace(component.semantic_key(), component);
  }
  std::unordered_set<std::string> seen;
  std::vector<std::pair<size_t, size_t>> ranges;
  ranges.reserve(components.size());
  for (const auto& entry : components) {
    const std::string key = entry.component.semantic_key();
    const auto expected_it = expected.find(key);
    if (expected_it == expected.end() || !(entry.component == expected_it->second)) {
      return base::error::InvalidArgument("Layout contains an unknown or incompatible component.");
    }
    if (!seen.insert(key).second) {
      return base::error::InvalidArgument("Layout contains a duplicate component.");
    }
    if (entry.size_bytes != entry.component.byte_size() ||
        entry.offset_bytes > total_bytes || entry.size_bytes > total_bytes - entry.offset_bytes) {
      return base::error::InvalidArgument("Layout component has invalid size or bounds.");
    }
    ranges.emplace_back(entry.offset_bytes, entry.offset_bytes + entry.size_bytes);
  }
  std::sort(ranges.begin(), ranges.end());
  for (size_t index = 1; index < ranges.size(); ++index) {
    if (ranges[index].first < ranges[index - 1].second) {
      return base::error::InvalidArgument("Layout component byte ranges overlap.");
    }
  }
  return base::error::Success();
}

base::Status MakePackedLayout(const PageSchema& schema,
                              const std::string& layout_id,
                              uint64_t epoch,
                              const std::vector<size_t>& component_order,
                              const std::vector<int64_t>& virtual_block_ids,
                              LayoutDescriptor* layout) {
  if (layout == nullptr) {
    return base::error::InvalidArgument("Layout output is null.");
  }
  base::Status schema_status = schema.validate();
  if (!schema_status) {
    return schema_status;
  }
  if (component_order.size() != schema.components.size() ||
      (!virtual_block_ids.empty() && virtual_block_ids.size() != component_order.size())) {
    return base::error::InvalidArgument("Packed layout order or virtual block list has wrong size.");
  }

  LayoutDescriptor candidate;
  candidate.layout_id = layout_id;
  candidate.epoch = epoch;
  std::vector<bool> used(schema.components.size(), false);
  for (size_t position = 0; position < component_order.size(); ++position) {
    const size_t component_index = component_order[position];
    if (component_index >= schema.components.size() || used[component_index]) {
      return base::error::InvalidArgument("Packed layout order is not a permutation.");
    }
    used[component_index] = true;
    const auto& component = schema.components[component_index];
    const size_t bytes = component.byte_size();
    if (bytes > std::numeric_limits<size_t>::max() - candidate.total_bytes) {
      return base::error::InvalidArgument("Packed layout byte size overflows size_t.");
    }
    const int64_t block_id = virtual_block_ids.empty()
                                 ? static_cast<int64_t>(component_index)
                                 : virtual_block_ids[position];
    candidate.components.push_back({component, candidate.total_bytes, bytes, block_id});
    candidate.total_bytes += bytes;
  }
  base::Status status = candidate.validate(schema);
  if (!status) {
    return status;
  }
  *layout = std::move(candidate);
  return base::error::Success();
}

}  // namespace cache
