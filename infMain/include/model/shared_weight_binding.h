#ifndef KUIPER_INCLUDE_MODEL_SHARED_WEIGHT_BINDING_H_
#define KUIPER_INCLUDE_MODEL_SHARED_WEIGHT_BINDING_H_

#include <cstdint>
#include <memory>

#include "base/base.h"

namespace model {

struct SharedWeightBinding {
  const void* host_file_base = nullptr;
  void* device_file_base = nullptr;
  uint64_t file_bytes = 0;
  base::DataType dtype = base::DataType::kDataTypeUnknown;
  uint64_t owner_incarnation = 0;
  uint64_t allocation_id = 0;
  uint32_t generation = 0;
  std::shared_ptr<void> capsule;

  bool valid() const {
    return host_file_base != nullptr && device_file_base != nullptr && file_bytes != 0 &&
           dtype != base::DataType::kDataTypeUnknown && owner_incarnation != 0 &&
           allocation_id != 0 && generation != 0 && capsule != nullptr;
  }
};

struct SharedWeightBindingReport {
  bool enabled = false;
  uint64_t imported_logical_bytes = 0;
  uint64_t bound_tensor_views = 0;
  uint64_t bound_tensor_logical_bytes = 0;
  uint64_t unique_tensor_ranges = 0;
  uint64_t unique_tensor_bytes = 0;
  bool attention_views_bound = false;
  bool embedding_view_bound = false;
  bool output_view_bound = false;
};

}  // namespace model

#endif  // KUIPER_INCLUDE_MODEL_SHARED_WEIGHT_BINDING_H_
