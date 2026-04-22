// Common serving capacity information shared by model serving demos.
#ifndef KUIPER_INCLUDE_SERVING_SERVING_CAPACITY_H_
#define KUIPER_INCLUDE_SERVING_SERVING_CAPACITY_H_

#include <cstddef>
#include <cstdint>
#include "base/base.h"

namespace serving {

struct ServingCapacityInfo {
  int32_t max_batch_size = 0;
  int32_t block_size = 0;
  int32_t total_kv_blocks = 0;
  int32_t free_kv_blocks = 0;
  int32_t layer_num = 0;
  int32_t model_dim = 0;
  int32_t head_num = 0;
  int32_t kv_head_num = 0;
  int32_t head_size = 0;
  int32_t kv_dim = 0;
  int32_t hidden_dim = 0;
  int32_t vocab_size = 0;
  base::DataType runtime_data_type = base::DataType::kDataTypeUnknown;
  size_t device_free_memory_bytes = 0;
  size_t device_total_memory_bytes = 0;
  size_t kv_bytes_per_token = 0;
  size_t kv_bytes_per_block_per_layer = 0;
  size_t total_kv_pool_bytes = 0;
  size_t workspace_bytes_per_token = 0;
  int32_t serving_workspace_token_capacity = 0;
  size_t serving_workspace_reserved_bytes = 0;
};

}  // namespace serving

#endif  // KUIPER_INCLUDE_SERVING_SERVING_CAPACITY_H_
