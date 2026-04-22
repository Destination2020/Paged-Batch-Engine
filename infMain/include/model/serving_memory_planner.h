// Shared helpers for dynamic KV cache sizing and serving workspace budgeting.
// Models provide their own workspace-bytes-per-token formula, then call this
// planner to translate that footprint into a safe token/block budget.
#ifndef KUIPER_INCLUDE_MODEL_SERVING_MEMORY_PLANNER_H_
#define KUIPER_INCLUDE_MODEL_SERVING_MEMORY_PLANNER_H_

#include <cstddef>
#include <cstdint>
#include "base/backend_runtime.h"
#include "base/kv_cache_format.h"

namespace model {

inline constexpr double kMinKVCacheMemoryUtilization = 0.01;
inline constexpr double kMaxKVCacheMemoryUtilization = 0.95;
inline constexpr size_t kMinKVCacheSafetyReserveBytes = 256ull * 1024ull * 1024ull;

struct DynamicKVCacheSizingConfig {
  int32_t layer_num = 0;
  int32_t block_size = 0;
  int32_t num_kv_heads = 0;
  int32_t head_size = 0;
  int32_t min_dynamic_kv_blocks = 1;
  double kv_cache_memory_utilization = 0.80;
};

struct ServingWorkspaceProfile {
  // Token-proportional activations/logits/qkv scratch in the model runtime
  // dtype, typically coming from reshape-able serving buffers.
  size_t activation_bytes_per_token = 0;
  // Token-proportional auxiliary scratch that is not naturally expressed in the
  // runtime dtype, e.g. fp32 split-kv partial accumulators.
  size_t aux_bytes_per_token = 0;

  size_t total_bytes_per_token() const {
    return activation_bytes_per_token + aux_bytes_per_token;
  }
};

struct WorkspaceTokenSizingConfig {
  size_t workspace_bytes_per_token = 0;
  size_t kv_bytes_per_block_per_layer = 0;
  int32_t layer_num = 0;
  int32_t requested_token_capacity = 0;
  int32_t min_token_capacity = 1;
  int32_t min_dynamic_kv_blocks = 1;
};

double clamp_kv_cache_memory_utilization(double utilization);

size_t kv_cache_bytes_per_block_per_layer(const base::KVCacheStorageSpec& spec,
                                          int32_t block_size,
                                          int32_t num_kv_heads,
                                          int32_t head_size);

size_t kv_cache_safety_reserve_bytes(size_t free_bytes);

size_t serving_workspace_bytes_for_tokens(size_t workspace_bytes_per_token,
                                          int32_t token_capacity);

int32_t estimate_dynamic_kv_blocks_per_layer(const base::KVCacheStorageSpec& storage_spec,
                                             const DynamicKVCacheSizingConfig& config,
                                             const base::DeviceMemoryInfo& memory_info);

int32_t resolve_serving_workspace_token_capacity(
    const WorkspaceTokenSizingConfig& config,
    const base::DeviceMemoryInfo& memory_info);

}  // namespace model

#endif  // KUIPER_INCLUDE_MODEL_SERVING_MEMORY_PLANNER_H_
