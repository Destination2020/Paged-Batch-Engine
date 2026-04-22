// Shared helpers for dynamic KV cache sizing and serving workspace budgeting.
#include "model/serving_memory_planner.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <glog/logging.h>

namespace model {

double clamp_kv_cache_memory_utilization(double utilization) {
  return std::min(kMaxKVCacheMemoryUtilization,
                  std::max(kMinKVCacheMemoryUtilization, utilization));
}

size_t kv_cache_bytes_per_block_per_layer(const base::KVCacheStorageSpec& spec,
                                          int32_t block_size,
                                          int32_t num_kv_heads,
                                          int32_t head_size) {
  const size_t tokens = static_cast<size_t>(block_size);
  const size_t kv_heads = static_cast<size_t>(num_kv_heads);
  const size_t head_dim = static_cast<size_t>(head_size);
  size_t bytes = 2u * tokens * kv_heads * head_dim * base::DataTypeSize(spec.storage_dtype);
  if (spec.has_scales()) {
    bytes += 2u * tokens * kv_heads * base::DataTypeSize(spec.scale_dtype);
  }
  return bytes;
}

size_t kv_cache_safety_reserve_bytes(size_t free_bytes) {
  return std::max(
      kMinKVCacheSafetyReserveBytes,
      static_cast<size_t>(std::ceil(static_cast<double>(free_bytes) * 0.05)));
}

size_t serving_workspace_bytes_for_tokens(size_t workspace_bytes_per_token,
                                          int32_t token_capacity) {
  return workspace_bytes_per_token * static_cast<size_t>(std::max(0, token_capacity));
}

int32_t estimate_dynamic_kv_blocks_per_layer(const base::KVCacheStorageSpec& storage_spec,
                                             const DynamicKVCacheSizingConfig& config,
                                             const base::DeviceMemoryInfo& memory_info) {
  const double clamped_utilization =
      clamp_kv_cache_memory_utilization(config.kv_cache_memory_utilization);
  if (clamped_utilization != config.kv_cache_memory_utilization) {
    LOG(WARNING) << "Clamped KV cache memory utilization from "
                 << config.kv_cache_memory_utilization << " to " << clamped_utilization;
  }

  const size_t bytes_per_block_per_layer =
      kv_cache_bytes_per_block_per_layer(storage_spec, config.block_size,
                                         config.num_kv_heads, config.head_size);
  CHECK_GT(bytes_per_block_per_layer, 0);
  CHECK_GT(config.layer_num, 0);

  const size_t free_bytes = memory_info.free_bytes;
  const size_t total_bytes = memory_info.total_bytes;
  const size_t requested_kv_bytes =
      static_cast<size_t>(std::floor(static_cast<double>(free_bytes) * clamped_utilization));
  const size_t safety_reserve = kv_cache_safety_reserve_bytes(free_bytes);
  if (free_bytes <= safety_reserve) {
    LOG(WARNING) << "KV cache budget is too small after safety reserve: free_bytes="
                 << free_bytes << " safety_reserve=" << safety_reserve
                 << ". Falling back to minimum KV blocks.";
    return config.min_dynamic_kv_blocks;
  }

  const size_t max_safe_kv_bytes = free_bytes - safety_reserve;
  const size_t kv_budget_bytes = std::min(requested_kv_bytes, max_safe_kv_bytes);
  if (kv_budget_bytes < requested_kv_bytes) {
    LOG(WARNING) << "Requested KV cache budget exceeds safety boundary: requested_kv_bytes="
                 << requested_kv_bytes << ", max_safe_kv_bytes=" << max_safe_kv_bytes
                 << ", safety_reserve=" << safety_reserve
                 << ". Capping KV pool size.";
  }

  const size_t bytes_per_all_layer_block =
      static_cast<size_t>(config.layer_num) * bytes_per_block_per_layer;
  const size_t estimated_blocks = kv_budget_bytes / bytes_per_all_layer_block;
  const size_t max_int32_blocks =
      static_cast<size_t>(std::numeric_limits<int32_t>::max());
  const int32_t blocks = static_cast<int32_t>(
      std::min(max_int32_blocks,
               std::max<size_t>(config.min_dynamic_kv_blocks, estimated_blocks)));

  LOG(INFO) << "Dynamic KV cache sizing: free_bytes=" << free_bytes
            << ", total_bytes=" << total_bytes
            << ", kv_cache_memory_utilization=" << clamped_utilization
            << ", requested_kv_bytes=" << requested_kv_bytes
            << ", safety_reserve=" << safety_reserve
            << ", kv_budget_bytes=" << kv_budget_bytes
            << ", bytes_per_block_per_layer=" << bytes_per_block_per_layer
            << ", num_layers=" << config.layer_num
            << ", blocks_per_layer=" << blocks;
  return blocks;
}

int32_t resolve_serving_workspace_token_capacity(
    const WorkspaceTokenSizingConfig& config,
    const base::DeviceMemoryInfo& memory_info) {
  const int32_t requested_tokens =
      std::max(config.min_token_capacity, config.requested_token_capacity);
  if (config.workspace_bytes_per_token == 0) {
    return requested_tokens;
  }

  const size_t free_bytes = memory_info.free_bytes;
  const size_t minimum_kv_pool_bytes =
      static_cast<size_t>(config.min_dynamic_kv_blocks) *
      static_cast<size_t>(config.layer_num) * config.kv_bytes_per_block_per_layer;
  const size_t safety_reserve = kv_cache_safety_reserve_bytes(free_bytes);
  const size_t non_workspace_reserve = safety_reserve + minimum_kv_pool_bytes;

  if (free_bytes <= non_workspace_reserve) {
    LOG(WARNING) << "Device memory is tight before serving workspace allocation: free_bytes="
                 << free_bytes << ", non_workspace_reserve=" << non_workspace_reserve
                 << ". Falling back to minimum workspace token capacity="
                 << config.min_token_capacity;
    return config.min_token_capacity;
  }

  const size_t max_workspace_bytes = free_bytes - non_workspace_reserve;
  const size_t requested_workspace_bytes =
      serving_workspace_bytes_for_tokens(config.workspace_bytes_per_token, requested_tokens);
  if (requested_workspace_bytes <= max_workspace_bytes) {
    return requested_tokens;
  }

  const int32_t max_safe_tokens = static_cast<int32_t>(
      std::min<size_t>(static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                       max_workspace_bytes / config.workspace_bytes_per_token));
  const int32_t resolved_tokens =
      std::max(config.min_token_capacity,
               std::min(requested_tokens, max_safe_tokens));
  LOG(WARNING) << "Clamped serving workspace token capacity from " << requested_tokens
               << " to " << resolved_tokens
               << " to keep room for KV cache and safety reserve. free_bytes="
               << free_bytes
               << ", requested_workspace_bytes=" << requested_workspace_bytes
               << ", max_workspace_bytes=" << max_workspace_bytes
               << ", workspace_bytes_per_token=" << config.workspace_bytes_per_token
               << ", safety_reserve=" << safety_reserve
               << ", minimum_kv_pool_bytes=" << minimum_kv_pool_bytes;
  return resolved_tokens;
}

}  // namespace model
