#pragma once
#include "serving/pd_handoff.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <numeric>
#include <utility>

#include <cuda_runtime_api.h>
#if defined(KUIPER_ENABLE_NCCL)
#include <nccl.h>
#endif

#include "base/alloc.h"
#include "base/nvtx_utils.h"

namespace serving::handoff_detail {
#if defined(KUIPER_ENABLE_NCCL)
std::string nccl_error_string(const char* stage, ncclResult_t result);
#endif
base::Status nccl_unavailable_status();
base::MemcpyKind copy_kind_for_devices(base::DeviceType src, base::DeviceType dst);
const uint8_t* component_address(const base::KVBlockPayloadPtrs& payload,
                                 cache::ComponentKind kind);
base::Status bind_manifest_transfers(
    const KVBlockManifest& manifest, base::KVCacheManager* source_manager,
    base::KVCacheManager* destination_manager, cache::TransferPlanner* planner,
    std::vector<cache::BoundTransfer>* transfers);
base::Status validate_cuda_kv_manifest(const KVBlockManifest& manifest,
                                       base::KVCacheManager* src_kv_manager,
                                       base::KVCacheManager* dst_kv_manager,
                                       const char* connector_name);
base::Status validate_cuda_kv_layer_mapping(const KVBlockMapping& mapping,
                                            const KVPoolDescriptor& src_pool,
                                            const KVPoolDescriptor& dst_pool,
                                            base::KVCacheManager* src_kv_manager,
                                            base::KVCacheManager* dst_kv_manager,
                                            const char* connector_name);
base::Status validate_remote_nccl_options(
    const KVBlockManifest& manifest,
    const RemoteNcclKVTransferOptions& options);
}
