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

#include "pd_handoff_internal.h"
namespace serving::handoff_detail {


#if defined(KUIPER_ENABLE_NCCL)
std::string nccl_error_string(const char* stage, ncclResult_t result) {
  return std::string(stage) + " failed: " + ncclGetErrorString(result) +
         " (" + std::to_string(static_cast<int>(result)) + ")";
}
#endif

base::Status nccl_unavailable_status() {
  return base::error::FunctionNotImplement(
      "NCCL support is not enabled. Install NCCL development headers/libraries "
      "and configure with -DKUIPER_ENABLE_NCCL=ON.");
}

base::MemcpyKind copy_kind_for_devices(base::DeviceType src, base::DeviceType dst) {
  if (src == base::DeviceType::kDeviceCPU && dst == base::DeviceType::kDeviceCPU) {
    return base::MemcpyKind::kMemcpyCPU2CPU;
  }
  if (src == base::DeviceType::kDeviceCPU && dst == base::DeviceType::kDeviceCUDA) {
    return base::MemcpyKind::kMemcpyCPU2CUDA;
  }
  if (src == base::DeviceType::kDeviceCUDA && dst == base::DeviceType::kDeviceCPU) {
    return base::MemcpyKind::kMemcpyCUDA2CPU;
  }
  if (src == base::DeviceType::kDeviceCUDA && dst == base::DeviceType::kDeviceCUDA) {
    return base::MemcpyKind::kMemcpyCUDA2CUDA;
  }
  return base::MemcpyKind::kMemcpyCPU2CPU;
}

const uint8_t* component_address(const base::KVBlockPayloadPtrs& payload,
                                 cache::ComponentKind kind) {
  switch (kind) {
    case cache::ComponentKind::kKey: return static_cast<const uint8_t*>(payload.key);
    case cache::ComponentKind::kValue: return static_cast<const uint8_t*>(payload.value);
    case cache::ComponentKind::kKeyScale:
      return static_cast<const uint8_t*>(payload.key_scale);
    case cache::ComponentKind::kValueScale:
      return static_cast<const uint8_t*>(payload.value_scale);
  }
  return nullptr;
}

base::Status bind_manifest_transfers(
    const KVBlockManifest& manifest, base::KVCacheManager* source_manager,
    base::KVCacheManager* destination_manager, cache::TransferPlanner* planner,
    std::vector<cache::BoundTransfer>* transfers) {
  if (!source_manager || !destination_manager || !planner || !transfers ||
      !transfers->empty())
    return base::error::InvalidArgument("invalid semantic transfer binding arguments");
  const int32_t layers = source_manager->num_layers();
  if (layers != destination_manager->num_layers() ||
      static_cast<int32_t>(manifest.layer_mappings.size()) != layers)
    return base::error::InvalidArgument("semantic transfer layer coverage mismatch");
  std::vector<const KVBlockMapping*> mappings(layers, nullptr);
  size_t page_count = 0;
  bool have_page_count = false;
  for (const auto& mapping : manifest.layer_mappings) {
    if (mapping.layer_idx < 0 || mapping.layer_idx >= layers ||
        mappings[mapping.layer_idx] ||
        mapping.src_block_ids.size() != mapping.dst_block_ids.size())
      return base::error::InvalidArgument("semantic transfer mapping is incomplete or duplicate");
    if (have_page_count) {
      if (page_count != mapping.src_block_ids.size())
        return base::error::InvalidArgument("semantic transfer page coverage mismatch");
    } else {
      page_count = mapping.src_block_ids.size();
      have_page_count = true;
    }
    mappings[mapping.layer_idx] = &mapping;
  }
  if (!page_count) return base::error::InvalidArgument("semantic transfer has no pages");
  for (auto* mapping : mappings)
    if (!mapping || mapping->src_block_ids.size() != page_count)
      return base::error::InvalidArgument("semantic transfer missing layer coverage");

  cache::PageSchema source_schema, destination_schema;
  auto& source_pool = source_manager->allocator_mut(0);
  auto& destination_pool = destination_manager->allocator_mut(0);
  auto status = cache::MakeKVPageSchema(1, layers, source_manager->block_size(),
      source_pool.num_kv_heads(), source_pool.head_size(), source_pool.storage_spec(),
      &source_schema);
  if (!status) return status;
  status = cache::MakeKVPageSchema(1, layers, destination_manager->block_size(),
      destination_pool.num_kv_heads(), destination_pool.head_size(),
      destination_pool.storage_spec(), &destination_schema);
  if (!status) return status;
  std::shared_ptr<const cache::TransferPlan> plan;
  status = planner->compile(source_schema, destination_schema,
                            cache::TransferPlacement::kExactDirect, &plan);
  if (!status) return status;
  std::vector<size_t> source_order(source_schema.components.size());
  std::vector<size_t> destination_order(destination_schema.components.size());
  std::iota(source_order.begin(), source_order.end(), 0);
  std::iota(destination_order.begin(), destination_order.end(), 0);
  cache::LayoutDescriptor source_layout, destination_layout;
  status = cache::MakePackedLayout(source_schema, "pd-source", 1, source_order, {},
                                   &source_layout);
  if (!status) return status;
  status = cache::MakePackedLayout(destination_schema, "pd-destination", 1,
                                   destination_order, {}, &destination_layout);
  if (!status) return status;

  std::vector<cache::BoundTransfer> pending;
  pending.reserve(page_count);
  for (size_t page = 0; page < page_count; ++page) {
    cache::ConstComponentEndpoint source_endpoint;
    cache::MutableComponentEndpoint destination_endpoint;
    source_endpoint.layout = &source_layout;
    destination_endpoint.layout = &destination_layout;
    for (const auto& component : source_schema.components) {
      auto payload = source_manager->allocator_mut(component.layer)
                         .get_block_payload_ptrs(mappings[component.layer]->src_block_ids[page]);
      source_endpoint.components.push_back(
          {component, component_address(payload, component.kind), component.byte_size()});
    }
    for (const auto& component : destination_schema.components) {
      auto payload = destination_manager->allocator_mut(component.layer)
                         .get_block_payload_ptrs(mappings[component.layer]->dst_block_ids[page]);
      destination_endpoint.components.push_back(
          {component, const_cast<uint8_t*>(component_address(payload, component.kind)),
           component.byte_size()});
    }
    cache::BoundTransfer bound;
    status = cache::BoundTransfer::BindComponents(
        *plan, source_schema, destination_schema, source_endpoint,
        destination_endpoint, &bound);
    if (!status) return status;
    pending.push_back(std::move(bound));
  }
  *transfers = std::move(pending);
  return base::error::Success();
}

base::Status validate_cuda_kv_manifest(const KVBlockManifest& manifest,
                                       base::KVCacheManager* src_kv_manager,
                                       base::KVCacheManager* dst_kv_manager,
                                       const char* connector_name) {
  if (src_kv_manager->num_layers() != manifest.src_pool.layer_num ||
      dst_kv_manager->num_layers() != manifest.dst_pool.layer_num) {
    return base::error::InvalidArgument(std::string(connector_name) +
                                        " layer_num mismatch");
  }
  if (src_kv_manager->block_size() != manifest.src_pool.block_size ||
      dst_kv_manager->block_size() != manifest.dst_pool.block_size) {
    return base::error::InvalidArgument(std::string(connector_name) +
                                        " block_size mismatch");
  }
  for (const auto& mapping : manifest.layer_mappings) {
    base::BlockAllocator& src_allocator = src_kv_manager->allocator_mut(mapping.layer_idx);
    base::BlockAllocator& dst_allocator = dst_kv_manager->allocator_mut(mapping.layer_idx);
    if (src_allocator.device_type() != base::DeviceType::kDeviceCUDA ||
        dst_allocator.device_type() != base::DeviceType::kDeviceCUDA) {
      return base::error::InvalidArgument(std::string(connector_name) +
                                          " requires CUDA KV pools");
    }
    if (src_allocator.storage_mode() != manifest.src_pool.storage_mode ||
        dst_allocator.storage_mode() != manifest.dst_pool.storage_mode) {
      return base::error::InvalidArgument(std::string(connector_name) +
                                          " storage_mode mismatch");
    }
    if (src_allocator.storage_dtype() != dst_allocator.storage_dtype() ||
        src_allocator.scale_dtype() != dst_allocator.scale_dtype()) {
      return base::error::InvalidArgument(std::string(connector_name) +
                                          " dtype mismatch");
    }
    if (src_allocator.key_value_bytes_per_block() != dst_allocator.key_value_bytes_per_block() ||
        src_allocator.scale_bytes_per_block() != dst_allocator.scale_bytes_per_block()) {
      return base::error::InvalidArgument(std::string(connector_name) +
                                          " block byte size mismatch");
    }
  }
  return base::error::Success();
}

base::Status validate_cuda_kv_layer_mapping(const KVBlockMapping& mapping,
                                            const KVPoolDescriptor& src_pool,
                                            const KVPoolDescriptor& dst_pool,
                                            base::KVCacheManager* src_kv_manager,
                                            base::KVCacheManager* dst_kv_manager,
                                            const char* connector_name) {
  if (mapping.layer_idx < 0 || mapping.layer_idx >= src_pool.layer_num) {
    return base::error::InvalidArgument(std::string(connector_name) +
                                        " layer_idx out of range");
  }
  if (mapping.src_block_ids.size() != mapping.dst_block_ids.size()) {
    return base::error::InvalidArgument(std::string(connector_name) +
                                        " src/dst block mapping size mismatch");
  }
  if (mapping.src_block_ids.empty()) {
    return base::error::InvalidArgument(std::string(connector_name) +
                                        " block mapping cannot be empty");
  }
  base::BlockAllocator& src_allocator = src_kv_manager->allocator_mut(mapping.layer_idx);
  base::BlockAllocator& dst_allocator = dst_kv_manager->allocator_mut(mapping.layer_idx);
  if (src_allocator.device_type() != base::DeviceType::kDeviceCUDA ||
      dst_allocator.device_type() != base::DeviceType::kDeviceCUDA) {
    return base::error::InvalidArgument(std::string(connector_name) +
                                        " requires CUDA KV pools");
  }
  if (src_allocator.storage_mode() != src_pool.storage_mode ||
      dst_allocator.storage_mode() != dst_pool.storage_mode) {
    return base::error::InvalidArgument(std::string(connector_name) +
                                        " storage_mode mismatch");
  }
  if (src_allocator.storage_dtype() != dst_allocator.storage_dtype() ||
      src_allocator.scale_dtype() != dst_allocator.scale_dtype()) {
    return base::error::InvalidArgument(std::string(connector_name) +
                                        " dtype mismatch");
  }
  if (src_allocator.key_value_bytes_per_block() != dst_allocator.key_value_bytes_per_block() ||
      src_allocator.scale_bytes_per_block() != dst_allocator.scale_bytes_per_block()) {
    return base::error::InvalidArgument(std::string(connector_name) +
                                        " block byte size mismatch");
  }
  return base::error::Success();
}

base::Status validate_remote_nccl_options(
    const KVBlockManifest& manifest,
    const RemoteNcclKVTransferOptions& options) {
  base::Status status = manifest.validate();
  if (!status) {
    return status;
  }
  if (options.kv_manager == nullptr) {
    return base::error::InvalidArgument("remote nccl connector KV manager is null");
  }
#if defined(KUIPER_ENABLE_NCCL)
  if (options.nccl_unique_id.size() != sizeof(ncclUniqueId)) {
    return base::error::InvalidArgument("remote nccl connector invalid unique id size");
  }
#endif
  const KVPoolDescriptor& local_pool =
      options.role == RemoteNcclKVTransferRole::kProducer ? manifest.src_pool
                                                          : manifest.dst_pool;
  if (local_pool.device_id != options.device_id) {
    return base::error::InvalidArgument("remote nccl connector device_id mismatch");
  }
  if (options.kv_manager->num_layers() != local_pool.layer_num) {
    return base::error::InvalidArgument("remote nccl connector layer_num mismatch");
  }
  if (options.kv_manager->block_size() != local_pool.block_size) {
    return base::error::InvalidArgument("remote nccl connector block_size mismatch");
  }
  for (const auto& mapping : manifest.layer_mappings) {
    if (mapping.layer_idx < 0 || mapping.layer_idx >= local_pool.layer_num) {
      return base::error::InvalidArgument("remote nccl connector layer_idx out of range");
    }
    if (mapping.src_block_ids.size() != mapping.dst_block_ids.size()) {
      return base::error::InvalidArgument(
          "remote nccl connector src/dst block count mismatch");
    }
    base::BlockAllocator& allocator =
        options.kv_manager->allocator_mut(mapping.layer_idx);
    if (allocator.device_type() != base::DeviceType::kDeviceCUDA) {
      return base::error::InvalidArgument(
          "remote nccl connector requires CUDA KV pool");
    }
    if (allocator.storage_mode() != local_pool.storage_mode) {
      return base::error::InvalidArgument(
          "remote nccl connector storage_mode mismatch");
    }
    const auto& local_ids =
        options.role == RemoteNcclKVTransferRole::kProducer
            ? mapping.src_block_ids
            : mapping.dst_block_ids;
    for (int32_t block_id : local_ids) {
      if (block_id < 0 || block_id >= allocator.num_total_blocks()) {
        return base::error::InvalidArgument(
            "remote nccl connector block id out of range");
      }
    }
  }
  return base::error::Success();
}

}
namespace serving {


int64_t KVPoolDescriptor::bytes_per_block_per_layer() const {
  const auto storage = base::MakeKVCacheStorageSpec(dtype, storage_mode);
  const int64_t payload_element_bytes =
      static_cast<int64_t>(base::DataTypeSize(storage.storage_dtype));
  if (payload_element_bytes <= 0 || block_size <= 0 || kv_head_num <= 0 ||
      head_size <= 0) {
    return 0;
  }
  int64_t bytes = static_cast<int64_t>(2) * block_size * kv_head_num *
                  head_size * payload_element_bytes;
  if (storage.has_scales()) {
    bytes += static_cast<int64_t>(2) * block_size * kv_head_num *
             static_cast<int64_t>(base::DataTypeSize(storage.scale_dtype));
  }
  return bytes;
}

bool KVPoolDescriptor::compatible_with(const KVPoolDescriptor& other) const {
  return layer_num == other.layer_num &&
         block_size == other.block_size &&
         kv_head_num == other.kv_head_num &&
         head_size == other.head_size &&
         dtype == other.dtype &&
         storage_mode == other.storage_mode;
}

base::Status KVBlockManifest::validate() const {
  if (client_request_id.empty()) {
    return base::error::InvalidArgument("pd handoff manifest missing client_request_id");
  }
  if (!handoff_id.valid()) {
    return base::error::InvalidArgument("pd handoff manifest missing handoff_id");
  }
  if (prompt_tokens <= 0) {
    return base::error::InvalidArgument("pd handoff manifest prompt_tokens must be positive");
  }
  if (computed_tokens <= 0 || computed_tokens > prompt_tokens) {
    return base::error::InvalidArgument("pd handoff manifest computed_tokens is invalid");
  }
  if (!src_pool.compatible_with(dst_pool)) {
    return base::error::InvalidArgument("pd handoff source/destination KV pools are incompatible");
  }
  if (src_pool.layer_num <= 0) {
    return base::error::InvalidArgument("pd handoff KV pool layer_num must be positive");
  }
  if (static_cast<int32_t>(layer_mappings.size()) != src_pool.layer_num) {
    return base::error::InvalidArgument("pd handoff layer mapping count does not match layer_num");
  }
  for (const auto& mapping : layer_mappings) {
    if (mapping.layer_idx < 0 || mapping.layer_idx >= src_pool.layer_num) {
      return base::error::InvalidArgument("pd handoff layer_idx out of range");
    }
    if (mapping.src_block_ids.size() != mapping.dst_block_ids.size()) {
      return base::error::InvalidArgument("pd handoff src/dst block mapping size mismatch");
    }
    if (mapping.src_block_ids.empty()) {
      return base::error::InvalidArgument("pd handoff block mapping cannot be empty");
    }
  }
  return base::error::Success();
}

KVTransferStatus KVTransferStatus::Pending() { return {KVTransferState::kPending, ""}; }
KVTransferStatus KVTransferStatus::Completed() { return {KVTransferState::kCompleted, ""}; }
KVTransferStatus KVTransferStatus::Failed(std::string message) {
  return {KVTransferState::kFailed, std::move(message)};
}
KVTransferStatus KVTransferStatus::Cancelled(std::string message) {
  return {KVTransferState::kCancelled, std::move(message)};
}

bool KVTransferStatus::done() const { return state != KVTransferState::kPending; }
bool KVTransferStatus::ok() const { return state == KVTransferState::kCompleted; }

}
