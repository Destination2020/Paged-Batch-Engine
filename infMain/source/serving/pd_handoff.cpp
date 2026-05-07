#include "serving/pd_handoff.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <utility>

#include <cuda_runtime_api.h>
#include <nccl.h>

#include "base/alloc.h"

namespace serving {
namespace {

std::string nccl_error_string(const char* stage, ncclResult_t result) {
  return std::string(stage) + " failed: " + ncclGetErrorString(result) +
         " (" + std::to_string(static_cast<int>(result)) + ")";
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
  if (options.nccl_unique_id.size() != sizeof(ncclUniqueId)) {
    return base::error::InvalidArgument("remote nccl connector invalid unique id size");
  }
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

}  // namespace

int64_t KVPoolDescriptor::bytes_per_block_per_layer() const {
  const int32_t bytes = static_cast<int32_t>(base::DataTypeSize(dtype));
  if (bytes <= 0 || block_size <= 0 || kv_head_num <= 0 || head_size <= 0) {
    return 0;
  }
  // key + value. FP8 scale storage is tracked by storage_mode but intentionally
  // excluded from this first descriptor size estimate.
  return static_cast<int64_t>(2) * block_size * kv_head_num * head_size * bytes;
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

base::Status InProcNoCopyKVTransferConnector::submit(const KVBlockManifest& manifest,
                                                     HandoffId* handle) {
  if (handle == nullptr) {
    return base::error::InvalidArgument("pd connector submit handle is null");
  }
  base::Status status = manifest.validate();
  if (!status) {
    return status;
  }
  *handle = {next_handoff_id_++};
  transfers_.push_back({*handle, KVTransferStatus::Completed()});
  return base::error::Success();
}

KVTransferStatus InProcNoCopyKVTransferConnector::poll(HandoffId handle) {
  auto it = std::find_if(transfers_.begin(), transfers_.end(), [&](const auto& item) {
    return item.first.value == handle.value;
  });
  if (it == transfers_.end()) {
    return KVTransferStatus::Failed("pd connector unknown handoff id");
  }
  return it->second;
}

void InProcNoCopyKVTransferConnector::cancel(HandoffId handle, const std::string& reason) {
  auto it = std::find_if(transfers_.begin(), transfers_.end(), [&](const auto& item) {
    return item.first.value == handle.value;
  });
  if (it != transfers_.end() && !it->second.done()) {
    it->second = KVTransferStatus::Cancelled(reason);
  }
}

InProcKVBlockCopyConnector::InProcKVBlockCopyConnector(
    base::KVCacheManager* src_kv_manager,
    base::KVCacheManager* dst_kv_manager,
    void* transfer_queue,
    bool need_sync)
    : src_kv_manager_(src_kv_manager),
      dst_kv_manager_(dst_kv_manager),
      transfer_queue_(transfer_queue),
      need_sync_(need_sync) {
  CHECK_NE(src_kv_manager_, nullptr);
  CHECK_NE(dst_kv_manager_, nullptr);
}

InProcKVBlockCopyConnector::~InProcKVBlockCopyConnector() {
  for (auto& transfer : transfers_) {
    if (transfer.completion_event != nullptr) {
      cudaEventDestroy(static_cast<cudaEvent_t>(transfer.completion_event));
      transfer.completion_event = nullptr;
    }
  }
}

base::Status InProcKVBlockCopyConnector::submit(const KVBlockManifest& manifest,
                                                HandoffId* handle) {
  if (handle == nullptr) {
    return base::error::InvalidArgument("pd block copy connector submit handle is null");
  }
  base::Status status = manifest.validate();
  if (!status) {
    return status;
  }

  *handle = {next_handoff_id_++};
  status = copy_blocks(manifest);
  if (!status) {
    transfers_.push_back({*handle, KVTransferStatus::Failed(status.get_err_msg()), nullptr});
    return status;
  }
  void* completion_event = nullptr;
  if (transfer_queue_ != nullptr && !need_sync_) {
    cudaEvent_t event = nullptr;
    const cudaError_t event_status = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    if (event_status != cudaSuccess) {
      transfers_.push_back({*handle,
                            KVTransferStatus::Failed(cudaGetErrorString(event_status)),
                            nullptr});
      return base::error::InternalError(cudaGetErrorString(event_status));
    }
    const cudaError_t record_status = cudaEventRecord(
        event, static_cast<cudaStream_t>(transfer_queue_));
    if (record_status != cudaSuccess) {
      cudaEventDestroy(event);
      transfers_.push_back({*handle,
                            KVTransferStatus::Failed(cudaGetErrorString(record_status)),
                            nullptr});
      return base::error::InternalError(cudaGetErrorString(record_status));
    }
    completion_event = event;
  }
  transfers_.push_back({*handle,
                        completion_event == nullptr ? KVTransferStatus::Completed()
                                                    : KVTransferStatus::Pending(),
                        completion_event});
  return status;
}

KVTransferStatus InProcKVBlockCopyConnector::poll(HandoffId handle) {
  auto it = std::find_if(transfers_.begin(), transfers_.end(), [&](const auto& item) {
    return item.handle.value == handle.value;
  });
  if (it == transfers_.end()) {
    return KVTransferStatus::Failed("pd block copy connector unknown handoff id");
  }
  return poll_transfer(static_cast<size_t>(std::distance(transfers_.begin(), it)));
}

void InProcKVBlockCopyConnector::cancel(HandoffId handle, const std::string& reason) {
  auto it = std::find_if(transfers_.begin(), transfers_.end(), [&](const auto& item) {
    return item.handle.value == handle.value;
  });
  if (it != transfers_.end() && !it->status.done()) {
    if (it->completion_event != nullptr) {
      cudaEventDestroy(static_cast<cudaEvent_t>(it->completion_event));
      it->completion_event = nullptr;
    }
    it->status = KVTransferStatus::Cancelled(reason);
  }
}

KVTransferStatus InProcKVBlockCopyConnector::poll_transfer(size_t index) {
  auto& transfer = transfers_.at(index);
  if (transfer.status.done() || transfer.completion_event == nullptr) {
    return transfer.status;
  }
  const cudaError_t status = cudaEventQuery(static_cast<cudaEvent_t>(transfer.completion_event));
  if (status == cudaSuccess) {
    cudaEventDestroy(static_cast<cudaEvent_t>(transfer.completion_event));
    transfer.completion_event = nullptr;
    transfer.status = KVTransferStatus::Completed();
  } else if (status != cudaErrorNotReady) {
    cudaEventDestroy(static_cast<cudaEvent_t>(transfer.completion_event));
    transfer.completion_event = nullptr;
    transfer.status = KVTransferStatus::Failed(cudaGetErrorString(status));
  }
  return transfer.status;
}

base::Status InProcKVBlockCopyConnector::copy_blocks(const KVBlockManifest& manifest) {
  if (src_kv_manager_->num_layers() != manifest.src_pool.layer_num ||
      dst_kv_manager_->num_layers() != manifest.dst_pool.layer_num) {
    return base::error::InvalidArgument("pd block copy connector layer_num mismatch");
  }
  if (src_kv_manager_->block_size() != manifest.src_pool.block_size ||
      dst_kv_manager_->block_size() != manifest.dst_pool.block_size) {
    return base::error::InvalidArgument("pd block copy connector block_size mismatch");
  }

  for (const auto& mapping : manifest.layer_mappings) {
    base::BlockAllocator& src_allocator = src_kv_manager_->allocator_mut(mapping.layer_idx);
    base::BlockAllocator& dst_allocator = dst_kv_manager_->allocator_mut(mapping.layer_idx);
    if (src_allocator.storage_mode() != manifest.src_pool.storage_mode ||
        dst_allocator.storage_mode() != manifest.dst_pool.storage_mode) {
      return base::error::InvalidArgument("pd block copy connector storage_mode mismatch");
    }
    if (src_allocator.storage_dtype() != dst_allocator.storage_dtype() ||
        src_allocator.scale_dtype() != dst_allocator.scale_dtype()) {
      return base::error::InvalidArgument("pd block copy connector dtype mismatch");
    }
    if (src_allocator.key_value_bytes_per_block() != dst_allocator.key_value_bytes_per_block() ||
        src_allocator.scale_bytes_per_block() != dst_allocator.scale_bytes_per_block()) {
      return base::error::InvalidArgument("pd block copy connector block byte size mismatch");
    }
    const bool use_runtime_copy = src_allocator.device_type() != base::DeviceType::kDeviceCPU ||
                                  dst_allocator.device_type() != base::DeviceType::kDeviceCPU;
    const base::MemcpyKind memcpy_kind = copy_kind_for_devices(src_allocator.device_type(),
                                                               dst_allocator.device_type());
    auto cuda_allocator = use_runtime_copy ? base::CUDADeviceAllocatorFactory::get_instance() : nullptr;
    for (size_t block_idx = 0; block_idx < mapping.src_block_ids.size(); ++block_idx) {
      const auto src = src_allocator.get_block_payload_ptrs(mapping.src_block_ids[block_idx]);
      const auto dst = dst_allocator.get_block_payload_ptrs(mapping.dst_block_ids[block_idx]);
      if (use_runtime_copy) {
        cuda_allocator->memcpy(src.key, dst.key, src.key_value_bytes, memcpy_kind,
                               transfer_queue_, need_sync_);
        cuda_allocator->memcpy(src.value, dst.value, src.key_value_bytes, memcpy_kind,
                               transfer_queue_, need_sync_);
        if (src.scale_bytes > 0) {
          cuda_allocator->memcpy(src.key_scale, dst.key_scale, src.scale_bytes, memcpy_kind,
                                 transfer_queue_, need_sync_);
          cuda_allocator->memcpy(src.value_scale, dst.value_scale, src.scale_bytes, memcpy_kind,
                                 transfer_queue_, need_sync_);
        }
      } else {
        std::memcpy(dst.key, src.key, src.key_value_bytes);
        std::memcpy(dst.value, src.value, src.key_value_bytes);
        if (src.scale_bytes > 0) {
          std::memcpy(dst.key_scale, src.key_scale, src.scale_bytes);
          std::memcpy(dst.value_scale, src.value_scale, src.scale_bytes);
        }
      }
    }
  }
  return base::error::Success();
}

CudaP2PKVTransferConnector::CudaP2PKVTransferConnector(
    base::KVCacheManager* src_kv_manager,
    base::KVCacheManager* dst_kv_manager,
    int32_t src_device_id,
    int32_t dst_device_id,
    void* transfer_queue,
    bool need_sync)
    : src_kv_manager_(src_kv_manager),
      dst_kv_manager_(dst_kv_manager),
      src_device_id_(src_device_id),
      dst_device_id_(dst_device_id),
      transfer_queue_(transfer_queue),
      need_sync_(need_sync) {
  CHECK_NE(src_kv_manager_, nullptr);
  CHECK_NE(dst_kv_manager_, nullptr);

  if (src_device_id_ == dst_device_id_) {
    peer_copy_enabled_ = false;
    return;
  }
  int can_access_peer = 0;
  cudaError_t status = cudaDeviceCanAccessPeer(&can_access_peer,
                                               dst_device_id_,
                                               src_device_id_);
  if (status != cudaSuccess) {
    cudaGetLastError();
    peer_copy_enabled_ = false;
    return;
  }
  peer_copy_enabled_ = can_access_peer != 0;
  if (peer_copy_enabled_) {
    int previous_device = 0;
    cudaGetDevice(&previous_device);
    if (cudaSetDevice(dst_device_id_) == cudaSuccess) {
      status = cudaDeviceEnablePeerAccess(src_device_id_, 0);
      if (status != cudaSuccess && status != cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
        peer_copy_enabled_ = false;
      } else if (status == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
      }
    } else {
      cudaGetLastError();
      peer_copy_enabled_ = false;
    }
    cudaSetDevice(previous_device);
  }
}

CudaP2PKVTransferConnector::~CudaP2PKVTransferConnector() {
  for (auto& transfer : transfers_) {
    if (transfer.completion_event != nullptr) {
      cudaEventDestroy(static_cast<cudaEvent_t>(transfer.completion_event));
      transfer.completion_event = nullptr;
    }
  }
}

base::Status CudaP2PKVTransferConnector::submit(const KVBlockManifest& manifest,
                                                HandoffId* handle) {
  if (handle == nullptr) {
    return base::error::InvalidArgument("pd p2p connector submit handle is null");
  }
  base::Status status = manifest.validate();
  if (!status) {
    return status;
  }

  *handle = {next_handoff_id_++};
  status = copy_blocks(manifest);
  if (!status) {
    transfers_.push_back({*handle, KVTransferStatus::Failed(status.get_err_msg()), nullptr});
    return status;
  }

  void* completion_event = nullptr;
  if (transfer_queue_ != nullptr && !need_sync_) {
    int previous_device = 0;
    cudaGetDevice(&previous_device);
    cudaSetDevice(dst_device_id_);
    cudaEvent_t event = nullptr;
    const cudaError_t event_status = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    if (event_status != cudaSuccess) {
      cudaSetDevice(previous_device);
      transfers_.push_back({*handle,
                            KVTransferStatus::Failed(cudaGetErrorString(event_status)),
                            nullptr});
      return base::error::InternalError(cudaGetErrorString(event_status));
    }
    const cudaError_t record_status =
        cudaEventRecord(event, static_cast<cudaStream_t>(transfer_queue_));
    if (record_status != cudaSuccess) {
      cudaEventDestroy(event);
      cudaSetDevice(previous_device);
      transfers_.push_back({*handle,
                            KVTransferStatus::Failed(cudaGetErrorString(record_status)),
                            nullptr});
      return base::error::InternalError(cudaGetErrorString(record_status));
    }
    cudaSetDevice(previous_device);
    completion_event = event;
  }
  transfers_.push_back({*handle,
                        completion_event == nullptr ? KVTransferStatus::Completed()
                                                    : KVTransferStatus::Pending(),
                        completion_event});
  return status;
}

KVTransferStatus CudaP2PKVTransferConnector::poll(HandoffId handle) {
  auto it = std::find_if(transfers_.begin(), transfers_.end(), [&](const auto& item) {
    return item.handle.value == handle.value;
  });
  if (it == transfers_.end()) {
    return KVTransferStatus::Failed("pd p2p connector unknown handoff id");
  }
  return poll_transfer(static_cast<size_t>(std::distance(transfers_.begin(), it)));
}

void CudaP2PKVTransferConnector::cancel(HandoffId handle, const std::string& reason) {
  auto it = std::find_if(transfers_.begin(), transfers_.end(), [&](const auto& item) {
    return item.handle.value == handle.value;
  });
  if (it != transfers_.end() && !it->status.done()) {
    if (it->completion_event != nullptr) {
      cudaEventDestroy(static_cast<cudaEvent_t>(it->completion_event));
      it->completion_event = nullptr;
    }
    it->status = KVTransferStatus::Cancelled(reason);
  }
}

KVTransferStatus CudaP2PKVTransferConnector::poll_transfer(size_t index) {
  auto& transfer = transfers_.at(index);
  if (transfer.status.done() || transfer.completion_event == nullptr) {
    return transfer.status;
  }
  const cudaError_t status = cudaEventQuery(static_cast<cudaEvent_t>(transfer.completion_event));
  if (status == cudaSuccess) {
    cudaEventDestroy(static_cast<cudaEvent_t>(transfer.completion_event));
    transfer.completion_event = nullptr;
    transfer.status = KVTransferStatus::Completed();
  } else if (status != cudaErrorNotReady) {
    cudaEventDestroy(static_cast<cudaEvent_t>(transfer.completion_event));
    transfer.completion_event = nullptr;
    transfer.status = KVTransferStatus::Failed(cudaGetErrorString(status));
  }
  return transfer.status;
}

base::Status CudaP2PKVTransferConnector::copy_blocks(const KVBlockManifest& manifest) {
  if (src_kv_manager_->num_layers() != manifest.src_pool.layer_num ||
      dst_kv_manager_->num_layers() != manifest.dst_pool.layer_num) {
    return base::error::InvalidArgument("pd p2p connector layer_num mismatch");
  }
  if (src_kv_manager_->block_size() != manifest.src_pool.block_size ||
      dst_kv_manager_->block_size() != manifest.dst_pool.block_size) {
    return base::error::InvalidArgument("pd p2p connector block_size mismatch");
  }

  int previous_device = 0;
  cudaGetDevice(&previous_device);
  cudaError_t set_status = cudaSetDevice(dst_device_id_);
  if (set_status != cudaSuccess) {
    cudaGetLastError();
    return base::error::InternalError(std::string("cudaSetDevice for P2P dst failed: ") +
                                      cudaGetErrorString(set_status));
  }

  for (const auto& mapping : manifest.layer_mappings) {
    base::BlockAllocator& src_allocator = src_kv_manager_->allocator_mut(mapping.layer_idx);
    base::BlockAllocator& dst_allocator = dst_kv_manager_->allocator_mut(mapping.layer_idx);
    if (src_allocator.device_type() != base::DeviceType::kDeviceCUDA ||
        dst_allocator.device_type() != base::DeviceType::kDeviceCUDA) {
      cudaSetDevice(previous_device);
      return base::error::InvalidArgument("pd p2p connector requires CUDA KV pools");
    }
    if (src_allocator.storage_mode() != manifest.src_pool.storage_mode ||
        dst_allocator.storage_mode() != manifest.dst_pool.storage_mode) {
      cudaSetDevice(previous_device);
      return base::error::InvalidArgument("pd p2p connector storage_mode mismatch");
    }
    if (src_allocator.storage_dtype() != dst_allocator.storage_dtype() ||
        src_allocator.scale_dtype() != dst_allocator.scale_dtype()) {
      cudaSetDevice(previous_device);
      return base::error::InvalidArgument("pd p2p connector dtype mismatch");
    }
    if (src_allocator.key_value_bytes_per_block() != dst_allocator.key_value_bytes_per_block() ||
        src_allocator.scale_bytes_per_block() != dst_allocator.scale_bytes_per_block()) {
      cudaSetDevice(previous_device);
      return base::error::InvalidArgument("pd p2p connector block byte size mismatch");
    }

    for (size_t block_idx = 0; block_idx < mapping.src_block_ids.size(); ++block_idx) {
      const auto src = src_allocator.get_block_payload_ptrs(mapping.src_block_ids[block_idx]);
      const auto dst = dst_allocator.get_block_payload_ptrs(mapping.dst_block_ids[block_idx]);
      auto copy_payload = [&](const void* src_ptr, void* dst_ptr, size_t bytes) -> base::Status {
        if (bytes == 0) {
          return base::error::Success();
        }
        cudaError_t copy_status = cudaSuccess;
        if (peer_copy_enabled_) {
          if (transfer_queue_ != nullptr) {
            copy_status = cudaMemcpyPeerAsync(dst_ptr, dst_device_id_,
                                              src_ptr, src_device_id_,
                                              bytes,
                                              static_cast<cudaStream_t>(transfer_queue_));
          } else {
            copy_status = cudaMemcpyPeer(dst_ptr, dst_device_id_,
                                         src_ptr, src_device_id_, bytes);
          }
        } else if (transfer_queue_ != nullptr) {
          copy_status = cudaMemcpyAsync(dst_ptr, src_ptr, bytes,
                                        cudaMemcpyDeviceToDevice,
                                        static_cast<cudaStream_t>(transfer_queue_));
        } else {
          copy_status = cudaMemcpy(dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToDevice);
        }
        if (copy_status != cudaSuccess) {
          return base::error::InternalError(cudaGetErrorString(copy_status));
        }
        return base::error::Success();
      };
      base::Status status = copy_payload(src.key, dst.key, src.key_value_bytes);
      if (!status) {
        cudaSetDevice(previous_device);
        return status;
      }
      status = copy_payload(src.value, dst.value, src.key_value_bytes);
      if (!status) {
        cudaSetDevice(previous_device);
        return status;
      }
      if (src.scale_bytes > 0) {
        status = copy_payload(src.key_scale, dst.key_scale, src.scale_bytes);
        if (!status) {
          cudaSetDevice(previous_device);
          return status;
        }
        status = copy_payload(src.value_scale, dst.value_scale, src.scale_bytes);
        if (!status) {
          cudaSetDevice(previous_device);
          return status;
        }
      }
    }
  }

  if (need_sync_) {
    const cudaError_t sync_status =
        transfer_queue_ != nullptr
            ? cudaStreamSynchronize(static_cast<cudaStream_t>(transfer_queue_))
            : cudaDeviceSynchronize();
    if (sync_status != cudaSuccess) {
      cudaSetDevice(previous_device);
      return base::error::InternalError(cudaGetErrorString(sync_status));
    }
  }
  cudaSetDevice(previous_device);
  return base::error::Success();
}

NcclKVBlockTransferConnector::NcclKVBlockTransferConnector(
    base::KVCacheManager* src_kv_manager,
    base::KVCacheManager* dst_kv_manager,
    int32_t src_device_id,
    int32_t dst_device_id,
    void* transfer_queue,
    bool need_sync)
    : src_kv_manager_(src_kv_manager),
      dst_kv_manager_(dst_kv_manager),
      src_device_id_(src_device_id),
      dst_device_id_(dst_device_id),
      transfer_queue_(transfer_queue),
      need_sync_(need_sync) {
  CHECK_NE(src_kv_manager_, nullptr);
  CHECK_NE(dst_kv_manager_, nullptr);
}

NcclKVBlockTransferConnector::~NcclKVBlockTransferConnector() {
  for (auto& transfer : transfers_) {
    if (transfer.completion_event != nullptr) {
      cudaEventDestroy(static_cast<cudaEvent_t>(transfer.completion_event));
      transfer.completion_event = nullptr;
    }
  }

  int previous_device = 0;
  cudaGetDevice(&previous_device);
  if (src_stream_ != nullptr && owns_src_stream_) {
    cudaSetDevice(src_device_id_);
    cudaStreamSynchronize(static_cast<cudaStream_t>(src_stream_));
  }
  if (dst_stream_ != nullptr && owns_dst_stream_) {
    cudaSetDevice(dst_device_id_);
    cudaStreamSynchronize(static_cast<cudaStream_t>(dst_stream_));
  }
  if (src_comm_ != nullptr) {
    cudaSetDevice(src_device_id_);
    ncclCommDestroy(static_cast<ncclComm_t>(src_comm_));
    src_comm_ = nullptr;
  }
  if (dst_comm_ != nullptr) {
    cudaSetDevice(dst_device_id_);
    ncclCommDestroy(static_cast<ncclComm_t>(dst_comm_));
    dst_comm_ = nullptr;
  }
  if (src_stream_ != nullptr && owns_src_stream_) {
    cudaSetDevice(src_device_id_);
    cudaStreamDestroy(static_cast<cudaStream_t>(src_stream_));
    src_stream_ = nullptr;
  }
  if (dst_stream_ != nullptr && owns_dst_stream_) {
    cudaSetDevice(dst_device_id_);
    cudaStreamDestroy(static_cast<cudaStream_t>(dst_stream_));
    dst_stream_ = nullptr;
  }
  cudaSetDevice(previous_device);
}

base::Status NcclKVBlockTransferConnector::ensure_comms() {
  if (comms_initialized_) {
    return base::error::Success();
  }
  if (src_device_id_ == dst_device_id_) {
    return base::error::InvalidArgument("pd nccl connector requires two distinct devices");
  }

  int previous_device = 0;
  cudaGetDevice(&previous_device);

  cudaStream_t src_stream = nullptr;
  cudaError_t cuda_status = cudaSetDevice(src_device_id_);
  if (cuda_status != cudaSuccess) {
    cudaSetDevice(previous_device);
    return base::error::InternalError(std::string("cudaSetDevice for NCCL src failed: ") +
                                      cudaGetErrorString(cuda_status));
  }
  cuda_status = cudaStreamCreateWithFlags(&src_stream, cudaStreamNonBlocking);
  if (cuda_status != cudaSuccess) {
    cudaSetDevice(previous_device);
    return base::error::InternalError(std::string("cudaStreamCreate for NCCL src failed: ") +
                                      cudaGetErrorString(cuda_status));
  }
  src_stream_ = src_stream;
  owns_src_stream_ = true;

  if (transfer_queue_ != nullptr) {
    dst_stream_ = transfer_queue_;
    owns_dst_stream_ = false;
  } else {
    cudaStream_t dst_stream = nullptr;
    cuda_status = cudaSetDevice(dst_device_id_);
    if (cuda_status != cudaSuccess) {
      cudaSetDevice(src_device_id_);
      cudaStreamDestroy(src_stream);
      src_stream_ = nullptr;
      owns_src_stream_ = false;
      cudaSetDevice(previous_device);
      return base::error::InternalError(std::string("cudaSetDevice for NCCL dst failed: ") +
                                        cudaGetErrorString(cuda_status));
    }
    cuda_status = cudaStreamCreateWithFlags(&dst_stream, cudaStreamNonBlocking);
    if (cuda_status != cudaSuccess) {
      cudaSetDevice(src_device_id_);
      cudaStreamDestroy(src_stream);
      src_stream_ = nullptr;
      owns_src_stream_ = false;
      cudaSetDevice(previous_device);
      return base::error::InternalError(std::string("cudaStreamCreate for NCCL dst failed: ") +
                                        cudaGetErrorString(cuda_status));
    }
    dst_stream_ = dst_stream;
    owns_dst_stream_ = true;
  }

  ncclComm_t comms[2] = {nullptr, nullptr};
  int devices[2] = {src_device_id_, dst_device_id_};
  const ncclResult_t nccl_status = ncclCommInitAll(comms, 2, devices);
  if (nccl_status != ncclSuccess) {
    if (owns_src_stream_ && src_stream_ != nullptr) {
      cudaSetDevice(src_device_id_);
      cudaStreamDestroy(static_cast<cudaStream_t>(src_stream_));
    }
    if (owns_dst_stream_ && dst_stream_ != nullptr) {
      cudaSetDevice(dst_device_id_);
      cudaStreamDestroy(static_cast<cudaStream_t>(dst_stream_));
    }
    src_stream_ = nullptr;
    dst_stream_ = nullptr;
    owns_src_stream_ = false;
    owns_dst_stream_ = false;
    cudaSetDevice(previous_device);
    return base::error::InternalError(nccl_error_string("ncclCommInitAll", nccl_status));
  }

  src_comm_ = comms[0];
  dst_comm_ = comms[1];
  comms_initialized_ = true;
  cudaSetDevice(previous_device);
  return base::error::Success();
}

base::Status NcclKVBlockTransferConnector::submit(const KVBlockManifest& manifest,
                                                  HandoffId* handle) {
  if (handle == nullptr) {
    return base::error::InvalidArgument("pd nccl connector submit handle is null");
  }
  base::Status status = manifest.validate();
  if (!status) {
    return status;
  }

  *handle = {next_handoff_id_++};
  status = ensure_comms();
  if (status) {
    status = copy_blocks(manifest);
  }
  if (!status) {
    transfers_.push_back({*handle, KVTransferStatus::Failed(status.get_err_msg()), nullptr});
    return status;
  }

  void* completion_event = nullptr;
  if (!need_sync_) {
    int previous_device = 0;
    cudaGetDevice(&previous_device);
    cudaSetDevice(dst_device_id_);
    cudaEvent_t event = nullptr;
    const cudaError_t event_status = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    if (event_status != cudaSuccess) {
      cudaSetDevice(previous_device);
      transfers_.push_back({*handle,
                            KVTransferStatus::Failed(cudaGetErrorString(event_status)),
                            nullptr});
      return base::error::InternalError(cudaGetErrorString(event_status));
    }
    const cudaError_t record_status =
        cudaEventRecord(event, static_cast<cudaStream_t>(dst_stream_));
    if (record_status != cudaSuccess) {
      cudaEventDestroy(event);
      cudaSetDevice(previous_device);
      transfers_.push_back({*handle,
                            KVTransferStatus::Failed(cudaGetErrorString(record_status)),
                            nullptr});
      return base::error::InternalError(cudaGetErrorString(record_status));
    }
    cudaSetDevice(previous_device);
    completion_event = event;
  }
  transfers_.push_back({*handle,
                        completion_event == nullptr ? KVTransferStatus::Completed()
                                                    : KVTransferStatus::Pending(),
                        completion_event});
  return status;
}

KVTransferStatus NcclKVBlockTransferConnector::poll(HandoffId handle) {
  auto it = std::find_if(transfers_.begin(), transfers_.end(), [&](const auto& item) {
    return item.handle.value == handle.value;
  });
  if (it == transfers_.end()) {
    return KVTransferStatus::Failed("pd nccl connector unknown handoff id");
  }
  return poll_transfer(static_cast<size_t>(std::distance(transfers_.begin(), it)));
}

void NcclKVBlockTransferConnector::cancel(HandoffId handle, const std::string& reason) {
  auto it = std::find_if(transfers_.begin(), transfers_.end(), [&](const auto& item) {
    return item.handle.value == handle.value;
  });
  if (it != transfers_.end() && !it->status.done()) {
    if (it->completion_event != nullptr) {
      cudaEventDestroy(static_cast<cudaEvent_t>(it->completion_event));
      it->completion_event = nullptr;
    }
    it->status = KVTransferStatus::Cancelled(reason);
  }
}

KVTransferStatus NcclKVBlockTransferConnector::poll_transfer(size_t index) {
  auto& transfer = transfers_.at(index);
  if (transfer.status.done() || transfer.completion_event == nullptr) {
    return transfer.status;
  }
  const cudaError_t status = cudaEventQuery(static_cast<cudaEvent_t>(transfer.completion_event));
  if (status == cudaSuccess) {
    cudaEventDestroy(static_cast<cudaEvent_t>(transfer.completion_event));
    transfer.completion_event = nullptr;
    transfer.status = KVTransferStatus::Completed();
  } else if (status != cudaErrorNotReady) {
    cudaEventDestroy(static_cast<cudaEvent_t>(transfer.completion_event));
    transfer.completion_event = nullptr;
    transfer.status = KVTransferStatus::Failed(cudaGetErrorString(status));
  }
  return transfer.status;
}

base::Status NcclKVBlockTransferConnector::copy_blocks(const KVBlockManifest& manifest) {
  base::Status status = validate_cuda_kv_manifest(
      manifest, src_kv_manager_, dst_kv_manager_, "pd nccl connector");
  if (!status) {
    return status;
  }

  int previous_device = 0;
  cudaGetDevice(&previous_device);

  auto transfer_payload = [&](const void* src_ptr, void* dst_ptr, size_t bytes) -> base::Status {
    if (bytes == 0) {
      return base::error::Success();
    }
    ncclResult_t nccl_status = ncclGroupStart();
    if (nccl_status != ncclSuccess) {
      return base::error::InternalError(nccl_error_string("ncclGroupStart", nccl_status));
    }
    cudaSetDevice(src_device_id_);
    const ncclResult_t send_status =
        ncclSend(src_ptr, bytes, ncclUint8, 1,
                 static_cast<ncclComm_t>(src_comm_),
                 static_cast<cudaStream_t>(src_stream_));
    cudaSetDevice(dst_device_id_);
    const ncclResult_t recv_status =
        ncclRecv(dst_ptr, bytes, ncclUint8, 0,
                 static_cast<ncclComm_t>(dst_comm_),
                 static_cast<cudaStream_t>(dst_stream_));
    nccl_status = ncclGroupEnd();
    if (send_status != ncclSuccess) {
      return base::error::InternalError(nccl_error_string("ncclSend", send_status));
    }
    if (recv_status != ncclSuccess) {
      return base::error::InternalError(nccl_error_string("ncclRecv", recv_status));
    }
    if (nccl_status != ncclSuccess) {
      return base::error::InternalError(nccl_error_string("ncclGroupEnd", nccl_status));
    }
    return base::error::Success();
  };

  for (const auto& mapping : manifest.layer_mappings) {
    base::BlockAllocator& src_allocator = src_kv_manager_->allocator_mut(mapping.layer_idx);
    base::BlockAllocator& dst_allocator = dst_kv_manager_->allocator_mut(mapping.layer_idx);
    for (size_t block_idx = 0; block_idx < mapping.src_block_ids.size(); ++block_idx) {
      const auto src = src_allocator.get_block_payload_ptrs(mapping.src_block_ids[block_idx]);
      const auto dst = dst_allocator.get_block_payload_ptrs(mapping.dst_block_ids[block_idx]);
      status = transfer_payload(src.key, dst.key, src.key_value_bytes);
      if (!status) {
        cudaSetDevice(previous_device);
        return status;
      }
      status = transfer_payload(src.value, dst.value, src.key_value_bytes);
      if (!status) {
        cudaSetDevice(previous_device);
        return status;
      }
      if (src.scale_bytes > 0) {
        status = transfer_payload(src.key_scale, dst.key_scale, src.scale_bytes);
        if (!status) {
          cudaSetDevice(previous_device);
          return status;
        }
        status = transfer_payload(src.value_scale, dst.value_scale, src.scale_bytes);
        if (!status) {
          cudaSetDevice(previous_device);
          return status;
        }
      }
    }
  }

  if (need_sync_) {
    cudaSetDevice(src_device_id_);
    cudaError_t sync_status = cudaStreamSynchronize(static_cast<cudaStream_t>(src_stream_));
    if (sync_status != cudaSuccess) {
      cudaSetDevice(previous_device);
      return base::error::InternalError(cudaGetErrorString(sync_status));
    }
    cudaSetDevice(dst_device_id_);
    sync_status = cudaStreamSynchronize(static_cast<cudaStream_t>(dst_stream_));
    if (sync_status != cudaSuccess) {
      cudaSetDevice(previous_device);
      return base::error::InternalError(cudaGetErrorString(sync_status));
    }
  }
  cudaSetDevice(previous_device);
  return base::error::Success();
}

base::Status run_remote_nccl_kv_block_transfer(
    const KVBlockManifest& manifest,
    const RemoteNcclKVTransferOptions& options) {
  base::Status status = validate_remote_nccl_options(manifest, options);
  if (!status) {
    return status;
  }

  int previous_device = 0;
  cudaGetDevice(&previous_device);
  cudaError_t cuda_status = cudaSetDevice(options.device_id);
  if (cuda_status != cudaSuccess) {
    return base::error::InternalError(
        std::string("cudaSetDevice for remote NCCL failed: ") +
        cudaGetErrorString(cuda_status));
  }

  cudaStream_t stream = static_cast<cudaStream_t>(options.stream);
  bool owns_stream = false;
  if (stream == nullptr) {
    cuda_status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (cuda_status != cudaSuccess) {
      cudaSetDevice(previous_device);
      return base::error::InternalError(
          std::string("cudaStreamCreate for remote NCCL failed: ") +
          cudaGetErrorString(cuda_status));
    }
    owns_stream = true;
  }

  ncclUniqueId unique_id;
  std::memcpy(&unique_id, options.nccl_unique_id.data(), sizeof(unique_id));
  ncclComm_t comm = nullptr;
  const int rank =
      options.role == RemoteNcclKVTransferRole::kProducer ? 0 : 1;
  ncclResult_t nccl_status = ncclCommInitRank(&comm, 2, unique_id, rank);
  if (nccl_status != ncclSuccess) {
    if (owns_stream) {
      cudaStreamDestroy(stream);
    }
    cudaSetDevice(previous_device);
    return base::error::InternalError(
        nccl_error_string("ncclCommInitRank", nccl_status));
  }

  auto close_comm = [&]() {
    if (comm != nullptr) {
      ncclCommDestroy(comm);
      comm = nullptr;
    }
    if (owns_stream && stream != nullptr) {
      cudaStreamDestroy(stream);
      stream = nullptr;
    }
    cudaSetDevice(previous_device);
  };

  auto transfer_payload = [&](const void* ptr, size_t bytes) -> base::Status {
    if (bytes == 0) {
      return base::error::Success();
    }
    const int peer_rank =
        options.role == RemoteNcclKVTransferRole::kProducer ? 1 : 0;
    ncclResult_t result = ncclSuccess;
    if (options.role == RemoteNcclKVTransferRole::kProducer) {
      result = ncclSend(ptr, bytes, ncclUint8, peer_rank, comm, stream);
      if (result != ncclSuccess) {
        return base::error::InternalError(
            nccl_error_string("ncclSend", result));
      }
    } else {
      result = ncclRecv(const_cast<void*>(ptr), bytes, ncclUint8, peer_rank,
                        comm, stream);
      if (result != ncclSuccess) {
        return base::error::InternalError(
            nccl_error_string("ncclRecv", result));
      }
    }
    return base::error::Success();
  };

  for (const auto& mapping : manifest.layer_mappings) {
    base::BlockAllocator& allocator =
        options.kv_manager->allocator_mut(mapping.layer_idx);
    const auto& local_block_ids =
        options.role == RemoteNcclKVTransferRole::kProducer
            ? mapping.src_block_ids
            : mapping.dst_block_ids;
    for (int32_t block_id : local_block_ids) {
      const auto ptrs = allocator.get_block_payload_ptrs(block_id);
      status = transfer_payload(ptrs.key, ptrs.key_value_bytes);
      if (!status) {
        close_comm();
        return status;
      }
      status = transfer_payload(ptrs.value, ptrs.key_value_bytes);
      if (!status) {
        close_comm();
        return status;
      }
      if (ptrs.scale_bytes > 0) {
        status = transfer_payload(ptrs.key_scale, ptrs.scale_bytes);
        if (!status) {
          close_comm();
          return status;
        }
        status = transfer_payload(ptrs.value_scale, ptrs.scale_bytes);
        if (!status) {
          close_comm();
          return status;
        }
      }
    }
  }

  if (options.need_sync) {
    cuda_status = cudaStreamSynchronize(stream);
    if (cuda_status != cudaSuccess) {
      close_comm();
      return base::error::InternalError(
          std::string("cudaStreamSynchronize for remote NCCL failed: ") +
          cudaGetErrorString(cuda_status));
    }
  }
  close_comm();
  return base::error::Success();
}

RemoteNcclLayerKVTransferConnector::RemoteNcclLayerKVTransferConnector(
    RemoteNcclKVTransferOptions options)
    : options_(std::move(options)) {
  CHECK_NE(options_.kv_manager, nullptr);
}

RemoteNcclLayerKVTransferConnector::~RemoteNcclLayerKVTransferConnector() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& record : layer_records_) {
      if (record.completion_event != nullptr) {
        cudaEventDestroy(static_cast<cudaEvent_t>(record.completion_event));
        record.completion_event = nullptr;
      }
    }
  }

  int previous_device = 0;
  cudaGetDevice(&previous_device);
  cudaSetDevice(options_.device_id);
  if (stream_ != nullptr) {
    cudaStreamSynchronize(static_cast<cudaStream_t>(stream_));
  }
  if (comm_ != nullptr) {
    ncclCommDestroy(static_cast<ncclComm_t>(comm_));
    comm_ = nullptr;
  }
  if (owns_stream_ && stream_ != nullptr) {
    cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
    stream_ = nullptr;
  }
  cudaSetDevice(previous_device);
}

base::Status RemoteNcclLayerKVTransferConnector::validate_request(
    const LayerKVTransferRequest& request) const {
  if (request.client_request_id.empty()) {
    return base::error::InvalidArgument(
        "remote layer nccl connector missing client_request_id");
  }
  if (!request.handoff_id.valid()) {
    return base::error::InvalidArgument(
        "remote layer nccl connector missing handoff_id");
  }
  if (request.prompt_tokens <= 0) {
    return base::error::InvalidArgument(
        "remote layer nccl connector prompt_tokens must be positive");
  }
  if (request.computed_tokens <= 0 ||
      request.computed_tokens > request.prompt_tokens) {
    return base::error::InvalidArgument(
        "remote layer nccl connector computed_tokens is invalid");
  }
  if (!request.src_pool.compatible_with(request.dst_pool)) {
    return base::error::InvalidArgument(
        "remote layer nccl connector source/destination pools are incompatible");
  }
  const KVPoolDescriptor& local_pool =
      options_.role == RemoteNcclKVTransferRole::kProducer ? request.src_pool
                                                           : request.dst_pool;
  const base::RequestId local_request_id =
      options_.role == RemoteNcclKVTransferRole::kProducer
          ? request.src_request_id
          : request.dst_request_id;
  if (local_request_id < 0 ||
      !options_.kv_manager->is_valid_request(local_request_id)) {
    return base::error::InvalidArgument(
        "remote layer nccl connector local request is invalid");
  }
  if (local_pool.device_id != options_.device_id) {
    return base::error::InvalidArgument(
        "remote layer nccl connector device_id mismatch");
  }
  if (options_.kv_manager->num_layers() != local_pool.layer_num) {
    return base::error::InvalidArgument(
        "remote layer nccl connector layer_num mismatch");
  }
  if (options_.kv_manager->block_size() != local_pool.block_size) {
    return base::error::InvalidArgument(
        "remote layer nccl connector block_size mismatch");
  }
  if (options_.nccl_unique_id.size() != sizeof(ncclUniqueId)) {
    return base::error::InvalidArgument(
        "remote layer nccl connector invalid unique id size");
  }
  if (options_.role == RemoteNcclKVTransferRole::kConsumer &&
      options_.kv_manager->get_context_len(local_request_id) <
          request.computed_tokens) {
    return base::error::InvalidArgument(
        "remote layer nccl connector destination context is insufficient");
  }
  for (int32_t layer_idx = 0; layer_idx < local_pool.layer_num; ++layer_idx) {
    base::BlockAllocator& allocator =
        options_.kv_manager->allocator_mut(layer_idx);
    if (allocator.device_type() != base::DeviceType::kDeviceCUDA) {
      return base::error::InvalidArgument(
          "remote layer nccl connector requires CUDA KV pool");
    }
    if (allocator.storage_mode() != local_pool.storage_mode) {
      return base::error::InvalidArgument(
          "remote layer nccl connector storage_mode mismatch");
    }
  }
  return base::error::Success();
}

int32_t RemoteNcclLayerKVTransferConnector::required_blocks(int32_t tokens) const {
  if (tokens <= 0) {
    return 0;
  }
  return (tokens + request_.dst_pool.block_size - 1) /
         request_.dst_pool.block_size;
}

base::Status RemoteNcclLayerKVTransferConnector::ensure_comm() {
  if (comm_initialized_) {
    return base::error::Success();
  }
  cudaError_t cuda_status = cudaSetDevice(options_.device_id);
  if (cuda_status != cudaSuccess) {
    return base::error::InternalError(
        std::string("cudaSetDevice for remote layer NCCL failed: ") +
        cudaGetErrorString(cuda_status));
  }
  stream_ = options_.stream;
  if (stream_ == nullptr) {
    cudaStream_t stream = nullptr;
    cuda_status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (cuda_status != cudaSuccess) {
      return base::error::InternalError(
          std::string("cudaStreamCreate for remote layer NCCL failed: ") +
          cudaGetErrorString(cuda_status));
    }
    stream_ = stream;
    owns_stream_ = true;
  }

  ncclUniqueId unique_id;
  std::memcpy(&unique_id, options_.nccl_unique_id.data(), sizeof(unique_id));
  ncclComm_t comm = nullptr;
  const int rank =
      options_.role == RemoteNcclKVTransferRole::kProducer ? 0 : 1;
  const ncclResult_t nccl_status = ncclCommInitRank(&comm, 2, unique_id, rank);
  if (nccl_status != ncclSuccess) {
    if (owns_stream_ && stream_ != nullptr) {
      cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
      stream_ = nullptr;
      owns_stream_ = false;
    }
    return base::error::InternalError(
        nccl_error_string("ncclCommInitRank", nccl_status));
  }
  comm_ = comm;
  comm_initialized_ = true;
  return base::error::Success();
}

base::Status RemoteNcclLayerKVTransferConnector::prepare(
    const LayerKVTransferRequest& request) {
  base::Status status = validate_request(request);
  if (!status) {
    return status;
  }

  std::lock_guard<std::mutex> lock(mu_);
  request_ = request;
  prepared_ = true;
  cancelled_ = false;
  cancel_reason_.clear();
  layer_records_.assign(
      options_.role == RemoteNcclKVTransferRole::kProducer
          ? request.src_pool.layer_num
          : request.dst_pool.layer_num,
      LayerRecord{});
  cv_.notify_all();
  return base::error::Success();
}

base::Status RemoteNcclLayerKVTransferConnector::copy_layer(
    int32_t layer_idx,
    void** completion_event) {
  CHECK_NE(completion_event, nullptr);
  *completion_event = nullptr;

  base::Status status = ensure_comm();
  if (!status) {
    return status;
  }

  const base::RequestId local_request_id =
      options_.role == RemoteNcclKVTransferRole::kProducer
          ? request_.src_request_id
          : request_.dst_request_id;
  const int32_t current_tokens = std::min(
      options_.kv_manager->get_context_len(local_request_id),
      request_.computed_tokens);
  const int32_t blocks = required_blocks(current_tokens);
  if (blocks <= 0) {
    return base::error::InvalidArgument(
        "remote layer nccl connector has no blocks to transfer");
  }
  const auto& local_blocks =
      options_.kv_manager->get_block_ids(local_request_id, layer_idx);
  if (static_cast<int32_t>(local_blocks.size()) < blocks) {
    return base::error::InvalidArgument(
        "remote layer nccl connector local block count is insufficient");
  }

  int previous_device = 0;
  cudaGetDevice(&previous_device);
  cudaSetDevice(options_.device_id);

  auto transfer_payload = [&](void* ptr, size_t bytes) -> base::Status {
    if (bytes == 0) {
      return base::error::Success();
    }
    const int peer_rank =
        options_.role == RemoteNcclKVTransferRole::kProducer ? 1 : 0;
    ncclResult_t nccl_status = ncclSuccess;
    if (options_.role == RemoteNcclKVTransferRole::kProducer) {
      nccl_status = ncclSend(ptr, bytes, ncclUint8, peer_rank,
                             static_cast<ncclComm_t>(comm_),
                             static_cast<cudaStream_t>(stream_));
      if (nccl_status != ncclSuccess) {
        return base::error::InternalError(
            nccl_error_string("ncclSend", nccl_status));
      }
    } else {
      nccl_status = ncclRecv(ptr, bytes, ncclUint8, peer_rank,
                             static_cast<ncclComm_t>(comm_),
                             static_cast<cudaStream_t>(stream_));
      if (nccl_status != ncclSuccess) {
        return base::error::InternalError(
            nccl_error_string("ncclRecv", nccl_status));
      }
    }
    return base::error::Success();
  };

  base::BlockAllocator& allocator = options_.kv_manager->allocator_mut(layer_idx);
  for (int32_t block_idx = 0; block_idx < blocks; ++block_idx) {
    const auto ptrs = allocator.get_block_payload_ptrs(local_blocks[block_idx]);
    base::Status status = transfer_payload(ptrs.key, ptrs.key_value_bytes);
    if (!status) {
      cudaSetDevice(previous_device);
      return status;
    }
    status = transfer_payload(ptrs.value, ptrs.key_value_bytes);
    if (!status) {
      cudaSetDevice(previous_device);
      return status;
    }
    if (ptrs.scale_bytes > 0) {
      status = transfer_payload(ptrs.key_scale, ptrs.scale_bytes);
      if (!status) {
        cudaSetDevice(previous_device);
        return status;
      }
      status = transfer_payload(ptrs.value_scale, ptrs.scale_bytes);
      if (!status) {
        cudaSetDevice(previous_device);
        return status;
      }
    }
  }

  if (options_.need_sync) {
    const cudaError_t sync_status =
        cudaStreamSynchronize(static_cast<cudaStream_t>(stream_));
    if (sync_status != cudaSuccess) {
      cudaSetDevice(previous_device);
      return base::error::InternalError(
          std::string("cudaStreamSynchronize for remote layer NCCL failed: ") +
          cudaGetErrorString(sync_status));
    }
    cudaSetDevice(previous_device);
    return base::error::Success();
  }

  cudaEvent_t event = nullptr;
  cudaError_t event_status = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
  if (event_status != cudaSuccess) {
    cudaSetDevice(previous_device);
    return base::error::InternalError(cudaGetErrorString(event_status));
  }
  event_status = cudaEventRecord(event, static_cast<cudaStream_t>(stream_));
  if (event_status != cudaSuccess) {
    cudaEventDestroy(event);
    cudaSetDevice(previous_device);
    return base::error::InternalError(cudaGetErrorString(event_status));
  }
  *completion_event = event;
  cudaSetDevice(previous_device);
  return base::error::Success();
}

base::Status RemoteNcclLayerKVTransferConnector::save_kv_layer(int32_t layer_idx) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (!prepared_) {
      return base::error::InvalidArgument(
          "remote layer nccl connector is not prepared");
    }
    if (cancelled_) {
      return base::error::InternalError(cancel_reason_);
    }
    if (layer_idx < 0 || layer_idx >= static_cast<int32_t>(layer_records_.size())) {
      return base::error::InvalidArgument(
          "remote layer nccl connector layer_idx out of range");
    }
    if (layer_records_[layer_idx].completion_event != nullptr) {
      cudaEventDestroy(static_cast<cudaEvent_t>(
          layer_records_[layer_idx].completion_event));
      layer_records_[layer_idx].completion_event = nullptr;
    }
    layer_records_[layer_idx].status = KVTransferStatus::Pending();
  }

  void* completion_event = nullptr;
  base::Status status = copy_layer(layer_idx, &completion_event);
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (!status) {
      if (completion_event != nullptr) {
        cudaEventDestroy(static_cast<cudaEvent_t>(completion_event));
      }
      layer_records_[layer_idx].status =
          KVTransferStatus::Failed(status.get_err_msg());
      layer_records_[layer_idx].completion_event = nullptr;
      cv_.notify_all();
      return status;
    }
    layer_records_[layer_idx].completion_event = completion_event;
    layer_records_[layer_idx].transferred_tokens = request_.computed_tokens;
    layer_records_[layer_idx].status =
        completion_event == nullptr ? KVTransferStatus::Completed()
                                    : KVTransferStatus::Pending();
    cv_.notify_all();
  }
  return base::error::Success();
}

base::Status RemoteNcclLayerKVTransferConnector::wait_for_layer_load(
    int32_t layer_idx,
    void* consumer_queue) {
  std::unique_lock<std::mutex> lock(mu_);
  if (!prepared_) {
    return base::error::InvalidArgument(
        "remote layer nccl connector is not prepared");
  }
  if (layer_idx < 0 || layer_idx >= static_cast<int32_t>(layer_records_.size())) {
    return base::error::InvalidArgument(
        "remote layer nccl connector layer_idx out of range");
  }
  while (true) {
    if (cancelled_) {
      return base::error::InternalError(cancel_reason_);
    }
    KVTransferStatus status = poll_layer_unlocked(layer_idx);
    if (status.ok()) {
      void* event = layer_records_[layer_idx].completion_event;
      lock.unlock();
      if (consumer_queue != nullptr && event != nullptr) {
        int previous_device = 0;
        cudaGetDevice(&previous_device);
        cudaSetDevice(options_.device_id);
        const cudaError_t wait_status =
            cudaStreamWaitEvent(static_cast<cudaStream_t>(consumer_queue),
                                static_cast<cudaEvent_t>(event), 0);
        cudaSetDevice(previous_device);
        if (wait_status != cudaSuccess) {
          return base::error::InternalError(cudaGetErrorString(wait_status));
        }
      }
      return base::error::Success();
    }
    if (status.state == KVTransferState::kFailed ||
        status.state == KVTransferState::kCancelled) {
      return base::error::InternalError(status.error);
    }
    cv_.wait_for(lock, std::chrono::milliseconds(1));
  }
}

KVTransferStatus RemoteNcclLayerKVTransferConnector::poll_layer_unlocked(
    int32_t layer_idx) {
  if (layer_idx < 0 || layer_idx >= static_cast<int32_t>(layer_records_.size())) {
    return KVTransferStatus::Failed(
        "remote layer nccl connector layer_idx out of range");
  }
  auto& record = layer_records_[layer_idx];
  if (record.status.done() || record.completion_event == nullptr) {
    return record.status;
  }
  const cudaError_t status =
      cudaEventQuery(static_cast<cudaEvent_t>(record.completion_event));
  if (status == cudaSuccess) {
    cudaEventDestroy(static_cast<cudaEvent_t>(record.completion_event));
    record.completion_event = nullptr;
    record.status = KVTransferStatus::Completed();
  } else if (status != cudaErrorNotReady) {
    cudaEventDestroy(static_cast<cudaEvent_t>(record.completion_event));
    record.completion_event = nullptr;
    record.status = KVTransferStatus::Failed(cudaGetErrorString(status));
  }
  return record.status;
}

KVTransferStatus RemoteNcclLayerKVTransferConnector::poll_layer(int32_t layer_idx) {
  std::lock_guard<std::mutex> lock(mu_);
  return poll_layer_unlocked(layer_idx);
}

KVTransferStatus RemoteNcclLayerKVTransferConnector::poll() {
  std::lock_guard<std::mutex> lock(mu_);
  if (!prepared_) {
    return KVTransferStatus::Pending();
  }
  bool all_done = true;
  for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_records_.size());
       ++layer_idx) {
    const KVTransferStatus status = poll_layer_unlocked(layer_idx);
    if (status.state == KVTransferState::kFailed ||
        status.state == KVTransferState::kCancelled) {
      return status;
    }
    all_done = all_done && status.ok();
  }
  return all_done ? KVTransferStatus::Completed() : KVTransferStatus::Pending();
}

void RemoteNcclLayerKVTransferConnector::cancel(const std::string& reason) {
  std::lock_guard<std::mutex> lock(mu_);
  cancelled_ = true;
  cancel_reason_ =
      reason.empty() ? "remote layer nccl connector cancelled" : reason;
  for (auto& record : layer_records_) {
    if (record.status.done()) {
      continue;
    }
    if (record.completion_event != nullptr) {
      cudaEventDestroy(static_cast<cudaEvent_t>(record.completion_event));
      record.completion_event = nullptr;
    }
    record.status = KVTransferStatus::Cancelled(cancel_reason_);
  }
  cv_.notify_all();
}

NcclLayerKVTransferConnector::NcclLayerKVTransferConnector(
    base::KVCacheManager* src_kv_manager,
    base::KVCacheManager* dst_kv_manager,
    int32_t src_device_id,
    int32_t dst_device_id,
    void* src_ready_queue,
    void* dst_transfer_queue,
    bool need_sync)
    : src_kv_manager_(src_kv_manager),
      dst_kv_manager_(dst_kv_manager),
      src_device_id_(src_device_id),
      dst_device_id_(dst_device_id),
      src_ready_queue_(src_ready_queue),
      dst_transfer_queue_(dst_transfer_queue),
      need_sync_(need_sync) {
  CHECK_NE(src_kv_manager_, nullptr);
  CHECK_NE(dst_kv_manager_, nullptr);
}

NcclLayerKVTransferConnector::~NcclLayerKVTransferConnector() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& record : layer_records_) {
      if (record.completion_event != nullptr) {
        cudaEventDestroy(static_cast<cudaEvent_t>(record.completion_event));
        record.completion_event = nullptr;
      }
    }
  }

  int previous_device = 0;
  cudaGetDevice(&previous_device);
  if (src_stream_ != nullptr && owns_src_stream_) {
    cudaSetDevice(src_device_id_);
    cudaStreamSynchronize(static_cast<cudaStream_t>(src_stream_));
  }
  if (dst_stream_ != nullptr && owns_dst_stream_) {
    cudaSetDevice(dst_device_id_);
    cudaStreamSynchronize(static_cast<cudaStream_t>(dst_stream_));
  }
  if (src_comm_ != nullptr) {
    cudaSetDevice(src_device_id_);
    ncclCommDestroy(static_cast<ncclComm_t>(src_comm_));
    src_comm_ = nullptr;
  }
  if (dst_comm_ != nullptr) {
    cudaSetDevice(dst_device_id_);
    ncclCommDestroy(static_cast<ncclComm_t>(dst_comm_));
    dst_comm_ = nullptr;
  }
  if (src_stream_ != nullptr && owns_src_stream_) {
    cudaSetDevice(src_device_id_);
    cudaStreamDestroy(static_cast<cudaStream_t>(src_stream_));
    src_stream_ = nullptr;
  }
  if (dst_stream_ != nullptr && owns_dst_stream_) {
    cudaSetDevice(dst_device_id_);
    cudaStreamDestroy(static_cast<cudaStream_t>(dst_stream_));
    dst_stream_ = nullptr;
  }
  cudaSetDevice(previous_device);
}

base::Status NcclLayerKVTransferConnector::validate_request(
    const LayerKVTransferRequest& request) const {
  if (request.client_request_id.empty()) {
    return base::error::InvalidArgument("pd layer connector missing client_request_id");
  }
  if (!request.handoff_id.valid()) {
    return base::error::InvalidArgument("pd layer connector missing handoff_id");
  }
  if (request.src_request_id < 0 || !src_kv_manager_->is_valid_request(request.src_request_id)) {
    return base::error::InvalidArgument("pd layer connector source request is invalid");
  }
  if (request.dst_request_id < 0 || !dst_kv_manager_->is_valid_request(request.dst_request_id)) {
    return base::error::InvalidArgument("pd layer connector destination request is invalid");
  }
  if (request.prompt_tokens <= 0) {
    return base::error::InvalidArgument("pd layer connector prompt_tokens must be positive");
  }
  if (request.computed_tokens <= 0 || request.computed_tokens > request.prompt_tokens) {
    return base::error::InvalidArgument("pd layer connector computed_tokens is invalid");
  }
  if (!request.src_pool.compatible_with(request.dst_pool)) {
    return base::error::InvalidArgument("pd layer connector source/destination pools are incompatible");
  }
  if (request.src_pool.layer_num != src_kv_manager_->num_layers() ||
      request.dst_pool.layer_num != dst_kv_manager_->num_layers()) {
    return base::error::InvalidArgument("pd layer connector layer_num mismatch");
  }
  if (request.src_pool.block_size != src_kv_manager_->block_size() ||
      request.dst_pool.block_size != dst_kv_manager_->block_size()) {
    return base::error::InvalidArgument("pd layer connector block_size mismatch");
  }
  if (dst_kv_manager_->get_context_len(request.dst_request_id) < request.computed_tokens) {
    return base::error::InvalidArgument("pd layer connector destination KV context length is insufficient");
  }
  return base::error::Success();
}

int32_t NcclLayerKVTransferConnector::required_blocks(int32_t tokens) const {
  if (tokens <= 0) {
    return 0;
  }
  return (tokens + request_.dst_pool.block_size - 1) / request_.dst_pool.block_size;
}

base::Status NcclLayerKVTransferConnector::prepare(
    const LayerKVTransferRequest& request) {
  base::Status status = validate_request(request);
  if (!status) {
    return status;
  }
  status = ensure_comms();
  if (!status) {
    return status;
  }

  std::lock_guard<std::mutex> lock(mu_);
  request_ = request;
  layer_records_.assign(request.dst_pool.layer_num, LayerRecord{});
  prepared_ = true;
  cancelled_ = false;
  cancel_reason_.clear();
  cv_.notify_all();
  return base::error::Success();
}

base::Status NcclLayerKVTransferConnector::ensure_comms() {
  if (comms_initialized_) {
    return base::error::Success();
  }
  if (src_device_id_ == dst_device_id_) {
    return base::error::InvalidArgument("pd layer nccl connector requires two distinct devices");
  }

  int previous_device = 0;
  cudaGetDevice(&previous_device);

  cudaStream_t src_stream = nullptr;
  cudaError_t cuda_status = cudaSetDevice(src_device_id_);
  if (cuda_status != cudaSuccess) {
    cudaSetDevice(previous_device);
    return base::error::InternalError(std::string("cudaSetDevice for layer NCCL src failed: ") +
                                      cudaGetErrorString(cuda_status));
  }
  cuda_status = cudaStreamCreateWithFlags(&src_stream, cudaStreamNonBlocking);
  if (cuda_status != cudaSuccess) {
    cudaSetDevice(previous_device);
    return base::error::InternalError(std::string("cudaStreamCreate for layer NCCL src failed: ") +
                                      cudaGetErrorString(cuda_status));
  }
  src_stream_ = src_stream;
  owns_src_stream_ = true;

  if (dst_transfer_queue_ != nullptr) {
    dst_stream_ = dst_transfer_queue_;
    owns_dst_stream_ = false;
  } else {
    cudaStream_t dst_stream = nullptr;
    cuda_status = cudaSetDevice(dst_device_id_);
    if (cuda_status != cudaSuccess) {
      cudaSetDevice(src_device_id_);
      cudaStreamDestroy(src_stream);
      src_stream_ = nullptr;
      owns_src_stream_ = false;
      cudaSetDevice(previous_device);
      return base::error::InternalError(std::string("cudaSetDevice for layer NCCL dst failed: ") +
                                        cudaGetErrorString(cuda_status));
    }
    cuda_status = cudaStreamCreateWithFlags(&dst_stream, cudaStreamNonBlocking);
    if (cuda_status != cudaSuccess) {
      cudaSetDevice(src_device_id_);
      cudaStreamDestroy(src_stream);
      src_stream_ = nullptr;
      owns_src_stream_ = false;
      cudaSetDevice(previous_device);
      return base::error::InternalError(std::string("cudaStreamCreate for layer NCCL dst failed: ") +
                                        cudaGetErrorString(cuda_status));
    }
    dst_stream_ = dst_stream;
    owns_dst_stream_ = true;
  }

  ncclComm_t comms[2] = {nullptr, nullptr};
  int devices[2] = {src_device_id_, dst_device_id_};
  const ncclResult_t nccl_status = ncclCommInitAll(comms, 2, devices);
  if (nccl_status != ncclSuccess) {
    if (owns_src_stream_ && src_stream_ != nullptr) {
      cudaSetDevice(src_device_id_);
      cudaStreamDestroy(static_cast<cudaStream_t>(src_stream_));
    }
    if (owns_dst_stream_ && dst_stream_ != nullptr) {
      cudaSetDevice(dst_device_id_);
      cudaStreamDestroy(static_cast<cudaStream_t>(dst_stream_));
    }
    src_stream_ = nullptr;
    dst_stream_ = nullptr;
    owns_src_stream_ = false;
    owns_dst_stream_ = false;
    cudaSetDevice(previous_device);
    return base::error::InternalError(nccl_error_string("ncclCommInitAll", nccl_status));
  }

  src_comm_ = comms[0];
  dst_comm_ = comms[1];
  comms_initialized_ = true;
  cudaSetDevice(previous_device);
  return base::error::Success();
}

base::Status NcclLayerKVTransferConnector::enqueue_source_ready_wait() {
  if (src_ready_queue_ == nullptr || src_stream_ == nullptr) {
    return base::error::Success();
  }
  int previous_device = 0;
  cudaGetDevice(&previous_device);
  cudaSetDevice(src_device_id_);
  cudaEvent_t event = nullptr;
  cudaError_t cuda_status = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
  if (cuda_status != cudaSuccess) {
    cudaSetDevice(previous_device);
    return base::error::InternalError(cudaGetErrorString(cuda_status));
  }
  cuda_status = cudaEventRecord(event, static_cast<cudaStream_t>(src_ready_queue_));
  if (cuda_status != cudaSuccess) {
    cudaEventDestroy(event);
    cudaSetDevice(previous_device);
    return base::error::InternalError(cudaGetErrorString(cuda_status));
  }
  cuda_status = cudaStreamWaitEvent(static_cast<cudaStream_t>(src_stream_), event, 0);
  cudaEventDestroy(event);
  if (cuda_status != cudaSuccess) {
    cudaSetDevice(previous_device);
    return base::error::InternalError(cudaGetErrorString(cuda_status));
  }
  cudaSetDevice(previous_device);
  return base::error::Success();
}

base::Status NcclLayerKVTransferConnector::save_kv_layer(int32_t layer_idx) {
  LayerKVTransferRequest request;
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (!prepared_) {
      return base::error::InvalidArgument("pd layer connector is not prepared");
    }
    if (cancelled_) {
      return base::error::InternalError(cancel_reason_);
    }
    if (layer_idx < 0 || layer_idx >= static_cast<int32_t>(layer_records_.size())) {
      return base::error::InvalidArgument("pd layer connector layer_idx out of range");
    }
    request = request_;
  }

  base::Status status = enqueue_source_ready_wait();
  if (!status) {
    std::lock_guard<std::mutex> lock(mu_);
    layer_records_[layer_idx].status = KVTransferStatus::Failed(status.get_err_msg());
    cv_.notify_all();
    return status;
  }

  const int32_t current_tokens = std::min(
      src_kv_manager_->get_context_len(request.src_request_id),
      request.computed_tokens);
  int32_t previous_transferred_tokens = 0;
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (current_tokens <= layer_records_[layer_idx].transferred_tokens) {
      return base::error::Success();
    }
    previous_transferred_tokens = layer_records_[layer_idx].transferred_tokens;
    if (layer_records_[layer_idx].completion_event != nullptr) {
      cudaEventDestroy(static_cast<cudaEvent_t>(
          layer_records_[layer_idx].completion_event));
      layer_records_[layer_idx].completion_event = nullptr;
    }
    layer_records_[layer_idx].status = KVTransferStatus::Pending();
  }
  const int32_t blocks = required_blocks(current_tokens);
  const int32_t start_block =
      previous_transferred_tokens / request.src_pool.block_size;
  if (start_block >= blocks) {
    std::lock_guard<std::mutex> lock(mu_);
    layer_records_[layer_idx].transferred_tokens = current_tokens;
    layer_records_[layer_idx].status =
        current_tokens >= request.computed_tokens ? KVTransferStatus::Completed()
                                                  : KVTransferStatus::Pending();
    cv_.notify_all();
    return base::error::Success();
  }
  KVBlockMapping mapping;
  mapping.layer_idx = layer_idx;
  const auto& src_blocks = src_kv_manager_->get_block_ids(request.src_request_id, layer_idx);
  const auto& dst_blocks = dst_kv_manager_->get_block_ids(request.dst_request_id, layer_idx);
  if (static_cast<int32_t>(src_blocks.size()) < blocks ||
      static_cast<int32_t>(dst_blocks.size()) < blocks) {
    status = base::error::InvalidArgument("pd layer connector block count is insufficient");
    std::lock_guard<std::mutex> lock(mu_);
    layer_records_[layer_idx].status = KVTransferStatus::Failed(status.get_err_msg());
    cv_.notify_all();
    return status;
  }
  mapping.src_block_ids.assign(src_blocks.begin() + start_block,
                               src_blocks.begin() + blocks);
  mapping.dst_block_ids.assign(dst_blocks.begin() + start_block,
                               dst_blocks.begin() + blocks);

  void* completion_event = nullptr;
  status = copy_layer(mapping, &completion_event);
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (!status) {
      if (completion_event != nullptr) {
        cudaEventDestroy(static_cast<cudaEvent_t>(completion_event));
      }
      layer_records_[layer_idx].status = KVTransferStatus::Failed(status.get_err_msg());
      layer_records_[layer_idx].completion_event = nullptr;
      cv_.notify_all();
      return status;
    }
    layer_records_[layer_idx].completion_event = completion_event;
    layer_records_[layer_idx].transferred_tokens = current_tokens;
    layer_records_[layer_idx].status =
        completion_event == nullptr && current_tokens >= request.computed_tokens
            ? KVTransferStatus::Completed()
            : KVTransferStatus::Pending();
    cv_.notify_all();
  }
  return base::error::Success();
}

base::Status NcclLayerKVTransferConnector::copy_layer(const KVBlockMapping& mapping,
                                                      void** completion_event) {
  CHECK_NE(completion_event, nullptr);
  *completion_event = nullptr;

  base::Status status = validate_cuda_kv_layer_mapping(
      mapping, request_.src_pool, request_.dst_pool,
      src_kv_manager_, dst_kv_manager_, "pd layer nccl connector");
  if (!status) {
    return status;
  }

  int previous_device = 0;
  cudaGetDevice(&previous_device);

  auto transfer_payload = [&](const void* src_ptr, void* dst_ptr, size_t bytes) -> base::Status {
    if (bytes == 0) {
      return base::error::Success();
    }
    ncclResult_t nccl_status = ncclGroupStart();
    if (nccl_status != ncclSuccess) {
      return base::error::InternalError(nccl_error_string("ncclGroupStart", nccl_status));
    }
    cudaSetDevice(src_device_id_);
    const ncclResult_t send_status =
        ncclSend(src_ptr, bytes, ncclUint8, 1,
                 static_cast<ncclComm_t>(src_comm_),
                 static_cast<cudaStream_t>(src_stream_));
    cudaSetDevice(dst_device_id_);
    const ncclResult_t recv_status =
        ncclRecv(dst_ptr, bytes, ncclUint8, 0,
                 static_cast<ncclComm_t>(dst_comm_),
                 static_cast<cudaStream_t>(dst_stream_));
    nccl_status = ncclGroupEnd();
    if (send_status != ncclSuccess) {
      return base::error::InternalError(nccl_error_string("ncclSend", send_status));
    }
    if (recv_status != ncclSuccess) {
      return base::error::InternalError(nccl_error_string("ncclRecv", recv_status));
    }
    if (nccl_status != ncclSuccess) {
      return base::error::InternalError(nccl_error_string("ncclGroupEnd", nccl_status));
    }
    return base::error::Success();
  };

  base::BlockAllocator& src_allocator = src_kv_manager_->allocator_mut(mapping.layer_idx);
  base::BlockAllocator& dst_allocator = dst_kv_manager_->allocator_mut(mapping.layer_idx);
  for (size_t block_idx = 0; block_idx < mapping.src_block_ids.size(); ++block_idx) {
    const auto src = src_allocator.get_block_payload_ptrs(mapping.src_block_ids[block_idx]);
    const auto dst = dst_allocator.get_block_payload_ptrs(mapping.dst_block_ids[block_idx]);
    status = transfer_payload(src.key, dst.key, src.key_value_bytes);
    if (!status) {
      cudaSetDevice(previous_device);
      return status;
    }
    status = transfer_payload(src.value, dst.value, src.key_value_bytes);
    if (!status) {
      cudaSetDevice(previous_device);
      return status;
    }
    if (src.scale_bytes > 0) {
      status = transfer_payload(src.key_scale, dst.key_scale, src.scale_bytes);
      if (!status) {
        cudaSetDevice(previous_device);
        return status;
      }
      status = transfer_payload(src.value_scale, dst.value_scale, src.scale_bytes);
      if (!status) {
        cudaSetDevice(previous_device);
        return status;
      }
    }
  }

  if (need_sync_) {
    cudaSetDevice(src_device_id_);
    cudaError_t sync_status = cudaStreamSynchronize(static_cast<cudaStream_t>(src_stream_));
    if (sync_status != cudaSuccess) {
      cudaSetDevice(previous_device);
      return base::error::InternalError(cudaGetErrorString(sync_status));
    }
    cudaSetDevice(dst_device_id_);
    sync_status = cudaStreamSynchronize(static_cast<cudaStream_t>(dst_stream_));
    if (sync_status != cudaSuccess) {
      cudaSetDevice(previous_device);
      return base::error::InternalError(cudaGetErrorString(sync_status));
    }
    cudaSetDevice(previous_device);
    return base::error::Success();
  }

  cudaSetDevice(dst_device_id_);
  cudaEvent_t event = nullptr;
  cudaError_t event_status = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
  if (event_status != cudaSuccess) {
    cudaSetDevice(previous_device);
    return base::error::InternalError(cudaGetErrorString(event_status));
  }
  event_status = cudaEventRecord(event, static_cast<cudaStream_t>(dst_stream_));
  if (event_status != cudaSuccess) {
    cudaEventDestroy(event);
    cudaSetDevice(previous_device);
    return base::error::InternalError(cudaGetErrorString(event_status));
  }
  *completion_event = event;
  cudaSetDevice(previous_device);
  return base::error::Success();
}

KVTransferStatus NcclLayerKVTransferConnector::poll_layer_unlocked(int32_t layer_idx) {
  if (layer_idx < 0 || layer_idx >= static_cast<int32_t>(layer_records_.size())) {
    return KVTransferStatus::Failed("pd layer connector layer_idx out of range");
  }
  auto& record = layer_records_[layer_idx];
  if (record.status.done() || record.completion_event == nullptr) {
    return record.status;
  }
  const cudaError_t status =
      cudaEventQuery(static_cast<cudaEvent_t>(record.completion_event));
  if (status == cudaSuccess) {
    cudaEventDestroy(static_cast<cudaEvent_t>(record.completion_event));
    record.completion_event = nullptr;
    record.status = record.transferred_tokens >= request_.computed_tokens
                        ? KVTransferStatus::Completed()
                        : KVTransferStatus::Pending();
  } else if (status != cudaErrorNotReady) {
    cudaEventDestroy(static_cast<cudaEvent_t>(record.completion_event));
    record.completion_event = nullptr;
    record.status = KVTransferStatus::Failed(cudaGetErrorString(status));
  }
  return record.status;
}

KVTransferStatus NcclLayerKVTransferConnector::poll_layer(int32_t layer_idx) {
  std::lock_guard<std::mutex> lock(mu_);
  return poll_layer_unlocked(layer_idx);
}

KVTransferStatus NcclLayerKVTransferConnector::poll() {
  std::lock_guard<std::mutex> lock(mu_);
  if (!prepared_) {
    return KVTransferStatus::Pending();
  }
  bool all_done = true;
  for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_records_.size()); ++layer_idx) {
    const KVTransferStatus status = poll_layer_unlocked(layer_idx);
    if (status.state == KVTransferState::kFailed ||
        status.state == KVTransferState::kCancelled) {
      return status;
    }
    all_done = all_done && status.ok();
  }
  return all_done ? KVTransferStatus::Completed() : KVTransferStatus::Pending();
}

base::Status NcclLayerKVTransferConnector::wait_for_layer_load(
    int32_t layer_idx,
    void* consumer_queue) {
  std::unique_lock<std::mutex> lock(mu_);
  if (!prepared_) {
    return base::error::InvalidArgument("pd layer connector is not prepared");
  }
  if (layer_idx < 0 || layer_idx >= static_cast<int32_t>(layer_records_.size())) {
    return base::error::InvalidArgument("pd layer connector layer_idx out of range");
  }
  while (true) {
    if (cancelled_) {
      return base::error::InternalError(cancel_reason_);
    }
    KVTransferStatus status = poll_layer_unlocked(layer_idx);
    if (status.ok()) {
      void* event = layer_records_[layer_idx].completion_event;
      lock.unlock();
      if (consumer_queue != nullptr && event != nullptr) {
        int previous_device = 0;
        cudaGetDevice(&previous_device);
        cudaSetDevice(dst_device_id_);
        const cudaError_t wait_status =
            cudaStreamWaitEvent(static_cast<cudaStream_t>(consumer_queue),
                                static_cast<cudaEvent_t>(event), 0);
        cudaSetDevice(previous_device);
        if (wait_status != cudaSuccess) {
          return base::error::InternalError(cudaGetErrorString(wait_status));
        }
      }
      return base::error::Success();
    }
    if (status.state == KVTransferState::kFailed) {
      return base::error::InternalError(status.error);
    }
    if (status.state == KVTransferState::kCancelled) {
      return base::error::InternalError(status.error);
    }
    cv_.wait_for(lock, std::chrono::milliseconds(1));
  }
}

void NcclLayerKVTransferConnector::cancel(const std::string& reason) {
  std::lock_guard<std::mutex> lock(mu_);
  cancelled_ = true;
  cancel_reason_ = reason.empty() ? "pd layer connector cancelled" : reason;
  for (auto& record : layer_records_) {
    if (record.status.done()) {
      continue;
    }
    if (record.completion_event != nullptr) {
      cudaEventDestroy(static_cast<cudaEvent_t>(record.completion_event));
      record.completion_event = nullptr;
    }
    record.status = KVTransferStatus::Cancelled(cancel_reason_);
  }
  cv_.notify_all();
}

}  // namespace serving
