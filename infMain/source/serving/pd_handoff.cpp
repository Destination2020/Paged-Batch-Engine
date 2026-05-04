#include "serving/pd_handoff.h"

#include <algorithm>
#include <cstring>
#include <utility>

#include <cuda_runtime_api.h>

#include "base/alloc.h"

namespace serving {
namespace {

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

}  // namespace serving
