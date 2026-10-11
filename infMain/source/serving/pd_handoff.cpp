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
namespace serving {
using namespace handoff_detail;
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

KVTransferStatus InProcNoCopyKVTransferConnector::drain(HandoffId handle) {
  return poll(handle);
}

void InProcNoCopyKVTransferConnector::release(HandoffId handle) {
  auto it = std::find_if(transfers_.begin(), transfers_.end(), [&](const auto& item) {
    return item.first.value == handle.value;
  });
  if (it != transfers_.end() && it->second.done()) transfers_.erase(it);
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
      if (cudaEventSynchronize(static_cast<cudaEvent_t>(transfer.completion_event)) != cudaSuccess)
        std::terminate();
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
    it->cancel_requested = true;
    it->cancel_reason = reason;
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
    transfer.status = transfer.cancel_requested
                          ? KVTransferStatus::Cancelled(transfer.cancel_reason)
                          : KVTransferStatus::Completed();
  } else if (status != cudaErrorNotReady) {
    cudaEventDestroy(static_cast<cudaEvent_t>(transfer.completion_event));
    transfer.completion_event = nullptr;
    transfer.status = KVTransferStatus::Failed(cudaGetErrorString(status));
  }
  return transfer.status;
}

KVTransferStatus InProcKVBlockCopyConnector::drain(HandoffId handle) {
  auto it = std::find_if(transfers_.begin(), transfers_.end(), [&](const auto& item) {
    return item.handle.value == handle.value;
  });
  if (it == transfers_.end()) return KVTransferStatus::Failed("unknown handoff id");
  if (it->completion_event != nullptr &&
      cudaEventSynchronize(static_cast<cudaEvent_t>(it->completion_event)) != cudaSuccess)
    return KVTransferStatus::Failed("CUDA block copy drain failed");
  return poll_transfer(static_cast<size_t>(std::distance(transfers_.begin(), it)));
}

void InProcKVBlockCopyConnector::release(HandoffId handle) {
  auto it = std::find_if(transfers_.begin(), transfers_.end(), [&](const auto& item) {
    return item.handle.value == handle.value;
  });
  if (it != transfers_.end() && it->status.done() && it->completion_event == nullptr)
    transfers_.erase(it);
}

base::Status InProcKVBlockCopyConnector::copy_blocks(const KVBlockManifest& manifest) {
  std::vector<cache::BoundTransfer> transfers;
  auto status = bind_manifest_transfers(manifest, src_kv_manager_, dst_kv_manager_,
                                        &planner_, &transfers);
  if (!status) return status;
  const auto src_device = src_kv_manager_->allocator(0).device_type();
  const auto dst_device = dst_kv_manager_->allocator(0).device_type();
  const bool runtime_copy = src_device != base::DeviceType::kDeviceCPU ||
                            dst_device != base::DeviceType::kDeviceCPU;
  const auto kind = copy_kind_for_devices(src_device, dst_device);
  auto allocator = runtime_copy ? base::CUDADeviceAllocatorFactory::get_instance() : nullptr;
  for (const auto& transfer : transfers) {
    for (const auto& operation : transfer.operations()) {
      if (runtime_copy)
        allocator->memcpy(operation.source, operation.destination, operation.bytes,
                          kind, transfer_queue_, need_sync_);
      else
        std::memcpy(operation.destination, operation.source, operation.bytes);
    }
  }
  return base::error::Success();
}

} // namespace serving
