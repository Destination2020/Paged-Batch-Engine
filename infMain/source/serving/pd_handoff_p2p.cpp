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
      if (cudaEventSynchronize(static_cast<cudaEvent_t>(transfer.completion_event)) != cudaSuccess)
        std::terminate();
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
    it->cancel_requested = true;
    it->cancel_reason = reason;
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

KVTransferStatus CudaP2PKVTransferConnector::drain(HandoffId handle) {
  auto it = std::find_if(transfers_.begin(), transfers_.end(), [&](const auto& item) {
    return item.handle.value == handle.value;
  });
  if (it == transfers_.end()) return KVTransferStatus::Failed("unknown P2P handoff id");
  if (it->completion_event != nullptr &&
      cudaEventSynchronize(static_cast<cudaEvent_t>(it->completion_event)) != cudaSuccess)
    return KVTransferStatus::Failed("CUDA P2P drain failed");
  return poll_transfer(static_cast<size_t>(std::distance(transfers_.begin(), it)));
}

void CudaP2PKVTransferConnector::release(HandoffId handle) {
  auto it = std::find_if(transfers_.begin(), transfers_.end(), [&](const auto& item) {
    return item.handle.value == handle.value;
  });
  if (it != transfers_.end() && it->status.done() && it->completion_event == nullptr)
    transfers_.erase(it);
}

base::Status CudaP2PKVTransferConnector::copy_blocks(const KVBlockManifest& manifest) {
  auto status = validate_cuda_kv_manifest(
      manifest, src_kv_manager_, dst_kv_manager_, "pd p2p connector");
  if (!status) return status;
  std::vector<cache::BoundTransfer> bound_transfers;
  status = bind_manifest_transfers(manifest, src_kv_manager_, dst_kv_manager_,
                                   &planner_, &bound_transfers);
  if (!status) return status;

  int previous_device = 0;
  cudaGetDevice(&previous_device);
  cudaError_t set_status = cudaSetDevice(dst_device_id_);
  if (set_status != cudaSuccess) {
    cudaGetLastError();
    return base::error::InternalError(std::string("cudaSetDevice for P2P dst failed: ") +
                                      cudaGetErrorString(set_status));
  }

  for (const auto& transfer : bound_transfers) {
    for (const auto& operation : transfer.operations()) {
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
      status = copy_payload(operation.source, operation.destination, operation.bytes);
      if (!status) {
        cudaSetDevice(previous_device);
        return status;
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

} // namespace serving
