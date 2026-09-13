#include "cache/page_migration.h"
#ifndef KUIPER_CPU_ONLY
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <vector>
namespace cache {
namespace {
class DeviceGuard {
 public:
  explicit DeviceGuard(int device) noexcept {
    valid_ = cudaGetDevice(&old_) == cudaSuccess && cudaSetDevice(device) == cudaSuccess;
  }
  ~DeviceGuard() { if (valid_) cudaSetDevice(old_); }
  bool valid() const { return valid_; }
 private:
  int old_ = 0;
  bool valid_ = false;
};
class CudaTransferBackend final : public TransferBackend {
 public:
  CudaTransferBackend(size_t capacity, int device, void* dependency)
      : device_(device), dependency_(static_cast<cudaEvent_t>(dependency)), slots_(capacity) {
    if (device_ < 0 && cudaGetDevice(&device_) != cudaSuccess)
      throw std::runtime_error("CUDA migration current device unavailable");
    DeviceGuard guard(device_);
    if (!guard.valid()) throw std::runtime_error("CUDA migration device unavailable");
    for (auto& slot : slots_) {
      if (cudaStreamCreateWithFlags(&slot.stream, cudaStreamNonBlocking) != cudaSuccess ||
          cudaEventCreateWithFlags(&slot.fence, cudaEventDisableTiming) != cudaSuccess) {
        destroy();
        throw std::runtime_error("CUDA migration stream/event allocation failed");
      }
    }
  }
  ~CudaTransferBackend() override {
    DeviceGuard guard(device_);
    if (!guard.valid()) std::terminate();
    for (auto& slot : slots_) {
      if (slot.id && cudaStreamSynchronize(slot.stream) != cudaSuccess) std::terminate();
    }
    destroy();
  }
  bool supports(base::DeviceType device) const override { return device == base::DeviceType::kDeviceCUDA; }
  BackendResult submit(MigrationId id, TransferDirection direction,
                       const BoundTransfer& transfer) noexcept override {
    DeviceGuard guard(device_);
    if (!guard.valid()) return BackendResult::kFailedSafe;
    if (find(id)) return BackendResult::kUnknown;
    // Validate every physical endpoint before enqueueing the first component.
    for (const auto& op : transfer.operations()) {
      const void* gpu = direction == TransferDirection::kToHost ? op.source : op.destination;
      const void* host = direction == TransferDirection::kToHost ? op.destination : op.source;
      cudaPointerAttributes gpu_attr{}, host_attr{};
      if (cudaPointerGetAttributes(&gpu_attr, gpu) != cudaSuccess ||
          cudaPointerGetAttributes(&host_attr, host) != cudaSuccess ||
          gpu_attr.type != cudaMemoryTypeDevice || gpu_attr.device != device_ ||
          host_attr.type != cudaMemoryTypeHost)
        return BackendResult::kFailedSafe;
    }
    Slot* slot = nullptr;
    for (auto& candidate : slots_) if (!candidate.id) { slot = &candidate; break; }
    if (!slot) return BackendResult::kFailedSafe;
    slot->id = id;
    slot->failed = false;
    slot->safe = false;
    if (dependency_ && cudaStreamWaitEvent(slot->stream, dependency_, 0) != cudaSuccess) {
      slot->failed = true;
      return BackendResult::kUnknown;
    }
    for (const auto& op : transfer.operations()) {
      const auto kind = direction == TransferDirection::kToHost ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
      if (cudaMemcpyAsync(op.destination, op.source, op.bytes, kind, slot->stream) != cudaSuccess) {
        slot->failed = true;
        return BackendResult::kUnknown;
      }
    }
    if (cudaEventRecord(slot->fence, slot->stream) != cudaSuccess) {
      slot->failed = true;
      return BackendResult::kUnknown;
    }
    return BackendResult::kPending;
  }
  BackendResult poll(MigrationId id) noexcept override {
    DeviceGuard guard(device_);
    auto* slot = find(id);
    if (!guard.valid() || !slot || slot->failed) return BackendResult::kUnknown;
    const auto result = cudaEventQuery(slot->fence);
    if (result == cudaErrorNotReady) return BackendResult::kPending;
    if (result != cudaSuccess) { slot->failed = true; return BackendResult::kUnknown; }
    slot->safe = true;
    return BackendResult::kSucceeded;
  }
  BackendResult drain(MigrationId id) noexcept override {
    DeviceGuard guard(device_);
    auto* slot = find(id);
    if (!guard.valid() || !slot || cudaStreamSynchronize(slot->stream) != cudaSuccess)
      return BackendResult::kUnknown;
    slot->safe = true;
    return slot->failed ? BackendResult::kFailedSafe : BackendResult::kSucceeded;
  }
  void release(MigrationId id) noexcept override {
    auto* slot = find(id);
    if (!slot) return; // Rejected before any stream submission.
    if (!slot->safe) std::terminate();
    slot->id = 0;
  }
 private:
  struct Slot {
    MigrationId id = 0;
    cudaStream_t stream = nullptr;
    cudaEvent_t fence = nullptr;
    bool failed = false, safe = false;
  };
  Slot* find(MigrationId id) noexcept {
    for (auto& slot : slots_) if (slot.id == id) return &slot;
    return nullptr;
  }
  void destroy() noexcept {
    for (auto& slot : slots_) {
      if (slot.fence) cudaEventDestroy(slot.fence);
      if (slot.stream) cudaStreamDestroy(slot.stream);
      slot.fence = nullptr; slot.stream = nullptr;
    }
  }
  int device_;
  cudaEvent_t dependency_;
  std::vector<Slot> slots_;
};
}
std::unique_ptr<TransferBackend> MakeCudaTransferBackend(size_t slots, int device, void* dependency) {
  return std::make_unique<CudaTransferBackend>(slots, device, dependency);
}
}  // namespace cache
#endif
