#ifndef KUIPER_INCLUDE_BASE_CUDA_BACKEND_RUNTIME_H_
#define KUIPER_INCLUDE_BASE_CUDA_BACKEND_RUNTIME_H_

#include <memory>
#include "base/backend_runtime.h"
#include "base/cuda_config.h"

namespace base {

class CudaRuntimeEvent final : public RuntimeEvent {
 public:
  explicit CudaRuntimeEvent(void* native_event);
  ~CudaRuntimeEvent() override;

  BackendType backend_type() const override;
  void* native_event() const;

 private:
  void* native_event_ = nullptr;
};

class CudaBackendRuntime final : public BackendRuntime {
 public:
  explicit CudaBackendRuntime(size_t cublas_lt_workspace_bytes = 4u * 1024u * 1024u);

  BackendType backend_type() const override;

  Status initialize(DeviceContext* context) override;

  std::shared_ptr<DeviceAllocator> device_allocator() override;

  std::shared_ptr<DeviceAllocator> host_allocator() override;

  std::shared_ptr<DeviceAllocator> pinned_host_allocator() override;

  Status copy(const CopyParams& params) override;

  Status memset_zero(const MemsetParams& params) override;

  Status synchronize(DeviceContext* context) override;

  Status synchronize_queue(void* queue) override;

  Status create_event(std::shared_ptr<RuntimeEvent>* out,
                      bool disable_timing = true) override;

  Status record_event(const std::shared_ptr<RuntimeEvent>& event,
                      void* queue) override;

  Status wait_event(const std::shared_ptr<RuntimeEvent>& event) override;

  Status query_memory(DeviceMemoryInfo* out) const override;

 private:
  size_t cublas_lt_workspace_bytes_ = 0;
};

std::shared_ptr<kernel::CudaConfig> cuda_config_from_device_context(
    const std::shared_ptr<DeviceContext>& context);

Status initialize_cuda_device_context(std::shared_ptr<DeviceContext>* context,
                                      int device_id = 0,
                                      size_t cublas_lt_workspace_bytes = 4u * 1024u * 1024u);

}  // namespace base

#endif  // KUIPER_INCLUDE_BASE_CUDA_BACKEND_RUNTIME_H_
