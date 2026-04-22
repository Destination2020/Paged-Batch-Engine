#ifndef KUIPER_INCLUDE_BASE_DEVICE_CONTEXT_H_
#define KUIPER_INCLUDE_BASE_DEVICE_CONTEXT_H_

#include <memory>
#include "base/backend_type.h"

namespace base {

class BackendRuntime;

struct DeviceContext {
  BackendType backend = BackendType::kUnknown;
  int device_id = 0;
  void* compute_queue = nullptr;
  void* transfer_queue = nullptr;
  void* blas_handle = nullptr;
  void* blaslt_handle = nullptr;

  // Backend-owned native context. CUDA currently stores kernel::CudaConfig here
  // while callers migrate away from direct CUDA-specific fields.
  std::shared_ptr<void> native_context;
  std::shared_ptr<BackendRuntime> runtime;
};

}  // namespace base

#endif  // KUIPER_INCLUDE_BASE_DEVICE_CONTEXT_H_
