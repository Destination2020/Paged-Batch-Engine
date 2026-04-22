#include "base/cuda_backend_runtime.h"

#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <string>
#include "base/alloc.h"
#include "base/cuda_init_utils.h"
#include "base/device_context.h"

namespace base {

CudaRuntimeEvent::CudaRuntimeEvent(void* native_event)
    : native_event_(native_event) {}

CudaRuntimeEvent::~CudaRuntimeEvent() {
  if (native_event_ != nullptr) {
    cudaEventDestroy(static_cast<cudaEvent_t>(native_event_));
    native_event_ = nullptr;
  }
}

BackendType CudaRuntimeEvent::backend_type() const {
  return BackendType::kCUDA;
}

void* CudaRuntimeEvent::native_event() const {
  return native_event_;
}

CudaBackendRuntime::CudaBackendRuntime(size_t cublas_lt_workspace_bytes)
    : cublas_lt_workspace_bytes_(cublas_lt_workspace_bytes) {}

BackendType CudaBackendRuntime::backend_type() const {
  return BackendType::kCUDA;
}

Status CudaBackendRuntime::initialize(DeviceContext* context) {
  CHECK_NE(context, nullptr);

  std::shared_ptr<kernel::CudaConfig> cuda_config;
  auto status = initialize_cuda_config(&cuda_config, context->device_id,
                                       cublas_lt_workspace_bytes_);
  if (!status) {
    return status;
  }

  context->backend = BackendType::kCUDA;
  context->compute_queue = cuda_config->stream;
  context->transfer_queue = cuda_config->stream;
  context->blas_handle = cuda_config->cublas_handle;
  context->blaslt_handle = cuda_config->cublas_lt_handle;
  context->native_context = cuda_config;
  return error::Success();
}

std::shared_ptr<DeviceAllocator> CudaBackendRuntime::device_allocator() {
  return CUDADeviceAllocatorFactory::get_instance();
}

std::shared_ptr<DeviceAllocator> CudaBackendRuntime::host_allocator() {
  return CPUDeviceAllocatorFactory::get_instance();
}

std::shared_ptr<DeviceAllocator> CudaBackendRuntime::pinned_host_allocator() {
  return PinnedCPUDeviceAllocatorFactory::get_instance();
}

Status CudaBackendRuntime::copy(const CopyParams& params) {
  if (params.byte_size == 0) {
    return error::Success();
  }
  if (params.src == nullptr || params.dst == nullptr) {
    return error::InvalidArgument("CudaBackendRuntime::copy got null pointer.");
  }

  MemcpyKind kind = MemcpyKind::kMemcpyCPU2CPU;
  switch (params.direction) {
    case CopyDirection::kHostToHost:
      kind = MemcpyKind::kMemcpyCPU2CPU;
      break;
    case CopyDirection::kHostToDevice:
      kind = MemcpyKind::kMemcpyCPU2CUDA;
      break;
    case CopyDirection::kDeviceToHost:
      kind = MemcpyKind::kMemcpyCUDA2CPU;
      break;
    case CopyDirection::kDeviceToDevice:
      kind = MemcpyKind::kMemcpyCUDA2CUDA;
      break;
    default:
      return error::InvalidArgument("CudaBackendRuntime::copy got unknown direction.");
  }

  device_allocator()->memcpy(params.src, params.dst, params.byte_size, kind,
                             params.queue, params.need_sync);
  return error::Success();
}

Status CudaBackendRuntime::memset_zero(const MemsetParams& params) {
  if (params.byte_size == 0) {
    return error::Success();
  }
  if (params.dst == nullptr) {
    return error::InvalidArgument("CudaBackendRuntime::memset_zero got null pointer.");
  }
  if (params.value != 0) {
    return error::InvalidArgument("CudaBackendRuntime::memset_zero only supports zero.");
  }

  device_allocator()->memset_zero(params.dst, params.byte_size, params.queue,
                                  params.need_sync);
  return error::Success();
}

Status CudaBackendRuntime::synchronize(DeviceContext* context) {
  if (context != nullptr && context->compute_queue != nullptr) {
    return synchronize_queue(context->compute_queue);
  }

  const cudaError_t status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    return error::InternalError(std::string("cudaDeviceSynchronize failed: ") +
                                cudaGetErrorString(status));
  }
  return error::Success();
}

Status CudaBackendRuntime::synchronize_queue(void* queue) {
  if (queue == nullptr) {
    return error::InvalidArgument("CudaBackendRuntime::synchronize_queue got null queue.");
  }

  cudaStream_t stream = static_cast<cudaStream_t>(queue);
  const cudaError_t status = cudaStreamSynchronize(stream);
  if (status != cudaSuccess) {
    return error::InternalError(std::string("cudaStreamSynchronize failed: ") +
                                cudaGetErrorString(status));
  }
  return error::Success();
}

Status CudaBackendRuntime::create_event(std::shared_ptr<RuntimeEvent>* out,
                                        bool disable_timing) {
  CHECK_NE(out, nullptr);
  unsigned int flags = disable_timing ? cudaEventDisableTiming : cudaEventDefault;
  cudaEvent_t event = nullptr;
  const cudaError_t status = cudaEventCreateWithFlags(&event, flags);
  if (status != cudaSuccess) {
    return error::InternalError(std::string("cudaEventCreateWithFlags failed: ") +
                                cudaGetErrorString(status));
  }
  *out = std::make_shared<CudaRuntimeEvent>(event);
  return error::Success();
}

Status CudaBackendRuntime::record_event(const std::shared_ptr<RuntimeEvent>& event,
                                        void* queue) {
  if (event == nullptr) {
    return error::InvalidArgument("CudaBackendRuntime::record_event got null event.");
  }
  if (queue == nullptr) {
    return error::InvalidArgument("CudaBackendRuntime::record_event got null queue.");
  }

  auto cuda_event = std::dynamic_pointer_cast<CudaRuntimeEvent>(event);
  if (cuda_event == nullptr || cuda_event->native_event() == nullptr) {
    return error::InvalidArgument("CudaBackendRuntime::record_event got non-CUDA event.");
  }

  const cudaError_t status = cudaEventRecord(
      static_cast<cudaEvent_t>(cuda_event->native_event()),
      static_cast<cudaStream_t>(queue));
  if (status != cudaSuccess) {
    return error::InternalError(std::string("cudaEventRecord failed: ") +
                                cudaGetErrorString(status));
  }
  return error::Success();
}

Status CudaBackendRuntime::wait_event(const std::shared_ptr<RuntimeEvent>& event) {
  if (event == nullptr) {
    return error::InvalidArgument("CudaBackendRuntime::wait_event got null event.");
  }

  auto cuda_event = std::dynamic_pointer_cast<CudaRuntimeEvent>(event);
  if (cuda_event == nullptr || cuda_event->native_event() == nullptr) {
    return error::InvalidArgument("CudaBackendRuntime::wait_event got non-CUDA event.");
  }

  const cudaError_t status =
      cudaEventSynchronize(static_cast<cudaEvent_t>(cuda_event->native_event()));
  if (status != cudaSuccess) {
    return error::InternalError(std::string("cudaEventSynchronize failed: ") +
                                cudaGetErrorString(status));
  }
  return error::Success();
}

Status CudaBackendRuntime::query_memory(DeviceMemoryInfo* out) const {
  CHECK_NE(out, nullptr);
  const cudaError_t status = cudaMemGetInfo(&out->free_bytes, &out->total_bytes);
  if (status != cudaSuccess) {
    return error::InternalError(std::string("cudaMemGetInfo failed: ") +
                                cudaGetErrorString(status));
  }
  return error::Success();
}

std::shared_ptr<kernel::CudaConfig> cuda_config_from_device_context(
    const std::shared_ptr<DeviceContext>& context) {
  if (context == nullptr || context->backend != BackendType::kCUDA ||
      context->native_context == nullptr) {
    return nullptr;
  }
  return std::static_pointer_cast<kernel::CudaConfig>(context->native_context);
}

Status initialize_cuda_device_context(std::shared_ptr<DeviceContext>* context,
                                      int device_id,
                                      size_t cublas_lt_workspace_bytes) {
  CHECK_NE(context, nullptr);

  auto local_context = std::make_shared<DeviceContext>();
  local_context->backend = BackendType::kCUDA;
  local_context->device_id = device_id;
  local_context->runtime =
      std::make_shared<CudaBackendRuntime>(cublas_lt_workspace_bytes);

  auto status = local_context->runtime->initialize(local_context.get());
  if (!status) {
    return status;
  }

  *context = std::move(local_context);
  return error::Success();
}

Status initialize_device_context(std::shared_ptr<DeviceContext>* context,
                                 DeviceType device_type,
                                 int device_id,
                                 size_t cublas_lt_workspace_bytes) {
  CHECK_NE(context, nullptr);

  switch (device_type) {
    case DeviceType::kDeviceCPU: {
      auto local_context = std::make_shared<DeviceContext>();
      local_context->backend = BackendType::kCPU;
      local_context->device_id = device_id;
      *context = std::move(local_context);
      return error::Success();
    }
    case DeviceType::kDeviceCUDA:
      return initialize_cuda_device_context(context, device_id,
                                            cublas_lt_workspace_bytes);
    case DeviceType::kDeviceHIP:
      return error::FunctionNotImplement(
          "HIP device context initialization is reserved but not implemented yet.");
    case DeviceType::kDeviceTPU:
      return error::FunctionNotImplement(
          "TPU device context initialization is reserved but not implemented yet.");
    default:
      return error::FunctionNotImplement(
          "Device context initialization is not implemented for device type " +
          std::to_string(static_cast<int>(device_type)));
  }
}

}  // namespace base
