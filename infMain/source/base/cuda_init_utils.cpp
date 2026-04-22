#include "base/cuda_init_utils.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>

#include <memory>
#include <sstream>
#include <utility>

namespace base {

namespace {

std::string cublas_status_to_string(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
#if defined(CUBLAS_STATUS_NOT_SUPPORTED)
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if defined(CUBLAS_STATUS_LICENSE_ERROR)
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
    default:
      return "CUBLAS_STATUS_UNKNOWN(" + std::to_string(static_cast<int>(status)) + ")";
  }
}

Status cuda_runtime_error(const std::string& stage, cudaError_t status) {
  std::ostringstream os;
  os << stage << " failed: " << cudaGetErrorString(status)
     << " (" << static_cast<int>(status) << ")";
  return error::InternalError(os.str());
}

Status cublas_error(const std::string& stage, cublasStatus_t status) {
  std::ostringstream os;
  os << stage << " failed: " << cublas_status_to_string(status)
     << " (" << static_cast<int>(status) << ")";
  return error::InternalError(os.str());
}

}  // namespace

Status initialize_cuda_config(std::shared_ptr<kernel::CudaConfig>* config,
                              int device_id,
                              size_t cublas_lt_workspace_bytes) {
  CHECK_NE(config, nullptr);

  const cudaError_t set_device_status = cudaSetDevice(device_id);
  if (set_device_status != cudaSuccess) {
    return cuda_runtime_error("cudaSetDevice(" + std::to_string(device_id) + ")",
                              set_device_status);
  }

  auto local_config = std::make_shared<kernel::CudaConfig>();

  const cudaError_t stream_status = cudaStreamCreate(&local_config->stream);
  if (stream_status != cudaSuccess) {
    return cuda_runtime_error("cudaStreamCreate", stream_status);
  }

  const cublasStatus_t cublas_status = cublasCreate(&local_config->cublas_handle);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    return cublas_error("cublasCreate", cublas_status);
  }

  const cublasStatus_t set_stream_status =
      cublasSetStream(local_config->cublas_handle, local_config->stream);
  if (set_stream_status != CUBLAS_STATUS_SUCCESS) {
    return cublas_error("cublasSetStream", set_stream_status);
  }

  const cublasStatus_t cublas_lt_status = cublasLtCreate(&local_config->cublas_lt_handle);
  if (cublas_lt_status != CUBLAS_STATUS_SUCCESS) {
    return cublas_error("cublasLtCreate", cublas_lt_status);
  }

  if (cublas_lt_workspace_bytes > 0) {
    void* cublas_lt_workspace = nullptr;
    const cudaError_t workspace_status =
        cudaMalloc(&cublas_lt_workspace, cublas_lt_workspace_bytes);
    if (workspace_status == cudaSuccess && cublas_lt_workspace != nullptr) {
      local_config->cublas_lt_workspace.reset(
          cublas_lt_workspace,
          [](void* ptr) {
            if (ptr != nullptr) {
              cudaFree(ptr);
            }
          });
      local_config->cublas_lt_workspace_bytes = cublas_lt_workspace_bytes;
    } else {
      // cuBLASLt workspace is an optimization, not a hard requirement.
      LOG(WARNING) << "cudaMalloc for cuBLASLt workspace failed: "
                   << cudaGetErrorString(workspace_status)
                   << " (" << static_cast<int>(workspace_status)
                   << "), requested_bytes=" << cublas_lt_workspace_bytes
                   << ". Continuing without cuBLASLt workspace.";
      cudaGetLastError();
    }
  }

  *config = std::move(local_config);
  return error::Success();
}

}  // namespace base
