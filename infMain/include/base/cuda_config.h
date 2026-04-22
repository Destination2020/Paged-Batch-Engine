#ifndef BLAS_HELPER_H
#define BLAS_HELPER_H
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime_api.h>
#include <memory>
namespace kernel {
struct CudaConfig {
  cudaStream_t stream = nullptr;
  cublasHandle_t cublas_handle = nullptr;
  cublasLtHandle_t cublas_lt_handle = nullptr;
  std::shared_ptr<void> cublas_lt_workspace;
  size_t cublas_lt_workspace_bytes = 0;
  ~CudaConfig() {
    cublas_lt_workspace.reset();
    if (cublas_lt_handle) {
      cublasLtDestroy(cublas_lt_handle);
    }
    if (cublas_handle) {
      cublasDestroy(cublas_handle);
    }
    if (stream) {
      cudaStreamDestroy(stream);
    }
  }
};
}  // namespace kernel
#endif  // BLAS_HELPER_H
