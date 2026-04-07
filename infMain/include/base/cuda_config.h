#ifndef BLAS_HELPER_H
#define BLAS_HELPER_H
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
namespace kernel {
struct CudaConfig {
  cudaStream_t stream = nullptr;
  cublasHandle_t cublas_handle = nullptr;
  ~CudaConfig() {
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
