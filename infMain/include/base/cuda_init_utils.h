// Shared CUDA/cuBLAS initialization helpers for model runtimes.
#ifndef KUIPER_INCLUDE_BASE_CUDA_INIT_UTILS_H_
#define KUIPER_INCLUDE_BASE_CUDA_INIT_UTILS_H_

#include <memory>
#include <string>
#include "base/base.h"
#include "base/cuda_config.h"

namespace base {

Status initialize_cuda_config(std::shared_ptr<kernel::CudaConfig>* config,
                              int device_id = 0,
                              size_t cublas_lt_workspace_bytes = 4u * 1024u * 1024u);

}  // namespace base

#endif  // KUIPER_INCLUDE_BASE_CUDA_INIT_UTILS_H_
