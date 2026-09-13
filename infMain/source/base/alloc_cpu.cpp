// Updated on March 15, 2026
#include <glog/logging.h>
#ifndef KUIPER_CPU_ONLY
#include <cuda_runtime_api.h>
#endif
#include <cstdlib>
#include "base/alloc.h"

#if (defined(_POSIX_ADVISORY_INFO) && (_POSIX_ADVISORY_INFO >= 200112L))
#define KUIPER_HAVE_POSIX_MEMALIGN
#endif

namespace base {
CPUDeviceAllocator::CPUDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCPU) {
}

void* CPUDeviceAllocator::allocate(size_t byte_size) const {
  if (!byte_size) {
    return nullptr;
  }
#ifdef KUIPER_HAVE_POSIX_MEMALIGN
  void* data = nullptr;
  const size_t alignment = (byte_size >= size_t(1024)) ? size_t(32) : size_t(16);
  int status = posix_memalign((void**)&data,
                              ((alignment >= sizeof(void*)) ? alignment : sizeof(void*)),
                              byte_size);
  if (status != 0) {
    return nullptr;
  }
  return data;
#else
  void* data = malloc(byte_size);
  return data;
#endif
}

void CPUDeviceAllocator::release(void* ptr) const {
  if (ptr) {
    free(ptr);
  }
}

PinnedCPUDeviceAllocator::PinnedCPUDeviceAllocator()
    : DeviceAllocator(DeviceType::kDeviceCPU) {
}

void* PinnedCPUDeviceAllocator::allocate(size_t byte_size) const {
  if (!byte_size) {
    return nullptr;
  }
#ifdef KUIPER_CPU_ONLY
  LOG(FATAL) << "Pinned CUDA allocation unavailable in CPU-only build";
  return nullptr;
#else
  void* data = nullptr;
  const cudaError_t status = cudaMallocHost(&data, byte_size);
  if (status != cudaSuccess) {
    LOG(ERROR) << "cudaMallocHost failed for " << byte_size
               << " bytes: " << cudaGetErrorString(status);
    return nullptr;
  }
  return data;
#endif
}

void PinnedCPUDeviceAllocator::release(void* ptr) const {
  if (!ptr) {
    return;
  }
#ifdef KUIPER_CPU_ONLY
  LOG(FATAL) << "Pinned CUDA release unavailable in CPU-only build";
#else
  const cudaError_t status = cudaFreeHost(ptr);
  CHECK_EQ(status, cudaSuccess) << "cudaFreeHost failed: " << cudaGetErrorString(status);
#endif
}

std::shared_ptr<CPUDeviceAllocator> CPUDeviceAllocatorFactory::instance = nullptr;
std::shared_ptr<PinnedCPUDeviceAllocator> PinnedCPUDeviceAllocatorFactory::instance = nullptr;
}  // namespace base
