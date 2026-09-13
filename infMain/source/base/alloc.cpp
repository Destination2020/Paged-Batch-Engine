// Updated on March 15, 2026
#include "base/alloc.h"
#include <cstring>
#ifndef KUIPER_CPU_ONLY
#include <cuda_runtime_api.h>
#endif
namespace base {
void DeviceAllocator::memcpy(const void* src_ptr, void* dest_ptr, size_t byte_size,
                             MemcpyKind memcpy_kind, void* stream, bool need_sync) const {
  CHECK_NE(src_ptr, nullptr);
  CHECK_NE(dest_ptr, nullptr);
  if (!byte_size) {
    return;
  }

#ifdef KUIPER_CPU_ONLY
  CHECK(memcpy_kind == MemcpyKind::kMemcpyCPU2CPU) << "CUDA copy unavailable in CPU-only build";
  std::memcpy(dest_ptr, src_ptr, byte_size);
#else
  cudaStream_t stream_ = nullptr;
  if (stream) {
    stream_ = static_cast<CUstream_st*>(stream);
  }
  if (memcpy_kind == MemcpyKind::kMemcpyCPU2CPU) {
    std::memcpy(dest_ptr, src_ptr, byte_size);
  } else if (memcpy_kind == MemcpyKind::kMemcpyCPU2CUDA) {
    if (!stream_) {
      cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice);
    } else {
      cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice, stream_);
    }
  } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CPU) {
    if (!stream_) {
      cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost);
    } else {
      cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost, stream_);
    }
  } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CUDA) {
    if (!stream_) {
      cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice);
    } else {
      cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice, stream_);
    }
  } else {
    LOG(FATAL) << "Unknown memcpy kind: " << int(memcpy_kind);
  }
  if (need_sync) {
    cudaDeviceSynchronize();
  }
#endif
}

void DeviceAllocator::memset_zero(void* ptr, size_t byte_size, void* stream,
                                  bool need_sync) {
  CHECK(device_type_ != base::DeviceType::kDeviceUnknown);
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    std::memset(ptr, 0, byte_size);
  } else {
#ifdef KUIPER_CPU_ONLY
    LOG(FATAL) << "Device memset unavailable in CPU-only build";
#else
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      cudaMemsetAsync(ptr, 0, byte_size, stream_);
    } else {
      cudaMemset(ptr, 0, byte_size);
    }
    if (need_sync) {
      cudaDeviceSynchronize();
    }
#endif
  }
}

}  // namespace base
