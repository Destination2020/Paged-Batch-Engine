
#include <cuda_runtime_api.h>
#include "base/alloc.h"
namespace base {

CUDADeviceAllocator::CUDADeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCUDA) {}

CUDADeviceAllocator::~CUDADeviceAllocator() { release_all_cached(); }

void* CUDADeviceAllocator::allocate(size_t byte_size) const {
  std::lock_guard<std::mutex> lock(mutex_);
  int id = -1;
  cudaError_t state = cudaGetDevice(&id);
  CHECK(state == cudaSuccess);
  if (byte_size > 1024 * 1024) {
    auto& big_buffers = big_buffers_map_[id];
    int sel_id = -1;
    for (int i = 0; i < big_buffers.size(); i++) {
      if (big_buffers[i].byte_size >= byte_size && !big_buffers[i].busy &&
          big_buffers[i].byte_size - byte_size < 1 * 1024 * 1024) {
        if (sel_id == -1 || big_buffers[sel_id].byte_size > big_buffers[i].byte_size) {
          sel_id = i;
        }
      }
    }
    if (sel_id != -1) {
      big_buffers[sel_id].busy = true;
      return big_buffers[sel_id].data;
    }

    const auto limit = byte_limits_.find(id);
    if (limit != byte_limits_.end() &&
        byte_size > limit->second - reserved_bytes_map_[id]) {
      LOG(ERROR) << "Private CUDA allocation budget exhausted: reserved="
                 << reserved_bytes_map_[id] << " requested=" << byte_size
                 << " limit=" << limit->second;
      return nullptr;
    }
    void* ptr = nullptr;
    state = cudaMalloc(&ptr, byte_size);
    if (cudaSuccess != state) {
      char buf[256];
      snprintf(buf, 256,
               "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory "
               "left on  device.",
               byte_size >> 20);
      LOG(ERROR) << buf;
      return nullptr;
    }
    big_buffers.emplace_back(ptr, byte_size, true);
    reserved_bytes_map_[id] += byte_size;
    return ptr;
  }

  auto& cuda_buffers = cuda_buffers_map_[id];
  for (int i = 0; i < cuda_buffers.size(); i++) {
    if (cuda_buffers[i].byte_size >= byte_size && !cuda_buffers[i].busy) {
      cuda_buffers[i].busy = true;
      no_busy_cnt_[id] -= cuda_buffers[i].byte_size;
      return cuda_buffers[i].data;
    }
  }
  const auto limit = byte_limits_.find(id);
  if (limit != byte_limits_.end() &&
      byte_size > limit->second - reserved_bytes_map_[id]) {
    LOG(ERROR) << "Private CUDA allocation budget exhausted: reserved="
               << reserved_bytes_map_[id] << " requested=" << byte_size
               << " limit=" << limit->second;
    return nullptr;
  }
  void* ptr = nullptr;
  state = cudaMalloc(&ptr, byte_size);
  if (cudaSuccess != state) {
    char buf[256];
    snprintf(buf, 256,
             "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory "
             "left on  device.",
             byte_size >> 20);
    LOG(ERROR) << buf;
    return nullptr;
  }
  cuda_buffers.emplace_back(ptr, byte_size, true);
  reserved_bytes_map_[id] += byte_size;
  return ptr;
}

void CUDADeviceAllocator::release(void* ptr) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!ptr) {
    return;
  }
  cudaError_t state = cudaSuccess;
  for (auto& it : cuda_buffers_map_) {
    if (no_busy_cnt_[it.first] > 1024 * 1024 * 1024) {
      auto& cuda_buffers = it.second;
      std::vector<CudaMemoryBuffer> temp;
      for (int i = 0; i < cuda_buffers.size(); i++) {
        if (!cuda_buffers[i].busy) {
          state = cudaSetDevice(it.first);
          state = cudaFree(cuda_buffers[i].data);
          reserved_bytes_map_[it.first] -= cuda_buffers[i].byte_size;
          CHECK(state == cudaSuccess)
              << "Error: CUDA error when release memory on device " << it.first;
        } else {
          temp.push_back(cuda_buffers[i]);
        }
      }
      cuda_buffers.clear();
      it.second = temp;
      no_busy_cnt_[it.first] = 0;
    }
  }

  for (auto& it : cuda_buffers_map_) {
    auto& cuda_buffers = it.second;
    for (int i = 0; i < cuda_buffers.size(); i++) {
      if (cuda_buffers[i].data == ptr) {
        no_busy_cnt_[it.first] += cuda_buffers[i].byte_size;
        cuda_buffers[i].busy = false;
        return;
      }
    }
  }
  // Large allocations can exist before the first small allocation. Search
  // their own device map so returning a cached buffer never frees it behind
  // the allocator's accounting or prevents its later reuse.
  for (auto& it : big_buffers_map_) {
    auto& big_buffers = it.second;
    for (int i = 0; i < big_buffers.size(); i++) {
      if (big_buffers[i].data == ptr) {
        big_buffers[i].busy = false;
        return;
      }
    }
  }
  state = cudaFree(ptr);
  CHECK(state == cudaSuccess) << "Error: CUDA error when release memory on device";
}

void CUDADeviceAllocator::release_all_cached() const {
  std::lock_guard<std::mutex> lock(mutex_);
  int original_device = -1;
  const cudaError_t get_device_status = cudaGetDevice(&original_device);
  if (get_device_status != cudaSuccess) {
    // Static allocator destruction may run after the CUDA runtime has already
    // torn down its primary contexts. The driver owns the remaining process
    // allocations at that point; avoid issuing invalid cleanup calls.
    cuda_buffers_map_.clear();
    big_buffers_map_.clear();
    no_busy_cnt_.clear();
    reserved_bytes_map_.clear();
    return;
  }
  auto release_map = [](auto& buffers_by_device, auto& reserved_bytes) {
    for (auto& [device, buffers] : buffers_by_device) {
      const cudaError_t set_status = cudaSetDevice(device);
      if (set_status != cudaSuccess) {
        LOG(ERROR) << "Unable to select CUDA device " << device
                   << " while releasing allocator cache: "
                   << cudaGetErrorString(set_status);
        continue;
      }
      for (auto& buffer : buffers) {
        if (buffer.data == nullptr) continue;
        const cudaError_t free_status = cudaFree(buffer.data);
        if (free_status != cudaSuccess) {
          LOG(ERROR) << "Unable to release cached CUDA allocation on device "
                     << device << ": " << cudaGetErrorString(free_status);
          continue;
        }
        if (reserved_bytes[device] >= buffer.byte_size)
          reserved_bytes[device] -= buffer.byte_size;
        buffer.data = nullptr;
      }
      buffers.clear();
    }
  };
  release_map(cuda_buffers_map_, reserved_bytes_map_);
  release_map(big_buffers_map_, reserved_bytes_map_);
  cuda_buffers_map_.clear();
  big_buffers_map_.clear();
  no_busy_cnt_.clear();
  reserved_bytes_map_.clear();
  cudaSetDevice(original_device);
}

void CUDADeviceAllocator::set_byte_limit(int device, size_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!bytes) { byte_limits_.erase(device); return; }
  CHECK_LE(reserved_bytes_map_[device], bytes) << "Already allocated beyond private CUDA budget";
  byte_limits_[device] = bytes;
}
size_t CUDADeviceAllocator::reserved_bytes(int device) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = reserved_bytes_map_.find(device);
  return it == reserved_bytes_map_.end() ? 0 : it->second;
}
std::shared_ptr<CUDADeviceAllocator> CUDADeviceAllocatorFactory::instance = nullptr;

}  // namespace base
