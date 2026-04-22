#ifndef KUIPER_INCLUDE_BASE_BACKEND_RUNTIME_H_
#define KUIPER_INCLUDE_BASE_BACKEND_RUNTIME_H_

#include <cstddef>
#include <memory>
#include "base/backend_type.h"
#include "base/base.h"

namespace base {

class DeviceAllocator;
class RuntimeEvent;
struct DeviceContext;

struct DeviceMemoryInfo {
  size_t free_bytes = 0;
  size_t total_bytes = 0;
};

enum class CopyDirection : uint8_t {
  kUnknown = 0,
  kHostToHost = 1,
  kHostToDevice = 2,
  kDeviceToHost = 3,
  kDeviceToDevice = 4,
};

struct CopyParams {
  const void* src = nullptr;
  void* dst = nullptr;
  size_t byte_size = 0;
  CopyDirection direction = CopyDirection::kUnknown;
  void* queue = nullptr;
  bool need_sync = false;
};

struct MemsetParams {
  void* dst = nullptr;
  size_t byte_size = 0;
  int value = 0;
  void* queue = nullptr;
  bool need_sync = false;
};

class RuntimeEvent {
 public:
  virtual ~RuntimeEvent() = default;
  virtual BackendType backend_type() const = 0;
};

class BackendRuntime {
 public:
  virtual ~BackendRuntime() = default;

  virtual BackendType backend_type() const = 0;
  virtual Status initialize(DeviceContext* context) = 0;

  virtual std::shared_ptr<DeviceAllocator> device_allocator() = 0;
  virtual std::shared_ptr<DeviceAllocator> host_allocator() = 0;
  virtual std::shared_ptr<DeviceAllocator> pinned_host_allocator() = 0;

  virtual Status copy(const CopyParams& params) = 0;
  virtual Status memset_zero(const MemsetParams& params) = 0;
  virtual Status synchronize(DeviceContext* context) = 0;
  virtual Status synchronize_queue(void* queue) = 0;
  virtual Status create_event(std::shared_ptr<RuntimeEvent>* out,
                              bool disable_timing = true) = 0;
  virtual Status record_event(const std::shared_ptr<RuntimeEvent>& event,
                              void* queue) = 0;
  virtual Status wait_event(const std::shared_ptr<RuntimeEvent>& event) = 0;
  virtual Status query_memory(DeviceMemoryInfo* out) const = 0;
};

Status initialize_device_context(std::shared_ptr<DeviceContext>* context,
                                 DeviceType device_type,
                                 int device_id = 0,
                                 size_t cublas_lt_workspace_bytes = 4u * 1024u * 1024u);

}  // namespace base

#endif  // KUIPER_INCLUDE_BASE_BACKEND_RUNTIME_H_
