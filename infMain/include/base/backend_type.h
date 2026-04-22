#ifndef KUIPER_INCLUDE_BASE_BACKEND_TYPE_H_
#define KUIPER_INCLUDE_BASE_BACKEND_TYPE_H_

#include <cstdint>

namespace base {

enum class BackendType : uint8_t {
  kUnknown = 0,
  kCPU = 1,
  kCUDA = 2,
  kHIP = 3,
  kTPU = 4,
};

}  // namespace base

#endif  // KUIPER_INCLUDE_BASE_BACKEND_TYPE_H_
