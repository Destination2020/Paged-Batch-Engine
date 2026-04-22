#ifndef KUIPER_INCLUDE_BASE_NVTX_UTILS_H_
#define KUIPER_INCLUDE_BASE_NVTX_UTILS_H_

#include <cstdint>
#include <string>

#if defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define KUIPER_HAVE_NVTX 1
#elif __has_include(<nvToolsExt.h>)
#include <nvToolsExt.h>
#define KUIPER_HAVE_NVTX 1
#else
#define KUIPER_HAVE_NVTX 0
#endif
#else
#define KUIPER_HAVE_NVTX 0
#endif

namespace base {
namespace nvtx {

constexpr uint32_t kColorDefault = 0xFF7F7F7F;
constexpr uint32_t kColorStep = 0xFF3B82F6;
constexpr uint32_t kColorSchedule = 0xFF10B981;
constexpr uint32_t kColorMetadata = 0xFFF59E0B;
constexpr uint32_t kColorForward = 0xFFEF4444;
constexpr uint32_t kColorLayer = 0xFF8B5CF6;
constexpr uint32_t kColorSample = 0xFFEC4899;
constexpr uint32_t kColorProcess = 0xFF14B8A6;
constexpr uint32_t kColorMemcpy = 0xFFF97316;

class ScopedRange {
 public:
  explicit ScopedRange(const char* message, uint32_t color = kColorDefault) {
    begin(message, color);
  }

  explicit ScopedRange(const std::string& message, uint32_t color = kColorDefault)
      : owned_message_(message) {
    begin(owned_message_.c_str(), color);
  }

  ScopedRange(const ScopedRange&) = delete;
  ScopedRange& operator=(const ScopedRange&) = delete;

  ScopedRange(ScopedRange&& other) noexcept
      : pushed_(other.pushed_),
        owned_message_(std::move(other.owned_message_)) {
    other.pushed_ = false;
  }

  ScopedRange& operator=(ScopedRange&& other) noexcept {
    if (this != &other) {
      end();
      pushed_ = other.pushed_;
      owned_message_ = std::move(other.owned_message_);
      other.pushed_ = false;
    }
    return *this;
  }

  ~ScopedRange() { end(); }

 private:
  void begin(const char* message, uint32_t color) {
#if KUIPER_HAVE_NVTX
    nvtxEventAttributes_t event{};
    event.version = NVTX_VERSION;
    event.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    event.colorType = NVTX_COLOR_ARGB;
    event.color = color;
    event.messageType = NVTX_MESSAGE_TYPE_ASCII;
    event.message.ascii = message;
    nvtxRangePushEx(&event);
    pushed_ = true;
#else
    (void)message;
    (void)color;
#endif
  }

  void end() {
#if KUIPER_HAVE_NVTX
    if (pushed_) {
      nvtxRangePop();
      pushed_ = false;
    }
#endif
  }

  bool pushed_ = false;
  std::string owned_message_;
};

inline void Mark(const char* message, uint32_t color = kColorDefault) {
#if KUIPER_HAVE_NVTX
  nvtxEventAttributes_t event{};
  event.version = NVTX_VERSION;
  event.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  event.colorType = NVTX_COLOR_ARGB;
  event.color = color;
  event.messageType = NVTX_MESSAGE_TYPE_ASCII;
  event.message.ascii = message;
  nvtxMarkEx(&event);
#else
  (void)message;
  (void)color;
#endif
}

}  // namespace nvtx
}  // namespace base

#endif  // KUIPER_INCLUDE_BASE_NVTX_UTILS_H_
