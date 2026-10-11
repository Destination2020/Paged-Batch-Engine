#pragma once
#include <csignal>

namespace serving {
// One serving loop per process. Signal handlers only publish a flag; CUDA,
// locks, joins and resource destruction stay on ordinary application threads.
inline volatile std::sig_atomic_t serving_stop_requested = 0;
inline void request_serving_stop(int) { serving_stop_requested = 1; }
class ScopedServingStopSignals {
 public:
  ScopedServingStopSignals() {
    serving_stop_requested = 0;
    previous_int_ = std::signal(SIGINT, request_serving_stop);
    previous_term_ = std::signal(SIGTERM, request_serving_stop);
  }
  ~ScopedServingStopSignals() {
    std::signal(SIGINT, previous_int_);
    std::signal(SIGTERM, previous_term_);
  }
 private:
  using Handler = void (*)(int);
  Handler previous_int_, previous_term_;
};
}  // namespace serving
