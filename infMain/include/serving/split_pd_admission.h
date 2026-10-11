#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>

namespace serving {
// Decode-owner-thread accounting. Separate quotas prevent a long output in one
// lane from consuming the other lane's decode slots or future KV capacity.
class SplitPDAdmission {
 public:
  SplitPDAdmission(std::array<int, 2> slots, std::array<int64_t, 2> tokens)
      : slots_(slots), tokens_(tokens) {}
  static int64_t charge(int64_t prompt, int64_t output, int block_size) {
    return ((prompt + output + block_size - 1) / block_size) * block_size;
  }
  bool fits_request(int lane, int64_t tokens) const {
    return tokens > 0 && tokens <= tokens_.at(lane);
  }
  bool can_acquire(int lane, int64_t tokens) const {
    return fits_request(lane, tokens) && active_.at(lane) < slots_.at(lane) &&
           used_.at(lane) <= tokens_.at(lane) - tokens;
  }
  void acquire(int lane, int64_t tokens) {
    if (!can_acquire(lane, tokens)) throw std::logic_error("split PD admission exceeded");
    ++active_[lane];
    used_[lane] += tokens;
  }
  void release(int lane, int64_t tokens) {
    if (active_.at(lane) <= 0 || used_.at(lane) < tokens)
      throw std::logic_error("split PD admission double release");
    --active_[lane];
    used_[lane] -= tokens;
  }
  int active(int lane) const { return active_.at(lane); }
  int64_t used(int lane) const { return used_.at(lane); }

 private:
  std::array<int, 2> slots_, active_{};
  std::array<int64_t, 2> tokens_, used_{};
};
}  // namespace serving
