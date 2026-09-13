// Per-request generation controls shared by online serving and Scheduler.
#ifndef KUIPER_INCLUDE_SERVING_GENERATION_CONFIG_H_
#define KUIPER_INCLUDE_SERVING_GENERATION_CONFIG_H_

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace serving {

// Counter-based SplitMix64 mapping. Uses only seed and committed generation step;
// batch order, physical request handles and retries do not consume random state.
inline float request_sample_uniform(uint64_t seed, uint64_t counter) {
  uint64_t z = seed + UINT64_C(0x9e3779b97f4a7c15) * (counter + 1);
  z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
  z ^= z >> 31;
  return static_cast<float>(z >> 40) * (1.0f / 16777216.0f);
}

struct SamplingConfig {
  uint64_t seed = 0xC0FFEEu;
  double temperature = 0.0;
  double top_p = 1.0;
  int32_t top_k = 0;
  double repetition_penalty = 1.0;
  std::vector<std::string> stop;

  void normalize() {
    temperature = std::max(0.0, temperature);
    top_p = std::min(1.0, std::max(0.0, top_p));
    top_k = std::max(0, top_k);
    repetition_penalty = std::max(0.0, repetition_penalty);
  }
};

struct GenerationConfig {
  GenerationConfig() = default;

  GenerationConfig(int32_t max_new_tokens_value,
                   int32_t min_new_tokens_value = 0,
                   bool ignore_eos_value = false,
                   int32_t priority_value = 0)
      : max_new_tokens(max_new_tokens_value),
        min_new_tokens(min_new_tokens_value),
        ignore_eos(ignore_eos_value),
        priority(priority_value) {
    normalize();
  }

  int32_t max_new_tokens = 128;
  int32_t min_new_tokens = 0;
  bool ignore_eos = false;
  int32_t priority = 0;
  SamplingConfig sampling;

  void normalize() {
    sampling.normalize();
    max_new_tokens = std::max(1, max_new_tokens);
    min_new_tokens = std::max(0, std::min(min_new_tokens, max_new_tokens));
  }
};

}  // namespace serving

#endif  // KUIPER_INCLUDE_SERVING_GENERATION_CONFIG_H_
