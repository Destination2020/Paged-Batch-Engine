// Per-request sequence state for continuous batching
#ifndef KUIPER_INCLUDE_SERVING_SEQUENCE_STATE_H_
#define KUIPER_INCLUDE_SERVING_SEQUENCE_STATE_H_

#include <cstdint>
#include <vector>
#include "base/kv_cache_manager.h"

namespace serving {

struct SequenceState {
  base::RequestId request_id = -1;
  std::vector<int32_t> prompt_tokens;
  std::vector<int32_t> output_tokens;
  int32_t next_token = -1;
  bool finished = false;

  int32_t context_len() const {
    return static_cast<int32_t>(prompt_tokens.size() + output_tokens.size());
  }
  bool prefill_done() const { return next_token != -1; }
};

}  // namespace serving

#endif  // KUIPER_INCLUDE_SERVING_SEQUENCE_STATE_H_
