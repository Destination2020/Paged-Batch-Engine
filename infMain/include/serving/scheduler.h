// Continuous batching scheduler
#ifndef KUIPER_INCLUDE_SERVING_SCHEDULER_H_
#define KUIPER_INCLUDE_SERVING_SCHEDULER_H_

#include <cstdint>
#include <deque>
#include <functional>
#include <vector>
#include "base/kv_cache_manager.h"
#include "serving/decode_batch.h"
#include "serving/sequence_state.h"

namespace model {
class Qwen2Model;
}  // namespace model

namespace serving {

class Scheduler {
 public:
  Scheduler(int32_t max_batch_size, base::KVCacheManager* kv_manager);

  // Submit a new request (prompt already encoded)
  void add_request(std::vector<int32_t> prompt_tokens);

  // One scheduling step:
  //   1. Admit waiting requests (prefill via model)
  //   2. Append decode slots for all running sequences
  //   3. Build GPU batch metadata
  DecodeBatchMetadata schedule_step(const model::Qwen2Model& model);

  // Process forward outputs: update sequence states
  void process_outputs(const std::vector<int32_t>& sampled_tokens,
                       const std::function<bool(int32_t)>& is_eos);

  // Pop completed sequences
  std::vector<SequenceState> pop_finished();

  bool has_active_requests() const;

  int32_t num_running() const { return static_cast<int32_t>(running_.size()); }
  int32_t num_waiting() const { return static_cast<int32_t>(waiting_.size()); }

 private:
  // Simple prefill: run model.predict() token-by-token to fill KV cache
  void prefill_sequence(SequenceState& seq, const model::Qwen2Model& model);

  // Build GPU metadata from running_ sequences
  DecodeBatchMetadata build_batch();

  int32_t max_batch_size_;
  base::KVCacheManager* kv_manager_;  // not owned

  std::deque<SequenceState> waiting_;
  std::vector<SequenceState> running_;
  std::vector<SequenceState> finished_;
};

}  // namespace serving

#endif  // KUIPER_INCLUDE_SERVING_SCHEDULER_H_
