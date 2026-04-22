// Continuous batching scheduler with token-budget scheduling
#ifndef KUIPER_INCLUDE_SERVING_SCHEDULER_H_
#define KUIPER_INCLUDE_SERVING_SCHEDULER_H_

#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <vector>
#include "base/kv_cache_manager.h"
#include "serving/mixed_batch.h"
#include "serving/sequence_state.h"

namespace serving {

class MixedBatchBuilder;

struct SchedulerConfig {
  int32_t max_num_seqs = 4;              // max concurrent sequences
  int32_t max_num_batched_tokens = 128;  // token budget per step
  int32_t prefill_chunk_cap = 64;        // max prefill tokens per request per step
};

// Scheduler output: what each request should do this step
struct SchedulerOutput {
  // All sequences participating this step (both decode and prefill)
  std::vector<SequenceState*> scheduled_seqs;

  // Per-sequence: how many tokens this step
  // decode seq → 1, prefill seq → chunk_size
  std::vector<int32_t> num_tokens_per_seq;

  int32_t total_tokens = 0;
  int32_t num_decode_seqs = 0;
  int32_t num_prefill_seqs = 0;
};

class Scheduler {
 public:
  explicit Scheduler(const SchedulerConfig& config, base::KVCacheManager* kv_manager);
  ~Scheduler();

  // Submit a new request
  void add_request(std::vector<int32_t> prompt_tokens, int32_t max_new_tokens);

  // One scheduling step:
  //   1. Allocate token budget to running (decode) sequences
  //   2. Allocate remaining budget to waiting/prefilling sequences
  //   3. Return SchedulerOutput describing what to do
  SchedulerOutput schedule_step();

  // After forward + sampling: update sequence states
  void process_outputs(const SchedulerOutput& output,
                       const MixedBatchMetadata& batch,
                       const SampledTokenView& sampled_tokens,
                       const std::function<bool(int32_t)>& is_eos);

  // Build full mixed-batch metadata (decode rows first, then prefill rows).
  MixedBatchMetadata build_mixed_batch(const SchedulerOutput& output,
                                       void* stream = nullptr) const;

  // Backward-compatible helper for the current decode-only execution path.
  MixedBatchMetadata build_decode_batch(const SchedulerOutput& output,
                                        void* stream = nullptr) const;

  // Retrieve completed sequences
  std::vector<SequenceState> pop_finished();

  // Reset latency accounting start time after offline request submission.
  void reset_request_arrival_times();

  bool has_active_requests() const;

 private:
  int32_t total_kv_token_capacity() const;
  void reap_finished_running();
  void reap_preempted_running();
  void preempt_sequence(SequenceState* seq, const std::string& reason);
  bool can_admit_waiting_sequence(const SequenceState& seq,
                                  int32_t desired_chunk,
                                  int32_t free_blocks,
                                  int32_t* chunk) const;
  bool reject_waiting_request_that_cannot_start();
  bool fail_stalled_prefill_request();

  SchedulerConfig config_;
  base::KVCacheManager* kv_manager_;
  mutable std::unique_ptr<MixedBatchBuilder> mixed_batch_builder_;
  int64_t next_client_request_id_ = 0;
  int32_t max_request_blocks_hint_ = 1;
  int64_t scheduler_step_ = 0;

  // waiting_: not yet started prefill (or partially prefilled, paused)
  // running_: all active sequences (both prefilling and decoding).
  // Use deque so pointers stored in SchedulerOutput stay valid across push_back.
  // finished_: completed sequences ready for pickup
  std::deque<SequenceState> waiting_;
  std::deque<SequenceState> running_;
  std::vector<SequenceState> finished_;
};

}  // namespace serving

#endif  // KUIPER_INCLUDE_SERVING_SCHEDULER_H_
