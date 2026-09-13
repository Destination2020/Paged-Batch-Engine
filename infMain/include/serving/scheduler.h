// Continuous batching scheduler with token-budget scheduling
#ifndef KUIPER_INCLUDE_SERVING_SCHEDULER_H_
#define KUIPER_INCLUDE_SERVING_SCHEDULER_H_

#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "base/kv_cache_manager.h"
#include "serving/generation_config.h"
#include "serving/mixed_batch.h"
#include "serving/request_checkpoint.h"
#include "serving/sequence_state.h"

namespace serving {

class MixedBatchBuilder;

enum class SchedulingPolicy {
  kFCFS,
  kPriority,
};

enum class PreemptionPolicy { kRecompute, kCheckpoint, kAuto };

struct SchedulerConfig {
  int32_t max_num_seqs = 4;              // max concurrent sequences
  int32_t max_num_batched_tokens = 128;  // token budget per step
  int32_t prefill_chunk_cap = 64;        // max prefill tokens per request per step
  SchedulingPolicy policy = SchedulingPolicy::kFCFS;
  PreemptionPolicy preemption_policy = PreemptionPolicy::kRecompute;
  int32_t long_prefill_token_threshold = 0;  // 0 disables long-prefill throttling
  int32_t max_partial_prefills = 0;          // 0 means no explicit limit
  int32_t max_long_partial_prefills = 0;     // 0 means no explicit limit
  size_t max_checkpoint_records = 64;
  size_t max_checkpoint_bytes = size_t{1} << 30;
  std::string checkpoint_model_namespace = "default";
};

// Scheduler output: what each request should do this step
struct SchedulerMetrics {
  int64_t schedule_steps = 0;
  int64_t no_progress_steps = 0;
  int64_t scheduled_decode_tokens = 0;
  int64_t scheduled_prefill_tokens = 0;
  int64_t scheduled_batched_tokens = 0;
  int64_t preemptions = 0;
  int64_t priority_preemptions = 0;
  int64_t decode_kv_preemptions = 0;
  int64_t checkpoint_preemptions = 0;
  int64_t recompute_preemptions = 0;
  int64_t checkpoint_fallbacks = 0;
  int64_t checkpoint_restores = 0;
  int64_t waiting_rejections = 0;
  int64_t stalled_prefill_failures = 0;
  int64_t cache_transfer_completions = 0;
  int64_t cache_restore_wait_steps = 0;
  int64_t waiting_queue_samples = 0;
  int64_t running_queue_samples = 0;
  int32_t max_waiting_queue = 0;
  int32_t max_running_queue = 0;
};

struct SchedulerOutput {
  // All sequences participating this step (both decode and prefill)
  std::vector<SequenceState*> scheduled_seqs;

  // Per-sequence: how many tokens this step
  // decode seq → 1, prefill seq → chunk_size
  std::vector<int32_t> num_tokens_per_seq;

  int32_t total_tokens = 0;
  int32_t num_decode_seqs = 0;
  int32_t num_prefill_seqs = 0;
  int32_t waiting_queue_size = 0;
  int32_t running_queue_size = 0;
  int32_t preemptions = 0;
};

class Scheduler {
 public:
  explicit Scheduler(const SchedulerConfig& config, base::KVCacheManager* kv_manager);
  ~Scheduler();

  // Submit a new request
  int64_t add_request(std::vector<int32_t> prompt_tokens,
                      GenerationConfig generation_config);

  int64_t add_decode_ready_request(base::RequestId request_id,
                                   std::vector<int32_t> prompt_tokens,
                                   GenerationConfig generation_config,
                                   int32_t computed_tokens,
                                   int32_t first_token);

  bool set_multimodal_checkpoint_state(
      int64_t client_request_id, std::vector<int32_t> positions,
      int64_t rope_delta, int32_t image_begin, int32_t feature_rows,
      int32_t hidden_size, std::string feature_content,
      std::string feature_representation, bool exact_dependency);
  int32_t request_computed_tokens(int64_t client_request_id) const;

  bool cancel_request(int64_t client_request_id, const std::string& reason);
  // Whole-step boundary API: no sequence returned by the previous
  // schedule_step may still be executing when suspend_request is called.
  bool suspend_request(int64_t client_request_id, uint64_t* revision);
  bool restore_request(int64_t client_request_id, uint64_t revision);
  bool checkpoint_manifest(int64_t client_request_id, uint64_t revision,
                           RequestCheckpointManifest* manifest) const;

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
  const SchedulerMetrics& metrics() const { return metrics_; }
  void reset_metrics() { metrics_ = SchedulerMetrics{}; }

 private:
  int32_t total_kv_token_capacity() const;
  int32_t reusable_free_blocks() const;
  void reap_finished_running();
  void reap_preempted_running();
  void preempt_sequence(SequenceState* seq, const std::string& reason);
  void maybe_restore_checkpointed_request();
  void reserve_metadata_capacity_for_request(int32_t prompt_tokens,
                                             const GenerationConfig& generation_config);
  void append_decode_to_output(SequenceState* seq, SchedulerOutput* output);
  void append_prefill_to_output(SequenceState* seq,
                                int32_t chunk,
                                SchedulerOutput* output);
  bool higher_priority(const SequenceState& lhs, const SequenceState& rhs) const;
  void enqueue_waiting_sequence(SequenceState seq);
  bool try_allocate_decode_slot(SequenceState* seq, std::string* preempt_reason);
  int32_t choose_prefill_chunk(const SequenceState& seq, int32_t remaining_budget) const;
  int32_t allocate_prefill_chunk(const SequenceState& seq,
                                 int32_t desired_chunk,
                                 int32_t* remaining_free_blocks) const;
  int32_t running_partial_prefills() const;
  int32_t running_long_partial_prefills() const;
  bool can_schedule_prefill_now(const SequenceState& seq) const;
  bool maybe_preempt_for_waiting_sequence(const SequenceState& waiting_seq,
                                          int32_t* remaining_free_blocks);
  int32_t min_unused_blocks_across_layers() const;
  int32_t estimate_decode_remaining_blocks(const SequenceState& seq) const;
  int32_t committed_running_decode_blocks() const;
  int32_t available_decode_admission_blocks() const;
  bool can_admit_decode_ready_sequence(const SequenceState& seq,
                                       int32_t available_blocks) const;
  void admit_decode_ready_sequence(SequenceState seq);
  void admit_waiting_decode_ready_requests();
  void schedule_decode_sequences(SchedulerOutput* output,
                                 int32_t* remaining_budget);
  void schedule_running_prefills(SchedulerOutput* output,
                                 int32_t* remaining_budget,
                                 int32_t* remaining_free_blocks);
  void admit_waiting_prefills(SchedulerOutput* output,
                              int32_t* remaining_budget,
                              int32_t* remaining_free_blocks);
  void handle_no_progress_step(const SchedulerOutput& output);
  void process_decode_output(SequenceState* seq,
                             int32_t output_index,
                             const MixedBatchMetadata& batch,
                             const SampledTokenView& sampled_tokens,
                             const std::function<bool(int32_t)>& is_eos,
                             std::chrono::steady_clock::time_point now,
                             int32_t* sample_idx);
  void process_prefill_output(SequenceState* seq,
                              int32_t output_index,
                              const MixedBatchMetadata& batch,
                              const SampledTokenView& sampled_tokens,
                              const std::function<bool(int32_t)>& is_eos,
                              std::chrono::steady_clock::time_point now,
                              int32_t* sample_idx);
  void maybe_publish_radix_cache(SequenceState* seq);
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
  RequestCheckpointStore checkpoint_store_;
  std::map<int64_t, CheckpointTicket> suspended_;
  std::map<int64_t, CheckpointTicket> restored_checkpoints_;
  SchedulerMetrics metrics_;
};

}  // namespace serving

#endif  // KUIPER_INCLUDE_SERVING_SCHEDULER_H_
