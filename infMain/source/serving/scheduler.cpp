// Continuous batching scheduler with token-budget scheduling
#include "serving/scheduler.h"
#include <algorithm>
#include <chrono>
#include <glog/logging.h>
#include "serving/mixed_batch_builder.h"

namespace {

int32_t blocks_for_tokens(int32_t num_tokens, int32_t block_size) {
  if (num_tokens <= 0) return 0;
  return (num_tokens + block_size - 1) / block_size;
}

int32_t additional_blocks_needed(int32_t current_tokens,
                                 int32_t appended_tokens,
                                 int32_t block_size) {
  if (appended_tokens <= 0) return 0;
  return blocks_for_tokens(current_tokens + appended_tokens, block_size) -
         blocks_for_tokens(current_tokens, block_size);
}

int32_t cap_prefill_chunk_by_blocks(int32_t current_tokens,
                                    int32_t desired_chunk,
                                    int32_t free_blocks,
                                    int32_t block_size) {
  if (desired_chunk <= 0) return 0;

  const int32_t used_in_last_block = current_tokens % block_size;
  const int32_t room_in_current_block =
      (current_tokens > 0 && used_in_last_block != 0) ? (block_size - used_in_last_block) : 0;
  const int32_t max_tokens_fit = room_in_current_block + free_blocks * block_size;
  return std::min(desired_chunk, max_tokens_fit);
}

int32_t additional_blocks_needed_for_tokens(int32_t current_tokens,
                                            int32_t target_tokens,
                                            int32_t block_size) {
  CHECK_GE(current_tokens, 0);
  CHECK_GE(target_tokens, current_tokens);
  return blocks_for_tokens(target_tokens, block_size) -
         blocks_for_tokens(current_tokens, block_size);
}

void move_finished_sequences_to_output(std::deque<serving::SequenceState>* running,
                                       std::vector<serving::SequenceState>* finished,
                                       base::KVCacheManager* kv_manager) {
  CHECK_NE(running, nullptr);
  CHECK_NE(finished, nullptr);
  CHECK_NE(kv_manager, nullptr);

  auto it = running->begin();
  while (it != running->end()) {
    if (it->finished) {
      kv_manager->free_request(it->request_id);
      finished->push_back(std::move(*it));
      it = running->erase(it);
    } else {
      ++it;
    }
  }
}

}  // namespace

namespace serving {

namespace {

void finish_sequence(SequenceState* seq, std::chrono::steady_clock::time_point now) {
  CHECK_NE(seq, nullptr);
  seq->next_token = -1;
  seq->mark_finished(now);
}

void fail_sequence(SequenceState* seq,
                   std::chrono::steady_clock::time_point now,
                   const std::string& reason) {
  CHECK_NE(seq, nullptr);
  seq->next_token = -1;
  seq->mark_failed(now, reason);
}

void record_sampled_token(SequenceState* seq,
                          int32_t token,
                          const std::function<bool(int32_t)>& is_eos,
                          std::chrono::steady_clock::time_point now) {
  CHECK_NE(seq, nullptr);
  seq->record_generated_token(now);
  const bool token_is_eos = is_eos(token);
  const bool should_stop_on_eos = !seq->ignore_eos &&
                                  seq->generated_tokens > seq->min_new_tokens &&
                                  token_is_eos;
  if (should_stop_on_eos) {
    finish_sequence(seq, now);
    return;
  }

  seq->next_token = token;
  if (!token_is_eos) {
    seq->output_tokens.push_back(token);
  }
  if (seq->generated_tokens >= seq->max_new_tokens) {
    finish_sequence(seq, now);
  }
}

}  // namespace

Scheduler::Scheduler(const SchedulerConfig& config, base::KVCacheManager* kv_manager)
    : config_(config), kv_manager_(kv_manager) {
  CHECK_NE(kv_manager_, nullptr);
  CHECK_GT(config_.max_num_seqs, 0);
  CHECK_GT(config_.max_num_batched_tokens, 0);
  mixed_batch_builder_ = std::make_unique<MixedBatchBuilder>(kv_manager_);
  max_request_blocks_hint_ = std::max(
      1, blocks_for_tokens(config_.prefill_chunk_cap, kv_manager_->block_size()));
  mixed_batch_builder_->reserve_metadata_capacity(
      config_.max_num_batched_tokens,
      config_.max_num_seqs,
      config_.max_num_seqs,
      config_.max_num_batched_tokens,
      config_.max_num_seqs * max_request_blocks_hint_);
}

Scheduler::~Scheduler() = default;

int64_t Scheduler::add_request(std::vector<int32_t> prompt_tokens,
                               int32_t max_new_tokens,
                               int32_t min_new_tokens,
                               bool ignore_eos) {
  const auto now = std::chrono::steady_clock::now();
  const int32_t total_tokens_hint = static_cast<int32_t>(prompt_tokens.size()) +
                                    std::max(1, max_new_tokens);
  max_request_blocks_hint_ = std::max(
      max_request_blocks_hint_,
      std::max(1, blocks_for_tokens(total_tokens_hint, kv_manager_->block_size())));
  CHECK(mixed_batch_builder_ != nullptr);
  mixed_batch_builder_->reserve_metadata_capacity(
      config_.max_num_batched_tokens,
      config_.max_num_seqs,
      config_.max_num_seqs,
      config_.max_num_batched_tokens,
      config_.max_num_seqs * max_request_blocks_hint_);

  SequenceState seq;
  seq.request_id = kv_manager_->register_request_with_radix_cache(prompt_tokens);
  seq.client_request_id = next_client_request_id_++;
  const int64_t client_request_id = seq.client_request_id;
  seq.prompt_tokens = std::move(prompt_tokens);
  seq.max_new_tokens = std::max(1, max_new_tokens);
  seq.min_new_tokens = std::max(0, std::min(min_new_tokens, seq.max_new_tokens));
  seq.ignore_eos = ignore_eos;
  seq.generated_tokens = 0;
  seq.next_token = -1;
  seq.finished = false;
  seq.failed = false;
  seq.first_token_recorded = false;
  seq.finished_time_recorded = false;
  seq.recompute_pending = false;
  seq.radix_cache_published = false;
  seq.status = SequenceStatus::kWaiting;
  seq.finish_reason.clear();
  seq.last_preempt_reason.clear();
  seq.preemption_count = 0;
  seq.ready_step = 0;
  seq.arrival_time = std::chrono::steady_clock::now();
  seq.num_prompt_tokens_computed = kv_manager_->get_context_len(seq.request_id);
  seq.scheduled_tokens = 0;

  const int32_t kv_capacity_tokens = total_kv_token_capacity();
  if (seq.remaining_prompt_tokens() > kv_capacity_tokens) {
    fail_sequence(&seq, now,
                  "prompt_exceeds_kv_capacity(prompt_tokens=" +
                      std::to_string(seq.prompt_tokens.size()) +
                      ", kv_capacity_tokens=" +
                      std::to_string(kv_capacity_tokens) + ")");
    kv_manager_->free_request(seq.request_id);
    finished_.push_back(std::move(seq));
    return client_request_id;
  }

  waiting_.push_back(std::move(seq));
  return client_request_id;
}

bool Scheduler::cancel_request(int64_t client_request_id, const std::string& reason) {
  const auto now = std::chrono::steady_clock::now();
  for (auto it = waiting_.begin(); it != waiting_.end(); ++it) {
    if (it->client_request_id != client_request_id) {
      continue;
    }
    fail_sequence(&*it, now, reason);
    kv_manager_->free_request(it->request_id);
    finished_.push_back(std::move(*it));
    waiting_.erase(it);
    return true;
  }

  for (auto& seq : running_) {
    if (seq.client_request_id != client_request_id) {
      continue;
    }
    fail_sequence(&seq, now, reason);
    reap_finished_running();
    return true;
  }
  return false;
}

SchedulerOutput Scheduler::schedule_step() {
  ++scheduler_step_;
  reap_finished_running();
  reap_preempted_running();

  SchedulerOutput output;
  int32_t remaining_budget = config_.max_num_batched_tokens;
  int32_t remaining_free_blocks = 0;

  // Phase 1: Reserve 1 token per running decode sequence
  for (auto& seq : running_) {
    if (remaining_budget <= 0) break;
    if (!seq.is_prefill()) {
      // Decode seq: 1 token
      // Append 1 KV slot for the decode token
      bool ok = kv_manager_->append_slot(seq.request_id);
      if (!ok) {
        preempt_sequence(&seq,
                         "decode_kv_exhausted(context_len=" +
                             std::to_string(kv_manager_->get_context_len(seq.request_id)) +
                             ", kv_capacity_tokens=" +
                             std::to_string(total_kv_token_capacity()) + ")");
        continue;
      }
      seq.status = SequenceStatus::kRunning;
      seq.scheduled_tokens = 1;
      output.scheduled_seqs.push_back(&seq);
      output.num_tokens_per_seq.push_back(1);
      output.num_decode_seqs++;
      output.total_tokens++;
      remaining_budget--;
    }
  }

  remaining_free_blocks =
      kv_manager_->num_free_blocks(0) + kv_manager_->radix_cache_evictable_blocks();

  // Phase 2: Allocate budget to running prefill sequences (continuing chunks)
  for (auto& seq : running_) {
    if (remaining_budget <= 0) break;
    if (seq.status == SequenceStatus::kPreempted) continue;
    if (seq.is_prefill()) {
      const int32_t desired_chunk = std::min({
          seq.remaining_prompt_tokens(),
          remaining_budget,
          config_.prefill_chunk_cap});
      const int32_t current_tokens = kv_manager_->get_context_len(seq.request_id);
      int32_t chunk = cap_prefill_chunk_by_blocks(
          current_tokens, desired_chunk, remaining_free_blocks, kv_manager_->block_size());
      if (chunk <= 0) continue;
      remaining_free_blocks -= additional_blocks_needed(
          current_tokens, chunk, kv_manager_->block_size());

      // Don't allocate KV slots here — prefill_chunk handles it one-at-a-time
      // to ensure attention only sees already-written positions
      seq.scheduled_tokens = chunk;
      seq.status = SequenceStatus::kRunning;
      output.scheduled_seqs.push_back(&seq);
      output.num_tokens_per_seq.push_back(chunk);
      output.num_prefill_seqs++;
      output.total_tokens += chunk;
      remaining_budget -= chunk;
    }
  }

  // Phase 3: Admit new waiting requests if budget and seq slots remain
  while (!waiting_.empty() && remaining_budget > 0 &&
         static_cast<int32_t>(running_.size()) < config_.max_num_seqs) {
    auto& seq = waiting_.front();
    if (seq.ready_step > scheduler_step_) {
      break;
    }

    // Cap the first prefill chunk by both token budget and block budget.
    const int32_t desired_chunk = std::min({
        seq.remaining_prompt_tokens(),
        remaining_budget,
        config_.prefill_chunk_cap});
    int32_t chunk = 0;
    if (!can_admit_waiting_sequence(seq, desired_chunk, remaining_free_blocks, &chunk)) {
      break;
    }
    if (chunk <= 0) break;
    const int32_t current_tokens = kv_manager_->get_context_len(seq.request_id);
    remaining_free_blocks -= additional_blocks_needed(current_tokens, chunk,
                                                      kv_manager_->block_size());

    // Move to running. Prefill will allocate/write KV slots token-by-token.
    running_.push_back(std::move(seq));
    waiting_.pop_front();
    auto& admitted = running_.back();

    admitted.status = SequenceStatus::kRunning;
    admitted.scheduled_tokens = chunk;
    output.scheduled_seqs.push_back(&admitted);
    output.num_tokens_per_seq.push_back(chunk);
    output.num_prefill_seqs++;
    output.total_tokens += chunk;
    remaining_budget -= chunk;
  }

  if (output.total_tokens == 0) {
    reap_finished_running();
    reap_preempted_running();
    if (reject_waiting_request_that_cannot_start()) {
      return output;
    }
    if (fail_stalled_prefill_request()) {
      return output;
    }
  }

  return output;
}

void Scheduler::process_outputs(const SchedulerOutput& output,
                                const MixedBatchMetadata& batch,
                                const SampledTokenView& sampled_tokens,
                                const std::function<bool(int32_t)>& is_eos) {
  CHECK_EQ(sampled_tokens.size(), static_cast<int32_t>(batch.sample_row_to_request.size()));

  const auto now = std::chrono::steady_clock::now();
  int32_t sample_idx = 0;
  for (size_t i = 0; i < output.scheduled_seqs.size(); ++i) {
    auto* seq = output.scheduled_seqs[i];

    if (static_cast<int32_t>(i) < output.num_decode_seqs) {
      // Decode sequence: update with sampled token
      CHECK_LT(sample_idx, sampled_tokens.size());
      CHECK_EQ(batch.sample_row_to_request[sample_idx], static_cast<int32_t>(i));
      int32_t token = sampled_tokens[sample_idx++];
      record_sampled_token(seq, token, is_eos, now);
    } else {
      // Prefill sequence: advance prompt progress
      seq->num_prompt_tokens_computed += seq->scheduled_tokens;
      CHECK_LE(seq->num_prompt_tokens_computed,
               seq->prefill_target_tokens());

      const bool prompt_finished =
          seq->num_prompt_tokens_computed == seq->prefill_target_tokens();
      if (prompt_finished && seq->recompute_pending) {
        seq->recompute_pending = false;
      }
      if (prompt_finished && !seq->recompute_pending &&
          !seq->radix_cache_published && seq->output_tokens.empty()) {
        kv_manager_->publish_radix_cache(seq->request_id, seq->prompt_tokens);
        seq->radix_cache_published = true;
      }
      if (prompt_finished && sample_idx < sampled_tokens.size() &&
          batch.sample_row_to_request[sample_idx] == static_cast<int32_t>(i)) {
        int32_t token = sampled_tokens[sample_idx++];
        record_sampled_token(seq, token, is_eos, now);
      }
    }
    seq->scheduled_tokens = 0;
  }

  CHECK_EQ(sample_idx, sampled_tokens.size());

  reap_finished_running();
}

std::vector<SequenceState> Scheduler::pop_finished() {
  std::vector<SequenceState> result;
  result.swap(finished_);
  return result;
}

void Scheduler::reset_request_arrival_times() {
  const auto now = std::chrono::steady_clock::now();
  for (auto& seq : waiting_) {
    seq.arrival_time = now;
    seq.first_token_recorded = false;
    seq.finished_time_recorded = false;
    seq.generated_tokens = 0;
    seq.failed = false;
    if (seq.status == SequenceStatus::kFailed) {
      seq.status = SequenceStatus::kWaiting;
    }
    if (!seq.recompute_pending) {
      seq.finish_reason.clear();
    }
  }
  for (auto& seq : running_) {
    seq.arrival_time = now;
    seq.first_token_recorded = false;
    seq.finished_time_recorded = false;
    seq.generated_tokens = 0;
    seq.failed = false;
    if (seq.status == SequenceStatus::kFailed) {
      seq.status = SequenceStatus::kRunning;
    }
    if (!seq.recompute_pending) {
      seq.finish_reason.clear();
    }
  }
}

bool Scheduler::has_active_requests() const {
  return !waiting_.empty() || !running_.empty();
}

int32_t Scheduler::total_kv_token_capacity() const {
  return std::max(0, kv_manager_->num_total_blocks(0)) * kv_manager_->block_size();
}

void Scheduler::reap_finished_running() {
  move_finished_sequences_to_output(&running_, &finished_, kv_manager_);
}

void Scheduler::reap_preempted_running() {
  auto it = running_.begin();
  while (it != running_.end()) {
    if (it->status == SequenceStatus::kPreempted) {
      waiting_.push_back(std::move(*it));
      it = running_.erase(it);
    } else {
      ++it;
    }
  }
}

void Scheduler::preempt_sequence(SequenceState* seq, const std::string& reason) {
  CHECK_NE(seq, nullptr);
  CHECK(!seq->finished);
  CHECK(!seq->failed);

  kv_manager_->free_request(seq->request_id);
  seq->request_id = kv_manager_->register_request_with_radix_cache(seq->prompt_tokens);
  seq->num_prompt_tokens_computed = kv_manager_->get_context_len(seq->request_id);
  seq->scheduled_tokens = 0;
  seq->next_token = -1;
  seq->recompute_pending = true;
  seq->radix_cache_published = false;
  seq->status = SequenceStatus::kPreempted;
  seq->last_preempt_reason = reason;
  seq->preemption_count++;
  seq->ready_step = scheduler_step_ + 1;
  LOG(WARNING) << "Preempted request " << seq->client_request_id
               << ": " << reason
               << ", recompute_tokens=" << seq->prefill_target_tokens()
               << ", preemption_count=" << seq->preemption_count;
}

bool Scheduler::can_admit_waiting_sequence(const SequenceState& seq,
                                           int32_t desired_chunk,
                                           int32_t free_blocks,
                                           int32_t* chunk) const {
  CHECK_NE(chunk, nullptr);
  *chunk = 0;
  if (desired_chunk <= 0) {
    return false;
  }

  const int32_t current_tokens = kv_manager_->get_context_len(seq.request_id);
  if (seq.recompute_pending) {
    const int32_t target_tokens = seq.prefill_target_tokens();
    CHECK_LE(current_tokens, target_tokens);

    // Recomputed requests need enough room to rebuild their full previous
    // context and append one future decode token. Without this stricter gate,
    // a preempted request can immediately re-enter running, consume all freed
    // blocks during recompute, then get preempted again on the next decode.
    const int32_t required_blocks =
        additional_blocks_needed_for_tokens(current_tokens, target_tokens + 1,
                                            kv_manager_->block_size());
    if (required_blocks > free_blocks) {
      return false;
    }
  }

  *chunk = cap_prefill_chunk_by_blocks(current_tokens, desired_chunk, free_blocks,
                                       kv_manager_->block_size());
  return *chunk > 0;
}

bool Scheduler::reject_waiting_request_that_cannot_start() {
  if (waiting_.empty()) {
    return false;
  }
  auto& seq = waiting_.front();
  if (seq.ready_step > scheduler_step_) {
    return false;
  }
  const int32_t desired_chunk = std::min({
      seq.remaining_prompt_tokens(),
      config_.max_num_batched_tokens,
      config_.prefill_chunk_cap});
  int32_t chunk = 0;
  if (can_admit_waiting_sequence(
          seq, desired_chunk,
          kv_manager_->num_free_blocks(0) + kv_manager_->radix_cache_evictable_blocks(),
          &chunk)) {
    return false;
  }

  if (seq.recompute_pending) {
    const int32_t required_tokens = seq.prefill_target_tokens() + 1;
    if (required_tokens <= total_kv_token_capacity()) {
      return false;
    }
  }

  auto failed = std::move(seq);
  waiting_.pop_front();
  std::string reason;
  if (failed.recompute_pending) {
    reason = "preempted_recompute_exceeds_kv_capacity(required_tokens=" +
             std::to_string(failed.prefill_target_tokens() + 1) +
             ", kv_capacity_tokens=" + std::to_string(total_kv_token_capacity()) + ")";
  } else {
    const int32_t reusable_free_blocks =
        kv_manager_->num_free_blocks(0) + kv_manager_->radix_cache_evictable_blocks();
    reason = "prefill_cannot_start(remaining_prompt_tokens=" +
             std::to_string(failed.remaining_prompt_tokens()) +
             ", free_kv_blocks=" + std::to_string(kv_manager_->num_free_blocks(0)) +
             ", radix_evictable_blocks=" +
             std::to_string(kv_manager_->radix_cache_evictable_blocks()) +
             ", reusable_kv_blocks=" + std::to_string(reusable_free_blocks) + ")";
  }
  fail_sequence(&failed, std::chrono::steady_clock::now(), reason);
  kv_manager_->free_request(failed.request_id);
  finished_.push_back(std::move(failed));
  return true;
}

bool Scheduler::fail_stalled_prefill_request() {
  for (auto& seq : running_) {
    if (!seq.is_prefill()) {
      continue;
    }
    const int32_t current_tokens = kv_manager_->get_context_len(seq.request_id);
    const int32_t desired_chunk = std::min({
        seq.remaining_prompt_tokens(),
        config_.max_num_batched_tokens,
        config_.prefill_chunk_cap});
    const int32_t chunk = cap_prefill_chunk_by_blocks(
        current_tokens, desired_chunk,
        kv_manager_->num_free_blocks(0) + kv_manager_->radix_cache_evictable_blocks(),
        kv_manager_->block_size());
    if (chunk > 0) {
      continue;
    }

    fail_sequence(&seq, std::chrono::steady_clock::now(),
                  "prefill_stalled_kv_exhausted(context_len=" +
                      std::to_string(current_tokens) +
                      ", remaining_prompt_tokens=" +
                      std::to_string(seq.remaining_prompt_tokens()) +
                      ", free_kv_blocks=" +
                      std::to_string(kv_manager_->num_free_blocks(0)) +
                      ", radix_evictable_blocks=" +
                      std::to_string(kv_manager_->radix_cache_evictable_blocks()) + ")");
    reap_finished_running();
    return true;
  }
  return false;
}

MixedBatchMetadata Scheduler::build_mixed_batch(const SchedulerOutput& output,
                                                void* stream) const {
  CHECK(mixed_batch_builder_ != nullptr);
  if (output.num_prefill_seqs == 0 && output.num_decode_seqs > 0) {
    return mixed_batch_builder_->build_decode(output, stream);
  }
  return mixed_batch_builder_->build(output, stream);
}

MixedBatchMetadata Scheduler::build_decode_batch(const SchedulerOutput& output,
                                                 void* stream) const {
  CHECK(mixed_batch_builder_ != nullptr);
  return mixed_batch_builder_->build_decode(output, stream);
}

}  // namespace serving
