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
  const auto& generation_config = seq->generation_config;
  const bool should_stop_on_eos = !generation_config.ignore_eos &&
                                  seq->generated_tokens > generation_config.min_new_tokens &&
                                  token_is_eos;
  if (should_stop_on_eos) {
    finish_sequence(seq, now);
    return;
  }

  seq->next_token = token;
  if (!token_is_eos) {
    seq->output_tokens.push_back(token);
  }
  if (seq->generated_tokens >= generation_config.max_new_tokens) {
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
                               GenerationConfig generation_config) {
  generation_config.normalize();
  const auto now = std::chrono::steady_clock::now();
  reserve_metadata_capacity_for_request(
      static_cast<int32_t>(prompt_tokens.size()), generation_config);

  SequenceState seq;
  seq.request_id = kv_manager_->register_request_with_radix_cache(prompt_tokens);
  seq.client_request_id = next_client_request_id_++;
  const int64_t client_request_id = seq.client_request_id;
  seq.prompt_tokens = std::move(prompt_tokens);
  seq.generation_config = generation_config;
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
  seq.computed_tokens = kv_manager_->get_context_len(seq.request_id);
  seq.scheduled_tokens = 0;

  const int32_t kv_capacity_tokens = total_kv_token_capacity();
  if (seq.remaining_tokens() > kv_capacity_tokens) {
    fail_sequence(&seq, now,
                  "prompt_exceeds_kv_capacity(prompt_tokens=" +
                      std::to_string(seq.prompt_tokens.size()) +
                      ", kv_capacity_tokens=" +
                      std::to_string(kv_capacity_tokens) + ")");
    kv_manager_->free_request(seq.request_id);
    finished_.push_back(std::move(seq));
    return client_request_id;
  }

  enqueue_waiting_sequence(std::move(seq));
  return client_request_id;
}

int64_t Scheduler::add_decode_ready_request(base::RequestId request_id,
                                            std::vector<int32_t> prompt_tokens,
                                            GenerationConfig generation_config,
                                            int32_t computed_tokens,
                                            int32_t first_token) {
  generation_config.normalize();
  const auto now = std::chrono::steady_clock::now();
  if (first_token < 0) {
    SequenceState failed;
    failed.request_id = request_id;
    failed.client_request_id = next_client_request_id_++;
    const int64_t client_request_id = failed.client_request_id;
    failed.prompt_tokens = std::move(prompt_tokens);
    failed.generation_config = generation_config;
    failed.arrival_time = now;
    fail_sequence(&failed, now, "decode_ready_missing_first_token");
    if (kv_manager_->is_valid_request(request_id)) {
      kv_manager_->free_request(request_id);
    }
    finished_.push_back(std::move(failed));
    return client_request_id;
  }
  if (!kv_manager_->is_valid_request(request_id)) {
    SequenceState failed;
    failed.request_id = request_id;
    failed.client_request_id = next_client_request_id_++;
    const int64_t client_request_id = failed.client_request_id;
    failed.prompt_tokens = std::move(prompt_tokens);
    failed.generation_config = generation_config;
    failed.arrival_time = now;
    fail_sequence(&failed, now, "decode_ready_invalid_kv_request");
    finished_.push_back(std::move(failed));
    return client_request_id;
  }

  reserve_metadata_capacity_for_request(
      static_cast<int32_t>(prompt_tokens.size()), generation_config);

  SequenceState seq;
  seq.request_id = request_id;
  seq.client_request_id = next_client_request_id_++;
  const int64_t client_request_id = seq.client_request_id;
  seq.prompt_tokens = std::move(prompt_tokens);
  seq.generation_config = generation_config;
  seq.generated_tokens = 1;
  seq.next_token = first_token;
  seq.output_tokens.push_back(first_token);
  seq.finished = false;
  seq.failed = false;
  seq.first_token_recorded = true;
  seq.finished_time_recorded = false;
  seq.recompute_pending = false;
  seq.radix_cache_published = true;
  seq.status = SequenceStatus::kRunning;
  seq.finish_reason.clear();
  seq.last_preempt_reason.clear();
  seq.preemption_count = 0;
  seq.ready_step = 0;
  seq.arrival_time = now;
  seq.first_token_time = now;
  seq.last_token_time = now;
  seq.computed_tokens = computed_tokens;
  seq.scheduled_tokens = 0;

  if (seq.computed_tokens != seq.target_tokens() ||
      kv_manager_->get_context_len(seq.request_id) != seq.computed_tokens) {
    fail_sequence(&seq, now,
                  "decode_ready_context_mismatch(computed_tokens=" +
                      std::to_string(seq.computed_tokens) +
                      ", target_tokens=" + std::to_string(seq.target_tokens()) +
                      ", kv_context_len=" +
                      std::to_string(kv_manager_->get_context_len(seq.request_id)) + ")");
    kv_manager_->free_request(seq.request_id);
    finished_.push_back(std::move(seq));
    return client_request_id;
  }

  if (seq.generated_tokens >= seq.generation_config.max_new_tokens) {
    finish_sequence(&seq, now);
    kv_manager_->free_request(seq.request_id);
    finished_.push_back(std::move(seq));
    return client_request_id;
  }

  running_.push_back(std::move(seq));
  metrics_.max_running_queue = std::max(
      metrics_.max_running_queue, static_cast<int32_t>(running_.size()));
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
  ++metrics_.schedule_steps;
  reap_finished_running();
  reap_preempted_running();

  SchedulerOutput output;
  const int64_t preemptions_before_step = metrics_.preemptions;
  int32_t remaining_budget = config_.max_num_batched_tokens;
  int32_t remaining_free_blocks = 0;

  schedule_decode_sequences(&output, &remaining_budget);
  remaining_free_blocks = reusable_free_blocks();
  schedule_running_prefills(&output, &remaining_budget, &remaining_free_blocks);
  admit_waiting_prefills(&output, &remaining_budget, &remaining_free_blocks);
  handle_no_progress_step(output);

  output.waiting_queue_size = static_cast<int32_t>(waiting_.size());
  output.running_queue_size = static_cast<int32_t>(running_.size());
  output.preemptions = static_cast<int32_t>(metrics_.preemptions - preemptions_before_step);
  metrics_.scheduled_decode_tokens += output.num_decode_seqs;
  metrics_.scheduled_prefill_tokens += output.total_tokens - output.num_decode_seqs;
  metrics_.scheduled_batched_tokens += output.total_tokens;
  metrics_.waiting_queue_samples += output.waiting_queue_size;
  metrics_.running_queue_samples += output.running_queue_size;
  metrics_.max_waiting_queue = std::max(metrics_.max_waiting_queue, output.waiting_queue_size);
  metrics_.max_running_queue = std::max(metrics_.max_running_queue, output.running_queue_size);
  if (output.total_tokens == 0) {
    ++metrics_.no_progress_steps;
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
    const int32_t output_index = static_cast<int32_t>(i);
    if (output_index < output.num_decode_seqs) {
      process_decode_output(seq, output_index, batch, sampled_tokens, is_eos, now,
                            &sample_idx);
    } else {
      process_prefill_output(seq, output_index, batch, sampled_tokens, is_eos, now,
                             &sample_idx);
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

int32_t Scheduler::reusable_free_blocks() const {
  return kv_manager_->num_free_blocks(0) + kv_manager_->radix_cache_evictable_blocks();
}

void Scheduler::reserve_metadata_capacity_for_request(
    int32_t prompt_tokens, const GenerationConfig& generation_config) {
  const int32_t total_tokens_hint = prompt_tokens + generation_config.max_new_tokens;
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
}

void Scheduler::append_decode_to_output(SequenceState* seq,
                                        SchedulerOutput* output) {
  CHECK_NE(seq, nullptr);
  CHECK_NE(output, nullptr);
  seq->status = SequenceStatus::kRunning;
  seq->scheduled_tokens = 1;
  output->scheduled_seqs.push_back(seq);
  output->num_tokens_per_seq.push_back(1);
  output->num_decode_seqs++;
  output->total_tokens++;
}

void Scheduler::append_prefill_to_output(SequenceState* seq,
                                         int32_t chunk,
                                         SchedulerOutput* output) {
  CHECK_NE(seq, nullptr);
  CHECK_NE(output, nullptr);
  CHECK_GT(chunk, 0);
  seq->status = SequenceStatus::kRunning;
  seq->scheduled_tokens = chunk;
  output->scheduled_seqs.push_back(seq);
  output->num_tokens_per_seq.push_back(chunk);
  output->num_prefill_seqs++;
  output->total_tokens += chunk;
}

bool Scheduler::higher_priority(const SequenceState& lhs,
                                const SequenceState& rhs) const {
  if (lhs.generation_config.priority != rhs.generation_config.priority) {
    return lhs.generation_config.priority > rhs.generation_config.priority;
  }
  return lhs.client_request_id < rhs.client_request_id;
}

void Scheduler::enqueue_waiting_sequence(SequenceState seq) {
  if (config_.policy == SchedulingPolicy::kFCFS) {
    waiting_.push_back(std::move(seq));
    return;
  }

  auto it = waiting_.begin();
  while (it != waiting_.end() && !higher_priority(seq, *it)) {
    ++it;
  }
  waiting_.insert(it, std::move(seq));
}

bool Scheduler::try_allocate_decode_slot(SequenceState* seq,
                                         std::string* preempt_reason) {
  CHECK_NE(seq, nullptr);
  CHECK_NE(preempt_reason, nullptr);
  if (kv_manager_->append_slot(seq->request_id)) {
    return true;
  }
  *preempt_reason = "decode_kv_exhausted(context_len=" +
                    std::to_string(kv_manager_->get_context_len(seq->request_id)) +
                    ", kv_capacity_tokens=" +
                    std::to_string(total_kv_token_capacity()) + ")";
  return false;
}

int32_t Scheduler::choose_prefill_chunk(const SequenceState& seq,
                                        int32_t remaining_budget) const {
  int32_t chunk_cap = config_.prefill_chunk_cap;
  if (config_.long_prefill_token_threshold > 0 &&
      seq.remaining_tokens() > config_.long_prefill_token_threshold) {
    chunk_cap = std::min(chunk_cap, config_.long_prefill_token_threshold);
  }
  return std::min({seq.remaining_tokens(), remaining_budget, chunk_cap});
}

int32_t Scheduler::allocate_prefill_chunk(const SequenceState& seq,
                                          int32_t desired_chunk,
                                          int32_t* remaining_free_blocks) const {
  CHECK_NE(remaining_free_blocks, nullptr);
  const int32_t current_tokens = kv_manager_->get_context_len(seq.request_id);
  const int32_t chunk = cap_prefill_chunk_by_blocks(
      current_tokens, desired_chunk, *remaining_free_blocks, kv_manager_->block_size());
  if (chunk <= 0) {
    return 0;
  }
  *remaining_free_blocks -= additional_blocks_needed(
      current_tokens, chunk, kv_manager_->block_size());
  return chunk;
}

int32_t Scheduler::running_partial_prefills() const {
  int32_t count = 0;
  for (const auto& seq : running_) {
    if (seq.is_prefill()) {
      ++count;
    }
  }
  return count;
}

int32_t Scheduler::running_long_partial_prefills() const {
  if (config_.long_prefill_token_threshold <= 0) {
    return 0;
  }
  int32_t count = 0;
  for (const auto& seq : running_) {
    if (seq.is_prefill() &&
        seq.remaining_tokens() > config_.long_prefill_token_threshold) {
      ++count;
    }
  }
  return count;
}

bool Scheduler::can_schedule_prefill_now(const SequenceState& seq) const {
  if (!seq.is_prefill()) {
    return false;
  }
  if (config_.max_partial_prefills > 0 &&
      running_partial_prefills() >= config_.max_partial_prefills &&
      seq.status != SequenceStatus::kRunning) {
    return false;
  }
  if (config_.max_long_partial_prefills > 0 &&
      config_.long_prefill_token_threshold > 0 &&
      seq.remaining_tokens() > config_.long_prefill_token_threshold &&
      running_long_partial_prefills() >= config_.max_long_partial_prefills &&
      seq.status != SequenceStatus::kRunning) {
    return false;
  }
  return true;
}

bool Scheduler::maybe_preempt_for_waiting_sequence(
    const SequenceState& waiting_seq, int32_t* remaining_free_blocks) {
  CHECK_NE(remaining_free_blocks, nullptr);
  if (config_.policy != SchedulingPolicy::kPriority) {
    return false;
  }

  auto victim = running_.end();
  for (auto it = running_.begin(); it != running_.end(); ++it) {
    if (it->status == SequenceStatus::kPreempted || it->finished || it->failed) {
      continue;
    }
    if (!higher_priority(waiting_seq, *it)) {
      continue;
    }
    if (victim == running_.end() || higher_priority(*victim, *it)) {
      victim = it;
    }
  }
  if (victim == running_.end()) {
    return false;
  }

  preempt_sequence(&*victim,
                   "priority_preempted_by_waiting_request(waiting_request_id=" +
                       std::to_string(waiting_seq.client_request_id) +
                       ", waiting_priority=" +
                       std::to_string(waiting_seq.generation_config.priority) +
                       ", victim_priority=" +
                       std::to_string(victim->generation_config.priority) + ")");
  reap_preempted_running();
  *remaining_free_blocks = reusable_free_blocks();
  return true;
}

void Scheduler::schedule_decode_sequences(SchedulerOutput* output,
                                          int32_t* remaining_budget) {
  CHECK_NE(output, nullptr);
  CHECK_NE(remaining_budget, nullptr);
  for (auto& seq : running_) {
    if (*remaining_budget <= 0) {
      break;
    }
    if (seq.is_prefill()) {
      continue;
    }

    std::string preempt_reason;
    if (!try_allocate_decode_slot(&seq, &preempt_reason)) {
      preempt_sequence(&seq, preempt_reason);
      continue;
    }
    append_decode_to_output(&seq, output);
    --(*remaining_budget);
  }
}

void Scheduler::schedule_running_prefills(SchedulerOutput* output,
                                          int32_t* remaining_budget,
                                          int32_t* remaining_free_blocks) {
  CHECK_NE(output, nullptr);
  CHECK_NE(remaining_budget, nullptr);
  CHECK_NE(remaining_free_blocks, nullptr);
  for (auto& seq : running_) {
    if (*remaining_budget <= 0) {
      break;
    }
    if (seq.status == SequenceStatus::kPreempted || !seq.is_prefill()) {
      continue;
    }

    if (!can_schedule_prefill_now(seq)) {
      continue;
    }
    const int32_t desired_chunk = choose_prefill_chunk(seq, *remaining_budget);
    const int32_t chunk = allocate_prefill_chunk(seq, desired_chunk, remaining_free_blocks);
    if (chunk <= 0) {
      continue;
    }
    append_prefill_to_output(&seq, chunk, output);
    *remaining_budget -= chunk;
  }
}

void Scheduler::admit_waiting_prefills(SchedulerOutput* output,
                                       int32_t* remaining_budget,
                                       int32_t* remaining_free_blocks) {
  CHECK_NE(output, nullptr);
  CHECK_NE(remaining_budget, nullptr);
  CHECK_NE(remaining_free_blocks, nullptr);
  while (!waiting_.empty() && *remaining_budget > 0 &&
         static_cast<int32_t>(running_.size()) < config_.max_num_seqs) {
    auto& seq = waiting_.front();
    if (seq.ready_step > scheduler_step_) {
      break;
    }

    if (!can_schedule_prefill_now(seq)) {
      break;
    }
    const int32_t desired_chunk = choose_prefill_chunk(seq, *remaining_budget);
    int32_t chunk = 0;
    if (!can_admit_waiting_sequence(seq, desired_chunk, *remaining_free_blocks, &chunk)) {
      if (maybe_preempt_for_waiting_sequence(seq, remaining_free_blocks) &&
          can_admit_waiting_sequence(seq, desired_chunk, *remaining_free_blocks, &chunk)) {
        // The high-priority waiting request can proceed after preemption.
      } else {
        break;
      }
    }
    if (chunk <= 0) {
      break;
    }

    chunk = allocate_prefill_chunk(seq, chunk, remaining_free_blocks);
    if (chunk <= 0) {
      break;
    }

    running_.push_back(std::move(seq));
    waiting_.pop_front();
    append_prefill_to_output(&running_.back(), chunk, output);
    *remaining_budget -= chunk;
  }
}

void Scheduler::handle_no_progress_step(const SchedulerOutput& output) {
  if (output.total_tokens != 0) {
    return;
  }
  reap_finished_running();
  reap_preempted_running();
  if (reject_waiting_request_that_cannot_start()) {
    return;
  }
  fail_stalled_prefill_request();
}


void Scheduler::reap_finished_running() {
  move_finished_sequences_to_output(&running_, &finished_, kv_manager_);
}

void Scheduler::reap_preempted_running() {
  auto it = running_.begin();
  while (it != running_.end()) {
    if (it->status == SequenceStatus::kPreempted) {
      enqueue_waiting_sequence(std::move(*it));
      it = running_.erase(it);
    } else {
      ++it;
    }
  }
}

void Scheduler::preempt_sequence(SequenceState* seq, const std::string& reason) {
  CHECK_NE(seq, nullptr);
  ++metrics_.preemptions;
  if (reason.find("priority_preempted") != std::string::npos) {
    ++metrics_.priority_preemptions;
  }
  if (reason.find("decode_kv_exhausted") != std::string::npos) {
    ++metrics_.decode_kv_preemptions;
  }
  CHECK(!seq->finished);
  CHECK(!seq->failed);

  kv_manager_->free_request(seq->request_id);
  seq->request_id = kv_manager_->register_request_with_radix_cache(seq->prompt_tokens);
  seq->computed_tokens = kv_manager_->get_context_len(seq->request_id);
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
               << ", recompute_tokens=" << seq->target_tokens()
               << ", preemption_count=" << seq->preemption_count;
}

void Scheduler::process_decode_output(
    SequenceState* seq,
    int32_t output_index,
    const MixedBatchMetadata& batch,
    const SampledTokenView& sampled_tokens,
    const std::function<bool(int32_t)>& is_eos,
    std::chrono::steady_clock::time_point now,
    int32_t* sample_idx) {
  CHECK_NE(seq, nullptr);
  CHECK_NE(sample_idx, nullptr);
  CHECK_LT(*sample_idx, sampled_tokens.size());
  CHECK_EQ(batch.sample_row_to_request[*sample_idx], output_index);
  const int32_t token = sampled_tokens[(*sample_idx)++];
  record_sampled_token(seq, token, is_eos, now);
}

void Scheduler::process_prefill_output(
    SequenceState* seq,
    int32_t output_index,
    const MixedBatchMetadata& batch,
    const SampledTokenView& sampled_tokens,
    const std::function<bool(int32_t)>& is_eos,
    std::chrono::steady_clock::time_point now,
    int32_t* sample_idx) {
  CHECK_NE(seq, nullptr);
  CHECK_NE(sample_idx, nullptr);
  seq->computed_tokens += seq->scheduled_tokens;
  CHECK_LE(seq->computed_tokens, seq->target_tokens());

  const bool prompt_finished =
      seq->computed_tokens == seq->target_tokens();
  if (!prompt_finished) {
    return;
  }

  if (seq->recompute_pending) {
    seq->recompute_pending = false;
  }
  maybe_publish_radix_cache(seq);

  if (*sample_idx < sampled_tokens.size() &&
      batch.sample_row_to_request[*sample_idx] == output_index) {
    const int32_t token = sampled_tokens[(*sample_idx)++];
    record_sampled_token(seq, token, is_eos, now);
  }
}

void Scheduler::maybe_publish_radix_cache(SequenceState* seq) {
  CHECK_NE(seq, nullptr);
  if (!seq->recompute_pending && !seq->radix_cache_published &&
      seq->output_tokens.empty()) {
    kv_manager_->publish_radix_cache(seq->request_id, seq->prompt_tokens);
    seq->radix_cache_published = true;
  }
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
    const int32_t target_tokens = seq.target_tokens();
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
      seq.remaining_tokens(),
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
    const int32_t required_tokens = seq.target_tokens() + 1;
    if (required_tokens <= total_kv_token_capacity()) {
      return false;
    }
  }

  auto failed = std::move(seq);
  waiting_.pop_front();
  std::string reason;
  if (failed.recompute_pending) {
    reason = "preempted_recompute_exceeds_kv_capacity(required_tokens=" +
             std::to_string(failed.target_tokens() + 1) +
             ", kv_capacity_tokens=" + std::to_string(total_kv_token_capacity()) + ")";
  } else {
    const int32_t reusable_free_blocks =
        kv_manager_->num_free_blocks(0) + kv_manager_->radix_cache_evictable_blocks();
    reason = "prefill_cannot_start(remaining_prompt_tokens=" +
             std::to_string(failed.remaining_tokens()) +
             ", free_kv_blocks=" + std::to_string(kv_manager_->num_free_blocks(0)) +
             ", radix_evictable_blocks=" +
             std::to_string(kv_manager_->radix_cache_evictable_blocks()) +
             ", reusable_kv_blocks=" + std::to_string(reusable_free_blocks) + ")";
  }
  ++metrics_.waiting_rejections;
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
        seq.remaining_tokens(),
        config_.max_num_batched_tokens,
        config_.prefill_chunk_cap});
    const int32_t chunk = cap_prefill_chunk_by_blocks(
        current_tokens, desired_chunk,
        kv_manager_->num_free_blocks(0) + kv_manager_->radix_cache_evictable_blocks(),
        kv_manager_->block_size());
    if (chunk > 0) {
      continue;
    }

    ++metrics_.stalled_prefill_failures;
    fail_sequence(&seq, std::chrono::steady_clock::now(),
                  "prefill_stalled_kv_exhausted(context_len=" +
                      std::to_string(current_tokens) +
                      ", remaining_prompt_tokens=" +
                      std::to_string(seq.remaining_tokens()) +
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
