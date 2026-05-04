#include "serving/serving_online_engine.h"

#include <glog/logging.h>
#include <chrono>
#include <deque>
#include <utility>

#include "base/base.h"
#include "serving/scheduler.h"
#include "serving/serving_benchmark_app.h"
#include "serving/serving_config.h"

namespace serving {

void OnlineRequestHandle::set_request_id(int64_t request_id) {
  std::lock_guard<std::mutex> lock(mu_);
  request_id_ = request_id;
}

int64_t OnlineRequestHandle::request_id() const {
  std::lock_guard<std::mutex> lock(mu_);
  return request_id_;
}

void OnlineRequestHandle::mark_cancelled() {
  std::lock_guard<std::mutex> lock(mu_);
  cancelled_ = true;
}

bool OnlineRequestHandle::cancelled() const {
  std::lock_guard<std::mutex> lock(mu_);
  return cancelled_;
}

void OnlineRequestHandle::push_token(const std::string& token_text) {
  std::lock_guard<std::mutex> lock(mu_);
  token_texts_.push_back(token_text);
  cv_.notify_all();
}

void OnlineRequestHandle::finish(std::string full_text, bool failed, std::string error) {
  std::lock_guard<std::mutex> lock(mu_);
  full_text_ = std::move(full_text);
  failed_ = failed;
  error_ = std::move(error);
  finished_ = true;
  cv_.notify_all();
}

bool OnlineRequestHandle::wait_next_token(std::string* token_text, bool* finished,
                                          bool* failed, std::string* error) {
  std::unique_lock<std::mutex> lock(mu_);
  cv_.wait(lock, [&]() { return next_idx_ < token_texts_.size() || finished_; });
  if (next_idx_ < token_texts_.size()) {
    *token_text = token_texts_[next_idx_++];
    *finished = false;
    *failed = false;
    error->clear();
    return true;
  }
  *finished = true;
  *failed = failed_;
  *error = error_;
  return false;
}

bool OnlineRequestHandle::wait_full_text(int32_t timeout_ms, std::string* text,
                                         bool* failed, std::string* error) {
  std::unique_lock<std::mutex> lock(mu_);
  const bool ready = timeout_ms > 0
      ? cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&]() { return finished_; })
      : (cv_.wait(lock, [&]() { return finished_; }), true);
  if (!ready) {
    return false;
  }
  *text = full_text_;
  *failed = failed_;
  *error = error_;
  return true;
}

OnlineServingEngine::OnlineServingEngine(ServingBenchmarkApp* app, const BenchConfig& config)
    : app_(app), config_(config) {}

OnlineServingEngine::~OnlineServingEngine() { stop(); }

void OnlineServingEngine::start() {
  SchedulerConfig sched_config;
  sched_config.max_num_seqs = app_->max_model_batch_size();
  sched_config.max_num_batched_tokens = config_.max_num_batched_tokens;
  sched_config.prefill_chunk_cap = config_.prefill_chunk_cap;
  sched_config.policy = config_.scheduling_policy;
  sched_config.long_prefill_token_threshold = config_.long_prefill_token_threshold;
  sched_config.max_partial_prefills = config_.max_partial_prefills;
  sched_config.max_long_partial_prefills = config_.max_long_partial_prefills;
  scheduler_ = std::make_unique<Scheduler>(sched_config, app_->kv_cache_manager());
  gpu_worker_ = std::make_unique<InProcGpuWorker>(app_, scheduler_.get());
  worker_ = std::thread([this]() { run_loop(); });
}

void OnlineServingEngine::stop() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    stopping_ = true;
  }
  cv_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
}

std::shared_ptr<OnlineRequestHandle> OnlineServingEngine::submit(
    const OnlineGenerateRequest& request, std::string* error) {
  auto handle = std::make_shared<OnlineRequestHandle>();
  auto prompt_tokens = app_->encode_prompt(request.prompt);
  if (config_.max_prompt_tokens > 0 &&
      static_cast<int32_t>(prompt_tokens.size()) > config_.max_prompt_tokens) {
    *error = "prompt_too_long";
    return nullptr;
  }
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (stopping_) {
      *error = "server_stopping";
      return nullptr;
    }
    if (pending_submissions_ >= config_.max_queue_size) {
      *error = "queue_full";
      return nullptr;
    }
    PendingSubmission submission;
    submission.prompt_tokens = std::move(prompt_tokens);
    submission.generation_config = request.generation_config;
    submission.stop = request.generation_config.sampling.stop;
    submission.generation_config.normalize();
    submission.handle = handle;
    submissions_.push_back(std::move(submission));
    ++pending_submissions_;
  }
  cv_.notify_all();
  return handle;
}

int32_t OnlineServingEngine::default_timeout_ms() const { return config_.request_timeout_ms; }

nlohmann::json OnlineServingEngine::metrics_json() const {
  const auto& metrics = scheduler_->metrics();
  nlohmann::json body;
  body["scheduler"]["steps"] = metrics.schedule_steps;
  body["scheduler"]["no_progress_steps"] = metrics.no_progress_steps;
  body["scheduler"]["decode_tokens"] = metrics.scheduled_decode_tokens;
  body["scheduler"]["prefill_tokens"] = metrics.scheduled_prefill_tokens;
  body["scheduler"]["batched_tokens"] = metrics.scheduled_batched_tokens;
  body["scheduler"]["preemptions"] = metrics.preemptions;
  body["scheduler"]["priority_preemptions"] = metrics.priority_preemptions;
  body["scheduler"]["decode_kv_preemptions"] = metrics.decode_kv_preemptions;
  body["scheduler"]["waiting_rejections"] = metrics.waiting_rejections;
  body["scheduler"]["stalled_prefill_failures"] = metrics.stalled_prefill_failures;
  body["scheduler"]["max_waiting_queue"] = metrics.max_waiting_queue;
  body["scheduler"]["max_running_queue"] = metrics.max_running_queue;
  return body;
}

void OnlineServingEngine::cancel(const std::shared_ptr<OnlineRequestHandle>& handle,
                                 const std::string& reason) {
  if (!handle) {
    return;
  }
  handle->mark_cancelled();
  {
    std::lock_guard<std::mutex> lock(mu_);
    cancellations_.push_back({handle->request_id(), reason});
  }
  cv_.notify_all();
}

void OnlineServingEngine::run_loop() {
  void* stream = app_->model_stream();
  while (true) {
    flush_submissions();
    if (!scheduler_->has_active_requests()) {
      std::unique_lock<std::mutex> lock(mu_);
      cv_.wait(lock, [&]() { return stopping_ || !submissions_.empty(); });
      if (stopping_ && submissions_.empty()) {
        break;
      }
      continue;
    }
    run_step(stream);
  }
}

void OnlineServingEngine::flush_submissions() {
  std::deque<PendingSubmission> local;
  {
    std::lock_guard<std::mutex> lock(mu_);
    local.swap(submissions_);
    pending_submissions_ -= static_cast<int32_t>(local.size());
  }
  while (!local.empty()) {
    auto submission = std::move(local.front());
    local.pop_front();
    const int64_t request_id = scheduler_->add_request(
        std::move(submission.prompt_tokens), submission.generation_config);
    submission.handle->set_request_id(request_id);
    stop_by_request_[request_id] = std::move(submission.stop);
    streamed_text_by_request_[request_id].clear();
    handles_[request_id] = submission.handle;
    if (submission.handle->cancelled()) {
      scheduler_->cancel_request(request_id, "client_disconnected");
    }
  }
}

void OnlineServingEngine::flush_cancellations() {
  std::vector<std::pair<int64_t, std::string>> local;
  {
    std::lock_guard<std::mutex> lock(mu_);
    local.swap(cancellations_);
  }
  for (const auto& [request_id, reason] : local) {
    if (request_id >= 0) {
      scheduler_->cancel_request(request_id, reason);
    }
  }
}

void OnlineServingEngine::run_step(void* stream) {
  flush_cancellations();
  SchedulerOutput sched_out = scheduler_->schedule_step();
  if (sched_out.total_tokens == 0) {
    publish_finished();
    return;
  }

  WorkerStepOutput worker_output;
  base::Status status = gpu_worker_->execute_step(sched_out, stream, &worker_output);
  if (!status) {
    LOG(ERROR) << "online gpu worker step failed: " << status.get_err_msg();
    return;
  }

  const SampledTokenView sampled_tokens = worker_output.sampled_token_view();
  publish_sampled_tokens(sched_out, worker_output.batch, sampled_tokens);
  scheduler_->process_outputs(
      sched_out, worker_output.batch, sampled_tokens,
      [&](int32_t token) { return app_->is_sentence_ending(token); });
  publish_finished();
}

void OnlineServingEngine::publish_sampled_tokens(
    const SchedulerOutput& output,
    const MixedBatchMetadata& batch,
    const SampledTokenView& sampled_tokens) {
  CHECK_EQ(sampled_tokens.size(), static_cast<int32_t>(batch.sample_row_to_request.size()));
  for (int32_t sample_idx = 0; sample_idx < sampled_tokens.size(); ++sample_idx) {
    const int32_t request_idx = batch.sample_row_to_request[sample_idx];
    CHECK_GE(request_idx, 0);
    CHECK_LT(request_idx, static_cast<int32_t>(output.scheduled_seqs.size()));
    const auto* seq = output.scheduled_seqs[request_idx];
    CHECK_NE(seq, nullptr);
    auto it = handles_.find(seq->client_request_id);
    if (it == handles_.end()) {
      continue;
    }
    const int32_t token = sampled_tokens[sample_idx];
    if (app_->is_sentence_ending(token)) {
      continue;
    }
    const std::string token_text = app_->decode_tokens({token});
    std::string& streamed_text = streamed_text_by_request_[seq->client_request_id];
    const std::string candidate_text = streamed_text + token_text;
    auto stop_it = stop_by_request_.find(seq->client_request_id);
    size_t stop_pos = std::string::npos;
    if (stop_it != stop_by_request_.end()) {
      for (const auto& stop : stop_it->second) {
        if (stop.empty()) {
          continue;
        }
        const size_t pos = candidate_text.find(stop);
        if (pos != std::string::npos) {
          stop_pos = std::min(stop_pos, pos);
        }
      }
    }
    if (stop_pos != std::string::npos) {
      const std::string trimmed_text = candidate_text.substr(0, stop_pos);
      if (trimmed_text.size() > streamed_text.size()) {
        it->second->push_token(trimmed_text.substr(streamed_text.size()));
      }
      it->second->finish(app_->postprocess_decoded_text(trimmed_text), false, "");
      scheduler_->cancel_request(seq->client_request_id, "stop_sequence");
      stop_by_request_.erase(seq->client_request_id);
      streamed_text_by_request_.erase(seq->client_request_id);
      handles_.erase(it);
      continue;
    }
    streamed_text = candidate_text;
    it->second->push_token(token_text);
  }
}

void OnlineServingEngine::publish_finished() {
  auto finished = scheduler_->pop_finished();
  for (const auto& seq : finished) {
    auto it = handles_.find(seq.client_request_id);
    if (it == handles_.end()) {
      continue;
    }
    if (it->second->cancelled()) {
      stop_by_request_.erase(seq.client_request_id);
      streamed_text_by_request_.erase(seq.client_request_id);
      handles_.erase(it);
      continue;
    }
    std::string full_text;
    if (!seq.failed) {
      full_text = app_->postprocess_decoded_text(app_->decode_tokens(seq.output_tokens));
    }
    it->second->finish(full_text, seq.failed, seq.finish_reason);
    stop_by_request_.erase(seq.client_request_id);
    streamed_text_by_request_.erase(seq.client_request_id);
    handles_.erase(it);
  }
}

}  // namespace serving
