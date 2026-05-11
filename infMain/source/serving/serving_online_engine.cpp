#include "serving/serving_online_engine.h"

#include <glog/logging.h>
#include <algorithm>
#include <chrono>
#include <deque>
#include <future>
#include <utility>
#include <unordered_map>
#include <cuda_runtime_api.h>
#if defined(KUIPER_ENABLE_NCCL)
#include <nccl.h>
#endif

#include "base/base.h"
#include "base/nvtx_utils.h"
#include "serving/decode_kv_reservation.h"
#include "serving/pd_handoff.h"
#include "serving/scheduler.h"
#include "serving/serving_benchmark_app.h"
#include "serving/serving_config.h"
#include "serving/serving_zmq_rpc.h"

namespace serving {

namespace {

void set_online_worker_device_or_die(const BenchConfig& config) {
  int32_t device_id = config.device_id;
  if (is_remote_pd_mode(config.pd_mode)) {
    device_id = config.decode_device_id;
  }
  const cudaError_t status = cudaSetDevice(device_id);
  CHECK_EQ(status, cudaSuccess)
      << "cudaSetDevice(" << device_id
      << ") failed in online worker thread: " << cudaGetErrorString(status);
}

SchedulerConfig make_online_scheduler_config(const ServingBenchmarkApp* app,
                                             const BenchConfig& config) {
  SchedulerConfig sched_config;
  sched_config.max_num_seqs = app->max_model_batch_size();
  sched_config.max_num_batched_tokens = config.max_num_batched_tokens;
  sched_config.prefill_chunk_cap = config.prefill_chunk_cap;
  sched_config.policy = config.scheduling_policy;
  sched_config.long_prefill_token_threshold =
      config.long_prefill_token_threshold;
  sched_config.max_partial_prefills = config.max_partial_prefills;
  sched_config.max_long_partial_prefills = config.max_long_partial_prefills;
  return sched_config;
}

}  // namespace

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

bool OnlineRequestHandle::wait_next_token_for(int32_t timeout_ms,
                                              std::string* token_text,
                                              bool* finished,
                                              bool* failed,
                                              std::string* error) {
  std::unique_lock<std::mutex> lock(mu_);
  const bool ready = cv_.wait_for(
      lock, std::chrono::milliseconds(std::max(1, timeout_ms)),
      [&]() { return next_idx_ < token_texts_.size() || finished_; });
  if (!ready) {
    *finished = false;
    *failed = false;
    error->clear();
    return false;
  }
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
  if (is_pd_mode(config_.pd_mode)) {
    if (is_dual_gpu_pd_mode(config_.pd_mode)) {
      CHECK(app_->pd_dual_gpu_supported())
          << "pd-mode=" << config_.pd_mode << " requires app-provided P/D models";
    }
    worker_ = std::thread([this]() { run_pd_loop(); });
    return;
  }

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
    if (is_pd_mode(config_.pd_mode)) {
      handle->set_request_id(next_pd_request_id_++);
    }
    submissions_.push_back(std::move(submission));
    ++pending_submissions_;
  }
  cv_.notify_all();
  return handle;
}

int32_t OnlineServingEngine::default_timeout_ms() const { return config_.request_timeout_ms; }

nlohmann::json OnlineServingEngine::metrics_json() const {
  nlohmann::json body;
  if (is_pd_mode(config_.pd_mode)) {
    std::lock_guard<std::mutex> lock(mu_);
    body["engine"]["mode"] = config_.pd_mode;
    body["engine"]["queued_requests"] = pending_submissions_;
    body["engine"]["active_requests"] = pd_active_requests_;
    body["engine"]["completed_requests"] = pd_completed_requests_;
    body["engine"]["failed_requests"] = pd_failed_requests_;
    body["engine"]["generated_tokens"] = pd_generated_tokens_;
    body["pd"]["mode"] = config_.pd_mode;
    body["pd"]["transfer_backend"] = pd_transfer_backend(config_.pd_mode);
    body["pd"]["prefill_device_id"] = config_.prefill_device_id;
    body["pd"]["decode_device_id"] = config_.decode_device_id;
    if (is_remote_pd_mode(config_.pd_mode)) {
      body["pd"]["prefill_zmq_endpoint"] = config_.prefill_zmq_endpoint;
    }
    return body;
  }

  const auto& metrics = scheduler_->metrics();
  body["engine"]["mode"] = "single";
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
  set_online_worker_device_or_die(config_);
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

void OnlineServingEngine::run_pd_loop() {
  set_online_worker_device_or_die(config_);
  if (is_remote_pd_mode(config_.pd_mode) &&
      (config_.pd_mode == "remote-zmq-nccl" ||
       config_.pd_mode == "remote-zmq-nccl-layer")) {
    run_remote_pd_batch_loop();
    return;
  }
  while (true) {
    PendingSubmission submission;
    {
      std::unique_lock<std::mutex> lock(mu_);
      cv_.wait(lock, [&]() { return stopping_ || !submissions_.empty(); });
      if (stopping_ && submissions_.empty()) {
        break;
      }
      submission = std::move(submissions_.front());
      submissions_.pop_front();
      --pending_submissions_;
      ++pd_active_requests_;
    }

    run_pd_submission(std::move(submission));

    {
      std::lock_guard<std::mutex> lock(mu_);
      --pd_active_requests_;
    }
  }
}

void OnlineServingEngine::run_remote_pd_batch_loop() {
  SchedulerConfig decode_config = make_online_scheduler_config(app_, config_);
  Scheduler decode_scheduler(decode_config, app_->kv_cache_manager());
  DecodeKVReservationManager reservation_manager(app_->kv_cache_manager(),
                                                 app_->pd_decode_kv_pool());
  ZmqRpcConfig prefill_rpc = make_prefill_zmq_rpc_config(config_);

  enum class RequestStage {
    kPollingPrefill,
    kTransferringKV,
    kDecoding,
  };

  struct RemoteRequestState {
    int64_t local_request_id = -1;
    int64_t prefill_request_id = -1;
    int64_t decode_client_id = -1;
    RequestStage stage = RequestStage::kPollingPrefill;
    std::vector<int32_t> prompt_tokens;
    GenerationConfig generation_config;
    std::vector<std::string> stop;
    std::shared_ptr<OnlineRequestHandle> handle;
    std::string streamed_text;
    RemotePrefillResult prefill;
    DecodeKVReservation reservation;
    KVBlockManifest manifest;
    std::string nccl_unique_id;
    std::future<base::Status> send_future;
    std::future<base::Status> recv_future;
  };

  auto request_response = [&](const nlohmann::json& request,
                              nlohmann::json* response) -> base::Status {
    const std::string type = request.value("type", std::string("unknown"));
    base::nvtx::ScopedRange range("decode_prefill_rpc:" + type,
                                  base::nvtx::kColorProcess);
    return zmq_request_response(prefill_rpc, request, response);
  };

  auto release_remote_prefill = [&](const RemotePrefillResult& prefill) {
    if (prefill.failed || !prefill.handoff_id.valid()) {
      return;
    }
    base::nvtx::ScopedRange range("remote_prefill_release",
                                  base::nvtx::kColorProcess);
    nlohmann::json release_response;
    request_response(
        {{"type", zmq_rpc_message_type_name(ZmqRpcMessageType::kKvRelease)},
         {"handoff_id", prefill.handoff_id.value}},
        &release_response);
  };

  auto finish_remote = [&](std::shared_ptr<RemoteRequestState> state,
                           bool failed,
                           const std::string& error,
                           const std::vector<int32_t>& output_tokens) {
    base::nvtx::ScopedRange range("remote_finish_request",
                                  base::nvtx::kColorProcess);
    if (state == nullptr || state->handle == nullptr) {
      return;
    }
    if (state->reservation.valid()) {
      reservation_manager.release(&state->reservation);
    }
    {
      std::lock_guard<std::mutex> lock(mu_);
      --pd_active_requests_;
      pd_generated_tokens_ += static_cast<int64_t>(output_tokens.size());
      if (failed) {
        ++pd_failed_requests_;
      } else {
        ++pd_completed_requests_;
      }
    }
    if (state->handle->cancelled()) {
      state->handle->finish(app_->postprocess_decoded_text(state->streamed_text),
                            false, "");
      return;
    }
    if (failed) {
      state->handle->finish("", true, error);
      return;
    }
    state->handle->finish(
        app_->postprocess_decoded_text(app_->decode_tokens(output_tokens)),
        false, "");
  };

  auto push_token = [&](RemoteRequestState* state, int32_t token) {
    base::nvtx::ScopedRange range("remote_push_token",
                                  base::nvtx::kColorProcess);
    if (state == nullptr || state->handle == nullptr ||
        state->handle->cancelled() || app_->is_sentence_ending(token)) {
      return;
    }
    const std::string token_text = app_->decode_tokens({token});
    const std::string candidate_text = state->streamed_text + token_text;
    size_t stop_pos = std::string::npos;
    for (const auto& stop : state->stop) {
      if (stop.empty()) {
        continue;
      }
      const size_t pos = candidate_text.find(stop);
      if (pos != std::string::npos) {
        stop_pos = std::min(stop_pos, pos);
      }
    }
    if (stop_pos != std::string::npos) {
      const std::string trimmed_text = candidate_text.substr(0, stop_pos);
      if (trimmed_text.size() > state->streamed_text.size()) {
        state->handle->push_token(
            trimmed_text.substr(state->streamed_text.size()));
      }
      state->handle->mark_cancelled();
      return;
    }
    state->streamed_text = candidate_text;
    state->handle->push_token(token_text);
  };

  std::unordered_map<int64_t, std::shared_ptr<RemoteRequestState>> active;
  std::unordered_map<int64_t, int64_t> decode_to_local;

  while (true) {
    std::deque<PendingSubmission> submissions;
    {
      std::unique_lock<std::mutex> lock(mu_);
      if (submissions_.empty() && active.empty() &&
          !decode_scheduler.has_active_requests()) {
        cv_.wait(lock, [&]() { return stopping_ || !submissions_.empty(); });
      }
      if (stopping_ && submissions_.empty() && active.empty() &&
          !decode_scheduler.has_active_requests()) {
        break;
      }
      submissions.swap(submissions_);
      pending_submissions_ -= static_cast<int32_t>(submissions.size());
    }

    while (!submissions.empty()) {
      PendingSubmission submission = std::move(submissions.front());
      submissions.pop_front();
      if (submission.handle == nullptr || submission.handle->cancelled()) {
        continue;
      }
      auto state = std::make_shared<RemoteRequestState>();
      state->local_request_id = submission.handle->request_id();
      state->prompt_tokens = std::move(submission.prompt_tokens);
      state->generation_config = submission.generation_config;
      state->stop = std::move(submission.stop);
      state->handle = submission.handle;
      {
        std::lock_guard<std::mutex> lock(mu_);
        ++pd_active_requests_;
      }

      nlohmann::json response;
      base::Status status;
      {
        base::nvtx::ScopedRange range("prefill_submit_rpc",
                                      base::nvtx::kColorProcess);
        status = request_response(
            {{"type",
              zmq_rpc_message_type_name(ZmqRpcMessageType::kPrefillSubmit)},
             {"prompt_tokens", state->prompt_tokens},
             {"generation_config",
              generation_config_to_json(state->generation_config)}},
            &response);
      }
      if (!status || !response.value("ok", false)) {
        finish_remote(state, true,
                      status ? response.value("error", "prefill_submit_failed")
                             : status.get_err_msg(),
                      {});
        continue;
      }
      state->prefill_request_id =
          response.value("prefill_request_id", int64_t{-1});
      if (state->prefill_request_id < 0) {
        finish_remote(state, true, "invalid_prefill_request_id", {});
        continue;
      }
      active[state->local_request_id] = state;
    }

    std::vector<int64_t> to_finish;
    for (auto& item : active) {
      auto state = item.second;
      if (state->stage != RequestStage::kPollingPrefill) {
        continue;
      }
      nlohmann::json response;
      base::Status status;
      {
        base::nvtx::ScopedRange range("prefill_poll_rpc",
                                      base::nvtx::kColorProcess);
        status = request_response(
            {{"type", zmq_rpc_message_type_name(ZmqRpcMessageType::kPrefillPoll)},
             {"prefill_request_id", state->prefill_request_id}},
            &response);
      }
      if (!status || !response.value("ok", false)) {
        finish_remote(state, true,
                      status ? response.value("error", "prefill_poll_failed")
                             : status.get_err_msg(),
                      {});
        to_finish.push_back(item.first);
        continue;
      }
      if (!response.value("ready", false)) {
        continue;
      }
      state->prefill =
          remote_prefill_result_from_json(response.at("prefill_result"));
      if (state->handle->cancelled()) {
        release_remote_prefill(state->prefill);
        finish_remote(state, false, "", {});
        to_finish.push_back(item.first);
        continue;
      }
      if (state->prefill.failed) {
        finish_remote(state, true, state->prefill.error,
                      state->prefill.output_tokens);
        to_finish.push_back(item.first);
        continue;
      }
      if (state->prefill.first_token < 0 ||
          state->prefill.computed_tokens <= 0 ||
          !state->prefill.handoff_id.valid()) {
        release_remote_prefill(state->prefill);
        finish_remote(state, true, "remote_prefill_missing_nccl_metadata", {});
        to_finish.push_back(item.first);
        continue;
      }
      push_token(state.get(), state->prefill.first_token);
      if (state->handle->cancelled()) {
        release_remote_prefill(state->prefill);
        finish_remote(state, false, "", {});
        to_finish.push_back(item.first);
        continue;
      }

      DecodeKVReservationRequest reservation_request;
      reservation_request.client_request_id =
          state->prefill.client_request_id.empty()
              ? GlobalRequestId{"remote-zmq-nccl"}
              : state->prefill.client_request_id;
      reservation_request.handoff_id = state->prefill.handoff_id;
      reservation_request.prompt_tokens =
          static_cast<int32_t>(state->prompt_tokens.size());
      reservation_request.computed_tokens = state->prefill.computed_tokens;
      reservation_request.first_token = state->prefill.first_token;
      reservation_request.src_pool = state->prefill.src_pool;
      reservation_request.src_block_ids_per_layer.resize(
          state->prefill.src_pool.layer_num);
      for (const auto& layer : state->prefill.layers) {
        if (layer.layer_idx >= 0 &&
            layer.layer_idx < static_cast<int32_t>(
                                  reservation_request
                                      .src_block_ids_per_layer.size())) {
          reservation_request.src_block_ids_per_layer[layer.layer_idx] =
              layer.src_block_ids;
        }
      }

      base::Status reserve_status;
      {
        base::nvtx::ScopedRange range("decode_reserve_kv",
                                      base::nvtx::kColorMetadata);
        reserve_status = reservation_manager.reserve(
            reservation_request, &state->reservation, &state->manifest);
      }
      if (!reserve_status) {
        release_remote_prefill(state->prefill);
        finish_remote(state, true, reserve_status.get_err_msg(), {});
        to_finish.push_back(item.first);
        continue;
      }

#if defined(KUIPER_ENABLE_NCCL)
      ncclUniqueId unique_id;
      ncclResult_t nccl_status = ncclGetUniqueId(&unique_id);
      if (nccl_status != ncclSuccess) {
        release_remote_prefill(state->prefill);
        finish_remote(state, true,
                      std::string("ncclGetUniqueId failed: ") +
                          ncclGetErrorString(nccl_status),
                      {});
        to_finish.push_back(item.first);
        continue;
      }
      state->nccl_unique_id.assign(reinterpret_cast<const char*>(&unique_id),
                                   sizeof(unique_id));
      const KVBlockManifest manifest = state->manifest;
      const std::string unique_id_bytes = state->nccl_unique_id;
      const BenchConfig config = config_;
      base::nvtx::Mark("kv_transfer_start", base::nvtx::kColorMemcpy);
      state->send_future = std::async(
          std::launch::async, [this, manifest, unique_id_bytes, config]() {
            base::nvtx::ScopedRange range("kv_send_rpc",
                                          base::nvtx::kColorMemcpy);
            ZmqRpcConfig transfer_rpc = make_prefill_zmq_rpc_config(config);
            nlohmann::json transfer_response;
            base::Status transfer_status = zmq_request_response(
                transfer_rpc,
                {{"type", zmq_rpc_message_type_name(ZmqRpcMessageType::kKvTransfer)},
                 {"backend", "nccl"},
                 {"manifest", kv_block_manifest_to_json(manifest)},
                 {"nccl_unique_id", binary_to_hex_json(unique_id_bytes)}},
                &transfer_response);
            if (!transfer_status) {
              return transfer_status;
            }
            if (!transfer_response.value("ok", false)) {
              return base::error::InternalError(
                  transfer_response.value(
                      "error", "remote_nccl_prefill_send_failed"));
            }
            return base::error::Success();
          });

      state->recv_future = std::async(
          std::launch::async,
          [this, manifest, unique_id_bytes]() {
            base::nvtx::ScopedRange range("kv_recv_nccl",
                                          base::nvtx::kColorMemcpy);
            RemoteNcclKVTransferOptions recv_options;
            recv_options.role = RemoteNcclKVTransferRole::kConsumer;
            recv_options.kv_manager = app_->kv_cache_manager();
            recv_options.device_id = manifest.dst_pool.device_id;
            recv_options.nccl_unique_id = unique_id_bytes;
            recv_options.stream = nullptr;
            recv_options.need_sync = true;
            return run_remote_nccl_kv_block_transfer(manifest, recv_options);
          });
      state->stage = RequestStage::kTransferringKV;
#else
      release_remote_prefill(state->prefill);
      finish_remote(state, true,
                    "NCCL support is not enabled. Reconfigure with "
                    "-DKUIPER_ENABLE_NCCL=ON for remote-zmq-nccl modes.",
                    {});
      to_finish.push_back(item.first);
      continue;
#endif
    }
    for (int64_t id : to_finish) {
      active.erase(id);
    }

    to_finish.clear();
    for (auto& item : active) {
      auto state = item.second;
      if (state->stage != RequestStage::kTransferringKV) {
        continue;
      }
      if (state->send_future.wait_for(std::chrono::milliseconds(0)) !=
              std::future_status::ready ||
          state->recv_future.wait_for(std::chrono::milliseconds(0)) !=
              std::future_status::ready) {
        continue;
      }
      const base::Status send_status = state->send_future.get();
      const base::Status recv_status = state->recv_future.get();
      if (state->handle->cancelled()) {
        release_remote_prefill(state->prefill);
        finish_remote(state, false, "", {});
        to_finish.push_back(item.first);
        continue;
      }
      if (!recv_status || !send_status) {
        release_remote_prefill(state->prefill);
        finish_remote(state, true,
                      !recv_status ? recv_status.get_err_msg()
                                   : send_status.get_err_msg(),
                      {});
        to_finish.push_back(item.first);
        continue;
      }

      int64_t decode_client_id = -1;
      {
        base::nvtx::ScopedRange range("decode_add_ready_request",
                                      base::nvtx::kColorSchedule);
        decode_client_id = decode_scheduler.add_decode_ready_request(
            state->reservation.decode_request_id, state->prompt_tokens,
            state->generation_config, state->reservation.reserved_tokens,
            state->prefill.first_token);
      }
      state->decode_client_id = decode_client_id;
      decode_to_local[decode_client_id] = state->local_request_id;
      state->reservation = {};
      state->stage = RequestStage::kDecoding;
    }
    for (int64_t id : to_finish) {
      active.erase(id);
    }

    SchedulerOutput output;
    {
      base::nvtx::ScopedRange range("decode_schedule",
                                    base::nvtx::kColorSchedule);
      output = decode_scheduler.schedule_step();
    }
    if (output.total_tokens > 0) {
      MixedBatchMetadata batch;
      {
        base::nvtx::ScopedRange range("decode_build_batch",
                                      base::nvtx::kColorMetadata);
        batch = decode_scheduler.build_decode_batch(output, app_->model_stream());
      }
      base::Status status;
      {
        base::nvtx::ScopedRange range("decode_forward",
                                      base::nvtx::kColorForward);
        status = app_->pd_forward_decode_batch(batch);
      }
      if (!status) {
        LOG(ERROR) << "remote PD decode batch failed: "
                   << status.get_err_msg();
        for (auto* seq : output.scheduled_seqs) {
          if (seq != nullptr) {
            decode_scheduler.cancel_request(seq->client_request_id,
                                            status.get_err_msg());
          }
        }
      } else {
        SampledTokenView sampled;
        {
          base::nvtx::ScopedRange range("decode_sample",
                                        base::nvtx::kColorSample);
          sampled = app_->pd_batch_sample_decode(batch, output);
        }
        {
          base::nvtx::ScopedRange range("decode_emit_tokens",
                                        base::nvtx::kColorProcess);
          for (int32_t i = 0; i < sampled.size(); ++i) {
            const int32_t request_idx = batch.sample_row_to_request[i];
            if (request_idx < 0 ||
                request_idx >=
                    static_cast<int32_t>(output.scheduled_seqs.size())) {
              continue;
            }
            const SequenceState* seq = output.scheduled_seqs[request_idx];
            if (seq == nullptr) {
              continue;
            }
            auto local_it = decode_to_local.find(seq->client_request_id);
            if (local_it == decode_to_local.end()) {
              continue;
            }
            auto active_it = active.find(local_it->second);
            if (active_it == active.end()) {
              continue;
            }
            push_token(active_it->second.get(), sampled.tokens[i]);
          }
        }
        {
          base::nvtx::ScopedRange range("decode_process_outputs",
                                        base::nvtx::kColorProcess);
          decode_scheduler.process_outputs(
              output, batch, sampled,
              [&](int32_t token) { return app_->is_sentence_ending(token); });
        }
      }
    }

    std::vector<SequenceState> finished;
    {
      base::nvtx::ScopedRange range("decode_pop_finished",
                                    base::nvtx::kColorProcess);
      finished = decode_scheduler.pop_finished();
    }
    for (const auto& seq : finished) {
      auto local_it = decode_to_local.find(seq.client_request_id);
      if (local_it == decode_to_local.end()) {
        continue;
      }
      auto active_it = active.find(local_it->second);
      if (active_it != active.end()) {
        release_remote_prefill(active_it->second->prefill);
        finish_remote(active_it->second, seq.failed, seq.finish_reason,
                      seq.output_tokens);
        active.erase(active_it);
      }
      decode_to_local.erase(local_it);
    }

    if (active.empty() && !decode_scheduler.has_active_requests()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
}

void OnlineServingEngine::run_pd_submission(PendingSubmission submission) {
  if (submission.handle == nullptr || submission.handle->cancelled()) {
    return;
  }

  std::string streamed_text;
  auto on_token = [&](int32_t token) {
    if (submission.handle->cancelled() || app_->is_sentence_ending(token)) {
      return;
    }
    const std::string token_text = app_->decode_tokens({token});
    const std::string candidate_text = streamed_text + token_text;
    size_t stop_pos = std::string::npos;
    for (const auto& stop : submission.stop) {
      if (stop.empty()) {
        continue;
      }
      const size_t pos = candidate_text.find(stop);
      if (pos != std::string::npos) {
        stop_pos = std::min(stop_pos, pos);
      }
    }
    if (stop_pos != std::string::npos) {
      const std::string trimmed_text = candidate_text.substr(0, stop_pos);
      if (trimmed_text.size() > streamed_text.size()) {
        submission.handle->push_token(trimmed_text.substr(streamed_text.size()));
      }
      submission.handle->mark_cancelled();
      return;
    }
    streamed_text = candidate_text;
    submission.handle->push_token(token_text);
  };
  const auto result =
      is_remote_pd_mode(config_.pd_mode)
          ? (config_.pd_mode == "remote-zmq-nccl" ||
                     config_.pd_mode == "remote-zmq-nccl-layer"
                 ? app_->run_remote_zmq_nccl_pd_generation(
                       std::move(submission.prompt_tokens),
                       submission.generation_config, on_token)
                 : app_->run_remote_zmq_cpu_pd_generation(
                       std::move(submission.prompt_tokens),
                       submission.generation_config, on_token))
          : app_->run_dual_gpu_pd_generation(
                std::move(submission.prompt_tokens), submission.generation_config,
                on_token);

  {
    std::lock_guard<std::mutex> lock(mu_);
    pd_generated_tokens_ += static_cast<int64_t>(result.output_tokens.size());
    if (result.failed) {
      ++pd_failed_requests_;
    } else {
      ++pd_completed_requests_;
    }
  }

  if (submission.handle->cancelled()) {
    submission.handle->finish(app_->postprocess_decoded_text(streamed_text),
                              false, "");
    return;
  }
  if (result.failed) {
    submission.handle->finish("", true, result.error);
    return;
  }
  submission.handle->finish(app_->postprocess_decoded_text(
                                app_->decode_tokens(result.output_tokens)),
                            false, "");
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
