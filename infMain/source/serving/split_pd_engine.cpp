#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <chrono>
#include <future>
#include <stdexcept>
#include "base/nvtx_utils.h"
#include "serving/decode_kv_reservation.h"
#include "serving/serving_online_engine.h"
#include "serving/serving_runtime.h"
#include "serving/serving_zmq_rpc.h"
#include "serving/split_pd_admission.h"

namespace serving {
namespace {
nlohmann::json exchange(const ZmqRpcConfig& rpc, const nlohmann::json& request) {
  nlohmann::json response;
  auto status = zmq_request_response(rpc, request, &response);
  if (!status) throw std::runtime_error("PD channel outcome unknown: " + status.get_err_msg());
  if (!response.value("ok", false))
    throw std::runtime_error(response.value("error", "PD RPC failed"));
  return response;
}

RemotePrefillResult prefill(const ZmqRpcConfig& rpc, std::vector<int32_t> tokens,
                            GenerationConfig generation,
                            std::shared_ptr<OnlineRequestHandle> handle) {
  auto submitted = exchange(rpc, {{"type", "prefill_submit"},
                                  {"prompt_tokens", tokens},
                                  {"generation_config", generation_config_to_json(generation)}});
  const int64_t id = submitted.at("prefill_request_id");
  for (;;) {
    if (handle->cancelled()) {
      exchange(rpc, {{"type", "cancel"}, {"prefill_request_id", id}});
      RemotePrefillResult result;
      result.failed = true;
      result.error = "cancelled";
      return result;
    }
    auto response = exchange(rpc, {{"type", "prefill_poll"}, {"prefill_request_id", id}});
    if (response.at("ready").get<bool>())
      return remote_prefill_result_from_json(response.at("prefill_result"));
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

// Both processes map the same owner epoch. Block IDs encode byte offsets, so
// D can resolve the P source through its own mapping. P retains the source
// lease until this copy completes and an explicit release is acknowledged.
uint64_t copy_shared_kv(ServingRuntime* app, const RemotePrefillResult& source,
                        DecodeKVReservation* reservation) {
  base::nvtx::ScopedRange range("split_pd_shared_kv_copy", base::nvtx::kColorMemcpy);
  auto* manager = app->kv_cache_manager();
  const auto& first = manager->allocator(0);
  if (!source.shared_arena_epoch || source.shared_arena_epoch != first.shared_arena_epoch())
    throw std::runtime_error("PD arena epoch mismatch");
  DecodeKVReservationRequest request;
  request.client_request_id = source.client_request_id;
  request.handoff_id = source.handoff_id;
  request.prompt_tokens = source.prompt_tokens.size();
  request.computed_tokens = source.computed_tokens;
  request.first_token = source.first_token;
  request.src_pool = source.src_pool;
  if (source.layers.size() != static_cast<size_t>(manager->num_layers()))
    throw std::runtime_error("PD layer count mismatch");
  request.src_block_ids_per_layer.resize(manager->num_layers());
  for (size_t index = 0; index < source.layers.size(); ++index) {
    const auto& layer = source.layers[index];
    const auto& allocator = manager->allocator(index);
    if (layer.layer_idx != static_cast<int>(index) || !allocator.uses_shared_arena())
      throw std::runtime_error("PD shared layout mismatch");
    for (int id : layer.src_block_ids)
      if (id < 0 ||
          (static_cast<uint64_t>(id) + 2 * manager->num_layers()) *
                  allocator.key_value_bytes_per_block() >
              allocator.shared_arena_bytes() ||
          id % (2 * manager->num_layers()) != 0)
        throw std::runtime_error("PD source block outside shared page layout");
    if (index > 0 && layer.src_block_ids != source.layers.front().src_block_ids)
      throw std::runtime_error("PD page-major layer IDs disagree");
    request.src_block_ids_per_layer[index] = layer.src_block_ids;
  }
  DecodeKVReservationManager reservations(manager, app->pd_decode_kv_pool());
  KVBlockManifest manifest;
  auto status = reservations.reserve(request, reservation, &manifest);
  if (!status) throw std::runtime_error(status.get_err_msg());
  uint64_t copied = 0;
  auto stream = static_cast<cudaStream_t>(app->model_stream());
  const auto& mapping = manifest.layer_mappings.front();
  const int page_stride = 2 * manager->num_layers();
  const size_t page_bytes = page_stride * first.key_value_bytes_per_block();
  for (const auto& layer : manifest.layer_mappings)
    if (layer.dst_block_ids != mapping.dst_block_ids)
      LOG(FATAL) << "PD destination is not page-major";
  // Copy all K/V planes in each page together and coalesce consecutive pages.
  // This avoids 2*layers CUDA API calls per page on long-input handoffs.
  for (size_t i = 0; i < mapping.src_block_ids.size();) {
    size_t end = i + 1;
    while (end < mapping.src_block_ids.size() &&
           mapping.src_block_ids[end] == mapping.src_block_ids[end - 1] + page_stride &&
           mapping.dst_block_ids[end] == mapping.dst_block_ids[end - 1] + page_stride)
      ++end;
    auto src = first.get_block_payload_ptrs(mapping.src_block_ids[i]);
    auto dst = first.get_block_payload_ptrs(mapping.dst_block_ids[i]);
    const size_t bytes = (end - i) * page_bytes;
    if (cudaMemcpyAsync(dst.key, src.key, bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess)
      LOG(FATAL) << "PD CUDA copy failed; discard the complete IPC domain";
    copied += bytes;
    i = end;
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess)
    LOG(FATAL) << "PD copy completion unknown; retain source until domain shutdown";
  if (!manager->commit_kv(reservation->decode_request_id, source.computed_tokens))
    throw std::runtime_error("PD destination KV commit failed");
  return copied;
}
}  // namespace

void OnlineServingEngine::run_split_pd_loop() {
  SchedulerConfig scheduler_config;
  scheduler_config.max_num_seqs = config_.pd.short_decode_slots + config_.pd.long_decode_slots;
  CHECK_LE(scheduler_config.max_num_seqs, app_->max_model_batch_size());
  scheduler_config.max_num_batched_tokens = config_.max_num_batched_tokens;
  CHECK_GE(scheduler_config.max_num_batched_tokens, scheduler_config.max_num_seqs);
  const auto capacity = app_->serving_capacity_info();
  const uint64_t promised =
      static_cast<uint64_t>(config_.pd.short_kv_tokens) + config_.pd.long_kv_tokens;
  CHECK_LE(promised * capacity.kv_bytes_per_token + (4ull * config_.kv.shared_kv_chunk_mib << 20),
           static_cast<uint64_t>(config_.kv.shared_kv_hard_limit_mib) << 20)
      << "Decode KV quota must cover both lane promises and chunk rounding";
  Scheduler decoder(scheduler_config, app_->kv_cache_manager());
  SplitPDAdmission admission({config_.pd.short_decode_slots, config_.pd.long_decode_slots},
                             {config_.pd.short_kv_tokens, config_.pd.long_kv_tokens});
  struct Request {
    PendingSubmission submission;
    int lane = 0;
    int64_t charged_tokens = 0;
    std::string text;
    bool stopped_by_sequence = false;
  };
  struct Lane {
    ZmqRpcConfig rpc;
    std::shared_ptr<Request> request;
    std::future<RemotePrefillResult> pending;
    std::future<void> releasing;
    uint64_t dispatched = 0, completed = 0, cancelled = 0, generated = 0;
  };
  std::array<Lane, 2> lanes;
  lanes[0].rpc = {config_.pd.short_prefill_endpoint, config_.ipc.engine_zmq_timeout_ms};
  lanes[1].rpc = {config_.pd.long_prefill_endpoint, config_.ipc.engine_zmq_timeout_ms};
  std::unordered_map<int64_t, std::shared_ptr<Request>> decoding;
  uint64_t copied_bytes = 0, decode_steps = 0, max_decode_batch = 0;
  std::deque<nlohmann::json> recent;

  auto finish = [&](const std::shared_ptr<Request>& request, bool failed, const std::string& reason,
                    const std::vector<int32_t>& tokens, int generated) {
    admission.release(request->lane, request->charged_tokens);
    auto& lane = lanes[request->lane];
    failed = failed && !request->submission.handle->cancelled() && !request->stopped_by_sequence;
    lane.generated += generated;
    if (request->submission.handle->cancelled())
      ++lane.cancelled;
    else
      ++lane.completed;
    recent.push_back(
        {{"request_id", request->submission.handle->request_id()},
         {"lane", request->lane == 0 ? "short" : "long"},
         {"prompt_tokens", request->submission.prompt_tokens.size()},
         {"requested_output_tokens", request->submission.generation_config.max_new_tokens},
         {"generated_tokens", generated},
         {"failed", failed},
         {"reason", reason}});
    if (recent.size() > 64) recent.pop_front();
    const std::string text =
        failed ? "" : app_->postprocess_decoded_text(app_->decode_tokens(tokens));
    request->submission.handle->finish(text, failed, reason);
    std::lock_guard<std::mutex> lock(mu_);
    --pd_active_requests_;
    pd_generated_tokens_ += generated;
    if (failed)
      ++pd_failed_requests_;
    else
      ++pd_completed_requests_;
  };
  auto emit = [&](Request& request, int token) {
    if (request.submission.handle->cancelled() || app_->is_sentence_ending(token)) return;
    const auto piece = app_->decode_tokens({token});
    const auto text = request.text + piece;
    size_t end = std::string::npos;
    for (const auto& stop : request.submission.stop)
      if (!stop.empty()) end = std::min(end, text.find(stop));
    if (end != std::string::npos) {
      if (end > request.text.size())
        request.submission.handle->push_token(
            text.substr(request.text.size(), end - request.text.size()));
      request.text = text.substr(0, end);
      request.submission.handle->finish(app_->postprocess_decoded_text(request.text), false,
                                        "stop_sequence");
      request.stopped_by_sequence = true;
    } else {
      request.submission.handle->push_token(piece);
      request.text = text;
    }
  };

  for (;;) {
    // RPCs run on one asynchronous channel per lane. No GPU owner loop or
    // coordinator mutex waits for a Prefill chunk or network round trip.
    for (int index = 0; index < 2; ++index) {
      auto& lane = lanes[index];
      if (lane.releasing.valid() &&
          lane.releasing.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        try {
          lane.releasing.get();
        } catch (const std::exception& error) {
          LOG(FATAL) << error.what() << "; PD domain fail-stop";
        }
      }
      if (lane.pending.valid() &&
          lane.pending.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        RemotePrefillResult result;
        try {
          result = lane.pending.get();
        } catch (const std::exception& error) {
          LOG(FATAL) << error.what() << "; PD domain fail-stop";
        }
        auto request = std::move(lane.request);
        DecodeKVReservation reservation;
        try {
          if (result.failed) {
            finish(request, !request->submission.handle->cancelled(), result.error, {}, 0);
          } else if (request->submission.handle->cancelled()) {
            finish(request, false, "cancelled", {}, 0);
          } else {
            if (result.prompt_tokens != request->submission.prompt_tokens || result.first_token < 0)
              throw std::runtime_error("PD request handoff identity mismatch");
            if (!request->submission.generation_config.ignore_eos &&
                request->submission.generation_config.min_new_tokens < 1 &&
                app_->is_sentence_ending(result.first_token)) {
              finish(request, false, "eos", {}, 1);
            } else {
              copied_bytes += copy_shared_kv(app_, result, &reservation);
              emit(*request, result.first_token);
              const auto id = decoder.add_decode_ready_request(
                  reservation.decode_request_id, request->submission.prompt_tokens,
                  request->submission.generation_config, result.computed_tokens,
                  result.first_token);
              reservation = {};
              decoding[id] = request;
            }
          }
        } catch (const std::exception& error) {
          if (reservation.valid())
            app_->kv_cache_manager()->free_request(reservation.decode_request_id);
          finish(request, true, error.what(), {}, 0);
        }
        if (!result.failed && result.handoff_id.valid()) {
          lane.releasing =
              std::async(std::launch::async, [rpc = lane.rpc, id = result.handoff_id.value] {
                exchange(rpc, {{"type", "kv_release"}, {"handoff_id", id}});
              });
        }
      }
      if (!lane.pending.valid() && !lane.releasing.valid()) {
        std::shared_ptr<Request> request;
        {
          std::lock_guard<std::mutex> lock(mu_);
          auto& queue = split_submissions_[index];
          while (!queue.empty() && queue.front().handle->cancelled()) {
            queue.front().handle->finish("", false, "cancelled");
            queue.pop_front();
            --pending_submissions_;
            ++lane.cancelled;
          }
          if (!queue.empty()) {
            const auto charge = SplitPDAdmission::charge(
                queue.front().prompt_tokens.size(), queue.front().generation_config.max_new_tokens,
                app_->kv_cache_manager()->block_size());
            if (!admission.fits_request(index, charge)) {
              queue.front().handle->finish("", true, "request_exceeds_lane_kv_budget");
              queue.pop_front();
              --pending_submissions_;
              ++pd_failed_requests_;
            } else if (admission.can_acquire(index, charge)) {
              request = std::make_shared<Request>();
              request->submission = std::move(queue.front());
              queue.pop_front();
              request->lane = index;
              request->charged_tokens = charge;
              --pending_submissions_;
              ++pd_active_requests_;
              admission.acquire(index, charge);
            }
          }
        }
        if (request) {
          lane.request = request;
          ++lane.dispatched;
          lane.pending =
              std::async(std::launch::async, prefill, lane.rpc, request->submission.prompt_tokens,
                         request->submission.generation_config, request->submission.handle);
        }
      }
    }
    for (const auto& item : decoding)
      if (item.second->submission.handle->cancelled() || item.second->stopped_by_sequence)
        decoder.cancel_request(item.first,
                               item.second->stopped_by_sequence ? "stop_sequence" : "cancelled");
    {
      // Metadata owns compute leases only through this step, before pop_finished
      // and metrics expose the returned capacity.
      auto output = decoder.schedule_step();
      if (output.total_tokens > 0) {
        CHECK_EQ(output.num_prefill_seqs, 0) << "Decode worker must never recompute Prefill";
        auto batch = decoder.build_decode_batch(output, app_->model_stream());
        auto status = app_->pd_forward_decode_batch(batch);
        if (!status) LOG(FATAL) << "PD decode forward failed: " << status.get_err_msg();
        auto sampled = app_->pd_batch_sample_decode(batch, output);
        for (int i = 0; i < sampled.size(); ++i) {
          const auto* seq = output.scheduled_seqs.at(batch.sample_row_to_request.at(i));
          emit(*decoding.at(seq->client_request_id), sampled.tokens[i]);
        }
        decoder.process_outputs(output, batch, sampled,
                                [&](int token) { return app_->is_sentence_ending(token); });
        ++decode_steps;
        max_decode_batch =
            std::max(max_decode_batch, static_cast<uint64_t>(output.num_decode_seqs));
      }
    }
    for (const auto& seq : decoder.pop_finished()) {
      auto request = decoding.at(seq.client_request_id);
      finish(request, seq.failed, seq.finish_reason, seq.output_tokens, seq.generated_tokens);
      decoding.erase(seq.client_request_id);
    }
    bool idle = decoding.empty();
    for (const auto& lane : lanes) idle = idle && !lane.pending.valid() && !lane.releasing.valid();
    std::unique_lock<std::mutex> lock(mu_);
    cancellations_.clear();  // Handles carry cancellation across P and D stages.
    split_pd_metrics_ = {{"short_prompt_tokens", config_.pd.short_prompt_tokens},
                         {"kv_device_copy_bytes", copied_bytes},
                         {"kv_host_copy_bytes", 0},
                         {"decode_steps", decode_steps},
                         {"max_decode_batch", max_decode_batch}};
    split_pd_metrics_["recent_requests"] = recent;
    for (int index = 0; index < 2; ++index) {
      const auto& lane = lanes[index];
      split_pd_metrics_[index == 0 ? "short" : "long"] = {
          {"active", admission.active(index)},     {"reserved_tokens", admission.used(index)},
          {"prefill_dispatched", lane.dispatched}, {"completed", lane.completed},
          {"cancelled", lane.cancelled},           {"generated_tokens", lane.generated}};
    }
    if (idle && pending_submissions_ == 0 && stopping_) break;
    if (idle && pending_submissions_ == 0)
      cv_.wait(lock, [&] { return stopping_ || pending_submissions_ > 0; });
    else if (decoding.empty())
      cv_.wait_for(lock, std::chrono::milliseconds(1));
  }
}
}  // namespace serving
