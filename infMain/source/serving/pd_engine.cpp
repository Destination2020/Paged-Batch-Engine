#include "serving/pd_engine.h"

#include <glog/logging.h>

#include <utility>

#include "serving/serving_benchmark_app.h"

namespace serving {

InProcPrefillEngine::InProcPrefillEngine(ServingBenchmarkApp* app,
                                         base::KVCacheManager* kv_manager,
                                         InProcPDEngineConfig config)
    : app_(app), kv_manager_(kv_manager), config_(std::move(config)) {
  CHECK_NE(app_, nullptr);
  CHECK_NE(kv_manager_, nullptr);
  scheduler_ = std::make_unique<Scheduler>(config_.scheduler_config, kv_manager_);
  worker_ = std::make_unique<InProcPrefillWorker>(app_, scheduler_.get());
}

base::Status InProcPrefillEngine::submit_request(
    std::vector<int32_t> prompt_tokens,
    GenerationConfig generation_config,
    PrefillSubmitResult* result) {
  if (result == nullptr) {
    return base::error::InvalidArgument("prefill engine submit result is null");
  }
  result->request_id = scheduler_->add_request(std::move(prompt_tokens), generation_config);
  return base::error::Success();
}

SchedulerOutput InProcPrefillEngine::schedule_step() {
  return scheduler_->schedule_step();
}

base::Status InProcPrefillEngine::execute_step(const SchedulerOutput& output,
                                               void* stream,
                                               PDWorkerStepOutput* step_output) {
  return worker_->execute_prefill_step(output, stream, step_output);
}

base::Status InProcPrefillEngine::execute_and_process_step(
    const SchedulerOutput& output,
    void* stream,
    PDWorkerStepOutput* step_output) {
  base::Status status = execute_step(output, stream, step_output);
  if (!status) {
    return status;
  }
  scheduler_->process_outputs(
      output, step_output->batch, step_output->sampled_token_view(),
      [&](int32_t token) { return app_->is_sentence_ending(token); });
  return base::error::Success();
}

InProcDecodeEngine::InProcDecodeEngine(ServingBenchmarkApp* app,
                                       base::KVCacheManager* kv_manager,
                                       InProcPDEngineConfig config)
    : app_(app), kv_manager_(kv_manager), config_(std::move(config)) {
  CHECK_NE(app_, nullptr);
  CHECK_NE(kv_manager_, nullptr);
  scheduler_ = std::make_unique<Scheduler>(config_.scheduler_config, kv_manager_);
  worker_ = std::make_unique<InProcDecodeWorker>(app_, scheduler_.get());
}

base::Status InProcDecodeEngine::submit_decode_ready_request(
    std::vector<int32_t> prompt_tokens,
    GenerationConfig generation_config,
    int64_t* request_id) {
  if (request_id == nullptr) {
    return base::error::InvalidArgument("decode engine request_id output is null");
  }
  *request_id = scheduler_->add_request(std::move(prompt_tokens), generation_config);
  return base::error::Success();
}

base::Status InProcDecodeEngine::submit_decode_ready_request(
    const DecodeKVReservation& reservation,
    std::vector<int32_t> prompt_tokens,
    GenerationConfig generation_config,
    DecodeReadySubmitResult* result) {
  return submit_decode_ready_request(reservation, std::move(prompt_tokens),
                                     generation_config, reservation.first_token,
                                     result);
}

base::Status InProcDecodeEngine::submit_decode_ready_request(
    const DecodeKVReservation& reservation,
    std::vector<int32_t> prompt_tokens,
    GenerationConfig generation_config,
    int32_t first_token,
    DecodeReadySubmitResult* result) {
  if (result == nullptr) {
    return base::error::InvalidArgument("decode-ready submit result is null");
  }
  if (!reservation.valid()) {
    return base::error::InvalidArgument("decode-ready reservation is invalid");
  }
  result->request_id = scheduler_->add_decode_ready_request(
      reservation.decode_request_id, std::move(prompt_tokens), generation_config,
      reservation.reserved_tokens, first_token);
  return base::error::Success();
}

SchedulerOutput InProcDecodeEngine::schedule_step() {
  return scheduler_->schedule_step();
}

base::Status InProcDecodeEngine::execute_step(const SchedulerOutput& output,
                                              void* stream,
                                              PDWorkerStepOutput* step_output) {
  return worker_->execute_decode_step(output, stream, step_output);
}

}  // namespace serving
