#include "serving/pd_worker.h"

#include <glog/logging.h>

#include "serving/serving_benchmark_app.h"

namespace serving {

PDWorkerStepKind classify_pd_worker_step(const SchedulerOutput& output) {
  if (output.total_tokens == 0) {
    return PDWorkerStepKind::kEmpty;
  }
  if (output.num_decode_seqs > 0 && output.num_prefill_seqs == 0) {
    return PDWorkerStepKind::kDecodeOnly;
  }
  if (output.num_decode_seqs == 0 && output.num_prefill_seqs > 0) {
    return PDWorkerStepKind::kPrefillOnly;
  }
  return PDWorkerStepKind::kMixed;
}

bool pd_worker_step_has_prefill(PDWorkerStepKind kind) {
  return kind == PDWorkerStepKind::kPrefillOnly || kind == PDWorkerStepKind::kMixed;
}

bool pd_worker_step_has_decode(PDWorkerStepKind kind) {
  return kind == PDWorkerStepKind::kDecodeOnly || kind == PDWorkerStepKind::kMixed;
}

InProcPrefillWorker::InProcPrefillWorker(ServingBenchmarkApp* app, Scheduler* scheduler)
    : app_(app), scheduler_(scheduler) {
  CHECK_NE(app_, nullptr);
  CHECK_NE(scheduler_, nullptr);
}

base::Status InProcPrefillWorker::execute_prefill_step(
    const SchedulerOutput& output,
    void* stream,
    PDWorkerStepOutput* step_output) {
  CHECK_NE(step_output, nullptr);
  step_output->batch = {};
  step_output->sampled_tokens.clear();
  const PDWorkerStepKind kind = classify_pd_worker_step(output);
  if (kind == PDWorkerStepKind::kEmpty) {
    return base::error::Success();
  }
  if (!pd_worker_step_has_prefill(kind)) {
    return base::error::InvalidArgument("prefill worker got a decode-only step");
  }

  step_output->batch = scheduler_->build_mixed_batch(output, stream);
  base::Status status = app_->forward_mixed_batch(step_output->batch);
  if (!status) {
    return status;
  }
  const SampledTokenView sampled = app_->batch_sample(step_output->batch, output);
  step_output->sampled_tokens.assign(sampled.tokens, sampled.tokens + sampled.size());
  return base::error::Success();
}

InProcDecodeWorker::InProcDecodeWorker(ServingBenchmarkApp* app, Scheduler* scheduler)
    : app_(app), scheduler_(scheduler) {
  CHECK_NE(app_, nullptr);
  CHECK_NE(scheduler_, nullptr);
}

base::Status InProcDecodeWorker::execute_decode_step(
    const SchedulerOutput& output,
    void* stream,
    PDWorkerStepOutput* step_output) {
  CHECK_NE(step_output, nullptr);
  step_output->batch = {};
  step_output->sampled_tokens.clear();
  const PDWorkerStepKind kind = classify_pd_worker_step(output);
  if (kind == PDWorkerStepKind::kEmpty) {
    return base::error::Success();
  }
  if (kind != PDWorkerStepKind::kDecodeOnly) {
    return base::error::InvalidArgument(
        "decode worker currently accepts decode-only steps; mixed steps stay on prefill/mixed path");
  }

  step_output->batch = scheduler_->build_decode_batch(output, stream);
  base::Status status = app_->forward_decode_batch(step_output->batch);
  if (!status) {
    return status;
  }
  const SampledTokenView sampled = app_->batch_sample(step_output->batch, output);
  step_output->sampled_tokens.assign(sampled.tokens, sampled.tokens + sampled.size());
  return base::error::Success();
}

InProcPDWorkerPair::InProcPDWorkerPair(ServingBenchmarkApp* app, Scheduler* scheduler)
    : prefill_worker_(app, scheduler), decode_worker_(app, scheduler) {}

base::Status InProcPDWorkerPair::execute_step(const SchedulerOutput& output,
                                              void* stream,
                                              PDWorkerStepOutput* step_output) {
  const PDWorkerStepKind kind = classify_pd_worker_step(output);
  if (kind == PDWorkerStepKind::kDecodeOnly) {
    return decode_worker_.execute_decode_step(output, stream, step_output);
  }
  return prefill_worker_.execute_prefill_step(output, stream, step_output);
}

}  // namespace serving
