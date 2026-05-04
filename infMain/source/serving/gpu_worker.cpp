#include "serving/gpu_worker.h"

#include <glog/logging.h>

#include <utility>

#include "serving/pd_worker.h"
#include "serving/serving_benchmark_app.h"

namespace serving {

InProcGpuWorker::InProcGpuWorker(ServingBenchmarkApp* app, Scheduler* scheduler)
    : app_(app), scheduler_(scheduler) {
  CHECK_NE(app_, nullptr);
  CHECK_NE(scheduler_, nullptr);
  pd_worker_pair_ = std::make_unique<InProcPDWorkerPair>(app_, scheduler_);
}

InProcGpuWorker::~InProcGpuWorker() = default;

base::Status InProcGpuWorker::execute_step(const SchedulerOutput& sched_out,
                                           void* stream,
                                           WorkerStepOutput* output) {
  CHECK_NE(output, nullptr);
  output->batch = {};
  output->sampled_tokens.clear();
  if (sched_out.total_tokens == 0) {
    return base::error::Success();
  }

  PDWorkerStepOutput pd_output;
  base::Status status = pd_worker_pair_->execute_step(sched_out, stream, &pd_output);
  if (!status) {
    return status;
  }
  output->batch = std::move(pd_output.batch);
  output->sampled_tokens = std::move(pd_output.sampled_tokens);
  return status;
}

}  // namespace serving
