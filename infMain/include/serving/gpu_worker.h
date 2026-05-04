#ifndef KUIPER_INCLUDE_SERVING_GPU_WORKER_H_
#define KUIPER_INCLUDE_SERVING_GPU_WORKER_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "base/base.h"
#include "serving/mixed_batch.h"
#include "serving/scheduler.h"

namespace serving {

class Scheduler;
class ServingBenchmarkApp;
class InProcPDWorkerPair;

struct WorkerStepOutput {
  MixedBatchMetadata batch;
  std::vector<int32_t> sampled_tokens;

  SampledTokenView sampled_token_view() const {
    return {sampled_tokens.data(), static_cast<int32_t>(sampled_tokens.size())};
  }
};

class GpuWorker {
 public:
  virtual ~GpuWorker() = default;

  virtual base::Status execute_step(const SchedulerOutput& sched_out,
                                    void* stream,
                                    WorkerStepOutput* output) = 0;
};

class InProcGpuWorker final : public GpuWorker {
 public:
  InProcGpuWorker(ServingBenchmarkApp* app, Scheduler* scheduler);
  ~InProcGpuWorker() override;

  base::Status execute_step(const SchedulerOutput& sched_out,
                            void* stream,
                            WorkerStepOutput* output) override;

 private:
  ServingBenchmarkApp* app_ = nullptr;
  Scheduler* scheduler_ = nullptr;
  std::unique_ptr<InProcPDWorkerPair> pd_worker_pair_;
};

}  // namespace serving

#endif  // KUIPER_INCLUDE_SERVING_GPU_WORKER_H_
