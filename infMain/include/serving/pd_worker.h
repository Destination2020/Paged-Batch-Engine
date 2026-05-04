#ifndef KUIPER_INCLUDE_SERVING_PD_WORKER_H_
#define KUIPER_INCLUDE_SERVING_PD_WORKER_H_

#include <cstdint>
#include <vector>

#include "base/base.h"
#include "serving/mixed_batch.h"
#include "serving/scheduler.h"

namespace serving {

class Scheduler;
class ServingBenchmarkApp;

enum class PDWorkerStepKind {
  kEmpty,
  kDecodeOnly,
  kPrefillOnly,
  kMixed,
};

PDWorkerStepKind classify_pd_worker_step(const SchedulerOutput& output);
bool pd_worker_step_has_prefill(PDWorkerStepKind kind);
bool pd_worker_step_has_decode(PDWorkerStepKind kind);

struct PDWorkerStepOutput {
  MixedBatchMetadata batch;
  std::vector<int32_t> sampled_tokens;

  SampledTokenView sampled_token_view() const {
    return {sampled_tokens.data(), static_cast<int32_t>(sampled_tokens.size())};
  }
};

class PrefillWorker {
 public:
  virtual ~PrefillWorker() = default;

  virtual base::Status execute_prefill_step(const SchedulerOutput& output,
                                            void* stream,
                                            PDWorkerStepOutput* step_output) = 0;
};

class DecodeWorker {
 public:
  virtual ~DecodeWorker() = default;

  virtual base::Status execute_decode_step(const SchedulerOutput& output,
                                           void* stream,
                                           PDWorkerStepOutput* step_output) = 0;
};

class InProcPrefillWorker final : public PrefillWorker {
 public:
  InProcPrefillWorker(ServingBenchmarkApp* app, Scheduler* scheduler);

  base::Status execute_prefill_step(const SchedulerOutput& output,
                                    void* stream,
                                    PDWorkerStepOutput* step_output) override;

 private:
  ServingBenchmarkApp* app_ = nullptr;
  Scheduler* scheduler_ = nullptr;
};

class InProcDecodeWorker final : public DecodeWorker {
 public:
  InProcDecodeWorker(ServingBenchmarkApp* app, Scheduler* scheduler);

  base::Status execute_decode_step(const SchedulerOutput& output,
                                   void* stream,
                                   PDWorkerStepOutput* step_output) override;

 private:
  ServingBenchmarkApp* app_ = nullptr;
  Scheduler* scheduler_ = nullptr;
};

class InProcPDWorkerPair final {
 public:
  InProcPDWorkerPair(ServingBenchmarkApp* app, Scheduler* scheduler);

  base::Status execute_step(const SchedulerOutput& output,
                            void* stream,
                            PDWorkerStepOutput* step_output);

 private:
  InProcPrefillWorker prefill_worker_;
  InProcDecodeWorker decode_worker_;
};

}  // namespace serving

#endif  // KUIPER_INCLUDE_SERVING_PD_WORKER_H_
