#include <gtest/gtest.h>

#include "serving/pd_worker.h"

namespace serving {

TEST(PDWorkerTest, ClassifiesEmptyDecodePrefillAndMixedSteps) {
  SchedulerOutput output;
  EXPECT_EQ(classify_pd_worker_step(output), PDWorkerStepKind::kEmpty);
  EXPECT_FALSE(pd_worker_step_has_prefill(PDWorkerStepKind::kEmpty));
  EXPECT_FALSE(pd_worker_step_has_decode(PDWorkerStepKind::kEmpty));

  output.total_tokens = 2;
  output.num_decode_seqs = 2;
  output.num_prefill_seqs = 0;
  EXPECT_EQ(classify_pd_worker_step(output), PDWorkerStepKind::kDecodeOnly);
  EXPECT_FALSE(pd_worker_step_has_prefill(PDWorkerStepKind::kDecodeOnly));
  EXPECT_TRUE(pd_worker_step_has_decode(PDWorkerStepKind::kDecodeOnly));

  output.total_tokens = 8;
  output.num_decode_seqs = 0;
  output.num_prefill_seqs = 1;
  EXPECT_EQ(classify_pd_worker_step(output), PDWorkerStepKind::kPrefillOnly);
  EXPECT_TRUE(pd_worker_step_has_prefill(PDWorkerStepKind::kPrefillOnly));
  EXPECT_FALSE(pd_worker_step_has_decode(PDWorkerStepKind::kPrefillOnly));

  output.total_tokens = 10;
  output.num_decode_seqs = 2;
  output.num_prefill_seqs = 1;
  EXPECT_EQ(classify_pd_worker_step(output), PDWorkerStepKind::kMixed);
  EXPECT_TRUE(pd_worker_step_has_prefill(PDWorkerStepKind::kMixed));
  EXPECT_TRUE(pd_worker_step_has_decode(PDWorkerStepKind::kMixed));
}

}  // namespace serving
