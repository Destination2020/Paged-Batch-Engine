#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include "serving/sequence_state.h"
#include "serving/serving_zmq_rpc.h"

TEST(RequestSamplingTest, BatchReorderAndRetryPreserveRandomInputs) {
  std::array<serving::SequenceState, 3> requests;
  for (size_t i = 0; i < requests.size(); ++i) {
    requests[i].generation_config.sampling.seed = 7000 + i;
  }
  const auto now = std::chrono::steady_clock::now();
  for (uint64_t step = 0; step < 1000; ++step) {
    for (int index : {2, 0, 1}) {
      auto& seq = requests[index];
      const auto expected = serving::request_sample_uniform(7000 + index, step);
      const auto draw = serving::request_sample_uniform(
          seq.generation_config.sampling.seed, seq.sampling_counter);
      EXPECT_EQ(draw, expected);
      EXPECT_GE(draw, 0.0f);
      EXPECT_LT(draw, 1.0f);
      // Physical handle replacement and repeated sampling before commit do not
      // consume another draw. This is also the pending-token checkpoint rule.
      seq.request_id += 123456789;
      EXPECT_EQ(serving::request_sample_uniform(seq.generation_config.sampling.seed,
                                               seq.sampling_counter), draw);
      seq.record_generated_token(now);
    }
  }
  EXPECT_NE(serving::request_sample_uniform(7000, 0),
            serving::request_sample_uniform(7001, 0));
  EXPECT_NE(serving::request_sample_uniform(7000, 0),
            serving::request_sample_uniform(7000, 1));
}

TEST(RequestSamplingTest, CopyOfProgressResumesRandomStream) {
  serving::SequenceState seq;
  seq.generation_config.sampling.seed = UINT64_C(0xf123456789abcdef);
  const auto now = std::chrono::steady_clock::now();
  for (int i = 0; i < 17; ++i) seq.record_generated_token(now);
  auto restored = seq;
  restored.request_id = INT64_C(1) << 42;
  restored.generated_tokens = 0;  // Metric reset must not reset random progress.
  for (int i = 0; i < 20; ++i) {
    EXPECT_EQ(serving::request_sample_uniform(seq.generation_config.sampling.seed,
                                             seq.sampling_counter),
              serving::request_sample_uniform(restored.generation_config.sampling.seed,
                                             restored.sampling_counter));
    seq.record_generated_token(now);
    restored.record_generated_token(now);
  }
}

TEST(RequestMetricsTest, PreservesUnevenTokenGapsSeparatelyFromRequestMean) {
  serving::SequenceState seq;
  const auto start = std::chrono::steady_clock::time_point{};
  seq.arrival_time = start;
  EXPECT_TRUE(seq.token_gaps_ms().empty());
  seq.record_generated_token(start + std::chrono::milliseconds(10));
  EXPECT_TRUE(seq.token_gaps_ms().empty());
  seq.record_generated_token(start + std::chrono::milliseconds(11));
  seq.record_generated_token(start + std::chrono::milliseconds(110));
  EXPECT_EQ(seq.token_gaps_ms(), (std::vector<double>{1.0, 99.0}));
  EXPECT_DOUBLE_EQ(seq.mean_token_gap_ms(), 50.0);
  EXPECT_DOUBLE_EQ(seq.itl_ms(), 50.0);
  EXPECT_DOUBLE_EQ(seq.ttft_ms(), 10.0);
  EXPECT_EQ(seq.token_times.size(), 3);
}

TEST(RequestWireTest, PreservesFullWidthSeedAndHandles) {
  serving::GenerationConfig config;
  config.sampling.seed = UINT64_C(0xf123456789abcdef);
  const auto restored = serving::generation_config_from_json(
      serving::generation_config_to_json(config));
  EXPECT_EQ(restored.sampling.seed, config.sampling.seed);
  serving::LayerKVTransferRequest request;
  request.src_request_id = (INT64_C(1) << 52) + 37;
  request.dst_request_id = (INT64_C(1) << 61) + 19;
  const auto decoded = serving::layer_kv_transfer_request_from_json(
      serving::layer_kv_transfer_request_to_json(request));
  EXPECT_EQ(decoded.src_request_id, request.src_request_id);
  EXPECT_EQ(decoded.dst_request_id, request.dst_request_id);
}
