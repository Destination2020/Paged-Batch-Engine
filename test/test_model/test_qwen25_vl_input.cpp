#include <gtest/gtest.h>

#include "model/multimodal_input.h"

namespace {

model::MultimodalSequencePlan OneImagePlan() {
  model::MultimodalSequencePlan plan;
  plan.expanded_token_ids.resize(70, 151655);
  for (auto& axis : plan.position_ids) {
    axis.resize(plan.expanded_token_ids.size());
  }
  plan.feature_refs.push_back({"image-content", "qwen25-vl-bf16", 64, 2048, 262144, "abc"});
  plan.media_spans.push_back({3, 67, 0, {1, 16, 16}});
  plan.decode_position = {70, -51};
  return plan;
}

TEST(Qwen25VLInputTest, AcceptsFrozenOneImageContract) {
  auto plan = OneImagePlan();
  EXPECT_EQ(model::ValidateMultimodalSequencePlan(plan), model::MultimodalInputError::kOk);
  EXPECT_EQ(plan.decode_position.next_token_offset, 70u);
  EXPECT_EQ(plan.decode_position.rope_delta, -51);
}

TEST(Qwen25VLInputTest, KeepsTokenOffsetsSeparateFromThreeAxisPositions) {
  auto plan = OneImagePlan();
  plan.position_ids[0][3] = 3;
  plan.position_ids[1][3] = 3;
  plan.position_ids[2][3] = 3;
  plan.position_ids[0][4] = 3;
  plan.position_ids[1][4] = 3;
  plan.position_ids[2][4] = 4;
  EXPECT_EQ(model::ValidateMultimodalSequencePlan(plan), model::MultimodalInputError::kOk);
  EXPECT_EQ(plan.media_spans[0].token_begin, 3u);
  EXPECT_NE(plan.position_ids[2][4], static_cast<int32_t>(plan.media_spans[0].token_begin));
}

TEST(Qwen25VLInputTest, RejectsFeatureAndSpanLengthMismatchBeforeCopy) {
  auto plan = OneImagePlan();
  plan.feature_refs[0].rows = 63;
  std::string reason;
  EXPECT_EQ(model::ValidateMultimodalSequencePlan(plan, &reason),
            model::MultimodalInputError::kFeatureLengthMismatch);
  EXPECT_FALSE(reason.empty());
}

TEST(Qwen25VLInputTest, RejectsOverlappingMediaSpans) {
  auto plan = OneImagePlan();
  plan.feature_refs.push_back({"second", "qwen25-vl-bf16", 8, 2048, 32768, "def"});
  plan.media_spans.push_back({60, 68, 1, {1, 4, 8}});
  EXPECT_EQ(model::ValidateMultimodalSequencePlan(plan),
            model::MultimodalInputError::kOverlappingSpan);
}

TEST(Qwen25VLInputTest, ValidatesBundleByteCoverage) {
  model::TensorBundleSchema bundle;
  bundle.total_bytes = 262168;
  bundle.components = {
      {"image_features", 4, {64, 2048}, 0, 262144, "features"},
      {"image_grid_thw", 2, {1, 3}, 262144, 24, "grid"},
  };
  EXPECT_EQ(model::ValidateTensorBundleSchema(bundle), model::MultimodalInputError::kOk);
  bundle.components[1].byte_length = 25;
  EXPECT_EQ(model::ValidateTensorBundleSchema(bundle),
            model::MultimodalInputError::kInvalidBundleComponent);
}

}  // namespace
