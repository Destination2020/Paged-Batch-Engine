#include <gtest/gtest.h>

#include "data/tensor_bundle.h"

namespace {

data::TensorBundleSchema ValidBundle() {
  data::TensorBundleSchema schema;
  schema.alignment = 16;
  schema.total_bytes = 96;
  schema.components = {
      {"features", data::TensorDType::kBFloat16, {4, 8}, 0, 64, {}},
      {"grid", data::TensorDType::kInt32, {3}, 64, 12, {}},
  };
  return schema;
}

TEST(TensorBundleTest, AcceptsAlignedDisjointComponentsWithPadding) {
  std::string reason;
  EXPECT_EQ(data::ValidateTensorBundle(ValidBundle(), &reason), data::DataError::kOk)
      << reason;
}

TEST(TensorBundleTest, RejectsShapeBytesOverlapAndDuplicateNames) {
  auto shape = ValidBundle();
  shape.components[0].byte_length = 62;
  EXPECT_EQ(data::ValidateTensorBundle(shape), data::DataError::kCoverageMismatch);

  auto overlap = ValidBundle();
  overlap.components[1].byte_offset = 48;
  EXPECT_EQ(data::ValidateTensorBundle(overlap), data::DataError::kCoverageMismatch);

  auto duplicate = ValidBundle();
  duplicate.components[1].name = "features";
  EXPECT_EQ(data::ValidateTensorBundle(duplicate), data::DataError::kInvalidArgument);
}

TEST(TensorBundleTest, RejectsOverflowBeforeAllocation) {
  auto schema = ValidBundle();
  schema.components[0].shape = {UINT64_MAX, 2};
  EXPECT_EQ(data::ValidateTensorBundle(schema), data::DataError::kCoverageMismatch);
}

}  // namespace
