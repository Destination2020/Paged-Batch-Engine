#include <gtest/gtest.h>

#include "data/protocol.h"

TEST(DataProtocolTest, DataRefHasStableRoundTripAndFixedWidth) {
  data::DataRef input;
  input.kind = data::DataKind::kTensorBundle;
  input.content.digest[0] = 0x12;
  input.content.digest[31] = 0x34;
  input.representation.digest[7] = 0x56;
  input.allocation = {99, 1234, 7};
  input.logical_bytes = 4096;
  std::vector<uint8_t> wire;
  ASSERT_EQ(data::EncodeDataRef(input, &wire), data::DataError::kOk);
  ASSERT_EQ(wire.size(), data::kDataRefWireBytes);
  data::DataRef output;
  ASSERT_EQ(data::DecodeDataRef(wire.data(), wire.size(), &output), data::DataError::kOk);
  EXPECT_EQ(output, input);
}

TEST(DataProtocolTest, RejectsVersionLengthAndReservedField) {
  data::DataRef input;
  input.kind = data::DataKind::kKVPage;
  input.allocation = {1, 2, 3};
  input.logical_bytes = 4;
  std::vector<uint8_t> wire;
  ASSERT_EQ(data::EncodeDataRef(input, &wire), data::DataError::kOk);
  data::DataRef output;
  EXPECT_EQ(data::DecodeDataRef(wire.data(), wire.size() - 1, &output),
            data::DataError::kInvalidArgument);
  wire[4] = 2;
  EXPECT_EQ(data::DecodeDataRef(wire.data(), wire.size(), &output),
            data::DataError::kUnsupportedVersion);
  wire[4] = 1;
  wire[7] = 1;
  EXPECT_EQ(data::DecodeDataRef(wire.data(), wire.size(), &output),
            data::DataError::kInvalidArgument);
}
