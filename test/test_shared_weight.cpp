#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

#include "data/shared_weight.h"
#include "op/layer.h"

namespace {

data::SharedWeightDescriptor Descriptor() {
  data::SharedWeightDescriptor value;
  value.state = data::SharedWeightState::kReady;
  value.owner_incarnation = 11;
  value.allocation_id = 7;
  value.generation = 3;
  value.device = 0;
  value.bytes = 4096;
  value.dtype = base::DataType::kDataTypeBf16;
  value.model_content[0] = 1;
  value.layout_identity[0] = 2;
  return value;
}

TEST(SharedWeightDescriptor, RoundTripsCompleteIdentity) {
  const auto source = Descriptor();
  std::vector<uint8_t> wire;
  ASSERT_TRUE(data::EncodeSharedWeightDescriptor(source, &wire));
  data::SharedWeightDescriptor decoded;
  ASSERT_TRUE(data::DecodeSharedWeightDescriptor(wire.data(), wire.size(), &decoded));
  EXPECT_EQ(decoded.owner_incarnation, source.owner_incarnation);
  EXPECT_EQ(decoded.allocation_id, source.allocation_id);
  EXPECT_EQ(decoded.generation, source.generation);
  EXPECT_EQ(decoded.device, source.device);
  EXPECT_EQ(decoded.bytes, source.bytes);
  EXPECT_EQ(decoded.dtype, source.dtype);
  EXPECT_EQ(decoded.model_content, source.model_content);
  EXPECT_EQ(decoded.layout_identity, source.layout_identity);
}

TEST(SharedWeightDescriptor, RejectsIncompleteAndCorruptPayloads) {
  auto source = Descriptor();
  source.state = data::SharedWeightState::kLoading;
  std::vector<uint8_t> wire;
  EXPECT_FALSE(data::EncodeSharedWeightDescriptor(source, &wire));
  source = Descriptor();
  ASSERT_TRUE(data::EncodeSharedWeightDescriptor(source, &wire));
  data::SharedWeightDescriptor decoded;
  EXPECT_FALSE(data::DecodeSharedWeightDescriptor(wire.data(), wire.size() - 1, &decoded));
  wire[0] ^= 0xff;
  EXPECT_FALSE(data::DecodeSharedWeightDescriptor(wire.data(), wire.size(), &decoded));
}

TEST(SharedWeightOwner, FailedLoadNeverPublishesDescriptor) {
  data::SharedWeightOwner owner;
  data::Digest256 content{};
  data::Digest256 layout{};
  std::string error;
  EXPECT_FALSE(owner.load_file("/definitely/not/a/pbe/model", 0, 19,
                               content, layout,
                               base::DataType::kDataTypeBf16, 4096, &error));
  EXPECT_EQ(owner.stats().state, data::SharedWeightState::kFailed);
  data::SharedWeightDescriptor descriptor;
  EXPECT_FALSE(owner.descriptor(&descriptor, &error));
}

TEST(SharedWeightBinding, BindsBorrowedViewAndRejectsBounds) {
  std::array<uint16_t, 32> host{};
  op::LayerParam layer(base::DeviceType::kDeviceCPU, op::LayerType::kLayerUnknown);
  layer.reset_weight_size(1);
  ASSERT_TRUE(layer.set_weight(0, {8}, host.data() + 4,
                               base::DeviceType::kDeviceCPU,
                               base::DataType::kDataTypeBf16));
  auto* device = reinterpret_cast<void*>(uintptr_t{0x100000});
  uint64_t views = 0;
  uint64_t bytes = 0;
  ASSERT_TRUE(layer.bind_external_weights(host.data(), device, sizeof(host),
                                          base::DataType::kDataTypeBf16,
                                          &views, &bytes));
  EXPECT_EQ(layer.get_weight(0).device_type(), base::DeviceType::kDeviceCUDA);
  EXPECT_EQ(layer.get_weight(0).get_buffer()->ptr(),
            reinterpret_cast<void*>(uintptr_t{0x100008}));
  EXPECT_EQ(views, 1);
  EXPECT_EQ(bytes, 16);

  op::LayerParam overflow(base::DeviceType::kDeviceCPU, op::LayerType::kLayerUnknown);
  overflow.reset_weight_size(1);
  ASSERT_TRUE(overflow.set_weight(0, {8}, host.data() + 28,
                                  base::DeviceType::kDeviceCPU,
                                  base::DataType::kDataTypeBf16));
  EXPECT_FALSE(overflow.bind_external_weights(host.data(), device, sizeof(host),
                                              base::DataType::kDataTypeBf16,
                                              nullptr, nullptr));
}

}  // namespace
