#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

#include "base/kv_cache_format.h"
#include "cache/layout_codec.h"
#include "cache/page_schema.h"
#include "cache/transfer_plan.h"

namespace cache {
namespace {

PageSchema MakePlainSchema() {
  PageSchema schema;
  EXPECT_TRUE(MakeKVPageSchema(
      1, 3, 4, 2, 8,
      base::MakeKVCacheStorageSpec(base::DataType::kDataTypeBf16,
                                   base::BlockStorageMode::kPlain),
      &schema));
  return schema;
}

PageSchema MakeFp8Schema() {
  PageSchema schema;
  EXPECT_TRUE(MakeKVPageSchema(
      1, 2, 4, 2, 8,
      base::MakeKVCacheStorageSpec(base::DataType::kDataTypeBf16,
                                   base::BlockStorageMode::kFp8E4M3PerTokenHead),
      &schema));
  return schema;
}

PageSchema MakeSchemaWithHeads(bool fp8, int32_t heads = 4,
                               uint32_t version = 1) {
  PageSchema schema;
  EXPECT_TRUE(MakeKVPageSchema(
      version, 2, 4, heads, 3,
      base::MakeKVCacheStorageSpec(
          base::DataType::kDataTypeBf16,
          fp8 ? base::BlockStorageMode::kFp8E4M3PerTokenHead
              : base::BlockStorageMode::kPlain),
      &schema));
  return schema;
}

PageSchema ShardHeads(PageSchema schema,
                      const std::vector<std::pair<int32_t, int32_t>>& ranges) {
  std::vector<ComponentDescriptor> sharded;
  for (const auto& component : schema.components) {
    for (const auto& range : ranges) {
      auto fragment = component;
      fragment.head_begin = range.first;
      fragment.head_count = range.second;
      sharded.push_back(fragment);
    }
  }
  schema.components = std::move(sharded);
  EXPECT_TRUE(schema.validate());
  return schema;
}

std::vector<size_t> IdentityOrder(size_t count) {
  std::vector<size_t> order(count);
  std::iota(order.begin(), order.end(), 0);
  return order;
}

const ComponentLayout& Find(const LayoutDescriptor& layout,
                            const ComponentDescriptor& component) {
  const auto found = std::find_if(
      layout.components.begin(), layout.components.end(), [&](const ComponentLayout& entry) {
        return entry.component.semantic_key() == component.semantic_key();
      });
  EXPECT_NE(found, layout.components.end());
  return *found;
}

void FillSemanticPattern(const LayoutDescriptor& layout, std::vector<uint8_t>* bytes) {
  for (const auto& entry : layout.components) {
    const uint8_t pattern = static_cast<uint8_t>(
        1 + entry.component.layer * 17 + static_cast<int>(entry.component.kind) * 3);
    std::fill(bytes->begin() + entry.offset_bytes,
              bytes->begin() + entry.offset_bytes + entry.size_bytes, pattern);
  }
}

void ExpectSemanticPayloadEqual(const LayoutDescriptor& left_layout,
                                const std::vector<uint8_t>& left,
                                const LayoutDescriptor& right_layout,
                                const std::vector<uint8_t>& right) {
  for (const auto& component : left_layout.components) {
    const auto& other = Find(right_layout, component.component);
    ASSERT_EQ(component.size_bytes, other.size_bytes);
    EXPECT_TRUE(std::equal(left.begin() + component.offset_bytes,
                           left.begin() + component.offset_bytes + component.size_bytes,
                           right.begin() + other.offset_bytes));
  }
}

void FillElementPattern(const LayoutDescriptor& layout,
                        std::vector<uint8_t>* bytes) {
  for (const auto& entry : layout.components) {
    const auto& component = entry.component;
    const size_t element_bytes = base::DataTypeSize(component.dtype);
    size_t offset = entry.offset_bytes;
    for (int32_t token = 0; token < component.token_count; ++token) {
      for (int32_t local_head = 0; local_head < component.head_count;
           ++local_head) {
        const int32_t head = component.head_begin + local_head;
        for (int32_t element = 0; element < component.elements_per_head;
             ++element) {
          for (size_t byte = 0; byte < element_bytes; ++byte) {
            (*bytes)[offset++] = static_cast<uint8_t>(
                1 + component.layer * 71 +
                static_cast<int32_t>(component.kind) * 31 + token * 13 +
                head * 7 + element * 3 + static_cast<int32_t>(byte));
          }
        }
      }
    }
    EXPECT_EQ(offset, entry.offset_bytes + entry.size_bytes);
  }
}

TEST(PageSchemaTest, RequiresCompleteKvAndScaleCoverage) {
  PageSchema plain = MakePlainSchema();
  EXPECT_TRUE(plain.validate());
  plain.components.pop_back();
  EXPECT_FALSE(plain.validate());

  PageSchema fp8 = MakeFp8Schema();
  EXPECT_TRUE(fp8.validate());
  fp8.components.erase(
      std::find_if(fp8.components.begin(), fp8.components.end(), [](const auto& component) {
        return component.kind == ComponentKind::kValueScale;
      }));
  EXPECT_FALSE(fp8.validate());

  PageSchema malformed = MakePlainSchema();
  malformed.storage.scale_dtype = base::DataType::kDataTypeFp32;
  EXPECT_FALSE(malformed.validate());
}

TEST(TransferPlanTest, ShuffledLayoutsAndVirtualBlocksRoundTrip) {
  const PageSchema schema = MakeFp8Schema();
  std::vector<size_t> source_order = IdentityOrder(schema.components.size());
  std::vector<size_t> host_order = source_order;
  std::reverse(host_order.begin(), host_order.end());
  std::vector<size_t> restored_order = {2, 5, 0, 7, 1, 6, 3, 4};
  std::vector<int64_t> source_blocks = {101, 203, 307, 409, 503, 601, 709, 809};
  std::vector<int64_t> host_blocks = {9001, 9002, 9003, 9004, 9005, 9006, 9007, 9008};
  std::vector<int64_t> restored_blocks = {41, 37, 31, 29, 23, 19, 17, 13};

  LayoutDescriptor source_layout;
  LayoutDescriptor host_layout;
  LayoutDescriptor restored_layout;
  ASSERT_TRUE(MakePackedLayout(schema, "gpu-layer-first", 11, source_order, source_blocks,
                               &source_layout));
  ASSERT_TRUE(MakePackedLayout(schema, "host-page-first", 12, host_order, host_blocks,
                               &host_layout));
  ASSERT_TRUE(MakePackedLayout(schema, "gpu-restored", 13, restored_order, restored_blocks,
                               &restored_layout));

  std::vector<uint8_t> source(source_layout.total_bytes);
  std::vector<uint8_t> host(host_layout.total_bytes, 0);
  std::vector<uint8_t> restored(restored_layout.total_bytes, 0);
  FillSemanticPattern(source_layout, &source);

  TransferPlanner planner;
  TransferPlan plan;
  ASSERT_TRUE(planner.plan(schema, schema, &plan));
  BoundTransfer pack;
  ASSERT_TRUE(BoundTransfer::Bind(plan, schema,
                                  {source.data(), source.size(), &source_layout},
                                  {host.data(), host.size(), &host_layout}, &pack));
  ASSERT_TRUE(ExecuteCpuTransfer(pack));
  EXPECT_EQ(pack.source_epoch(), 11u);
  EXPECT_EQ(pack.destination_epoch(), 12u);
  for (const auto& operation : pack.operations()) {
    EXPECT_EQ(operation.source_virtual_block_id,
              Find(source_layout, operation.component).virtual_block_id);
    EXPECT_EQ(operation.destination_virtual_block_id,
              Find(host_layout, operation.component).virtual_block_id);
  }

  BoundTransfer unpack;
  ASSERT_TRUE(BoundTransfer::Bind(plan, schema, {host.data(), host.size(), &host_layout},
                                  {restored.data(), restored.size(), &restored_layout}, &unpack));
  ASSERT_TRUE(ExecuteCpuTransfer(unpack));
  ExpectSemanticPayloadEqual(source_layout, source, restored_layout, restored);

  TransferPlan cached;
  ASSERT_TRUE(planner.plan(schema, schema, &cached));
  EXPECT_EQ(planner.cache_hits(), 1u);
  EXPECT_EQ(planner.cache_entries(), 1u);
}

TEST(TransferPlanTest, MissingComponentFailsBeforeAnyDestinationWrite) {
  const PageSchema schema = MakeFp8Schema();
  const std::vector<size_t> order = IdentityOrder(schema.components.size());
  LayoutDescriptor source_layout;
  LayoutDescriptor destination_layout;
  ASSERT_TRUE(MakePackedLayout(schema, "source", 1, order, {}, &source_layout));
  ASSERT_TRUE(MakePackedLayout(schema, "destination", 2, order, {}, &destination_layout));
  source_layout.components.pop_back();

  std::vector<uint8_t> source(source_layout.total_bytes, 0x27);
  std::vector<uint8_t> destination(destination_layout.total_bytes, 0xa5);
  const std::vector<uint8_t> original = destination;
  TransferPlanner planner;
  TransferPlan plan;
  ASSERT_TRUE(planner.plan(schema, schema, &plan));
  BoundTransfer bound;
  EXPECT_FALSE(BoundTransfer::Bind(plan, schema,
                                   {source.data(), source.size(), &source_layout},
                                   {destination.data(), destination.size(), &destination_layout},
                                   &bound));
  EXPECT_EQ(destination, original);

  source_layout.components.push_back(destination_layout.components.back());
  plan.operations.back() = plan.operations.front();
  EXPECT_FALSE(BoundTransfer::Bind(plan, schema,
                                   {source.data(), source.size(), &source_layout},
                                   {destination.data(), destination.size(), &destination_layout},
                                   &bound));
  EXPECT_EQ(destination, original);
}

TEST(TransferPlanTest, ShapeAndDtypeMismatchFailBeforeAnyDestinationWrite) {
  const PageSchema schema = MakePlainSchema();
  const std::vector<size_t> order = IdentityOrder(schema.components.size());
  LayoutDescriptor source_layout;
  LayoutDescriptor destination_layout;
  ASSERT_TRUE(MakePackedLayout(schema, "source", 1, order, {}, &source_layout));
  ASSERT_TRUE(MakePackedLayout(schema, "destination", 2, order, {}, &destination_layout));

  std::vector<uint8_t> source(source_layout.total_bytes, 0x33);
  std::vector<uint8_t> destination(destination_layout.total_bytes, 0x5a);
  const std::vector<uint8_t> original = destination;
  destination_layout.components.front().component.dtype = base::DataType::kDataTypeFp32;
  TransferPlanner planner;
  TransferPlan plan;
  ASSERT_TRUE(planner.plan(schema, schema, &plan));
  BoundTransfer bound;
  EXPECT_FALSE(BoundTransfer::Bind(plan, schema,
                                   {source.data(), source.size(), &source_layout},
                                   {destination.data(), destination.size(), &destination_layout},
                                   &bound));
  EXPECT_EQ(destination, original);

  PageSchema incompatible = schema;
  incompatible.head_size += 1;
  for (auto& component : incompatible.components) {
    component.elements_per_head += 1;
  }
  EXPECT_FALSE(planner.plan(schema, incompatible, &plan));
}

TEST(TransferPlanTest, IndependentAllocationsPackAndRestorePlainAndFp8) {
  for (const auto& schema : {MakePlainSchema(), MakeFp8Schema()}) {
    auto order = IdentityOrder(schema.components.size());
    LayoutDescriptor gpu_layout, host_layout, restored_layout;
    ASSERT_TRUE(MakePackedLayout(schema, "scatter-gpu", 31, order, {}, &gpu_layout));
    std::reverse(order.begin(), order.end());
    ASSERT_TRUE(MakePackedLayout(schema, "packed-host", 32, order, {}, &host_layout));
    ASSERT_TRUE(MakePackedLayout(schema, "restored-gpu", 33, order, {}, &restored_layout));
    std::vector<std::vector<uint8_t>> originals, restored;
    for (size_t i = 0; i < schema.components.size(); ++i) {
      originals.emplace_back(schema.components[i].byte_size(), static_cast<uint8_t>(i + 71));
      restored.emplace_back(schema.components[i].byte_size(), 0);
    }
    std::vector<uint8_t> host(host_layout.total_bytes, 0);
    ConstComponentEndpoint source{{}, &gpu_layout}, packed_source{{}, &host_layout};
    MutableComponentEndpoint packed{{}, &host_layout}, target{{}, &restored_layout};
    for (size_t i = 0; i < schema.components.size(); ++i) {
      const auto& component = schema.components[i];
      source.components.push_back({component, originals[i].data(), originals[i].size()});
      target.components.push_back({component, restored[i].data(), restored[i].size()});
      const auto& entry = Find(host_layout, component);
      packed.components.push_back({component, host.data() + entry.offset_bytes, entry.size_bytes});
      packed_source.components.push_back({component, host.data() + entry.offset_bytes, entry.size_bytes});
    }
    std::reverse(source.components.begin(), source.components.end());
    TransferPlanner planner;
    TransferPlan plan;
    ASSERT_TRUE(planner.plan(schema, schema, &plan));
    BoundTransfer pack, unpack;
    ASSERT_TRUE(BoundTransfer::BindComponents(plan, schema, source, packed, &pack));
    ASSERT_TRUE(ExecuteCpuTransfer(pack));
    ASSERT_TRUE(BoundTransfer::BindComponents(plan, schema, packed_source, target, &unpack));
    ASSERT_TRUE(ExecuteCpuTransfer(unpack));
    EXPECT_EQ(originals, restored);
    EXPECT_EQ(pack.source_epoch(), 31u);
    EXPECT_EQ(unpack.destination_epoch(), 33u);

    // Binding failures leave both destination bytes and an earlier binding intact.
    const auto before = host;
    const auto prior_count = pack.operations().size();
    const auto original_source = source;
    source.components.pop_back();
    EXPECT_FALSE(BoundTransfer::BindComponents(plan, schema, source, packed, &pack));
    source = original_source;
    source.components.back() = source.components.front();
    EXPECT_FALSE(BoundTransfer::BindComponents(plan, schema, source, packed, &pack));
    source = original_source;
    source.components.back().bytes = 1;
    EXPECT_FALSE(BoundTransfer::BindComponents(plan, schema, source, packed, &pack));
    source = original_source;
    source.components.back().component.dtype = base::DataType::kDataTypeFp32;
    EXPECT_FALSE(BoundTransfer::BindComponents(plan, schema, source, packed, &pack));
    source = original_source;
    auto malformed_target = packed;
    malformed_target.components.back().data = malformed_target.components.front().data;
    EXPECT_FALSE(BoundTransfer::BindComponents(plan, schema, source, malformed_target, &pack));
    plan.operations.back() = plan.operations.front();
    EXPECT_FALSE(BoundTransfer::BindComponents(plan, schema, source, packed, &pack));
    EXPECT_EQ(host, before);
    EXPECT_EQ(pack.operations().size(), prior_count);
    EXPECT_EQ(pack.source_epoch(), 31u);
  }
}

TEST(TransferPlanTest, HeadSplitMergeAndReorderPreserveElementSemantics) {
  for (bool fp8 : {false, true}) {
    const auto source_schema =
        ShardHeads(MakeSchemaWithHeads(fp8), {{0, 1}, {1, 3}});
    const auto destination_schema =
        ShardHeads(MakeSchemaWithHeads(fp8), {{0, 2}, {2, 2}});
    auto source_order = IdentityOrder(source_schema.components.size());
    auto destination_order = IdentityOrder(destination_schema.components.size());
    std::reverse(source_order.begin(), source_order.end());
    std::rotate(destination_order.begin(), destination_order.begin() + 3,
                destination_order.end());
    LayoutDescriptor source_layout;
    LayoutDescriptor destination_layout;
    ASSERT_TRUE(MakePackedLayout(source_schema, "tp-source", 41, source_order,
                                 {}, &source_layout));
    ASSERT_TRUE(MakePackedLayout(destination_schema, "tp-destination", 42,
                                 destination_order, {}, &destination_layout));
    std::vector<uint8_t> source(source_layout.total_bytes);
    FillElementPattern(source_layout, &source);
    std::vector<uint8_t> expected(destination_layout.total_bytes);
    FillElementPattern(destination_layout, &expected);

    for (const auto placement : {TransferPlacement::kExactDirect,
                                 TransferPlacement::kSenderPack,
                                 TransferPlacement::kReceiverUnpack}) {
      TransferPlanner planner;
      std::shared_ptr<const TransferPlan> plan;
      ASSERT_TRUE(planner.compile(source_schema, destination_schema, placement,
                                  &plan));
      std::vector<uint8_t> destination(destination_layout.total_bytes, 0xa5);
      BoundTransfer bound;
      ASSERT_TRUE(BoundTransfer::Bind(
          *plan, source_schema, destination_schema,
          {source.data(), source.size(), &source_layout},
          {destination.data(), destination.size(), &destination_layout},
          &bound));
      if (placement == TransferPlacement::kExactDirect) {
        ASSERT_TRUE(ExecuteCpuTransfer(bound));
      } else {
        std::vector<uint8_t> staging(bound.staging_bytes());
        ASSERT_TRUE(ExecuteCpuTransferStaged(bound, staging.data(),
                                             staging.size()));
      }
      EXPECT_EQ(destination, expected);
      EXPECT_EQ(bound.source_epoch(), 41u);
      EXPECT_EQ(bound.destination_epoch(), 42u);
      EXPECT_EQ(bound.placement(), placement);
    }
  }
}

TEST(TransferPlanTest, InvalidCoverageAndRepresentationFailBeforeWrite) {
  auto source_schema = ShardHeads(MakeSchemaWithHeads(false), {{0, 2}, {2, 2}});
  auto destination_schema =
      ShardHeads(MakeSchemaWithHeads(false), {{0, 1}, {1, 3}});
  TransferPlanner planner;
  std::shared_ptr<const TransferPlan> plan;
  ASSERT_TRUE(planner.compile(source_schema, destination_schema,
                              TransferPlacement::kExactDirect, &plan));

  auto order = IdentityOrder(source_schema.components.size());
  LayoutDescriptor source_layout;
  ASSERT_TRUE(MakePackedLayout(source_schema, "source", 1, order, {},
                               &source_layout));
  order = IdentityOrder(destination_schema.components.size());
  LayoutDescriptor destination_layout;
  ASSERT_TRUE(MakePackedLayout(destination_schema, "destination", 2, order,
                               {}, &destination_layout));
  std::vector<uint8_t> source(source_layout.total_bytes, 0x37);
  std::vector<uint8_t> destination(destination_layout.total_bytes, 0xa5);
  const auto original = destination;

  auto stale_source = source_schema;
  stale_source.version++;
  BoundTransfer bound;
  EXPECT_FALSE(BoundTransfer::Bind(
      *plan, stale_source, destination_schema,
      {source.data(), source.size(), &source_layout},
      {destination.data(), destination.size(), &destination_layout}, &bound));
  EXPECT_EQ(destination, original);

  auto missing = source_schema;
  missing.components.pop_back();
  EXPECT_FALSE(planner.compile(missing, destination_schema,
                               TransferPlacement::kExactDirect, &plan));
  auto duplicate = destination_schema;
  duplicate.components.push_back(duplicate.components.back());
  EXPECT_FALSE(planner.compile(source_schema, duplicate,
                               TransferPlacement::kExactDirect, &plan));
  auto different_dtype = destination_schema;
  different_dtype.storage = base::MakeKVCacheStorageSpec(
      base::DataType::kDataTypeBf16,
      base::BlockStorageMode::kFp8E4M3PerTokenHead);
  EXPECT_FALSE(planner.compile(source_schema, different_dtype,
                               TransferPlacement::kExactDirect, &plan));
  EXPECT_EQ(destination, original);
}

TEST(TransferPlanTest, ImmutableTemplateCacheIgnoresAddressesAndIsBounded) {
  const auto source = ShardHeads(MakeSchemaWithHeads(false), {{0, 2}, {2, 2}});
  const auto destination =
      ShardHeads(MakeSchemaWithHeads(false), {{0, 1}, {1, 3}});
  TransferPlanner planner(2, 1u << 20);
  std::shared_ptr<const TransferPlan> first;
  ASSERT_TRUE(planner.compile(source, destination,
                              TransferPlacement::kSenderPack, &first));
  for (int page = 0; page < 1000; ++page) {
    std::shared_ptr<const TransferPlan> current;
    ASSERT_TRUE(planner.compile(source, destination,
                                TransferPlacement::kSenderPack, &current));
    EXPECT_EQ(current.get(), first.get());
  }
  EXPECT_EQ(planner.cache_entries(), 1u);
  EXPECT_EQ(planner.cache_hits(), 1000u);
  EXPECT_GT(planner.cache_bytes(), 0u);

  std::shared_ptr<const TransferPlan> mode2;
  std::shared_ptr<const TransferPlan> mode3;
  ASSERT_TRUE(planner.compile(source, destination,
                              TransferPlacement::kReceiverUnpack, &mode2));
  ASSERT_TRUE(planner.compile(source, destination,
                              TransferPlacement::kExactDirect, &mode3));
  EXPECT_EQ(planner.cache_entries(), 2u);
  EXPECT_GE(planner.cache_evictions(), 1u);
  EXPECT_LE(planner.cache_bytes(), 1u << 20);
  EXPECT_NE(first.get(), mode2.get());
}

TEST(LayoutCodecTest, RejectsOverlappingOrNonPermutationLayouts) {
  const PageSchema schema = MakePlainSchema();
  std::vector<size_t> order = IdentityOrder(schema.components.size());
  order.back() = order.front();
  LayoutDescriptor layout;
  EXPECT_FALSE(MakePackedLayout(schema, "duplicate", 1, order, {}, &layout));

  order = IdentityOrder(schema.components.size());
  ASSERT_TRUE(MakePackedLayout(schema, "overlap", 1, order, {}, &layout));
  layout.components.back().offset_bytes = layout.components.front().offset_bytes;
  EXPECT_FALSE(layout.validate(schema));
}

}  // namespace
}  // namespace cache
