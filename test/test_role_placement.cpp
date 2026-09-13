#include <gtest/gtest.h>

#include "serving/role_placement.h"

namespace {
serving::RoleCostSnapshot Role(const char* name) {
  return {name, std::string(name) + "-inc", "model-v1", "bf16-v1", 1000,
          0, 0, 1 << 20, 0, 1024.0, 0.2, 1.0, 0.1};
}
serving::PlacementRequest Request() {
  serving::PlacementRequest request;
  request.model_revision = "model-v1";
  request.representation = "bf16-v1";
  request.required_bytes = 1024;
  request.prefix_bytes = 8192;
  request.prefill_tokens = 32;
  request.decode_tokens = 8;
  request.now_monotonic_ms = 1010;
  return request;
}
}  // namespace

TEST(RolePlacementTest, EmptyRemoteBeatsCachedButLongQueue) {
  auto cached = Role("cached");
  cached.queued_tokens = 100;
  cached.resident_prefix_bytes = 8192;
  auto remote = Role("remote");
  auto decision = serving::ChooseRole(Request(), {cached, remote});
  ASSERT_TRUE(decision.found);
  EXPECT_EQ(decision.worker, "remote");
}

TEST(RolePlacementTest, FreshCacheWinsWhenTransferDominates) {
  auto cached = Role("cached");
  cached.resident_prefix_bytes = 8192;
  auto remote = Role("remote");
  remote.bandwidth_bytes_per_ms = 64;
  auto decision = serving::ChooseRole(Request(), {remote, cached});
  ASSERT_TRUE(decision.found);
  EXPECT_EQ(decision.worker, "cached");
}

TEST(RolePlacementTest, StaleUnknownQueueIsNotTreatedAsZero) {
  auto stale = Role("stale");
  stale.observed_monotonic_ms = 0;
  stale.resident_prefix_bytes = 8192;
  auto fresh = Role("fresh");
  auto request = Request();
  request.stale_after_ms = 100;
  auto decision = serving::ChooseRole(request, {stale, fresh});
  ASSERT_TRUE(decision.found);
  EXPECT_EQ(decision.worker, "fresh");
  EXPECT_TRUE(decision.candidates[0].stale);
}

TEST(RolePlacementTest, FiltersBudgetIncarnationAndRetriesAnotherRole) {
  auto full = Role("full");
  full.admissible_bytes = 0;
  auto old = Role("old");
  old.model_revision = "model-v0";
  auto retry = Role("retry");
  auto request = Request();
  request.excluded_workers = {"retry"};
  EXPECT_FALSE(serving::ChooseRole(request, {full, old, retry}).found);
  request.excluded_workers.clear();
  auto decision = serving::ChooseRole(request, {full, old, retry});
  ASSERT_TRUE(decision.found);
  EXPECT_EQ(decision.incarnation, "retry-inc");
}

TEST(RolePlacementTest, HotspotLimitAndHysteresisAreBounded) {
  auto incumbent = Role("incumbent");
  incumbent.resident_prefix_bytes = 8192;
  auto challenger = Role("challenger");
  challenger.resident_prefix_bytes = 8192;
  challenger.rpc_layout_ms = 0.05;
  auto request = Request();
  request.incumbent = "challenger";
  request.hysteresis_ms = 0.1;
  EXPECT_EQ(serving::ChooseRole(request, {incumbent, challenger}).worker,
            "challenger");
  challenger.inflight_placements = 8;
  EXPECT_EQ(serving::ChooseRole(request, {incumbent, challenger}).worker,
            "incumbent");
}
