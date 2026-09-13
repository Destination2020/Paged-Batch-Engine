#include <gtest/gtest.h>

#include "cache/recovery_dependency.h"

namespace {
cache::RecoveryObject Media() {
  cache::RecoveryObject value;
  value.object_id = "media"; value.version = "sha256-v1";
  value.kind = cache::RecoveryKind::kPreserveUntilReleased;
  value.host_serviceable = true; value.bytes = 100;
  return value;
}
cache::RecoveryObject Feature() {
  cache::RecoveryObject value;
  value.object_id = "feature"; value.version = "encoder-v1";
  value.kind = cache::RecoveryKind::kRecomputable;
  value.continuation = cache::ContinuationRequirement::kExact;
  value.recipe = "encode(media,processor-v1)"; value.recipe_is_exact = false;
  value.dependencies = {{"media", "sha256-v1"}};
  value.gpu_serviceable = true; value.bytes = 200;
  return value;
}
}  // namespace

TEST(RecoveryDependencyTest, ExactFeatureKeepsLastReplicaWithoutProvenRecipe) {
  cache::RecoveryDependencyGraph graph;
  ASSERT_TRUE(graph.publish(Media()));
  ASSERT_TRUE(graph.publish(Feature()));
  std::string reason;
  EXPECT_FALSE(graph.can_remove_replica("feature", cache::RecoveryTier::kGpu, &reason));
  EXPECT_EQ(reason, "last_necessary_replica");
  EXPECT_EQ(graph.choose_pressure_action("feature", 0), cache::PressureAction::kKeep);
}

TEST(RecoveryDependencyTest, DemoteCommitsHostBeforeGpuMayDisappear) {
  cache::RecoveryDependencyGraph graph;
  ASSERT_TRUE(graph.publish(Media()));
  ASSERT_TRUE(graph.publish(Feature()));
  EXPECT_EQ(graph.choose_pressure_action("feature", 200),
            cache::PressureAction::kDemoteToHost);
  ASSERT_TRUE(graph.set_in_flight("feature", true));
  EXPECT_FALSE(graph.can_remove_replica("feature", cache::RecoveryTier::kGpu));
  ASSERT_TRUE(graph.set_residency("feature", cache::RecoveryTier::kHost, true));
  ASSERT_TRUE(graph.set_in_flight("feature", false));
  EXPECT_TRUE(graph.can_remove_replica("feature", cache::RecoveryTier::kGpu));
}

TEST(RecoveryDependencyTest, RejectsMissingOldAndCyclicDependencies) {
  cache::RecoveryDependencyGraph graph;
  std::string error;
  auto feature = Feature();
  EXPECT_FALSE(graph.publish(feature, &error));
  EXPECT_EQ(error, "missing_dependency");
  ASSERT_TRUE(graph.publish(Media()));
  feature.dependencies[0].required_version = "old";
  EXPECT_FALSE(graph.publish(feature, &error));
  EXPECT_EQ(error, "dependency_version_mismatch");
  feature.dependencies[0].required_version = "sha256-v1";
  ASSERT_TRUE(graph.publish(feature));
  auto media = Media();
  media.dependencies = {{"feature", "encoder-v1"}};
  EXPECT_FALSE(graph.publish(media, &error));
  EXPECT_EQ(error, "dependency_cycle");
}

TEST(RecoveryDependencyTest, HostFullDropsOnlyProvenRecomputableObjects) {
  cache::RecoveryDependencyGraph graph;
  ASSERT_TRUE(graph.publish(Media()));
  auto feature = Feature();
  feature.recipe_is_exact = true;
  ASSERT_TRUE(graph.publish(feature));
  EXPECT_EQ(graph.choose_pressure_action("feature", 0),
            cache::PressureAction::kDropAndRecompute);
  ASSERT_TRUE(graph.acquire("feature", true, false));
  EXPECT_EQ(graph.choose_pressure_action("feature", 1000), cache::PressureAction::kKeep);
  ASSERT_TRUE(graph.release("feature", true, false));
}

TEST(RecoveryDependencyTest, TenThousandAcquireReleaseCyclesReclaimCleanly) {
  cache::RecoveryDependencyGraph graph;
  ASSERT_TRUE(graph.publish(Media()));
  for (int i = 0; i < 10000; ++i) {
    ASSERT_TRUE(graph.acquire("media", i % 3 == 0, i % 3 == 1));
    ASSERT_TRUE(graph.release("media", i % 3 == 0, i % 3 == 1));
  }
  EXPECT_TRUE(graph.invariant_holds());
  EXPECT_TRUE(graph.erase("media"));
  EXPECT_EQ(graph.size(), 0u);
}

TEST(RecoveryDependencyTest, WatermarksHaveHysteresis) {
  cache::RecoveryWatermarks watermarks(90, 60);
  EXPECT_FALSE(watermarks.update(80));
  EXPECT_TRUE(watermarks.update(95));
  EXPECT_TRUE(watermarks.update(70));
  EXPECT_FALSE(watermarks.update(60));
}
