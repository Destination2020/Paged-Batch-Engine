#include <gtest/gtest.h>
#include "serving/node_memory_budget.h"

TEST(NodeMemoryBudgetTest, JointFailureRollsBackEveryPool) {
  serving::NodeMemoryBudget budget;
  ASSERT_TRUE(budget.set_capacity(serving::MemoryPool::kKV, 4));
  ASSERT_TRUE(budget.set_capacity(serving::MemoryPool::kBundles, 3));
  ASSERT_TRUE(budget.set_capacity(serving::MemoryPool::kStaging, 2));
  budget.freeze();
  ASSERT_TRUE(budget.reserve({{serving::MemoryPool::kKV, 2},
                              {serving::MemoryPool::kBundles, 2}}));
  EXPECT_FALSE(budget.reserve({{serving::MemoryPool::kKV, 1},
                               {serving::MemoryPool::kBundles, 2}}));
  EXPECT_EQ(budget.state(serving::MemoryPool::kKV).used, 2u);
  EXPECT_EQ(budget.state(serving::MemoryPool::kBundles).used, 2u);
  EXPECT_TRUE(budget.invariant_holds());
}

TEST(NodeMemoryBudgetTest, TwoGrowersMakeExplicitProgressWithoutHoldAndWait) {
  serving::NodeMemoryBudget budget;
  ASSERT_TRUE(budget.set_capacity(serving::MemoryPool::kKV, 4));
  budget.freeze();
  ASSERT_TRUE(budget.reserve({{serving::MemoryPool::kKV, 2}}));
  ASSERT_TRUE(budget.reserve({{serving::MemoryPool::kKV, 2}}));
  EXPECT_FALSE(budget.reserve({{serving::MemoryPool::kKV, 1}}));
  budget.release({{serving::MemoryPool::kKV, 2}});
  EXPECT_TRUE(budget.reserve({{serving::MemoryPool::kKV, 1}}));
  EXPECT_EQ(budget.state(serving::MemoryPool::kKV).used, 3u);
  EXPECT_FALSE(budget.set_capacity(serving::MemoryPool::kKV, 8));
  EXPECT_TRUE(budget.invariant_holds());
}
