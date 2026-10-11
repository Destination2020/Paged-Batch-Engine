#include <gtest/gtest.h>
#include "serving/serving_online_engine.h"

TEST(OnlineHandleTest, TerminalResultCannotBeOverwrittenByLateEvents) {
  serving::OnlineRequestHandle handle;
  handle.push_token("hello");
  handle.finish("hello", false, "");
  handle.push_token("late");
  handle.finish("", true, "late failure");
  std::string text, error;
  bool finished = false, failed = true;
  ASSERT_TRUE(handle.wait_next_token_for(0, &text, &finished, &failed, &error));
  EXPECT_EQ(text, "hello");
  EXPECT_FALSE(handle.wait_next_token_for(0, &text, &finished, &failed, &error));
  EXPECT_TRUE(finished);
  EXPECT_FALSE(failed);
  ASSERT_TRUE(handle.wait_full_text(1, &text, &failed, &error));
  EXPECT_EQ(text, "hello");
}
