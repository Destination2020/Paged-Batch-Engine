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

TEST(OnlineHandleTest, CancellationBlocksLateTokensAndCannotChangeCompletedResult) {
  serving::OnlineRequestHandle completed;
  completed.finish("done", false, "");
  completed.mark_cancelled();
  EXPECT_FALSE(completed.cancelled());
  serving::OnlineRequestHandle cancelled;
  cancelled.mark_cancelled();
  cancelled.push_token("late");
  cancelled.finish("", true, "cancelled");
  std::string text, error;
  bool done = false, failed = false;
  EXPECT_FALSE(cancelled.wait_next_token_for(0, &text, &done, &failed, &error));
  EXPECT_TRUE(done);
  EXPECT_TRUE(failed);
  EXPECT_TRUE(cancelled.cancelled());
}

#include "serving/serving_config_parser.h"

TEST(ServingConfigTest, ExistingFlagsPopulateIndependentDomains) {
  std::vector<std::string> args{"server", "--listen-port=18080", "--pd-mode=dual-gpu-p2p",
      "--decode-device-id=2", "--shared-kv-chunk-mib=4", "--quiet=true",
      "--engine-zmq-timeout-ms=1234"};
  std::vector<char*> argv;
  for (auto& arg : args) argv.push_back(arg.data());
  const auto config = serving::parse_bench_config(argv.size(), argv.data(), 1);
  EXPECT_EQ(config.http.listen_port, 18080);
  EXPECT_EQ(config.pd.decode_device_id, 2);
  EXPECT_EQ(config.kv.shared_kv_chunk_mib, 4);
  EXPECT_EQ(config.ipc.engine_zmq_timeout_ms, 1234);
  EXPECT_TRUE(config.benchmark.quiet);
}

TEST(ServingConfigTest, InvalidExternalValuesAreExceptionsNotProcessAborts) {
  for (std::string flag : {"--listen-port=80junk", "--quiet=maybe", "--seed=-1",
       "--host-cache-bytes=-1", "--temperature=nan", "--pd-mode=invalid",
       "--online-process-role=unknown", "--scheduling-policy=random"}) {
    char* argv[] = {flag.data()};
    EXPECT_THROW(serving::parse_bench_config(1, argv, 0), std::invalid_argument) << flag;
  }
}
