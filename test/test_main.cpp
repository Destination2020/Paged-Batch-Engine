// Updated on March 15, 2026
#include <glog/logging.h>
#include <gtest/gtest.h>

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  google::InitGoogleLogging("Kuiper");
  FLAGS_log_dir = "./log/";
  FLAGS_logtostderr = true;
  FLAGS_alsologtostderr = true;

  FLAGS_minloglevel = 0;

  LOG(INFO) << "Start Test...\n";
  return RUN_ALL_TESTS();
}