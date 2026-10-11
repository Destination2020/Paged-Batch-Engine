#include <gtest/gtest.h>
#include <cstdio>
#include <fstream>
#include <unistd.h>
#include "serving/serving_online_engine_pool.h"
#include "serving/serving_zmq_rpc.h"

TEST(MultiModelRouting, SingleRemoteModelIsTheHTTPDefault) {
  serving::BenchConfig config;
  config.model_id = "qwen25-1.5b";
  serving::ZmqOnlineEnginePool pool(config);
  EXPECT_EQ(pool.default_model_id(), config.model_id);
  EXPECT_EQ(pool.models_json().at("data").at(0).at("id"), config.model_id);
}

TEST(MultiModelRouting, IdentitySurvivesRPCSerialization) {
  serving::OnlineGenerateRequest request;
  request.model_id = "qwen25-1.5b";
  request.prompt = "hello";
  request.stream = true;
  request.generation_config.max_new_tokens = 16;
  const auto restored = serving::online_generate_request_from_json(
      serving::online_generate_request_to_json(request));
  EXPECT_EQ(restored.model_id, request.model_id);
  EXPECT_EQ(restored.prompt, request.prompt);
  EXPECT_EQ(restored.generation_config.max_new_tokens, 16);
}

TEST(MultiModelRouting, ListsModelsAndRejectsUnknownWithoutSendingRPC) {
  const auto path = "/tmp/pbe-registry-test-" + std::to_string(getpid()) + ".json";
  std::ofstream(path) << R"({"default_model":"large","models":[
      {"id":"small","endpoint":"ipc:///tmp/pbe-test-small.sock"},
      {"id":"large","endpoint":"ipc:///tmp/pbe-test-large.sock"}]})";
  serving::BenchConfig config;
  config.model_registry_path = path;
  serving::MultiModelOnlineEnginePool pool(config);
  EXPECT_EQ(pool.default_model_id(), "large");
  EXPECT_EQ(pool.models_json().at("data").size(), 2);
  serving::OnlineGenerateRequest request;
  request.model_id = "wrong-model";
  request.prompt = "hello";
  std::string error;
  EXPECT_EQ(pool.submit(request, &error), nullptr);
  EXPECT_EQ(error, "unknown_model");
  std::remove(path.c_str());
}

TEST(MultiModelRouting, RejectsDuplicateModelOrWorkerEndpoint) {
  const auto path = "/tmp/pbe-registry-duplicate-" + std::to_string(getpid()) + ".json";
  serving::BenchConfig config; config.model_registry_path = path;
  for (const auto& json : {
      R"({"models":[{"id":"a","endpoint":"ipc:///tmp/a"},{"id":"a","endpoint":"ipc:///tmp/b"}]})",
      R"({"models":[{"id":"a","endpoint":"ipc:///tmp/a"},{"id":"b","endpoint":"ipc:///tmp/a"}]})"}) {
    std::ofstream(path) << json;
    EXPECT_THROW(serving::MultiModelOnlineEnginePool pool(config), std::invalid_argument);
  }
  std::remove(path.c_str());
}

TEST(MultiModelRouting, SplitChineseCodePointIsBufferedUntilComplete) {
  serving::OnlineRequestHandle handle;
  handle.push_token("\xe4");
  std::string text, error; bool finished = false, failed = false;
  EXPECT_FALSE(handle.wait_next_token_for(1, &text, &finished, &failed, &error));
  handle.push_token("\xb8\xad");
  EXPECT_TRUE(handle.wait_next_token_for(1, &text, &finished, &failed, &error));
  EXPECT_EQ(text, "中");
  EXPECT_NO_THROW(nlohmann::json({{"text", text}}).dump());
  handle.push_token("\xe4");
  handle.finish("中\xe4", false, "");
  EXPECT_TRUE(handle.wait_full_text(1, &text, &failed, &error));
  EXPECT_EQ(text, "中\xef\xbf\xbd");
  EXPECT_NO_THROW(nlohmann::json({{"text", text}}).dump());
}
