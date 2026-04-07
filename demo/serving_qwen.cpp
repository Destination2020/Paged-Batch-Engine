// Continuous batching serving demo for Qwen2 Instruct
#include <glog/logging.h>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include "base/base.h"
#include "model/qwen2.h"
#include "serving/scheduler.h"

namespace {

std::string build_chatml_prompt(const std::string& user_prompt) {
  return "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful "
         "assistant.\n<|im_end|>\n<|im_start|>user\n" +
         user_prompt + "\n<|im_end|>\n<|im_start|>assistant\n";
}

std::string trim_response(std::string response) {
  const std::vector<std::string> stop_markers = {
      "<|im_end|>", "<|endoftext|>", "<|im_start|>"};
  for (const auto& marker : stop_markers) {
    auto pos = response.find(marker);
    if (pos != std::string::npos) response = response.substr(0, pos);
  }
  while (!response.empty() &&
         (response.back() == '\n' || response.back() == '\r' || response.back() == ' '))
    response.pop_back();
  return response;
}

}  // namespace

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);

  if (argc < 3) {
    LOG(INFO) << "Usage: ./serving_qwen <model.bin> <tokenizer.json> [prompt1] [prompt2] ...";
    return -1;
  }

  const std::string model_path = argv[1];
  const std::string tokenizer_path = argv[2];

  // Collect prompts (default if none provided)
  std::vector<std::string> prompts;
  if (argc > 3) {
    for (int i = 3; i < argc; ++i) {
      prompts.emplace_back(argv[i]);
    }
  } else {
    prompts = {"What is AI?", "Write a haiku about coding.", "Explain quantum computing briefly."};
  }

  // Init model
  model::Qwen2Model model(base::TokenizerType::kEncodeBpe, tokenizer_path, model_path, false);
  auto init_status = model.init(base::DeviceType::kDeviceCUDA);
  if (!init_status) {
    LOG(FATAL) << "Model init failed: " << init_status.get_err_msg();
    return -1;
  }

  // Create scheduler
  serving::Scheduler scheduler(
      model::model_max_batch_size,
      model.kv_cache_manager());

  // Submit all requests
  std::cout << "=== Submitting " << prompts.size() << " requests ===" << std::endl;
  for (size_t i = 0; i < prompts.size(); ++i) {
    std::string chatml = build_chatml_prompt(prompts[i]);
    auto tokens = model.encode(chatml);
    std::cout << "Request " << i << ": \"" << prompts[i]
              << "\" (" << tokens.size() << " tokens)" << std::endl;
    scheduler.add_request(std::move(tokens));
  }

  // Main serving loop
  std::cout << "\n=== Starting continuous batching ===" << std::endl;
  const auto start = std::chrono::steady_clock::now();
  int32_t total_decode_steps = 0;
  constexpr int32_t max_new_tokens = 256;
  int32_t step = 0;

  while (scheduler.has_active_requests() && step < max_new_tokens) {
    // Schedule: prefill new requests + build decode batch
    auto batch = scheduler.schedule_step(model);
    if (batch.batch_size == 0) break;

    // Forward
    auto status = model.forward_decode_batch(batch);
    CHECK(status) << "forward_decode_batch failed: " << status.get_err_msg();

    // Sample
    auto sampled = model.batch_sample(batch.batch_size);

    // Process outputs
    scheduler.process_outputs(sampled,
        [&](int32_t t) { return model.is_sentence_ending(t); });

    total_decode_steps += batch.batch_size;

    // Print finished sequences
    for (auto& seq : scheduler.pop_finished()) {
      std::string text = trim_response(model.decode(seq.output_tokens));
      std::cout << "\n--- Request " << seq.request_id << " finished ---\n"
                << text << "\n";
    }

    ++step;
  }

  const auto end = std::chrono::steady_clock::now();
  const double duration = std::chrono::duration<double>(end - start).count();

  std::cout << "\n=== Done ===" << std::endl;
  std::cout << "Steps: " << step << std::endl;
  std::cout << "Total decode tokens: " << total_decode_steps << std::endl;
  std::cout << "Duration: " << duration << "s" << std::endl;
  std::cout << "Throughput: " << total_decode_steps / duration << " tokens/s" << std::endl;

  return 0;
}
