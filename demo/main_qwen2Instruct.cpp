// Updated on March 24, 2026
#include <base/base.h>
#include <glog/logging.h>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include "model/qwen2.h"

namespace {

std::string join_prompt_from_argv(int argc, char* argv[]) {
  if (argc <= 3) {
    return "What is AI?";
  }

  std::string prompt;
  for (int i = 3; i < argc; ++i) {
    if (!prompt.empty()) {
      prompt += " ";
    }
    prompt += argv[i];
  }
  return prompt;
}

std::string build_chatml_prompt(const std::string& user_prompt) {
  return "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful "
         "assistant.\n<|im_end|>\n<|im_start|>user\n" +
         user_prompt + "\n<|im_end|>\n<|im_start|>assistant\n";
}

std::string trim_response(std::string response) {
  const std::vector<std::string> stop_markers = {
      "<|im_end|>",
      "<|endoftext|>",
      "<|im_start|>",
  };

  for (const auto& marker : stop_markers) {
    const size_t pos = response.find(marker);
    if (pos != std::string::npos) {
      response = response.substr(0, pos);
    }
  }

  while (!response.empty() && (response.back() == '\n' || response.back() == ' ')) {
    response.pop_back();
  }
  return response;
}

int32_t generate_response(const model::Qwen2Model& model, const std::string& prompt,
                          int32_t total_steps, std::string& response, bool need_output = false) {
  auto tokens = model.encode(prompt);
  const int32_t prompt_len = static_cast<int32_t>(tokens.size());
  LOG_IF(FATAL, tokens.empty()) << "The input tokens are empty.";

  int32_t pos = 0;
  int32_t next = tokens.at(pos);
  bool is_prompt = true;
  tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

  std::vector<int32_t> generated_tokens;
  while (pos < total_steps) {
    pos_tensor.index<int32_t>(0) = pos;
    if (pos < prompt_len - 1) {
      // Embed prompt tokens one at a time. The model's reusable serving workspace is
      // sized for the active batch, not for an arbitrarily long contiguous prompt.
      // Keeping a full-prompt embedding view here would make this simple demo fail
      // whenever prompt_len exceeds that workspace capacity.
      const auto token_embedding = model.embedding(std::vector<int32_t>{tokens.at(pos)});
      tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, false);
      model.predict(input, pos_tensor, is_prompt, next);
      next = tokens.at(pos + 1);
    } else {
      is_prompt = false;
      tokens = std::vector<int32_t>{next};
      const auto& token_embedding = model.embedding(tokens);
      tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
      if (model.is_sentence_ending(next)) {
        break;
      }
      if (next != 151643 && next != 151644 && next != 151645) {
        generated_tokens.push_back(next);
      }
    }
    pos += 1;
  }

  response = trim_response(model.decode(generated_tokens));
  if (need_output) {
    std::cout << response;
    std::cout.flush();
  }
  return std::min(pos, total_steps);
}

}  // namespace

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);

  if (argc < 3) {
    LOG(INFO) << "Usage: ./qwen_instruct_infer <model.bin> <tokenizer.json> [prompt...]";
    return -1;
  }

  const char* checkpoint_path = argv[1];
  const char* tokenizer_path = argv[2];
  const std::string user_prompt = join_prompt_from_argv(argc, argv);
  const std::string prompt = build_chatml_prompt(user_prompt);

  model::Qwen2Model model(base::TokenizerType::kEncodeBpe, tokenizer_path, checkpoint_path, false);
  model.set_use_paged_kv(true);
  auto init_status = model.init(base::DeviceType::kDeviceCUDA);
  if (!init_status) {
    LOG(FATAL) << "The model init failed, the error code is: " << init_status.get_err_code()
               << ", error message: " << init_status.get_err_msg();
  }

  std::cout << "User: " << user_prompt << "\nAssistant: ";
  std::cout.flush();

  std::string response;
  const auto start = std::chrono::steady_clock::now();
  const int32_t steps = generate_response(model, prompt, 512, response, true);
  const auto end = std::chrono::steady_clock::now();
  const auto duration = std::chrono::duration<double>(end - start).count();

  std::cout << "\nsteps:" << steps << std::endl;
  std::cout << "duration:" << duration << std::endl;
  std::cout << "steps/s:" << static_cast<double>(steps) / duration << std::endl;
  return 0;
}
