// Updated on March 24, 2026
#include <glog/logging.h>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <functional>
#include "base/base.h"
#include "model/qwen2.h"

namespace {

bool is_utf8_continuation(unsigned char c);
size_t longest_valid_utf8_prefix(const std::string& bytes);

struct ChatMessage {
  std::string role;
  std::string content;
};

struct GenerationConfig {
  int32_t max_new_tokens = 512;
  int32_t max_context_tokens = 4096;
};

struct Utf8StreamBuffer {
  std::string pending_bytes;
  std::string emitted_text;

  void push_piece(const std::string& piece, const std::function<void(const std::string&)>& emit_func = {}) {
    pending_bytes += piece;
    size_t flush_size = longest_valid_utf8_prefix(pending_bytes);
    if (flush_size == 0) {
      return;
    }
    std::string to_emit = pending_bytes.substr(0, flush_size);
    pending_bytes.erase(0, flush_size);
    if (!to_emit.empty() && emit_func) {
      emit_func(to_emit);
      emitted_text += to_emit;
    }
    return;
  }
};



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

  while (!response.empty() &&
         (response.back() == '\n' || response.back() == '\r' || response.back() == ' ')) {
    response.pop_back();
  }
  return response;
}

class ChatAssistant {
 public:
  ChatAssistant(std::string model_path, std::string tokenizer_path, std::string system_prompt,
                bool use_paged_kv = false)
      : model_path_(std::move(model_path)),
        tokenizer_path_(std::move(tokenizer_path)),
        system_prompt_(std::move(system_prompt)),
        use_paged_kv_(use_paged_kv) {}

  bool init() {
    try {
      model_ = std::make_unique<model::Qwen2Model>(base::TokenizerType::kEncodeBpe, tokenizer_path_,
                                                   model_path_, false);
      if (use_paged_kv_) {
        model_->set_use_paged_kv(true);
        LOG(INFO) << "Paged KV cache enabled";
      }
      auto init_status = model_->init(base::DeviceType::kDeviceCUDA);
      if (!init_status) {
        LOG(ERROR) << "模型初始化失败, 错误码: " << init_status.get_err_code()
                   << ", 错误信息: " << init_status.get_err_msg();
        return false;
      }
      reset_history();
      return true;
    } catch (const std::exception& e) {
      LOG(ERROR) << "初始化异常: " << e.what();
      return false;
    }
  }

  void reset_history() {
    history_.clear();
    history_.push_back({"system", system_prompt_});
  }

  void set_system_prompt(const std::string& system_prompt) {
    system_prompt_ = system_prompt;
    reset_history();
  }

  const std::vector<ChatMessage>& history() const { return history_; }

  std::string format_messages(const std::vector<ChatMessage>& messages) const {
    std::string prompt;
    for (const auto& message : messages) {
      prompt += "<|im_start|>" + message.role + "\n";
      prompt += message.content + "\n";
      prompt += "<|im_end|>\n";
    }
    prompt += "<|im_start|>assistant\n";
    return prompt;
  }

  std::pair<std::string, int32_t> chat(const std::string& user_input,
                                       const GenerationConfig& config, const std::function<void(const std::string&)>& emit_func = {}) {
    history_.push_back({"user", user_input});
    trim_history_to_fit(config.max_context_tokens);

    const std::string prompt = format_messages(history_);
    auto [response_text, steps] = generate(prompt, config.max_new_tokens, emit_func);
    history_.push_back({"assistant", response_text});
    return {response_text, steps};
  }

 private:
  void trim_history_to_fit(int32_t max_context_tokens) {
    if (!model_) {
      return;
    }

    while (history_.size() > 2) {
      const std::string prompt = format_messages(history_);
      const auto tokens = model_->encode(prompt);
      if (static_cast<int32_t>(tokens.size()) <= max_context_tokens) {
        return;
      }

      // Keep the system prompt and the most recent rounds.
      history_.erase(history_.begin() + 1, history_.begin() + 3);
    }

    const std::string prompt = format_messages(history_);
    const auto tokens = model_->encode(prompt);
    if (static_cast<int32_t>(tokens.size()) > max_context_tokens) {
      LOG(WARNING) << "当前 system prompt 太长, token 数为 " << tokens.size()
                   << ", 已超过限制 " << max_context_tokens;
    }
  }

  std::pair<std::string, int32_t> generate(const std::string& prompt, int32_t max_new_tokens, const std::function<void(const std::string&)>& emit_func = {}) {
    auto tokens = model_->encode(prompt);
    const int32_t prompt_len = static_cast<int32_t>(tokens.size());
    LOG_IF(FATAL, tokens.empty()) << "输入 tokens 为空。";

    int32_t pos = 0;
    int32_t next = tokens.at(pos);
    bool is_prompt = true;
    const auto& prompt_embedding = model_->embedding(tokens);
    tensor::Tensor pos_tensor = model_->get_buffer(model::ModelBufferType::kInputPos);

    std::vector<int32_t> response_tokens;
    Utf8StreamBuffer stream_buffer;
    while (pos < prompt_len + max_new_tokens) {
      pos_tensor.index<int32_t>(0) = pos;

      if (pos < prompt_len - 1) {
        tensor::Tensor input = model_->fill_input(pos_tensor, prompt_embedding, is_prompt);
        model_->predict(input, pos_tensor, is_prompt, next);
        next = tokens.at(pos + 1);
      } else {
        is_prompt = false;
        std::vector<int32_t> current_tokens = {next};
        const auto& token_embedding = model_->embedding(current_tokens);
        tensor::Tensor input = model_->fill_input(pos_tensor, token_embedding, is_prompt);
        model_->predict(input, pos_tensor, is_prompt, next);

        if (model_->is_sentence_ending(next)) {
          break;
        }

        if (next != 151643 && next != 151644 && next != 151645) {
          stream_buffer.push_piece(model_->decode(next), emit_func);
          response_tokens.push_back(next);
        }
      }

      pos += 1;
    }
    std::string final_text = trim_response(model_->decode(response_tokens));
    if (final_text.size() >= stream_buffer.emitted_text.size() &&
        final_text.compare(0, stream_buffer.emitted_text.size(), stream_buffer.emitted_text) == 0) {
        std::string tail = final_text.substr(stream_buffer.emitted_text.size());
        if (!tail.empty() && emit_func) {
          emit_func(tail);
        }
      }

    return {final_text, std::min(pos, prompt_len + max_new_tokens)};
  }

 private:
  std::string model_path_;
  std::string tokenizer_path_;
  std::string system_prompt_;
  bool use_paged_kv_ = false;
  std::vector<ChatMessage> history_;
  std::unique_ptr<model::Qwen2Model> model_;
};

void print_history(const std::vector<ChatMessage>& history) {
  std::cout << "\n=== Current History ===" << std::endl;
  for (const auto& msg : history) {
    std::cout << msg.role << ": " << msg.content << std::endl;
  }
  std::cout << "=======================\n" << std::endl;
}

void print_help() {
  std::cout << "Commands:\n"
            << "  /exit      Exit chat\n"
            << "  /clear     Clear chat history and keep the current system prompt\n"
            << "  /history   Print current chat history\n"
            << "  /system X  Replace the system prompt with X and clear history\n"
            << "  /help      Show this help message\n"
            << std::endl;
}


bool is_utf8_continuation(unsigned char c) {
  return (c & 0xC0) == 0x80;
}

size_t longest_valid_utf8_prefix(const std::string& s) {
  size_t i = 0;
  size_t last_good = 0;
  const size_t n = s.size();

  while (i < n) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    size_t char_len = 0;

    if ((c & 0x80) == 0x00) {
      char_len = 1;
    } else if ((c & 0xE0) == 0xC0) {
      if (c < 0xC2) {
        break;
      }
      char_len = 2;
    } else if ((c & 0xF0) == 0xE0) {
      char_len = 3;
    } else if ((c & 0xF8) == 0xF0) {
      if (c > 0xF4) {
        break;
      }
      char_len = 4;
    } else {
      break;
    }

    if (i + char_len > n) {
      break;
    }

    bool ok = true;
    for (size_t j = 1; j < char_len; ++j) {
      if (!is_utf8_continuation(static_cast<unsigned char>(s[i + j]))) {
        ok = false;
        break;
      }
    }
    if (!ok) {
      break;
    }

    if (char_len == 3) {
      unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
      if (c == 0xE0 && c1 < 0xA0) {
        break;
      }
      if (c == 0xED && c1 >= 0xA0) {
        break;
      }
    }

    if (char_len == 4) {
      unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
      if (c == 0xF0 && c1 < 0x90) {
        break;
      }
      if (c == 0xF4 && c1 > 0x8F) {
        break;
      }
    }

    i += char_len;
    last_good = i;
  }

  return last_good;
}

}  // namespace



int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);

  if (argc < 3) {
    LOG(INFO) << "Usage: ./qwen_instruct_chat <model.bin> <tokenizer.json> [system prompt]";
    return -1;
  }

  const std::string model_path = argv[1];
  const std::string tokenizer_path = argv[2];
  std::string system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
  bool use_paged_kv = false;

  for (int i = 3; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--paged") {
      use_paged_kv = true;
    } else if (system_prompt == "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.") {
      system_prompt = arg;
    }
  }

  ChatAssistant assistant(model_path, tokenizer_path, system_prompt, use_paged_kv);
  if (!assistant.init()) {
    LOG(FATAL) << "聊天助手初始化失败!";
    return -1;
  }

  GenerationConfig config;
  std::cout << "Qwen2 Instruct chat is ready.\n";
  std::cout << "Max new tokens: " << config.max_new_tokens
            << ", max context tokens: " << config.max_context_tokens << "\n";
  print_help();

  std::string user_input;
  while (true) {
    std::cout << "\nUser> " << std::flush;
    if (!std::getline(std::cin, user_input)) {
      break;
    }
    if (user_input.empty()) {
      continue;
    }

    if (user_input == "/exit" || user_input == "quit" || user_input == "exit") {
      break;
    }
    if (user_input == "/clear") {
      assistant.reset_history();
      std::cout << "History cleared." << std::endl;
      continue;
    }
    if (user_input == "/history") {
      print_history(assistant.history());
      continue;
    }
    if (user_input == "/help") {
      print_help();
      continue;
    }
    if (user_input.rfind("/system ", 0) == 0) {
      assistant.set_system_prompt(user_input.substr(8));
      std::cout << "System prompt updated and history cleared." << std::endl;
      continue;
    }

    std::cout << "\nAssistant> " << std::flush;
    const auto start = std::chrono::steady_clock::now();
    const auto [response, steps] = assistant.chat(user_input, config, [](const std::string& piece) { std::cout << piece << std::flush;});
    const auto end = std::chrono::steady_clock::now();
    const auto duration = std::chrono::duration<double>(end - start).count();

    //std::cout << response << std::endl;
    std::cout << "[steps: " << steps << ", duration: " << duration
              << "s, steps/s: " << static_cast<double>(steps) / duration << "]" << std::endl;
  }

  std::cout << "Chat ended." << std::endl;
  return 0;
}
