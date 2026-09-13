#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "base/alloc.h"
#include "model/qwen2.h"
#include "serving/scheduler.h"

namespace {

std::string RequiredEnv(const char* name) {
  const char* value = std::getenv(name);
  return value == nullptr ? std::string{} : std::string(value);
}

template <typename T>
std::vector<T> ReadBinary(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) return {};
  stream.seekg(0, std::ios::end);
  const auto bytes = stream.tellg();
  stream.seekg(0);
  if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(T)) != 0) return {};
  std::vector<T> result(static_cast<size_t>(bytes) / sizeof(T));
  stream.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(bytes));
  return stream ? result : std::vector<T>{};
}

float BFloat16ToFloat(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

TEST(Qwen25VLModelTest, OneImageMatchesReferenceLogitsAndGreedyDecode) {
  const std::string model_path = RequiredEnv("PBE_QWEN25_VL_MODEL");
  const std::string tokenizer_path = RequiredEnv("PBE_QWEN25_VL_TOKENIZER");
  const std::string fixture_dir = RequiredEnv("PBE_QWEN25_VL_FIXTURE_DIR");
  if (model_path.empty() || tokenizer_path.empty() || fixture_dir.empty()) {
    GTEST_SKIP() << "Set PBE_QWEN25_VL_MODEL, PBE_QWEN25_VL_TOKENIZER and "
                    "PBE_QWEN25_VL_FIXTURE_DIR";
  }

  std::ifstream manifest_stream(fixture_dir + "/reference.json");
  ASSERT_TRUE(manifest_stream);
  nlohmann::json manifest = nlohmann::json::parse(manifest_stream);
  auto case_it = std::find_if(manifest["cases"].begin(), manifest["cases"].end(),
                              [](const auto& item) { return item["case"] == "one_image_224"; });
  ASSERT_NE(case_it, manifest["cases"].end());
  const auto& reference = *case_it;
  const auto prompt = ReadBinary<int32_t>(fixture_dir + "/one_image_224.input_ids.i32.bin");
  const auto positions = ReadBinary<int32_t>(fixture_dir + "/one_image_224.position_ids.i32.bin");
  const auto embeddings = ReadBinary<uint16_t>(fixture_dir + "/one_image_224.inputs_embeds.bf16.bin");
  const auto expected_logits = ReadBinary<float>(fixture_dir + "/one_image_224.last_logits.f32.bin");
  const auto generation_logits =
      ReadBinary<float>(fixture_dir + "/one_image_224.generation_logits.f32.bin");
  const auto expected_tokens = reference["generated_token_ids"][0].get<std::vector<int32_t>>();
  const auto expected_margins = reference["greedy_top2_margin"][0].get<std::vector<double>>();
  const int64_t rope_delta = reference["rope_deltas"][0].get<int64_t>();
  ASSERT_FALSE(prompt.empty());
  ASSERT_EQ(positions.size(), prompt.size() * 3);
  ASSERT_EQ(embeddings.size(), prompt.size() * 2048);
  ASSERT_EQ(expected_logits.size(), 151936u);
  ASSERT_EQ(generation_logits.size(), expected_tokens.size() * expected_logits.size());
  ASSERT_EQ(expected_margins.size(), expected_tokens.size());

  model::Qwen2Model model(base::TokenizerType::kEncodeBpe, tokenizer_path, model_path, false);
  model.set_kv_cache_blocks_per_layer(128);
  model.set_serving_workspace_token_capacity(256);
  model.set_radix_cache_enabled(false);
  ASSERT_TRUE(model.init(base::DeviceType::kDeviceCUDA, 0));
  auto cuda_alloc = base::CUDADeviceAllocatorFactory::get_instance();
  tensor::Tensor embedding_gpu(base::DataType::kDataTypeBf16,
                               static_cast<int32_t>(prompt.size()), 2048, true, cuda_alloc);
  tensor::Tensor mrope_gpu(base::DataType::kDataTypeInt32, 3,
                           static_cast<int32_t>(prompt.size()), true, cuda_alloc);
  cuda_alloc->memcpy(embeddings.data(), embedding_gpu.ptr<uint16_t>(), embedding_gpu.byte_size(),
                     base::MemcpyKind::kMemcpyCPU2CUDA, nullptr, true);
  cuda_alloc->memcpy(positions.data(), mrope_gpu.ptr<int32_t>(), mrope_gpu.byte_size(),
                     base::MemcpyKind::kMemcpyCPU2CUDA, nullptr, true);

  serving::SchedulerConfig config;
  config.max_num_seqs = 1;
  config.max_num_batched_tokens = 256;
  config.prefill_chunk_cap = 256;
  serving::Scheduler scheduler(config, model.kv_cache_manager());
  serving::GenerationConfig generation(static_cast<int32_t>(expected_tokens.size()));
  generation.sampling.repetition_penalty = 1.05;
  scheduler.add_request(prompt, generation);
  std::vector<int32_t> actual_tokens;
  std::vector<size_t> divergent_steps;
  std::vector<double> step_max_errors;
  std::vector<double> step_mean_errors;
  std::vector<double> actual_top2_margins;
  std::vector<std::array<size_t, 2>> actual_top2_ids;
  std::vector<std::array<size_t, 2>> reference_top2_ids;
  bool checked_logits = false;
  while (scheduler.has_active_requests()) {
    auto output = scheduler.schedule_step();
    ASSERT_GT(output.total_tokens, 0);
    const bool decode = output.num_decode_seqs > 0 && output.num_prefill_seqs == 0;
    auto batch = decode ? scheduler.build_decode_batch(output, model.device_context()->compute_queue)
                        : scheduler.build_mixed_batch(output, model.device_context()->compute_queue);
    if (!decode) {
      ASSERT_EQ(batch.num_tokens, static_cast<int32_t>(prompt.size()));
      batch.input_embeddings_override = embedding_gpu;
      batch.mrope_positions = mrope_gpu;
    } else {
      const int32_t position = static_cast<int32_t>(prompt.size() + rope_delta +
                                                    actual_tokens.size() - 1);
      cuda_alloc->memcpy(&position, batch.positions.ptr<int32_t>(), sizeof(position),
                         base::MemcpyKind::kMemcpyCPU2CUDA,
                         model.device_context()->compute_queue, false);
    }
    const auto status = decode ? model.forward_decode_batch(batch)
                               : model.forward_mixed_batch(batch);
    ASSERT_TRUE(status) << status.get_err_msg();
    std::vector<float> actual_logits;
    ASSERT_TRUE(model.copy_logits_row_to_host(batch.num_tokens - 1, &actual_logits));
    {
      const size_t step = actual_tokens.size();
      const float* expected_step = generation_logits.data() + step * actual_logits.size();
      size_t actual_top = 0;
      size_t expected_top = 0;
      double step_max_abs = 0.0;
      double step_mean_abs = 0.0;
      double step_reference_abs = 0.0;
      for (size_t i = 0; i < actual_logits.size(); ++i) {
        const double error = std::abs(actual_logits[i] - expected_step[i]);
        step_max_abs = std::max(step_max_abs, error);
        step_mean_abs += error;
        step_reference_abs += std::abs(expected_step[i]);
        if (actual_logits[i] > actual_logits[actual_top]) actual_top = i;
        if (expected_step[i] > expected_step[expected_top]) expected_top = i;
      }
      step_mean_abs /= actual_logits.size();
      const auto top_two = [](const float* values, size_t count) {
        std::array<size_t, 2> ids{0, 1};
        if (values[ids[1]] > values[ids[0]]) std::swap(ids[0], ids[1]);
        for (size_t i = 2; i < count; ++i) {
          if (values[i] > values[ids[0]]) {
            ids[1] = ids[0];
            ids[0] = i;
          } else if (values[i] > values[ids[1]]) {
            ids[1] = i;
          }
        }
        return ids;
      };
      const auto actual_two = top_two(actual_logits.data(), actual_logits.size());
      const auto reference_two = top_two(expected_step, actual_logits.size());
      step_max_errors.push_back(step_max_abs);
      step_mean_errors.push_back(step_mean_abs);
      actual_top2_margins.push_back(
          actual_logits[actual_two[0]] - actual_logits[actual_two[1]]);
      actual_top2_ids.push_back(actual_two);
      reference_top2_ids.push_back(reference_two);
      std::cout << "QWEN25_VL_STEP step=" << step << " expected_top=" << expected_top
                << " actual_top=" << actual_top << " max_abs=" << step_max_abs
                << " mean_abs=" << step_mean_abs
                << " relative_mean=" << step_mean_abs /
                       std::max(1.0e-12, step_reference_abs / actual_logits.size()) << "\n";
      EXPECT_LT(step_max_abs, 0.75) << "step=" << step;
      EXPECT_LT(step_mean_abs, 0.20) << "step=" << step;
    }
    if (!checked_logits) {
      ASSERT_EQ(actual_logits.size(), expected_logits.size());
      double max_abs = 0.0;
      double mean_abs = 0.0;
      double reference_abs = 0.0;
      for (size_t i = 0; i < actual_logits.size(); ++i) {
        const double error = std::abs(actual_logits[i] - expected_logits[i]);
        max_abs = std::max(max_abs, error);
        mean_abs += error;
        reference_abs += std::abs(expected_logits[i]);
      }
      mean_abs /= actual_logits.size();
      std::cout << "QWEN25_VL_LOGITS max_abs=" << max_abs << " mean_abs=" << mean_abs
                << " relative_mean=" << mean_abs /
                       std::max(1.0e-12, reference_abs / actual_logits.size()) << "\n";
      EXPECT_LT(max_abs, 0.75);
      EXPECT_LT(mean_abs, 0.13);

      const auto reference_hidden =
          ReadBinary<float>(fixture_dir + "/one_image_224.hidden_final.f32.bin");
      ASSERT_EQ(reference_hidden.size(), prompt.size() * 2048);
      std::vector<float> actual_hidden;
      ASSERT_TRUE(model.copy_final_hidden_row_to_host(batch.num_tokens - 1, &actual_hidden));
      double hidden_max = 0.0;
      double hidden_mean = 0.0;
      double hidden_reference_abs = 0.0;
      const float* expected_hidden = reference_hidden.data() + (prompt.size() - 1) * 2048;
      for (size_t i = 0; i < actual_hidden.size(); ++i) {
        const double error = std::abs(actual_hidden[i] - expected_hidden[i]);
        hidden_max = std::max(hidden_max, error);
        hidden_mean += error;
        hidden_reference_abs += std::abs(expected_hidden[i]);
      }
      hidden_mean /= actual_hidden.size();
      std::cout << "QWEN25_VL_HIDDEN layer=final max_abs=" << hidden_max
                << " mean_abs=" << hidden_mean << " relative_mean="
                << hidden_mean /
                       std::max(1.0e-12, hidden_reference_abs / actual_hidden.size()) << "\n";
      EXPECT_LT(hidden_max, 1.5);
      EXPECT_LT(hidden_mean, 0.13);

      base::KVRequestSnapshot snapshot;
      ASSERT_TRUE(model.kv_cache_manager()->snapshot_request(batch.request_ids[0], &snapshot));
      ASSERT_EQ(snapshot.valid_tokens, static_cast<int32_t>(prompt.size()));
      for (int32_t layer : {0, 18, 35}) {
        const auto compare_kv = [&](const char* component,
                                    const std::vector<base::KVBlockSnapshot>& blocks,
                                    bool key) {
          const auto expected = ReadBinary<float>(fixture_dir + "/one_image_224.kv_" +
                                                   component + "_" + std::to_string(layer) +
                                                   ".f32.bin");
          ASSERT_EQ(expected.size(), prompt.size() * 2 * 128);
          double kv_max = 0.0;
          double kv_mean = 0.0;
          double kv_reference_abs = 0.0;
          size_t count = 0;
          for (size_t token = 0; token < prompt.size(); ++token) {
            const size_t block = token / 16;
            const size_t offset = token % 16;
            const auto& bytes = key ? blocks[block].key : blocks[block].value;
            for (size_t head = 0; head < 2; ++head) {
              for (size_t dim = 0; dim < 128; ++dim) {
                uint16_t raw = 0;
                const size_t element = (offset * 2 + head) * 128 + dim;
                std::memcpy(&raw, bytes.data() + element * sizeof(raw), sizeof(raw));
                const float actual = BFloat16ToFloat(raw);
                const float reference_value = expected[(head * prompt.size() + token) * 128 + dim];
                const double error = std::abs(actual - reference_value);
                kv_max = std::max(kv_max, error);
                kv_mean += error;
                kv_reference_abs += std::abs(reference_value);
                ++count;
              }
            }
          }
          kv_mean /= count;
          std::cout << "QWEN25_VL_KV layer=" << layer << " component=" << component
                    << " max_abs=" << kv_max << " mean_abs=" << kv_mean
                    << " relative_mean=" << kv_mean /
                           std::max(1.0e-12, kv_reference_abs / count) << "\n";
          EXPECT_LT(kv_max, 1.25);
          EXPECT_LT(kv_mean, 0.08);
        };
        ASSERT_LT(static_cast<size_t>(layer), snapshot.layers.size());
        compare_kv("key", snapshot.layers[layer].blocks, true);
        compare_kv("value", snapshot.layers[layer].blocks, false);
      }
      checked_logits = true;
    }
    auto sampled = model.batch_sample(batch, output);
    ASSERT_EQ(sampled.size(), 1);
    actual_tokens.push_back(sampled[0]);
    const size_t step = actual_tokens.size() - 1;
    if (sampled[0] != expected_tokens[step]) divergent_steps.push_back(step);
    // Keep the reference history after recording the real PBE greedy choice.
    // This isolates every incremental step from an earlier low-margin branch.
    std::vector<int32_t> teacher_token{expected_tokens[step]};
    serving::SampledTokenView teacher_view{teacher_token.data(), 1};
    scheduler.process_outputs(output, batch, teacher_view,
                              [&](int32_t token) { return model.is_sentence_ending(token); });
  }
  for (size_t step : divergent_steps) {
    std::cout << "QWEN25_VL_GREEDY_DIVERGENCE step=" << step
              << " reference=" << expected_tokens[step]
              << " pbe=" << actual_tokens[step]
              << " reference_top2_margin=" << expected_margins[step]
              << " pbe_top2_margin=" << actual_top2_margins[step]
              << " max_abs=" << step_max_errors[step]
              << " mean_abs=" << step_mean_errors[step] << "\n";
    EXPECT_LT(step_max_errors[step], 0.75) << "same-history logits error";
    EXPECT_LT(step_mean_errors[step], 0.20) << "same-history logits error";
    EXPECT_TRUE(expected_margins[step] <= 0.25 || actual_top2_margins[step] <= 0.25)
        << "high-confidence token divergence";
    const auto in_shared_top2 = [&](int32_t token) {
      return token == static_cast<int32_t>(actual_top2_ids[step][0]) ||
             token == static_cast<int32_t>(actual_top2_ids[step][1]) ||
             token == static_cast<int32_t>(reference_top2_ids[step][0]) ||
             token == static_cast<int32_t>(reference_top2_ids[step][1]);
    };
    EXPECT_TRUE(in_shared_top2(expected_tokens[step]));
    EXPECT_TRUE(in_shared_top2(actual_tokens[step]));
  }
}

TEST(Qwen25VLModelTest, TextResolutionTwoImageAndMixedPrefillMatchReference) {
  const std::string model_path = RequiredEnv("PBE_QWEN25_VL_MODEL");
  const std::string tokenizer_path = RequiredEnv("PBE_QWEN25_VL_TOKENIZER");
  const std::string fixture_dir = RequiredEnv("PBE_QWEN25_VL_FIXTURE_DIR");
  if (model_path.empty() || tokenizer_path.empty() || fixture_dir.empty()) {
    GTEST_SKIP() << "Set PBE_QWEN25_VL_MODEL, PBE_QWEN25_VL_TOKENIZER and "
                    "PBE_QWEN25_VL_FIXTURE_DIR";
  }

  model::Qwen2Model model(base::TokenizerType::kEncodeBpe, tokenizer_path, model_path, false);
  model.set_kv_cache_blocks_per_layer(128);
  model.set_serving_workspace_token_capacity(512);
  model.set_radix_cache_enabled(false);
  ASSERT_TRUE(model.init(base::DeviceType::kDeviceCUDA, 0));
  auto cuda_alloc = base::CUDADeviceAllocatorFactory::get_instance();

  const auto run_group = [&](const std::vector<std::string>& names) {
    std::vector<std::vector<int32_t>> prompts;
    std::vector<std::vector<int32_t>> positions;
    std::vector<std::vector<uint16_t>> embeddings;
    std::vector<std::vector<float>> expected_logits;
    size_t total_tokens = 0;
    for (const auto& name : names) {
      prompts.push_back(ReadBinary<int32_t>(fixture_dir + "/" + name + ".input_ids.i32.bin"));
      positions.push_back(ReadBinary<int32_t>(fixture_dir + "/" + name + ".position_ids.i32.bin"));
      embeddings.push_back(ReadBinary<uint16_t>(fixture_dir + "/" + name + ".inputs_embeds.bf16.bin"));
      expected_logits.push_back(ReadBinary<float>(fixture_dir + "/" + name + ".last_logits.f32.bin"));
      ASSERT_FALSE(prompts.back().empty());
      ASSERT_EQ(positions.back().size(), prompts.back().size() * 3);
      ASSERT_EQ(embeddings.back().size(), prompts.back().size() * 2048);
      ASSERT_EQ(expected_logits.back().size(), 151936u);
      total_tokens += prompts.back().size();
    }

    std::vector<uint16_t> packed_embeddings;
    packed_embeddings.reserve(total_tokens * 2048);
    for (const auto& value : embeddings)
      packed_embeddings.insert(packed_embeddings.end(), value.begin(), value.end());
    std::vector<int32_t> packed_positions;
    packed_positions.reserve(total_tokens * 3);
    for (size_t axis = 0; axis < 3; ++axis) {
      for (size_t request = 0; request < names.size(); ++request) {
        const size_t tokens = prompts[request].size();
        packed_positions.insert(packed_positions.end(),
                                positions[request].begin() + axis * tokens,
                                positions[request].begin() + (axis + 1) * tokens);
      }
    }

    tensor::Tensor embedding_gpu(base::DataType::kDataTypeBf16,
                                 static_cast<int32_t>(total_tokens), 2048, true, cuda_alloc);
    tensor::Tensor mrope_gpu(base::DataType::kDataTypeInt32, 3,
                             static_cast<int32_t>(total_tokens), true, cuda_alloc);
    cuda_alloc->memcpy(packed_embeddings.data(), embedding_gpu.ptr<uint16_t>(),
                       embedding_gpu.byte_size(), base::MemcpyKind::kMemcpyCPU2CUDA,
                       nullptr, true);
    cuda_alloc->memcpy(packed_positions.data(), mrope_gpu.ptr<int32_t>(),
                       mrope_gpu.byte_size(), base::MemcpyKind::kMemcpyCPU2CUDA,
                       nullptr, true);

    serving::SchedulerConfig config;
    config.max_num_seqs = static_cast<int32_t>(names.size());
    config.max_num_batched_tokens = 512;
    config.prefill_chunk_cap = 512;
    serving::Scheduler scheduler(config, model.kv_cache_manager());
    for (const auto& prompt : prompts) scheduler.add_request(prompt, serving::GenerationConfig(1));
    auto output = scheduler.schedule_step();
    ASSERT_EQ(output.total_tokens, static_cast<int32_t>(total_tokens));
    auto batch = scheduler.build_mixed_batch(output, model.device_context()->compute_queue);
    batch.input_embeddings_override = embedding_gpu;
    batch.mrope_positions = mrope_gpu;
    const auto status = model.forward_mixed_batch(batch);
    ASSERT_TRUE(status) << status.get_err_msg();
    ASSERT_EQ(batch.logits_row_indices.size(), names.size());
    for (size_t request = 0; request < names.size(); ++request) {
      std::vector<float> actual;
      ASSERT_TRUE(model.copy_logits_row_to_host(batch.logits_row_indices[request], &actual));
      double max_abs = 0.0;
      double mean_abs = 0.0;
      double reference_abs = 0.0;
      for (size_t token = 0; token < actual.size(); ++token) {
        const double error = std::abs(actual[token] - expected_logits[request][token]);
        max_abs = std::max(max_abs, error);
        mean_abs += error;
        reference_abs += std::abs(expected_logits[request][token]);
      }
      mean_abs /= actual.size();
      std::cout << "QWEN25_VL_PREFILL case=" << names[request]
                << " group=" << names.size() << " max_abs=" << max_abs
                << " mean_abs=" << mean_abs << " relative_mean="
                << mean_abs / std::max(1.0e-12, reference_abs / actual.size()) << "\n";
      EXPECT_LT(max_abs, 1.0);
      EXPECT_LT(mean_abs, 0.13);
    }
    auto sampled = model.batch_sample(batch, output);
    scheduler.process_outputs(output, batch, sampled,
                              [&](int32_t token) { return model.is_sentence_ending(token); });
    EXPECT_FALSE(scheduler.has_active_requests());
  };

  run_group({"text"});
  run_group({"one_image_wide"});
  run_group({"two_images"});
  run_group({"one_image_224", "one_image_wide"});
}

}  // namespace
