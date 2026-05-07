#include <gtest/gtest.h>

#include <memory>

#include "base/block_allocator.h"
#include "serving/decode_kv_reservation.h"
#include "serving/pd_engine.h"
#include "serving/pd_coordinator.h"
#include "serving/pd_handoff_builder.h"
#include "serving/serving_benchmark_app.h"

namespace serving {
namespace {

class StubServingApp final : public ServingBenchmarkApp {
 public:
  const char* usage_name() const override { return "stub"; }
  bool initialize_model(const std::string&, const std::string&, const BenchConfig&) override {
    return true;
  }
  int32_t max_model_batch_size() const override { return 4; }
  base::KVCacheManager* kv_cache_manager() const override { return nullptr; }
  ServingCapacityInfo serving_capacity_info() const override { return {}; }
  void* model_stream() const override { return nullptr; }
  std::vector<int32_t> encode_prompt(const std::string&) const override { return {}; }
  std::string decode_tokens(const std::vector<int32_t>&) const override { return {}; }
  bool is_sentence_ending(int32_t) const override { return false; }
  base::Status forward_mixed_batch(const MixedBatchMetadata& batch) const override {
    ++forward_mixed_calls;
    last_forward_mixed_tokens = batch.num_tokens;
    if (kv_manager_for_mixed_forward != nullptr) {
      for (int32_t i = 0; i < batch.num_requests; ++i) {
        if (batch.tokens_per_request[i] > 0) {
          CHECK(kv_manager_for_mixed_forward->append_slots(batch.request_ids[i],
                                                           batch.tokens_per_request[i]));
        }
      }
    }
    return base::error::Success();
  }
  base::Status forward_decode_batch(const MixedBatchMetadata& batch) const override {
    ++forward_decode_calls;
    last_forward_decode_tokens = batch.num_tokens;
    return base::error::Success();
  }
  SampledTokenView batch_sample(const MixedBatchMetadata& batch, const SchedulerOutput&) const override {
    sampled_tokens.assign(batch.sample_row_to_request.size(), 7);
    return {sampled_tokens.data(), static_cast<int32_t>(sampled_tokens.size())};
  }

  mutable int32_t forward_mixed_calls = 0;
  mutable int32_t forward_decode_calls = 0;
  mutable int32_t last_forward_mixed_tokens = 0;
  mutable int32_t last_forward_decode_tokens = 0;
  mutable base::KVCacheManager* kv_manager_for_mixed_forward = nullptr;
  mutable std::vector<int32_t> sampled_tokens;
};

std::unique_ptr<base::KVCacheManager> make_kv_manager(int32_t num_blocks = 16) {
  std::vector<std::unique_ptr<base::BlockAllocator>> allocators;
  allocators.emplace_back(std::make_unique<base::BlockAllocator>(
      num_blocks, 4, 2, 8, base::DataType::kDataTypeBf16,
      base::DeviceType::kDeviceCPU, base::BlockStorageMode::kPlain));
  allocators.emplace_back(std::make_unique<base::BlockAllocator>(
      num_blocks, 4, 2, 8, base::DataType::kDataTypeBf16,
      base::DeviceType::kDeviceCPU, base::BlockStorageMode::kPlain));
  return std::make_unique<base::KVCacheManager>(4, 2, std::move(allocators));
}

KVPoolDescriptor make_pool(int32_t device_id) {
  KVPoolDescriptor pool;
  pool.device_id = device_id;
  pool.layer_num = 2;
  pool.block_size = 4;
  pool.kv_head_num = 2;
  pool.head_size = 8;
  pool.dtype = base::DataType::kDataTypeBf16;
  pool.storage_mode = base::BlockStorageMode::kPlain;
  return pool;
}

InProcPDEngineConfig make_engine_config(int32_t device_id) {
  InProcPDEngineConfig config;
  config.scheduler_config.max_num_seqs = 4;
  config.scheduler_config.max_num_batched_tokens = 16;
  config.scheduler_config.prefill_chunk_cap = 4;
  config.kv_pool = make_pool(device_id);
  return config;
}

void fill_request_blocks(base::KVCacheManager* kv_manager,
                         base::RequestId request_id,
                         int32_t base_value) {
  for (int32_t layer_idx = 0; layer_idx < kv_manager->num_layers(); ++layer_idx) {
    const auto& block_ids = kv_manager->get_block_ids(request_id, layer_idx);
    auto& allocator = kv_manager->allocator_mut(layer_idx);
    for (int32_t block_pos = 0; block_pos < static_cast<int32_t>(block_ids.size()); ++block_pos) {
      const auto ptrs = allocator.get_block_payload_ptrs(block_ids[block_pos]);
      auto* key = static_cast<uint16_t*>(ptrs.key);
      auto* value = static_cast<uint16_t*>(ptrs.value);
      for (size_t i = 0; i < ptrs.key_value_bytes / sizeof(uint16_t); ++i) {
        key[i] = static_cast<uint16_t>(base_value + layer_idx * 100 + block_pos * 10 + i % 10);
        value[i] = static_cast<uint16_t>(base_value + 1000 + layer_idx * 100 + block_pos * 10 + i % 10);
      }
    }
  }
}

void expect_request_blocks_equal(base::KVCacheManager* src_kv_manager,
                                 base::RequestId src_request_id,
                                 base::KVCacheManager* dst_kv_manager,
                                 base::RequestId dst_request_id) {
  for (int32_t layer_idx = 0; layer_idx < src_kv_manager->num_layers(); ++layer_idx) {
    const auto& src_block_ids = src_kv_manager->get_block_ids(src_request_id, layer_idx);
    const auto& dst_block_ids = dst_kv_manager->get_block_ids(dst_request_id, layer_idx);
    ASSERT_EQ(src_block_ids.size(), dst_block_ids.size());
    auto& src_allocator = src_kv_manager->allocator_mut(layer_idx);
    auto& dst_allocator = dst_kv_manager->allocator_mut(layer_idx);
    for (int32_t block_pos = 0; block_pos < static_cast<int32_t>(src_block_ids.size()); ++block_pos) {
      const auto src = src_allocator.get_block_payload_ptrs(src_block_ids[block_pos]);
      const auto dst = dst_allocator.get_block_payload_ptrs(dst_block_ids[block_pos]);
      const auto* src_key = static_cast<const uint16_t*>(src.key);
      const auto* src_value = static_cast<const uint16_t*>(src.value);
      const auto* dst_key = static_cast<const uint16_t*>(dst.key);
      const auto* dst_value = static_cast<const uint16_t*>(dst.value);
      for (size_t i = 0; i < src.key_value_bytes / sizeof(uint16_t); ++i) {
        EXPECT_EQ(dst_key[i], src_key[i]);
        EXPECT_EQ(dst_value[i], src_value[i]);
      }
    }
  }
}

}  // namespace

TEST(PDEngineTest, PrefillAndDecodeEnginesExposeIndependentSchedulersAndKvManagers) {
  StubServingApp app;
  auto prefill_kv = make_kv_manager();
  auto decode_kv = make_kv_manager();

  InProcPrefillEngine prefill_engine(&app, prefill_kv.get(), make_engine_config(0));
  InProcDecodeEngine decode_engine(&app, decode_kv.get(), make_engine_config(1));

  EXPECT_NE(prefill_engine.scheduler(), decode_engine.scheduler());
  EXPECT_EQ(prefill_engine.kv_manager(), prefill_kv.get());
  EXPECT_EQ(decode_engine.kv_manager(), decode_kv.get());
  EXPECT_EQ(prefill_engine.kv_pool().device_id, 0);
  EXPECT_EQ(decode_engine.kv_pool().device_id, 1);
}

TEST(PDEngineTest, PrefillEngineOwnsPrefillSchedulerAdmission) {
  StubServingApp app;
  auto prefill_kv = make_kv_manager();
  InProcPrefillEngine prefill_engine(&app, prefill_kv.get(), make_engine_config(0));

  GenerationConfig generation_config;
  generation_config.max_new_tokens = 4;
  PrefillSubmitResult result;
  ASSERT_TRUE(prefill_engine.submit_request({1, 2, 3, 4, 5, 6}, generation_config, &result));
  EXPECT_GE(result.request_id, 0);

  SchedulerOutput output = prefill_engine.schedule_step();
  EXPECT_GT(output.total_tokens, 0);
  EXPECT_GT(output.num_prefill_seqs, 0);
  EXPECT_EQ(output.num_decode_seqs, 0);
  EXPECT_EQ(classify_pd_worker_step(output), PDWorkerStepKind::kPrefillOnly);
}

TEST(PDEngineTest, DecodeEngineRejectsNonDecodeOnlyExecutionAtWorkerBoundary) {
  StubServingApp app;
  auto decode_kv = make_kv_manager();
  InProcDecodeEngine decode_engine(&app, decode_kv.get(), make_engine_config(1));

  GenerationConfig generation_config;
  generation_config.max_new_tokens = 4;
  int64_t request_id = -1;
  ASSERT_TRUE(decode_engine.submit_decode_ready_request({1, 2, 3, 4}, generation_config,
                                                        &request_id));
  EXPECT_GE(request_id, 0);

  SchedulerOutput output = decode_engine.schedule_step();
  EXPECT_GT(output.num_prefill_seqs, 0);
  PDWorkerStepOutput step_output;
  EXPECT_FALSE(decode_engine.execute_step(output, nullptr, &step_output));
}

TEST(PDEngineTest, DecodeEngineInjectsReservationAsDecodeReadyRequest) {
  StubServingApp app;
  auto decode_kv = make_kv_manager();
  InProcDecodeEngine decode_engine(&app, decode_kv.get(), make_engine_config(1));
  DecodeKVReservationManager reservation_manager(decode_kv.get(), make_pool(1));

  DecodeKVReservationRequest reservation_request;
  reservation_request.client_request_id.value = "decode-ready";
  reservation_request.handoff_id.value = 31;
  reservation_request.prompt_tokens = 4;
  reservation_request.computed_tokens = 4;
  reservation_request.first_token = 9;
  reservation_request.src_pool = make_pool(0);
  reservation_request.src_block_ids_per_layer = {{10}, {20}};

  DecodeKVReservation reservation;
  KVBlockManifest manifest;
  ASSERT_TRUE(reservation_manager.reserve(reservation_request, &reservation, &manifest));

  GenerationConfig generation_config;
  generation_config.max_new_tokens = 4;
  DecodeReadySubmitResult result;
  ASSERT_TRUE(decode_engine.submit_decode_ready_request(
      reservation, {1, 2, 3, 4}, generation_config, reservation_request.first_token,
      &result));
  EXPECT_GE(result.request_id, 0);

  SchedulerOutput output = decode_engine.schedule_step();
  EXPECT_EQ(output.num_prefill_seqs, 0);
  EXPECT_EQ(output.num_decode_seqs, 1);
  EXPECT_EQ(output.total_tokens, 1);
  EXPECT_EQ(classify_pd_worker_step(output), PDWorkerStepKind::kDecodeOnly);
  ASSERT_EQ(output.scheduled_seqs.size(), 1);
  EXPECT_EQ(output.scheduled_seqs[0]->request_id, reservation.decode_request_id);
  EXPECT_EQ(output.scheduled_seqs[0]->next_token, reservation_request.first_token);
  EXPECT_EQ(output.scheduled_seqs[0]->generated_tokens, 1);
  EXPECT_EQ(output.scheduled_seqs[0]->output_tokens,
            std::vector<int32_t>({reservation_request.first_token}));
  EXPECT_TRUE(output.scheduled_seqs[0]->first_token_recorded);
  EXPECT_EQ(decode_kv->get_context_len(reservation.decode_request_id), 5);

  reservation_manager.release(&reservation);
}

TEST(PDEngineTest, InProcPrefillCopyDecodeEnginePipelineSchedulesDecodeOnly) {
  StubServingApp app;
  auto prefill_kv = make_kv_manager();
  auto decode_kv = make_kv_manager();
  app.kv_manager_for_mixed_forward = prefill_kv.get();
  InProcPrefillEngine prefill_engine(&app, prefill_kv.get(), make_engine_config(0));
  InProcDecodeEngine decode_engine(&app, decode_kv.get(), make_engine_config(1));

  const std::vector<int32_t> prompt_tokens = {1, 2, 3, 4};
  GenerationConfig generation_config;
  generation_config.max_new_tokens = 4;
  PrefillSubmitResult prefill_submit;
  ASSERT_TRUE(prefill_engine.submit_request(prompt_tokens, generation_config, &prefill_submit));

  SchedulerOutput prefill_output = prefill_engine.schedule_step();
  ASSERT_EQ(prefill_output.num_prefill_seqs, 1);
  ASSERT_EQ(prefill_output.num_decode_seqs, 0);
  ASSERT_EQ(prefill_output.scheduled_seqs.size(), 1);
  base::RequestId src_request_id = prefill_output.scheduled_seqs[0]->request_id;
  PDWorkerStepOutput prefill_step;
  ASSERT_TRUE(prefill_engine.execute_and_process_step(prefill_output, nullptr, &prefill_step));
  EXPECT_EQ(app.forward_mixed_calls, 1);
  EXPECT_EQ(app.last_forward_mixed_tokens, static_cast<int32_t>(prompt_tokens.size()));
  ASSERT_EQ(prefill_output.scheduled_seqs[0]->computed_tokens,
            static_cast<int32_t>(prompt_tokens.size()));
  ASSERT_FALSE(prefill_output.scheduled_seqs[0]->is_prefill());
  fill_request_blocks(prefill_kv.get(), src_request_id, 300);

  DecodeKVReservationManager reservation_manager(decode_kv.get(), decode_engine.kv_pool());
  PrefillHandoffBuildRequest handoff_build;
  handoff_build.scheduled_request_index = 0;
  handoff_build.client_request_id.value = "p-to-d";
  handoff_build.handoff_id.value = 41;
  PDHandoffBuilder handoff_builder(prefill_kv.get(), prefill_engine.kv_pool());
  DecodeKVReservationRequest reservation_request;
  ASSERT_TRUE(handoff_builder.build_decode_reservation_request(
      prefill_output, handoff_build, &reservation_request));
  EXPECT_EQ(reservation_request.first_token, 7);
  ASSERT_EQ(reservation_request.src_block_ids_per_layer.size(), 2);
  EXPECT_EQ(reservation_request.src_block_ids_per_layer[0],
            prefill_kv->get_block_ids(src_request_id, 0));

  DecodeKVReservation reservation;
  KVBlockManifest manifest;
  ASSERT_TRUE(reservation_manager.reserve(reservation_request, &reservation, &manifest));
  InProcKVBlockCopyConnector connector(prefill_kv.get(), decode_kv.get());
  PDCoordinator coordinator(&connector);
  PDHandoffState handoff_state;
  ASSERT_TRUE(coordinator.start_prefill_handoff(manifest, &handoff_state));
  ASSERT_TRUE(handoff_state.ready_for_decode());

  expect_request_blocks_equal(prefill_kv.get(), src_request_id,
                              decode_kv.get(), reservation.decode_request_id);

  DecodeReadySubmitResult decode_submit;
  ASSERT_TRUE(decode_engine.submit_decode_ready_request(
      reservation, prompt_tokens, generation_config, reservation_request.first_token,
      &decode_submit));
  SchedulerOutput decode_output = decode_engine.schedule_step();
  EXPECT_EQ(decode_output.num_prefill_seqs, 0);
  EXPECT_EQ(decode_output.num_decode_seqs, 1);
  EXPECT_EQ(decode_output.total_tokens, 1);
  EXPECT_EQ(classify_pd_worker_step(decode_output), PDWorkerStepKind::kDecodeOnly);
  ASSERT_EQ(decode_output.scheduled_seqs.size(), 1);
  EXPECT_EQ(decode_output.scheduled_seqs[0]->request_id, reservation.decode_request_id);
  EXPECT_EQ(decode_output.scheduled_seqs[0]->next_token, reservation_request.first_token);
  PDWorkerStepOutput decode_step;
  ASSERT_TRUE(decode_engine.execute_step(decode_output, nullptr, &decode_step));
  EXPECT_EQ(app.forward_decode_calls, 1);
  EXPECT_EQ(app.last_forward_decode_tokens, 1);

  reservation_manager.release(&reservation);
  prefill_kv->free_request(src_request_id);
}

}  // namespace serving
