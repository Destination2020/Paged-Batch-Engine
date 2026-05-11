#include <gtest/gtest.h>

#include <initializer_list>
#include <memory>
#include <vector>

#include "base/block_allocator.h"
#include "base/kv_cache_manager.h"
#include "serving/scheduler.h"
#include "tensor/tensor.h"

namespace {

std::vector<std::unique_ptr<base::BlockAllocator>> make_allocators(
    int32_t num_layers, int32_t num_blocks, int32_t block_size) {
  std::vector<std::unique_ptr<base::BlockAllocator>> allocators;
  allocators.reserve(num_layers);
  for (int32_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    allocators.push_back(std::make_unique<base::BlockAllocator>(
        num_blocks, block_size, 1, 8, base::DataType::kDataTypeFp32,
        base::DeviceType::kDeviceCPU));
  }
  return allocators;
}

std::vector<int32_t> Tokens(std::initializer_list<int32_t> values) {
  return std::vector<int32_t>(values);
}

std::vector<int32_t> TensorToIntVector(tensor::Tensor tensor) {
  tensor.to_host();
  const int32_t* ptr = tensor.ptr<int32_t>();
  return std::vector<int32_t>(ptr, ptr + tensor.size());
}

std::vector<int32_t> WarmPromptIntoRadixCache(base::KVCacheManager* kv_manager,
                                              const std::vector<int32_t>& prompt_tokens) {
  CHECK_NE(kv_manager, nullptr);
  const base::RequestId request_id = kv_manager->register_request();
  CHECK(kv_manager->append_slots(request_id, static_cast<int32_t>(prompt_tokens.size())));
  std::vector<int32_t> block_ids = kv_manager->get_block_ids(request_id, 0);
  kv_manager->publish_radix_cache(request_id, prompt_tokens);
  kv_manager->free_request(request_id);
  return block_ids;
}

}  // namespace

TEST(SchedulerRadixCacheTest, PartialHitBuildsCorrectPrefillMetadataAcrossChunks) {
  constexpr int32_t kBlockSize = 2;
  constexpr int32_t kNumLayers = 1;

  base::KVCacheManager kv_manager(
      kBlockSize, kNumLayers, make_allocators(kNumLayers, 16, kBlockSize));

  const auto warm_prompt = Tokens({1, 2, 3, 4, 5, 6, 7, 8});
  const std::vector<int32_t> cached_blocks =
      WarmPromptIntoRadixCache(&kv_manager, warm_prompt);
  ASSERT_EQ(cached_blocks.size(), 4);

  serving::SchedulerConfig config;
  config.max_num_seqs = 1;
  config.max_num_batched_tokens = 4;
  config.prefill_chunk_cap = 2;

  serving::Scheduler scheduler(config, &kv_manager);
  const auto partial_hit_prompt = Tokens({1, 2, 3, 4, 5, 6, 11, 12, 13, 14});
  scheduler.add_request(partial_hit_prompt, serving::GenerationConfig(4));

  const auto output1 = scheduler.schedule_step();
  ASSERT_EQ(output1.num_prefill_seqs, 1);
  ASSERT_EQ(output1.num_decode_seqs, 0);
  ASSERT_EQ(output1.total_tokens, 2);
  ASSERT_EQ(output1.num_tokens_per_seq.size(), 1u);
  ASSERT_EQ(output1.num_tokens_per_seq[0], 2);
  ASSERT_EQ(output1.scheduled_seqs.size(), 1u);

  auto* seq1 = output1.scheduled_seqs[0];
  ASSERT_NE(seq1, nullptr);
  EXPECT_EQ(seq1->computed_tokens, 6);
  EXPECT_EQ(kv_manager.get_context_len(seq1->request_id), 6);

  const auto& request_blocks_before_chunk = kv_manager.get_block_ids(seq1->request_id, 0);
  ASSERT_EQ(request_blocks_before_chunk.size(), 3u);
  EXPECT_EQ(request_blocks_before_chunk[0], cached_blocks[0]);
  EXPECT_EQ(request_blocks_before_chunk[1], cached_blocks[1]);
  EXPECT_EQ(request_blocks_before_chunk[2], cached_blocks[2]);

  auto batch1 = scheduler.build_mixed_batch(output1, nullptr);
  EXPECT_EQ(batch1.num_requests, 1);
  EXPECT_EQ(batch1.num_prefill_tokens, 2);
  EXPECT_TRUE(batch1.sample_row_to_request.empty());
  EXPECT_EQ(TensorToIntVector(batch1.token_ids), Tokens({11, 12}));
  EXPECT_EQ(TensorToIntVector(batch1.positions), Tokens({6, 7}));
  EXPECT_EQ(TensorToIntVector(batch1.seq_lens), Tokens({8}));
  EXPECT_EQ(TensorToIntVector(batch1.block_tables),
            std::vector<int32_t>({cached_blocks[0], cached_blocks[1], cached_blocks[2]}));

  ASSERT_TRUE(kv_manager.append_slots(seq1->request_id, output1.num_tokens_per_seq[0]));
  const auto& request_blocks_after_chunk1 = kv_manager.get_block_ids(seq1->request_id, 0);
  ASSERT_EQ(request_blocks_after_chunk1.size(), 4u);
  const int32_t first_private_block = request_blocks_after_chunk1[3];

  serving::SampledTokenView no_samples{nullptr, 0};
  scheduler.process_outputs(
      output1, batch1, no_samples,
      [](int32_t /*token*/) { return false; });

  const auto output2 = scheduler.schedule_step();
  ASSERT_EQ(output2.num_prefill_seqs, 1);
  ASSERT_EQ(output2.num_decode_seqs, 0);
  ASSERT_EQ(output2.total_tokens, 2);
  ASSERT_EQ(output2.num_tokens_per_seq.size(), 1u);
  ASSERT_EQ(output2.num_tokens_per_seq[0], 2);
  ASSERT_EQ(output2.scheduled_seqs.size(), 1u);

  auto* seq2 = output2.scheduled_seqs[0];
  ASSERT_NE(seq2, nullptr);
  EXPECT_EQ(seq2->computed_tokens, 8);
  EXPECT_EQ(kv_manager.get_context_len(seq2->request_id), 8);

  auto batch2 = scheduler.build_mixed_batch(output2, nullptr);
  EXPECT_EQ(batch2.num_requests, 1);
  EXPECT_EQ(batch2.num_prefill_tokens, 2);
  EXPECT_EQ(batch2.sample_row_to_request, std::vector<int32_t>({0}));
  EXPECT_EQ(TensorToIntVector(batch2.token_ids), Tokens({13, 14}));
  EXPECT_EQ(TensorToIntVector(batch2.positions), Tokens({8, 9}));
  EXPECT_EQ(TensorToIntVector(batch2.seq_lens), Tokens({10}));
  EXPECT_EQ(
      TensorToIntVector(batch2.block_tables),
      std::vector<int32_t>({cached_blocks[0], cached_blocks[1], cached_blocks[2],
                            first_private_block}));
}

TEST(SchedulerPolicyTest, PriorityPolicyAdmitsHigherPriorityWaitingRequestFirst) {
  constexpr int32_t kBlockSize = 4;
  constexpr int32_t kNumLayers = 1;
  base::KVCacheManager kv_manager(
      kBlockSize, kNumLayers, make_allocators(kNumLayers, 64, kBlockSize));

  serving::SchedulerConfig config;
  config.max_num_seqs = 2;
  config.max_num_batched_tokens = 4;
  config.prefill_chunk_cap = 4;
  config.policy = serving::SchedulingPolicy::kPriority;

  serving::Scheduler scheduler(config, &kv_manager);
  auto low_priority = serving::GenerationConfig(4, 0, false, 1);
  auto high_priority = serving::GenerationConfig(4, 0, false, 10);

  scheduler.add_request(Tokens({1, 2, 3, 4}), low_priority);
  scheduler.add_request(Tokens({5, 6, 7, 8}), high_priority);

  const auto output = scheduler.schedule_step();
  ASSERT_EQ(output.scheduled_seqs.size(), 1u);
  EXPECT_EQ(output.scheduled_seqs[0]->generation_config.priority, 10);
  EXPECT_EQ(output.num_tokens_per_seq[0], 4);
}

TEST(SchedulerPolicyTest, LongPrefillThresholdCapsChunkSize) {
  constexpr int32_t kBlockSize = 4;
  constexpr int32_t kNumLayers = 1;
  base::KVCacheManager kv_manager(
      kBlockSize, kNumLayers, make_allocators(kNumLayers, 64, kBlockSize));

  serving::SchedulerConfig config;
  config.max_num_seqs = 1;
  config.max_num_batched_tokens = 8;
  config.prefill_chunk_cap = 8;
  config.long_prefill_token_threshold = 3;

  serving::Scheduler scheduler(config, &kv_manager);
  scheduler.add_request(Tokens({1, 2, 3, 4, 5, 6, 7, 8}),
                        serving::GenerationConfig(4));

  const auto output = scheduler.schedule_step();
  ASSERT_EQ(output.scheduled_seqs.size(), 1u);
  EXPECT_EQ(output.num_prefill_seqs, 1);
  EXPECT_EQ(output.num_tokens_per_seq[0], 3);
}

TEST(SchedulerDecodeReadyTest, AdmissionRespectsMaxNumSeqs) {
  constexpr int32_t kBlockSize = 4;
  constexpr int32_t kNumLayers = 1;
  base::KVCacheManager kv_manager(
      kBlockSize, kNumLayers, make_allocators(kNumLayers, 64, kBlockSize));

  serving::SchedulerConfig config;
  config.max_num_seqs = 2;
  config.max_num_batched_tokens = 16;
  config.prefill_chunk_cap = 4;

  serving::Scheduler scheduler(config, &kv_manager);
  serving::GenerationConfig generation_config(4);

  std::vector<base::RequestId> request_ids;
  for (int i = 0; i < 4; ++i) {
    const base::RequestId request_id = kv_manager.register_request();
    ASSERT_TRUE(kv_manager.append_slots(request_id, 4));
    request_ids.push_back(request_id);
    scheduler.add_decode_ready_request(request_id, Tokens({1, 2, 3, 4}),
                                       generation_config, 4, 100 + i);
  }

  auto output = scheduler.schedule_step();
  EXPECT_EQ(output.num_decode_seqs, 2);
  EXPECT_EQ(output.total_tokens, 2);
  EXPECT_EQ(output.running_queue_size, 2);
  EXPECT_EQ(output.waiting_queue_size, 2);

  auto batch = scheduler.build_decode_batch(output, nullptr);
  std::vector<int32_t> sample_tokens = {200, 201};
  serving::SampledTokenView sampled{
      sample_tokens.data(), static_cast<int32_t>(sample_tokens.size())};
  scheduler.process_outputs(output, batch, sampled,
                            [](int32_t /*token*/) { return true; });
  scheduler.pop_finished();

  output = scheduler.schedule_step();
  EXPECT_EQ(output.num_decode_seqs, 2);
  EXPECT_EQ(output.running_queue_size, 2);
  EXPECT_EQ(output.waiting_queue_size, 0);
}

TEST(SchedulerDecodeReadyTest, AdmissionRespectsEstimatedRemainingBlocks) {
  constexpr int32_t kBlockSize = 4;
  constexpr int32_t kNumLayers = 1;
  base::KVCacheManager kv_manager(
      kBlockSize, kNumLayers, make_allocators(kNumLayers, 5, kBlockSize));

  serving::SchedulerConfig config;
  config.max_num_seqs = 4;
  config.max_num_batched_tokens = 16;
  config.prefill_chunk_cap = 4;

  serving::Scheduler scheduler(config, &kv_manager);
  serving::GenerationConfig generation_config(8);

  for (int i = 0; i < 2; ++i) {
    const base::RequestId request_id = kv_manager.register_request();
    ASSERT_TRUE(kv_manager.append_slots(request_id, 4));
    scheduler.add_decode_ready_request(request_id, Tokens({1, 2, 3, 4}),
                                       generation_config, 4, 10 + i);
  }

  auto output = scheduler.schedule_step();
  EXPECT_EQ(output.num_decode_seqs, 1);
  EXPECT_EQ(output.total_tokens, 1);
  EXPECT_EQ(output.running_queue_size, 1);
  EXPECT_EQ(output.waiting_queue_size, 1);
}
