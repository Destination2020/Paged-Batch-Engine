#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <vector>

#include "base/block_allocator.h"
#include "base/kv_cache_manager.h"
#include "serving/request_checkpoint.h"
#ifndef KUIPER_CPU_ONLY
#include <cuda_runtime_api.h>
#endif

namespace {

std::unique_ptr<base::KVCacheManager> make_checkpoint_manager(int blocks = 40) {
  std::vector<std::unique_ptr<base::BlockAllocator>> pools;
  for (int layer = 0; layer < 2; ++layer) {
    pools.emplace_back(std::make_unique<base::BlockAllocator>(
        blocks, 4, 2, 8, base::DataType::kDataTypeBf16,
        base::DeviceType::kDeviceCPU, base::BlockStorageMode::kPlain));
  }
  return std::make_unique<base::KVCacheManager>(4, 2, std::move(pools));
}

void fill_checkpoint_kv(base::KVCacheManager* manager, base::RequestId request,
                        uint8_t seed) {
  for (int layer = 0; layer < manager->num_layers(); ++layer) {
    auto& allocator = manager->allocator_mut(layer);
    const auto& ids = manager->get_block_ids(request, layer);
    for (size_t page = 0; page < ids.size(); ++page) {
      const auto payload = allocator.get_block_payload_ptrs(ids[page]);
      std::memset(payload.key, seed + layer * 17 + page, payload.key_value_bytes);
      std::memset(payload.value, seed + 91 + layer * 17 + page,
                  payload.key_value_bytes);
    }
  }
}

void expect_kv_equal(const base::KVRequestSnapshot& lhs,
                     const base::KVRequestSnapshot& rhs) {
  ASSERT_EQ(lhs.schema_version, rhs.schema_version);
  ASSERT_EQ(lhs.block_size, rhs.block_size);
  ASSERT_EQ(lhs.num_layers, rhs.num_layers);
  ASSERT_EQ(lhs.valid_tokens, rhs.valid_tokens);
  ASSERT_EQ(lhs.committed_tokens, rhs.committed_tokens);
  ASSERT_EQ(lhs.layers.size(), rhs.layers.size());
  for (size_t layer = 0; layer < lhs.layers.size(); ++layer) {
    ASSERT_EQ(lhs.layers[layer].blocks.size(), rhs.layers[layer].blocks.size());
    for (size_t page = 0; page < lhs.layers[layer].blocks.size(); ++page) {
      const auto& a = lhs.layers[layer].blocks[page];
      const auto& b = rhs.layers[layer].blocks[page];
      EXPECT_EQ(a.key, b.key);
      EXPECT_EQ(a.value, b.value);
      EXPECT_EQ(a.key_scale, b.key_scale);
      EXPECT_EQ(a.value_scale, b.value_scale);
    }
  }
}

serving::SequenceState make_sequence(base::RequestId request, int64_t client,
                                     int tokens) {
  serving::SequenceState sequence;
  sequence.request_id = request;
  sequence.client_request_id = client;
  sequence.prompt_tokens.resize(tokens, 11);
  sequence.output_tokens = {21, 22, 23};
  sequence.multimodal_positions = {0, 1, 2, 0, 3, 4};
  sequence.multimodal_rope_delta = -7;
  sequence.multimodal_image_begin = 2;
  sequence.multimodal_feature_rows = 4;
  sequence.multimodal_hidden_size = 2048;
  sequence.multimodal_feature_content = "feature-content-v1";
  sequence.multimodal_feature_representation = "qwen25-vl-bf16-v3";
  sequence.multimodal_exact_dependency = true;
  sequence.generated_tokens = 3;
  sequence.next_token = 23;
  sequence.sampling_counter = 3;
  sequence.emitted_cursor = 2;
  sequence.output_enqueued_cursor = 2;
  sequence.outbox = {{0, 0, 1, "alpha"}, {1, 1, 3, "beta"}};
  sequence.generation_config.sampling.seed = UINT64_C(0xf123456789abcdef);
  sequence.computed_tokens = tokens;
  sequence.status = serving::SequenceStatus::kRunning;
  return sequence;
}

TEST(RequestCheckpointTest, PrepareIsInvisibleAndPageBoundariesRestoreBitwise) {
  for (int tokens : {3, 4, 5}) {
    auto manager = make_checkpoint_manager();
    const auto source = manager->register_request();
    ASSERT_TRUE(manager->append_slots(source, tokens));
    fill_checkpoint_kv(manager.get(), source, static_cast<uint8_t>(tokens * 9));
    ASSERT_TRUE(manager->commit_kv(source, tokens));
    base::KVRequestSnapshot expected;
    ASSERT_TRUE(manager->snapshot_request(source, &expected));
    auto sequence = make_sequence(source, 100 + tokens, tokens);

    serving::RequestCheckpointStore store(8);
    serving::CheckpointTicket ticket;
    ASSERT_TRUE(store.prepare(sequence, "qwen2-test/bf16", manager.get(), &ticket));
    EXPECT_EQ(store.publication_object_count(), 1u);
    EXPECT_GT(store.publication_bytes(), 0u);
    serving::RequestCheckpointManifest manifest;
    EXPECT_FALSE(store.manifest(ticket, &manifest));
    ASSERT_TRUE(store.commit(ticket));
    ASSERT_TRUE(store.manifest(ticket, &manifest));
    EXPECT_EQ(manifest.valid_tokens, tokens);
    EXPECT_EQ(manifest.pending_next_token, 23);
    EXPECT_EQ(manifest.sampling_counter, 3u);
    EXPECT_EQ(manifest.emitted_cursor, 2u);
    EXPECT_EQ(manifest.output_enqueued_cursor, 2u);
    EXPECT_EQ(manifest.multimodal_position_values, 6u);
    EXPECT_TRUE(manifest.multimodal_exact_dependency);

    manager->free_request(source);
    ASSERT_TRUE(store.begin_restore(ticket, "qwen2-test/bf16", manager.get()));
    serving::SequenceState restored;
    ASSERT_TRUE(store.commit_restore(ticket, manager.get(), &restored));
    EXPECT_NE(restored.request_id, source);
    EXPECT_EQ(restored.next_token, sequence.next_token);
    EXPECT_EQ(restored.output_tokens, sequence.output_tokens);
    EXPECT_EQ(restored.sampling_counter, sequence.sampling_counter);
    EXPECT_EQ(restored.emitted_cursor, sequence.emitted_cursor);
    EXPECT_EQ(restored.output_enqueued_cursor, sequence.output_enqueued_cursor);
    EXPECT_EQ(restored.multimodal_positions, sequence.multimodal_positions);
    EXPECT_EQ(restored.multimodal_rope_delta, -7);
    EXPECT_EQ(restored.multimodal_image_begin, 2);
    EXPECT_EQ(restored.multimodal_feature_rows, 4);
    EXPECT_EQ(restored.multimodal_hidden_size, 2048);
    EXPECT_EQ(restored.multimodal_feature_content, "feature-content-v1");
    EXPECT_EQ(restored.multimodal_feature_representation, "qwen25-vl-bf16-v3");
    EXPECT_TRUE(restored.multimodal_exact_dependency);
    ASSERT_EQ(restored.outbox.size(), 2u);
    EXPECT_EQ(restored.outbox[1].text, "beta");
    EXPECT_EQ(restored.outbox[1].sampled_token_end, 3u);
    base::KVRequestSnapshot actual;
    ASSERT_TRUE(manager->snapshot_request(restored.request_id, &actual));
    expect_kv_equal(expected, actual);
    manager->free_request(restored.request_id);
    ASSERT_TRUE(store.erase(ticket));
    EXPECT_EQ(store.publication_object_count(), 0u);
    EXPECT_EQ(store.publication_bytes(), 0u);
  }
}

TEST(RequestCheckpointTest, RevisionsCancelStaleRestoreAndRelocateRepeatedly) {
  auto manager = make_checkpoint_manager();
  serving::RequestCheckpointStore store(12);
  auto current = manager->register_request();
  ASSERT_TRUE(manager->append_slots(current, 5));
  fill_checkpoint_kv(manager.get(), current, 31);
  ASSERT_TRUE(manager->commit_kv(current, 5));
  auto sequence = make_sequence(current, 700, 5);

  serving::CheckpointTicket revision1;
  ASSERT_TRUE(store.save(sequence, "model-A", manager.get(), &revision1));
  ASSERT_TRUE(store.begin_restore(revision1, "model-A", manager.get()));
  serving::CheckpointTicket revision2;
  sequence.sampling_counter = 9;
  sequence.emitted_cursor = 3;
  ASSERT_TRUE(store.save(sequence, "model-A", manager.get(), &revision2));
  EXPECT_GT(revision2.revision, revision1.revision);
  serving::SequenceState stale;
  EXPECT_FALSE(store.commit_restore(revision1, manager.get(), &stale));
  EXPECT_EQ(manager->num_active_requests(), 1);

  for (int relocation = 0; relocation < 3; ++relocation) {
    const auto old = current;
    manager->free_request(old);
    ASSERT_TRUE(store.begin_restore(revision2, "model-A", manager.get()));
    ASSERT_TRUE(store.commit_restore(revision2, manager.get(), &sequence));
    current = sequence.request_id;
    EXPECT_NE(current, old);
    EXPECT_EQ(sequence.sampling_counter, 9u);
    EXPECT_EQ(sequence.emitted_cursor, 3u);
    for (uint64_t offset = 0; offset < 10; ++offset) {
      EXPECT_EQ(serving::request_sample_uniform(
                    sequence.generation_config.sampling.seed,
                    sequence.sampling_counter + offset),
                serving::request_sample_uniform(UINT64_C(0xf123456789abcdef),
                                                 9 + offset));
    }
  }

  serving::CheckpointTicket revision3;
  ASSERT_TRUE(store.save(sequence, "model-A", manager.get(), &revision3));
  ASSERT_TRUE(store.begin_restore(revision3, "model-A", manager.get()));
  store.cancel_client(sequence.client_request_id, manager.get());
  serving::SequenceState cancelled;
  EXPECT_FALSE(store.commit_restore(revision3, manager.get(), &cancelled));
  EXPECT_EQ(manager->num_active_requests(), 1);
  manager->free_request(current);
  EXPECT_EQ(manager->num_active_requests(), 0);
  EXPECT_GE(store.stats().stale_restore_rejections, 1u);
  EXPECT_GE(store.stats().cancelled_restores, 1u);
}

TEST(RequestCheckpointTest, ModelOrSchemaMismatchCannotAllocateDestination) {
  auto manager = make_checkpoint_manager();
  const auto source = manager->register_request();
  ASSERT_TRUE(manager->append_slots(source, 4));
  ASSERT_TRUE(manager->commit_kv(source, 4));
  auto sequence = make_sequence(source, 800, 4);
  serving::RequestCheckpointStore store(2);
  serving::CheckpointTicket ticket;
  ASSERT_TRUE(store.save(sequence, "model-A", manager.get(), &ticket));
  const int before = manager->num_active_requests();
  EXPECT_FALSE(store.begin_restore(ticket, "model-B", manager.get()));
  EXPECT_EQ(manager->num_active_requests(), before);
  manager->free_request(source);
}

TEST(RequestCheckpointTest, OlderPrepareCannotReplaceNewerReadyRevision) {
  auto manager = make_checkpoint_manager();
  const auto source = manager->register_request();
  ASSERT_TRUE(manager->append_slots(source, 4));
  ASSERT_TRUE(manager->commit_kv(source, 4));
  auto sequence = make_sequence(source, 900, 4);
  serving::RequestCheckpointStore store(4);
  serving::CheckpointTicket revision1;
  serving::CheckpointTicket revision2;
  ASSERT_TRUE(store.prepare(sequence, "model-A", manager.get(), &revision1));
  sequence.sampling_counter++;
  ASSERT_TRUE(store.prepare(sequence, "model-A", manager.get(), &revision2));
  ASSERT_TRUE(store.commit(revision2));
  EXPECT_FALSE(store.commit(revision1));
  EXPECT_EQ(store.current_revision(sequence.client_request_id), revision2.revision);
  serving::RequestCheckpointManifest manifest;
  EXPECT_TRUE(store.manifest(revision2, &manifest));
  EXPECT_FALSE(store.manifest(revision1, &manifest));
  EXPECT_EQ(store.stats().stale_commit_rejections, 1u);
  manager->free_request(source);
}

TEST(RequestCheckpointTest, CancelRevokesPreparingTicketAndReclaimsMetadata) {
  auto manager = make_checkpoint_manager();
  const auto source = manager->register_request();
  ASSERT_TRUE(manager->append_slots(source, 4));
  ASSERT_TRUE(manager->commit_kv(source, 4));
  auto sequence = make_sequence(source, 901, 4);
  serving::RequestCheckpointStore store(4);
  serving::CheckpointTicket ticket;
  ASSERT_TRUE(store.prepare(sequence, "model-A", manager.get(), &ticket));
  EXPECT_GT(store.payload_bytes(), 0u);
  store.cancel_client(sequence.client_request_id, manager.get());
  EXPECT_FALSE(store.commit(ticket));
  EXPECT_EQ(store.size(), 0u);
  EXPECT_EQ(store.payload_bytes(), 0u);
  EXPECT_EQ(store.revision_metadata_size(), 0u);
  manager->free_request(source);
}

TEST(RequestCheckpointTest, ByteAndRecordAdmissionPreserveExistingRevision) {
  auto manager = make_checkpoint_manager();
  const auto source = manager->register_request();
  ASSERT_TRUE(manager->append_slots(source, 4));
  ASSERT_TRUE(manager->commit_kv(source, 4));
  auto sequence = make_sequence(source, 902, 4);

  serving::RequestCheckpointStore byte_limited(2, 1);
  serving::CheckpointTicket rejected;
  EXPECT_FALSE(byte_limited.prepare(sequence, "model-A", manager.get(), &rejected));
  EXPECT_EQ(byte_limited.size(), 0u);
  EXPECT_EQ(byte_limited.payload_bytes(), 0u);
  EXPECT_EQ(byte_limited.stats().byte_capacity_rejections, 1u);

  serving::RequestCheckpointStore record_limited(1);
  serving::CheckpointTicket ready;
  ASSERT_TRUE(record_limited.save(sequence, "model-A", manager.get(), &ready));
  serving::CheckpointTicket replacement;
  EXPECT_FALSE(record_limited.prepare(sequence, "model-A", manager.get(),
                                      &replacement));
  EXPECT_EQ(record_limited.current_revision(sequence.client_request_id),
            ready.revision);
  serving::RequestCheckpointManifest manifest;
  EXPECT_TRUE(record_limited.manifest(ready, &manifest));
  EXPECT_EQ(record_limited.stats().record_capacity_rejections, 1u);
  manager->free_request(source);
}

TEST(RequestCheckpointTest, LifecycleChurnLeavesRecordsAndMetadataBounded) {
  auto manager = make_checkpoint_manager();
  serving::RequestCheckpointStore store(8);
  for (int64_t lifecycle = 0; lifecycle < 10000; ++lifecycle) {
    const auto source = manager->register_request();
    auto sequence = make_sequence(source, 10000 + lifecycle, 0);
    sequence.computed_tokens = 0;
    serving::CheckpointTicket ticket;
    ASSERT_TRUE(store.save(sequence, "model-A", manager.get(), &ticket));
    store.cancel_client(sequence.client_request_id, manager.get());
    EXPECT_FALSE(store.commit(ticket));
    manager->free_request(source);
  }
  EXPECT_EQ(store.size(), 0u);
  EXPECT_EQ(store.payload_bytes(), 0u);
  EXPECT_EQ(store.revision_metadata_size(), 0u);
  EXPECT_LE(store.stats().max_records, 1u);
}

#ifndef KUIPER_CPU_ONLY
TEST(RequestCheckpointTest, CudaFp8SnapshotRestoresKvAndScalesAcrossNewBlocks) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
    GTEST_SKIP() << "A CUDA device is required";
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  std::vector<std::unique_ptr<base::BlockAllocator>> pools;
  pools.emplace_back(std::make_unique<base::BlockAllocator>(
      8, 4, 2, 8, base::DataType::kDataTypeBf16,
      base::DeviceType::kDeviceCUDA,
      base::BlockStorageMode::kFp8E4M3PerTokenHead));
  base::KVCacheManager manager(4, 1, std::move(pools));
  const auto source = manager.register_request();
  ASSERT_TRUE(manager.append_slots(source, 5));
  auto cuda_allocator = base::CUDADeviceAllocatorFactory::get_instance();
  for (int block_id : manager.get_block_ids(source, 0)) {
    const auto payload = manager.allocator_mut(0).get_block_payload_ptrs(block_id);
    std::vector<uint8_t> kv(payload.key_value_bytes, static_cast<uint8_t>(31 + block_id));
    std::vector<uint8_t> scales(payload.scale_bytes, static_cast<uint8_t>(71 + block_id));
    cuda_allocator->memcpy(kv.data(), payload.key, kv.size(),
                           base::MemcpyKind::kMemcpyCPU2CUDA, nullptr, true);
    cuda_allocator->memcpy(kv.data(), payload.value, kv.size(),
                           base::MemcpyKind::kMemcpyCPU2CUDA, nullptr, true);
    cuda_allocator->memcpy(scales.data(), payload.key_scale, scales.size(),
                           base::MemcpyKind::kMemcpyCPU2CUDA, nullptr, true);
    cuda_allocator->memcpy(scales.data(), payload.value_scale, scales.size(),
                           base::MemcpyKind::kMemcpyCPU2CUDA, nullptr, true);
  }
  ASSERT_TRUE(manager.commit_kv(source, 5));
  base::KVRequestSnapshot expected;
  ASSERT_TRUE(manager.snapshot_request(source, &expected));
  manager.free_request(source);
  base::RequestId restored = -1;
  ASSERT_TRUE(manager.restore_request_snapshot(expected, &restored));
  EXPECT_NE(restored, source);
  base::KVRequestSnapshot actual;
  ASSERT_TRUE(manager.snapshot_request(restored, &actual));
  expect_kv_equal(expected, actual);
  manager.free_request(restored);
}
#endif

}  // namespace
