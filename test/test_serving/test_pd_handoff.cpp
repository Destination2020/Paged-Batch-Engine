#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <memory>
#include <string>

#include "base/block_allocator.h"
#include "base/kv_cache_manager.h"
#include "serving/decode_kv_reservation.h"
#include "serving/pd_coordinator.h"
#include "serving/pd_handoff.h"

namespace serving {
namespace {

KVPoolDescriptor make_pool(int32_t device_id) {
  KVPoolDescriptor pool;
  pool.device_id = device_id;
  pool.layer_num = 2;
  pool.block_size = 16;
  pool.kv_head_num = 2;
  pool.head_size = 64;
  pool.dtype = base::DataType::kDataTypeBf16;
  pool.storage_mode = base::BlockStorageMode::kPlain;
  return pool;
}

KVBlockManifest make_manifest() {
  KVBlockManifest manifest;
  manifest.client_request_id.value = "req-1";
  manifest.handoff_id.value = 7;
  manifest.prompt_tokens = 32;
  manifest.computed_tokens = 32;
  manifest.first_token = 42;
  manifest.src_pool = make_pool(0);
  manifest.dst_pool = make_pool(1);
  manifest.layer_mappings = {
      {0, {1, 2}, {101, 102}},
      {1, {3, 4}, {103, 104}},
  };
  return manifest;
}

std::unique_ptr<base::KVCacheManager> make_kv_manager(int32_t num_blocks = 8) {
  std::vector<std::unique_ptr<base::BlockAllocator>> allocators;
  allocators.emplace_back(std::make_unique<base::BlockAllocator>(
      num_blocks, 16, 2, 64, base::DataType::kDataTypeBf16,
      base::DeviceType::kDeviceCPU, base::BlockStorageMode::kPlain));
  allocators.emplace_back(std::make_unique<base::BlockAllocator>(
      num_blocks, 16, 2, 64, base::DataType::kDataTypeBf16,
      base::DeviceType::kDeviceCPU, base::BlockStorageMode::kPlain));
  return std::make_unique<base::KVCacheManager>(16, 2, std::move(allocators));
}

std::unique_ptr<base::KVCacheManager> make_cuda_fp8_kv_manager(int32_t num_blocks = 8) {
  std::vector<std::unique_ptr<base::BlockAllocator>> allocators;
  allocators.emplace_back(std::make_unique<base::BlockAllocator>(
      num_blocks, 16, 2, 64, base::DataType::kDataTypeBf16,
      base::DeviceType::kDeviceCUDA, base::BlockStorageMode::kFp8E4M3PerTokenHead));
  allocators.emplace_back(std::make_unique<base::BlockAllocator>(
      num_blocks, 16, 2, 64, base::DataType::kDataTypeBf16,
      base::DeviceType::kDeviceCUDA, base::BlockStorageMode::kFp8E4M3PerTokenHead));
  return std::make_unique<base::KVCacheManager>(16, 2, std::move(allocators));
}

bool cuda_available() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

void fill_request_blocks(base::KVCacheManager* kv_manager,
                         base::RequestId request_id,
                         int32_t base_value) {
  for (int32_t layer_idx = 0; layer_idx < kv_manager->num_layers(); ++layer_idx) {
    const auto& block_ids = kv_manager->get_block_ids(request_id, layer_idx);
    auto& allocator = kv_manager->allocator_mut(layer_idx);
    for (int32_t block_pos = 0; block_pos < static_cast<int32_t>(block_ids.size()); ++block_pos) {
      const auto [key_ptr, value_ptr] = allocator.get_block_ptrs(block_ids[block_pos]);
      auto* key = static_cast<uint16_t*>(key_ptr);
      auto* value = static_cast<uint16_t*>(value_ptr);
      for (int32_t i = 0; i < 16 * 2 * 64; ++i) {
        key[i] = static_cast<uint16_t>(base_value + layer_idx * 1000 + block_pos * 10 + i % 10);
        value[i] = static_cast<uint16_t>(base_value + 5000 + layer_idx * 1000 + block_pos * 10 + i % 10);
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
    auto& src_allocator = src_kv_manager->allocator_mut(layer_idx);
    auto& dst_allocator = dst_kv_manager->allocator_mut(layer_idx);
    ASSERT_EQ(src_block_ids.size(), dst_block_ids.size());
    for (int32_t block_pos = 0; block_pos < static_cast<int32_t>(src_block_ids.size()); ++block_pos) {
      const auto [src_key_ptr, src_value_ptr] = src_allocator.get_block_ptrs(src_block_ids[block_pos]);
      const auto [dst_key_ptr, dst_value_ptr] = dst_allocator.get_block_ptrs(dst_block_ids[block_pos]);
      const auto* src_key = static_cast<const uint16_t*>(src_key_ptr);
      const auto* src_value = static_cast<const uint16_t*>(src_value_ptr);
      const auto* dst_key = static_cast<const uint16_t*>(dst_key_ptr);
      const auto* dst_value = static_cast<const uint16_t*>(dst_value_ptr);
      for (int32_t i = 0; i < 16 * 2 * 64; ++i) {
        EXPECT_EQ(dst_key[i], src_key[i]);
        EXPECT_EQ(dst_value[i], src_value[i]);
      }
    }
  }
}

void fill_cuda_fp8_request_blocks(base::KVCacheManager* kv_manager,
                                  base::RequestId request_id,
                                  int32_t base_value) {
  auto cuda_allocator = base::CUDADeviceAllocatorFactory::get_instance();
  for (int32_t layer_idx = 0; layer_idx < kv_manager->num_layers(); ++layer_idx) {
    const auto& block_ids = kv_manager->get_block_ids(request_id, layer_idx);
    auto& allocator = kv_manager->allocator_mut(layer_idx);
    for (int32_t block_pos = 0; block_pos < static_cast<int32_t>(block_ids.size()); ++block_pos) {
      const auto ptrs = allocator.get_block_payload_ptrs(block_ids[block_pos]);
      std::vector<int8_t> key_value(ptrs.key_value_bytes);
      std::vector<float> scales(ptrs.scale_bytes / sizeof(float));
      for (size_t i = 0; i < key_value.size(); ++i) {
        key_value[i] = static_cast<int8_t>((base_value + layer_idx * 11 + block_pos * 3 + i) % 127);
      }
      for (size_t i = 0; i < scales.size(); ++i) {
        scales[i] = static_cast<float>(base_value + layer_idx * 100 + block_pos * 10 + i) / 100.0f;
      }
      cuda_allocator->memcpy(key_value.data(), ptrs.key, ptrs.key_value_bytes,
                             base::MemcpyKind::kMemcpyCPU2CUDA, nullptr, true);
      cuda_allocator->memcpy(key_value.data(), ptrs.value, ptrs.key_value_bytes,
                             base::MemcpyKind::kMemcpyCPU2CUDA, nullptr, true);
      cuda_allocator->memcpy(scales.data(), ptrs.key_scale, ptrs.scale_bytes,
                             base::MemcpyKind::kMemcpyCPU2CUDA, nullptr, true);
      cuda_allocator->memcpy(scales.data(), ptrs.value_scale, ptrs.scale_bytes,
                             base::MemcpyKind::kMemcpyCPU2CUDA, nullptr, true);
    }
  }
}

void expect_cuda_fp8_request_blocks_equal(base::KVCacheManager* src_kv_manager,
                                          base::RequestId src_request_id,
                                          base::KVCacheManager* dst_kv_manager,
                                          base::RequestId dst_request_id) {
  auto cuda_allocator = base::CUDADeviceAllocatorFactory::get_instance();
  for (int32_t layer_idx = 0; layer_idx < src_kv_manager->num_layers(); ++layer_idx) {
    const auto& src_block_ids = src_kv_manager->get_block_ids(src_request_id, layer_idx);
    const auto& dst_block_ids = dst_kv_manager->get_block_ids(dst_request_id, layer_idx);
    auto& src_allocator = src_kv_manager->allocator_mut(layer_idx);
    auto& dst_allocator = dst_kv_manager->allocator_mut(layer_idx);
    ASSERT_EQ(src_block_ids.size(), dst_block_ids.size());
    for (int32_t block_pos = 0; block_pos < static_cast<int32_t>(src_block_ids.size()); ++block_pos) {
      const auto src = src_allocator.get_block_payload_ptrs(src_block_ids[block_pos]);
      const auto dst = dst_allocator.get_block_payload_ptrs(dst_block_ids[block_pos]);
      std::vector<int8_t> src_key(src.key_value_bytes);
      std::vector<int8_t> dst_key(dst.key_value_bytes);
      std::vector<int8_t> src_value(src.key_value_bytes);
      std::vector<int8_t> dst_value(dst.key_value_bytes);
      std::vector<float> src_key_scale(src.scale_bytes / sizeof(float));
      std::vector<float> dst_key_scale(dst.scale_bytes / sizeof(float));
      std::vector<float> src_value_scale(src.scale_bytes / sizeof(float));
      std::vector<float> dst_value_scale(dst.scale_bytes / sizeof(float));
      cuda_allocator->memcpy(src.key, src_key.data(), src.key_value_bytes,
                             base::MemcpyKind::kMemcpyCUDA2CPU, nullptr, true);
      cuda_allocator->memcpy(dst.key, dst_key.data(), dst.key_value_bytes,
                             base::MemcpyKind::kMemcpyCUDA2CPU, nullptr, true);
      cuda_allocator->memcpy(src.value, src_value.data(), src.key_value_bytes,
                             base::MemcpyKind::kMemcpyCUDA2CPU, nullptr, true);
      cuda_allocator->memcpy(dst.value, dst_value.data(), dst.key_value_bytes,
                             base::MemcpyKind::kMemcpyCUDA2CPU, nullptr, true);
      cuda_allocator->memcpy(src.key_scale, src_key_scale.data(), src.scale_bytes,
                             base::MemcpyKind::kMemcpyCUDA2CPU, nullptr, true);
      cuda_allocator->memcpy(dst.key_scale, dst_key_scale.data(), dst.scale_bytes,
                             base::MemcpyKind::kMemcpyCUDA2CPU, nullptr, true);
      cuda_allocator->memcpy(src.value_scale, src_value_scale.data(), src.scale_bytes,
                             base::MemcpyKind::kMemcpyCUDA2CPU, nullptr, true);
      cuda_allocator->memcpy(dst.value_scale, dst_value_scale.data(), dst.scale_bytes,
                             base::MemcpyKind::kMemcpyCUDA2CPU, nullptr, true);

      EXPECT_EQ(dst_key, src_key);
      EXPECT_EQ(dst_value, src_value);
      EXPECT_EQ(dst_key_scale, src_key_scale);
      EXPECT_EQ(dst_value_scale, src_value_scale);
    }
  }
}

}  // namespace

TEST(PDHandoffTest, ValidManifestPassesValidation) {
  KVBlockManifest manifest = make_manifest();
  EXPECT_TRUE(manifest.validate());
  EXPECT_EQ(manifest.src_pool.bytes_per_block_per_layer(),
            static_cast<int64_t>(2) * 16 * 2 * 64 * 2);
}

TEST(PDHandoffTest, IncompatiblePoolsFailValidation) {
  KVBlockManifest manifest = make_manifest();
  manifest.dst_pool.head_size = 128;
  EXPECT_FALSE(manifest.validate());
}

TEST(PDHandoffTest, NoCopyConnectorCompletesImmediately) {
  KVBlockManifest manifest = make_manifest();
  InProcNoCopyKVTransferConnector connector;
  HandoffId handle;
  ASSERT_TRUE(connector.submit(manifest, &handle));
  EXPECT_TRUE(handle.valid());
  KVTransferStatus status = connector.poll(handle);
  EXPECT_TRUE(status.done());
  EXPECT_TRUE(status.ok());
}

TEST(PDHandoffTest, CoordinatorMarksNoCopyTransferDecodeReady) {
  KVBlockManifest manifest = make_manifest();
  InProcNoCopyKVTransferConnector connector;
  PDCoordinator coordinator(&connector);

  PDHandoffState state;
  ASSERT_TRUE(coordinator.start_prefill_handoff(manifest, &state));
  EXPECT_TRUE(state.handoff_id.valid());
  EXPECT_EQ(state.phase, PDHandoffPhase::kDecodeReady);
  EXPECT_TRUE(state.ready_for_decode());
  EXPECT_TRUE(state.terminal());
}

TEST(PDHandoffTest, CoordinatorReportsInvalidManifestFailure) {
  KVBlockManifest manifest = make_manifest();
  manifest.client_request_id.value.clear();
  InProcNoCopyKVTransferConnector connector;
  PDCoordinator coordinator(&connector);

  PDHandoffState state;
  EXPECT_FALSE(coordinator.start_prefill_handoff(manifest, &state));
  EXPECT_EQ(state.phase, PDHandoffPhase::kFailed);
  EXPECT_FALSE(state.error.empty());
  EXPECT_TRUE(state.terminal());
}

TEST(PDHandoffTest, CoordinatorCancelBeforeCompletion) {
  class PendingConnector final : public KVTransferConnector {
   public:
    base::Status submit(const KVBlockManifest& manifest, HandoffId* handle) override {
      base::Status status = manifest.validate();
      if (!status) {
        return status;
      }
      *handle = {11};
      cancelled = false;
      return base::error::Success();
    }

    KVTransferStatus poll(HandoffId) override {
      return cancelled ? KVTransferStatus::Cancelled(cancel_reason)
                       : KVTransferStatus::Pending();
    }

    void cancel(HandoffId, const std::string& reason) override {
      cancelled = true;
      cancel_reason = reason;
    }

    bool cancelled = false;
    std::string cancel_reason;
  } connector;

  PDCoordinator coordinator(&connector);
  PDHandoffState state;
  ASSERT_TRUE(coordinator.start_prefill_handoff(make_manifest(), &state));
  EXPECT_EQ(state.phase, PDHandoffPhase::kTransferSubmitted);
  EXPECT_FALSE(state.ready_for_decode());

  coordinator.cancel(&state, "client cancelled");
  EXPECT_TRUE(connector.cancelled);
  EXPECT_EQ(state.phase, PDHandoffPhase::kCancelled);
  EXPECT_TRUE(state.terminal());
}

TEST(PDHandoffTest, DecodeReservationBuildsDstBlockManifest) {
  auto kv_manager = make_kv_manager();
  DecodeKVReservationManager reservation_manager(kv_manager.get(), make_pool(1));

  DecodeKVReservationRequest request;
  request.client_request_id.value = "req-reserve";
  request.handoff_id.value = 19;
  request.prompt_tokens = 40;
  request.computed_tokens = 32;
  request.first_token = 77;
  request.src_pool = make_pool(0);
  request.src_block_ids_per_layer = {{10, 11}, {20, 21}};

  DecodeKVReservation reservation;
  KVBlockManifest manifest;
  ASSERT_TRUE(reservation_manager.reserve(request, &reservation, &manifest));

  EXPECT_TRUE(reservation.valid());
  EXPECT_EQ(reservation.reserved_tokens, 32);
  EXPECT_EQ(kv_manager->get_context_len(reservation.decode_request_id), 32);
  ASSERT_EQ(reservation.dst_block_ids_per_layer.size(), 2);
  EXPECT_EQ(reservation.dst_block_ids_per_layer[0], std::vector<int32_t>({0, 1}));
  EXPECT_EQ(reservation.dst_block_ids_per_layer[1], std::vector<int32_t>({0, 1}));

  ASSERT_TRUE(manifest.validate());
  ASSERT_EQ(manifest.layer_mappings.size(), 2);
  EXPECT_EQ(manifest.layer_mappings[0].src_block_ids, std::vector<int32_t>({10, 11}));
  EXPECT_EQ(manifest.layer_mappings[0].dst_block_ids, std::vector<int32_t>({0, 1}));
  EXPECT_EQ(manifest.layer_mappings[1].src_block_ids, std::vector<int32_t>({20, 21}));
  EXPECT_EQ(manifest.layer_mappings[1].dst_block_ids, std::vector<int32_t>({0, 1}));

  reservation_manager.release(&reservation);
  EXPECT_FALSE(reservation.valid());
  EXPECT_EQ(kv_manager->num_active_requests(), 0);
}

TEST(PDHandoffTest, DecodeReservationFailsWhenDstBlocksInsufficient) {
  auto kv_manager = make_kv_manager(1);
  DecodeKVReservationManager reservation_manager(kv_manager.get(), make_pool(1));

  DecodeKVReservationRequest request;
  request.client_request_id.value = "req-reserve";
  request.handoff_id.value = 20;
  request.prompt_tokens = 40;
  request.computed_tokens = 32;
  request.first_token = 77;
  request.src_pool = make_pool(0);
  request.src_block_ids_per_layer = {{10, 11}, {20, 21}};

  DecodeKVReservation reservation;
  KVBlockManifest manifest;
  EXPECT_FALSE(reservation_manager.reserve(request, &reservation, &manifest));
  EXPECT_FALSE(reservation.valid());
  EXPECT_EQ(kv_manager->num_active_requests(), 0);
}

TEST(PDHandoffTest, InProcKVBlockCopyConnectorCopiesPlainBlocks) {
  auto src_kv_manager = make_kv_manager();
  auto dst_kv_manager = make_kv_manager();

  const base::RequestId src_request_id = src_kv_manager->register_request();
  ASSERT_TRUE(src_kv_manager->append_slots(src_request_id, 32));
  fill_request_blocks(src_kv_manager.get(), src_request_id, 100);

  DecodeKVReservationManager reservation_manager(dst_kv_manager.get(), make_pool(1));
  DecodeKVReservationRequest request;
  request.client_request_id.value = "req-copy";
  request.handoff_id.value = 21;
  request.prompt_tokens = 32;
  request.computed_tokens = 32;
  request.first_token = 88;
  request.src_pool = make_pool(0);
  request.src_block_ids_per_layer = {
      src_kv_manager->get_block_ids(src_request_id, 0),
      src_kv_manager->get_block_ids(src_request_id, 1),
  };

  DecodeKVReservation reservation;
  KVBlockManifest manifest;
  ASSERT_TRUE(reservation_manager.reserve(request, &reservation, &manifest));

  InProcKVBlockCopyConnector connector(src_kv_manager.get(), dst_kv_manager.get());
  HandoffId handle;
  ASSERT_TRUE(connector.submit(manifest, &handle));
  EXPECT_TRUE(connector.poll(handle).ok());
  expect_request_blocks_equal(src_kv_manager.get(), src_request_id,
                              dst_kv_manager.get(), reservation.decode_request_id);

  reservation_manager.release(&reservation);
  src_kv_manager->free_request(src_request_id);
}

TEST(PDHandoffTest, CoordinatorRunsReservationCopyDecodeReadyPipeline) {
  auto src_kv_manager = make_kv_manager();
  auto dst_kv_manager = make_kv_manager();

  const base::RequestId src_request_id = src_kv_manager->register_request();
  ASSERT_TRUE(src_kv_manager->append_slots(src_request_id, 32));
  fill_request_blocks(src_kv_manager.get(), src_request_id, 700);

  DecodeKVReservationManager reservation_manager(dst_kv_manager.get(), make_pool(1));
  DecodeKVReservationRequest request;
  request.client_request_id.value = "req-pipeline";
  request.handoff_id.value = 22;
  request.prompt_tokens = 32;
  request.computed_tokens = 32;
  request.first_token = 99;
  request.src_pool = make_pool(0);
  request.src_block_ids_per_layer = {
      src_kv_manager->get_block_ids(src_request_id, 0),
      src_kv_manager->get_block_ids(src_request_id, 1),
  };

  DecodeKVReservation reservation;
  KVBlockManifest manifest;
  ASSERT_TRUE(reservation_manager.reserve(request, &reservation, &manifest));
  ASSERT_TRUE(manifest.validate());

  InProcKVBlockCopyConnector connector(src_kv_manager.get(), dst_kv_manager.get());
  PDCoordinator coordinator(&connector);
  PDHandoffState state;
  ASSERT_TRUE(coordinator.start_prefill_handoff(manifest, &state));

  EXPECT_TRUE(state.handoff_id.valid());
  EXPECT_EQ(state.phase, PDHandoffPhase::kDecodeReady);
  EXPECT_TRUE(state.ready_for_decode());
  EXPECT_TRUE(state.terminal());
  EXPECT_TRUE(state.transfer_status.ok());
  expect_request_blocks_equal(src_kv_manager.get(), src_request_id,
                              dst_kv_manager.get(), reservation.decode_request_id);

  reservation_manager.release(&reservation);
  src_kv_manager->free_request(src_request_id);
}

TEST(PDHandoffTest, CudaFp8ConnectorCopiesKeyValueAndScaleBlocks) {
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA device is not available";
  }

  auto src_kv_manager = make_cuda_fp8_kv_manager();
  auto dst_kv_manager = make_cuda_fp8_kv_manager();

  const base::RequestId src_request_id = src_kv_manager->register_request();
  ASSERT_TRUE(src_kv_manager->append_slots(src_request_id, 32));
  fill_cuda_fp8_request_blocks(src_kv_manager.get(), src_request_id, 17);

  KVPoolDescriptor src_pool = make_pool(0);
  src_pool.storage_mode = base::BlockStorageMode::kFp8E4M3PerTokenHead;
  KVPoolDescriptor dst_pool = make_pool(0);
  dst_pool.storage_mode = base::BlockStorageMode::kFp8E4M3PerTokenHead;
  DecodeKVReservationManager reservation_manager(dst_kv_manager.get(), dst_pool);

  DecodeKVReservationRequest request;
  request.client_request_id.value = "req-cuda-fp8-copy";
  request.handoff_id.value = 23;
  request.prompt_tokens = 32;
  request.computed_tokens = 32;
  request.first_token = 101;
  request.src_pool = src_pool;
  request.src_block_ids_per_layer = {
      src_kv_manager->get_block_ids(src_request_id, 0),
      src_kv_manager->get_block_ids(src_request_id, 1),
  };

  DecodeKVReservation reservation;
  KVBlockManifest manifest;
  ASSERT_TRUE(reservation_manager.reserve(request, &reservation, &manifest));

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  InProcKVBlockCopyConnector connector(src_kv_manager.get(), dst_kv_manager.get(), stream, false);
  PDCoordinator coordinator(&connector);
  PDHandoffState state;
  ASSERT_TRUE(coordinator.start_prefill_handoff(manifest, &state));
  EXPECT_TRUE(state.handoff_id.valid());
  EXPECT_EQ(state.phase, PDHandoffPhase::kTransferSubmitted);
  EXPECT_FALSE(state.ready_for_decode());

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  ASSERT_TRUE(coordinator.advance(&state));
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

  EXPECT_EQ(state.phase, PDHandoffPhase::kDecodeReady);
  EXPECT_TRUE(state.ready_for_decode());
  EXPECT_TRUE(state.transfer_status.ok());
  expect_cuda_fp8_request_blocks_equal(src_kv_manager.get(), src_request_id,
                                       dst_kv_manager.get(), reservation.decode_request_id);

  reservation_manager.release(&reservation);
  src_kv_manager->free_request(src_request_id);
}

}  // namespace serving
