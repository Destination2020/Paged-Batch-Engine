#include <gtest/gtest.h>
#include <stdexcept>
#include "model/serving_memory_planner.h"
#if defined(QWEN2_SUPPORT)
TEST(ServingMemoryBudget, AbsoluteBudgetDoesNotScaleWithDeviceFreeMemory) {
  model::DynamicKVCacheSizingConfig config;
  config.layer_num = 24; config.block_size = 16;
  config.num_kv_heads = 2; config.head_size = 64;
  config.absolute_kv_budget_bytes = 64ull << 20;
  const auto storage = base::MakeKVCacheStorageSpec(base::DataType::kDataTypeBf16,
                                                   base::BlockStorageMode::kPlain);
  base::DeviceMemoryInfo memory{};
  memory.free_bytes = 2ull << 30; memory.total_bytes = 8ull << 30;
  auto count = model::estimate_dynamic_kv_blocks_per_layer(storage, config, memory);
  EXPECT_EQ(count, config.absolute_kv_budget_bytes / (24 * 16 * 2 * 64 * 2 * 2));
  memory.free_bytes = 5ull << 30;
  EXPECT_EQ(model::estimate_dynamic_kv_blocks_per_layer(storage, config, memory), count);
  memory.free_bytes = 32ull << 20;
  EXPECT_THROW(model::estimate_dynamic_kv_blocks_per_layer(storage, config, memory), std::invalid_argument);
  config.absolute_kv_budget_bytes = 1;
  EXPECT_THROW(model::estimate_dynamic_kv_blocks_per_layer(storage, config, memory), std::invalid_argument);
}
#endif

#include <cuda_runtime_api.h>
#include "base/alloc.h"
TEST(ServingMemoryBudget, CUDAAllocatorChargesCachedPhysicalBytesAndRejectsGrowth) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || !count) GTEST_SKIP() << "CUDA device required";
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  base::CUDADeviceAllocator allocator;
  allocator.set_byte_limit(0, 8192);
  void* first = allocator.allocate(4096);
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(allocator.reserved_bytes(0), 4096);
  EXPECT_EQ(allocator.allocate(6144), nullptr);
  allocator.release(first);
  EXPECT_EQ(allocator.reserved_bytes(0), 4096); // cached allocation is still physical
  void* reused = allocator.allocate(2048);
  EXPECT_EQ(reused, first);
  EXPECT_EQ(allocator.reserved_bytes(0), 4096);
  allocator.release(reused);
  allocator.release_all_cached();
  EXPECT_EQ(allocator.reserved_bytes(0), 0);
  // V4 explicit teardown returns physical bytes, but keeps the configured cap.
  EXPECT_EQ(allocator.allocate(8193), nullptr);
  void* after_flush = allocator.allocate(8192);
  ASSERT_NE(after_flush, nullptr);
  allocator.release(after_flush);
}

TEST(ServingMemoryBudget, CUDAAllocatorReusesLargeBufferBeforeAnySmallAllocation) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || !count) GTEST_SKIP() << "CUDA device required";
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  base::CUDADeviceAllocator allocator;
  allocator.set_byte_limit(0, 2ull << 20);
  void* first = allocator.allocate(2ull << 20);
  ASSERT_NE(first, nullptr);
  allocator.release(first);
  void* reused = allocator.allocate((2ull << 20) - 256);
  EXPECT_EQ(reused, first);
  EXPECT_EQ(allocator.reserved_bytes(0), 2ull << 20);
  allocator.release(reused);
}
