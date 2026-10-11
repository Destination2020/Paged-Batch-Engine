#include <gtest/gtest.h>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <vector>
#include "base/global_chunk_allocator.h"

TEST(GlobalChunkAllocator, SharesReturnedBytesAcrossDifferentLayouts) {
  base::GlobalChunkAllocator pool(12 * 1024, 7);
  pool.register_model("A", 8 * 1024);
  pool.register_model("B", 8 * 1024);
  const auto a = pool.acquire("A", "24x64", 6 * 1024, 512);
  const auto b = pool.acquire("B", "28x128", 4 * 1024, 1024);
  EXPECT_EQ(pool.free_bytes(), 2 * 1024);
  EXPECT_THROW(pool.acquire("B", "28x128", 4 * 1024, 1024), std::runtime_error);
  pool.release(a);
  const auto grown = pool.acquire("B", "28x128", 4 * 1024, 1024);
  EXPECT_EQ(grown.offset, a.offset);
  EXPECT_NE(grown.generation, a.generation);
  EXPECT_EQ(pool.leased_bytes("A"), 0);
  EXPECT_EQ(pool.leased_bytes("B"), 8 * 1024);
  EXPECT_THROW(pool.release(a), std::invalid_argument);
  pool.release(b); pool.release(grown);
  EXPECT_EQ(pool.free_bytes(), pool.bytes());
  EXPECT_EQ(pool.lease_count(), 0);
}

TEST(GlobalChunkAllocator, RejectsWrongEpochIdentityAndDuplicateRelease) {
  base::GlobalChunkAllocator pool(16384, 42);
  pool.register_model("A", 8192);
  auto live = pool.acquire("A", "layout", 4096, 256);
  for (int field = 0; field < 7; ++field) {
    auto stale = live;
    if (field == 0) ++stale.epoch;
    if (field == 1) ++stale.generation;
    if (field == 2) ++stale.offset;
    if (field == 3) ++stale.bytes;
    if (field == 4) stale.model = "B";
    if (field == 5) stale.layout = "wrong";
    if (field == 6) ++stale.id;
    EXPECT_THROW(pool.release(stale), std::invalid_argument);
    EXPECT_EQ(pool.leased_bytes("A"), 4096);
  }
  pool.release(live);
  EXPECT_THROW(pool.release(live), std::invalid_argument);
  EXPECT_EQ(pool.free_bytes(), 16384);
}

TEST(GlobalChunkAllocator, AlignmentFragmentationAndQuotaAreAtomic) {
  base::GlobalChunkAllocator pool(8192, 1);
  pool.register_model("A", 8192);
  pool.register_model("B", 2048);
  auto a = pool.acquire("A", "page3", 3072, 1536);
  auto b = pool.acquire("B", "page2", 2048, 2048);
  EXPECT_EQ(a.offset % 1536, 0);
  EXPECT_EQ(b.offset % 2048, 0);
  auto free = pool.free_bytes();
  EXPECT_THROW(pool.acquire("B", "page2", 2048, 2048), std::runtime_error);
  EXPECT_EQ(pool.free_bytes(), free);
  EXPECT_THROW(pool.acquire("A", "page4", 4096, 4096), std::runtime_error);
  EXPECT_EQ(pool.free_bytes(), free);
  EXPECT_THROW(pool.acquire("A", "bad", 100, 256), std::invalid_argument);
  pool.release(a); pool.release(b);
  auto full = pool.acquire("A", "full", 8192, 256);
  EXPECT_EQ(full.offset, 0);
}

TEST(GlobalChunkAllocator, RandomizedAccountingAndDisjointIntervals) {
  base::GlobalChunkAllocator pool(12 * 1024 * 1024, 77);
  pool.register_model("A", pool.bytes()); pool.register_model("B", pool.bytes());
  std::mt19937 rng(42);
  std::vector<base::ChunkLease> leases;
  for (int step = 0; step < 4000; ++step) {
    if (!leases.empty() && rng() % 3 == 0) {
      auto idx = rng() % leases.size();
      pool.release(leases[idx]); leases.erase(leases.begin() + idx);
    } else {
      const uint64_t page = (rng() % 2 ? 24 : 28) * 8192;
      try { leases.push_back(pool.acquire(rng() % 2 ? "A" : "B", "gqa", page * (1 + rng() % 4), page)); }
      catch (const std::runtime_error&) {}
    }
    uint64_t bytes = 0, a_bytes = 0, b_bytes = 0;
    std::sort(leases.begin(), leases.end(), [](const auto& a, const auto& b) { return a.offset < b.offset; });
    for (size_t i = 0; i < leases.size(); ++i) {
      const auto& lease = leases[i];
      EXPECT_LE(lease.offset + lease.bytes, pool.bytes());
      if (i) EXPECT_LE(leases[i - 1].offset + leases[i - 1].bytes, lease.offset);
      bytes += lease.bytes;
      (lease.model == "A" ? a_bytes : b_bytes) += lease.bytes;
    }
    EXPECT_EQ(bytes + pool.free_bytes(), pool.bytes());
    EXPECT_EQ(a_bytes, pool.leased_bytes("A"));
    EXPECT_EQ(b_bytes, pool.leased_bytes("B"));
  }
  for (const auto& lease : leases) pool.release(lease);
  EXPECT_EQ(pool.free_bytes(), pool.bytes());
}
