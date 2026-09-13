#include <gtest/gtest.h>
#include <algorithm>
#include "data/data_runtime.h"

namespace {
data::DataReservation Reservation(uint64_t operation, uint8_t content, uint64_t bytes = 16) {
  data::DataReservation value;
  value.operation = {77, operation};
  value.kind = data::DataKind::kTensorBundle;
  value.content.digest[0] = content;
  value.representation.digest[0] = 9;
  value.logical_bytes = bytes;
  return value;
}

TEST(LocalDataRuntimeTest, ReserveIsIdempotentAndPreflightsCapacity) {
  data::LocalDataRuntime runtime(77, 16, 1);
  data::AllocationHandle first, repeated;
  ASSERT_EQ(runtime.reserve(Reservation(1, 1), &first), data::DataError::kOk);
  ASSERT_EQ(runtime.reserve(Reservation(1, 1), &repeated), data::DataError::kOk);
  EXPECT_EQ(first, repeated);
  data::AllocationHandle rejected;
  EXPECT_EQ(runtime.reserve(Reservation(2, 2), &rejected), data::DataError::kCapacityExhausted);
}

TEST(LocalDataRuntimeTest, SealPublishesOnlyCompleteDataAndLeaseDefersWithdraw) {
  data::LocalDataRuntime runtime(77, 64, 4);
  auto reservation = Reservation(1, 1);
  data::AllocationHandle handle;
  ASSERT_EQ(runtime.reserve(reservation, &handle), data::DataError::kOk);
  uint8_t* bytes = nullptr;
  size_t size = 0;
  ASSERT_EQ(runtime.begin_write(handle, &bytes, &size), data::DataError::kOk);
  std::fill(bytes, bytes + size, 0x5a);
  data::DataRef ref;
  EXPECT_EQ(runtime.seal(handle, size - 1, "sum", &ref), data::DataError::kCoverageMismatch);
  data::DataLease missing;
  EXPECT_EQ(runtime.acquire(reservation.kind, reservation.content, reservation.representation, &missing),
            data::DataError::kNotReady);
  ASSERT_EQ(runtime.seal(handle, size, "sum", &ref), data::DataError::kOk);
  data::DataLease lease;
  ASSERT_EQ(runtime.acquire(reservation.kind, reservation.content, reservation.representation, &lease),
            data::DataError::kOk);
  EXPECT_EQ(lease.data()[0], 0x5a);
  ASSERT_EQ(runtime.release_producer(handle), data::DataError::kOk);
  ASSERT_EQ(runtime.withdraw(ref), data::DataError::kOk);
  EXPECT_EQ(runtime.bytes_used(), size);
  lease.reset();
  EXPECT_EQ(runtime.bytes_used(), 0u);
}

TEST(LocalDataRuntimeTest, ConcurrentProducersConvergeOnOneCanonicalObject) {
  data::LocalDataRuntime runtime(77, 64, 4);
  data::AllocationHandle a, b;
  ASSERT_EQ(runtime.reserve(Reservation(1, 3), &a), data::DataError::kOk);
  ASSERT_EQ(runtime.reserve(Reservation(2, 3), &b), data::DataError::kOk);
  data::DataRef winner, duplicate;
  ASSERT_EQ(runtime.seal(a, 16, "same", &winner), data::DataError::kOk);
  ASSERT_EQ(runtime.seal(b, 16, "same", &duplicate), data::DataError::kOk);
  EXPECT_EQ(winner, duplicate);
  EXPECT_EQ(runtime.bytes_used(), 16u);
  EXPECT_EQ(runtime.release_producer(b), data::DataError::kOk);
  EXPECT_EQ(runtime.object_count(), 1u);
}

TEST(LocalDataRuntimeTest, ConflictingCanonicalChecksumIsQuarantined) {
  data::LocalDataRuntime runtime(77, 64, 4);
  data::AllocationHandle a, b;
  ASSERT_EQ(runtime.reserve(Reservation(1, 4), &a), data::DataError::kOk);
  ASSERT_EQ(runtime.reserve(Reservation(2, 4), &b), data::DataError::kOk);
  data::DataRef ref;
  ASSERT_EQ(runtime.seal(a, 16, "one", &ref), data::DataError::kOk);
  EXPECT_EQ(runtime.seal(b, 16, "two", &ref), data::DataError::kChecksumMismatch);
  EXPECT_EQ(runtime.quarantined_bytes(), 16u);
  EXPECT_EQ(runtime.release_producer(b), data::DataError::kOk);
  EXPECT_EQ(runtime.quarantined_bytes(), 16u);
  EXPECT_EQ(runtime.reclaim_quarantine(b), data::DataError::kOk);
  EXPECT_EQ(runtime.quarantined_bytes(), 0u);
}

TEST(LocalDataRuntimeTest, TypedComputeAndIOLeasesDeferPhysicalReclaim) {
  data::LocalDataRuntime runtime(77, 64, 4);
  auto reservation = Reservation(1, 8);
  data::AllocationHandle handle;
  ASSERT_EQ(runtime.reserve(reservation, &handle), data::DataError::kOk);
  data::DataRef ref;
  ASSERT_EQ(runtime.seal(handle, 16, "typed", &ref), data::DataError::kOk);
  data::DataLease compute, io;
  ASSERT_EQ(runtime.ensure_local(reservation.kind, reservation.content,
                                 reservation.representation,
                                 data::DataLeaseKind::kCompute, &compute),
            data::DataError::kOk);
  ASSERT_EQ(runtime.acquire(reservation.kind, reservation.content,
                            reservation.representation,
                            data::DataLeaseKind::kIO, &io), data::DataError::kOk);
  ASSERT_EQ(runtime.withdraw(ref), data::DataError::kOk);
  ASSERT_EQ(runtime.release_producer(handle), data::DataError::kOk);
  compute.reset();
  EXPECT_EQ(runtime.bytes_used(), 16u);
  io.reset();
  EXPECT_EQ(runtime.bytes_used(), 0u);
}

TEST(LocalDataRuntimeTest, RejectsOldOwnerAndStaleGeneration) {
  data::LocalDataRuntime runtime(77, 64, 4);
  data::AllocationHandle handle;
  ASSERT_EQ(runtime.reserve(Reservation(1, 5), &handle), data::DataError::kOk);
  uint8_t* bytes = nullptr;
  size_t size = 0;
  auto wrong_owner = handle;
  wrong_owner.owner_incarnation = 76;
  EXPECT_EQ(runtime.begin_write(wrong_owner, &bytes, &size), data::DataError::kOwnerRestarted);
  auto stale = handle;
  ++stale.generation;
  EXPECT_EQ(runtime.begin_write(stale, &bytes, &size), data::DataError::kStaleGeneration);
}

TEST(LocalDataRuntimeTest, ProducerExitBeforeSealLeavesNoVisibleObject) {
  data::LocalDataRuntime runtime(77, 64, 4);
  const auto reservation = Reservation(41, 6);
  data::AllocationHandle handle;
  ASSERT_EQ(runtime.reserve(reservation, &handle), data::DataError::kOk);
  ASSERT_EQ(runtime.release_producer(handle), data::DataError::kOk);
  data::DataLease lease;
  EXPECT_EQ(runtime.acquire(reservation.kind, reservation.content,
                            reservation.representation, &lease), data::DataError::kNotReady);
  EXPECT_EQ(runtime.bytes_used(), 0u);
  EXPECT_EQ(runtime.object_count(), 0u);
}

TEST(LocalDataRuntimeTest, ServiceRestartRejectsEveryOldAllocationHandle) {
  data::AllocationHandle old;
  {
    data::LocalDataRuntime first(1001, 64, 4);
    auto own = Reservation(1, 7); own.operation.owner_incarnation = 1001;
    ASSERT_EQ(first.reserve(own, &old), data::DataError::kOk);
  }
  data::LocalDataRuntime restarted(1002, 64, 4);
  uint8_t* bytes=nullptr; size_t size=0;
  EXPECT_EQ(restarted.begin_write(old,&bytes,&size),data::DataError::kOwnerRestarted);
  EXPECT_EQ(restarted.bytes_used(),0u);
}
}  // namespace
