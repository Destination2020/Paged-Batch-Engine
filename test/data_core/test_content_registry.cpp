#include <gtest/gtest.h>
#include "data/content_registry.h"

TEST(ContentRegistryTest, RetentionSurvivesProducerAndTwoSequentialConsumers) {
  data::ContentRegistry registry(91, 1024, 8, 8);
  const std::vector<uint8_t> bytes{1,2,3,4,5};
  data::DataReservation r;r.operation={91,1};r.kind=data::DataKind::kKVPage;
  r.content=data::ContentIdFromBytes(bytes.data(),bytes.size());
  r.representation=data::RepresentationIdFromString("kv-test-v1");r.logical_bytes=bytes.size();
  data::AllocationHandle h;ASSERT_EQ(registry.reserve(r,&h),data::DataError::kOk);
  data::DataRef ref;ASSERT_EQ(registry.seal(h,bytes,data::DataChecksum(bytes.data(),bytes.size()),&ref),data::DataError::kOk);
  ASSERT_EQ(registry.release_producer(h),data::DataError::kOk);
  for(int consumer=0;consumer<2;++consumer){data::RemoteDataLease lease;ASSERT_EQ(registry.acquire(r.kind,r.content,r.representation,data::DataLeaseKind::kCompute,{700,static_cast<uint64_t>(consumer+1)},&lease),data::DataError::kOk);EXPECT_EQ(lease.bytes,bytes);ASSERT_EQ(registry.release(lease.token),data::DataError::kOk);}
  EXPECT_EQ(registry.stats().bytes_used,bytes.size());
  ASSERT_EQ(registry.withdraw(ref),data::DataError::kOk);
  EXPECT_EQ(registry.stats().bytes_used,0u);
}

TEST(ContentRegistryTest, LeaseCapacityIsBoundedAndReleaseIsExplicit) {
  data::ContentRegistry registry(92,1024,8,1);const std::vector<uint8_t>b{9};
  data::DataReservation r;r.operation={92,1};r.kind=data::DataKind::kKVPage;r.content=data::ContentIdFromBytes(b.data(),b.size());r.representation=data::RepresentationIdFromString("x");r.logical_bytes=1;
  data::AllocationHandle h;ASSERT_EQ(registry.reserve(r,&h),data::DataError::kOk);data::DataRef ref;ASSERT_EQ(registry.seal(h,b,data::DataChecksum(b.data(),b.size()),&ref),data::DataError::kOk);
  data::RemoteDataLease a,c;ASSERT_EQ(registry.acquire(r.kind,r.content,r.representation,data::DataLeaseKind::kRead,{701,1},&a),data::DataError::kOk);EXPECT_EQ(registry.acquire(r.kind,r.content,r.representation,data::DataLeaseKind::kRead,{702,1},&c),data::DataError::kCapacityExhausted);ASSERT_EQ(registry.release(a.token),data::DataError::kOk);EXPECT_EQ(registry.release(a.token),data::DataError::kOk);
}

TEST(ContentRegistryTest, TenThousandAcquireCancelCyclesReturnToSteadyState) {
  data::ContentRegistry registry(93, 1024, 8, 4);
  const std::vector<uint8_t> bytes{1,3,3,7};
  data::DataReservation r; r.operation={93,1}; r.kind=data::DataKind::kTensorBundle;
  r.content=data::ContentIdFromBytes(bytes.data(),bytes.size());
  r.representation=data::RepresentationIdFromString("churn-v1");
  r.logical_bytes=bytes.size();
  data::AllocationHandle h; ASSERT_EQ(registry.reserve(r,&h),data::DataError::kOk);
  data::DataRef ref; ASSERT_EQ(registry.seal(h,bytes,data::DataChecksum(bytes.data(),bytes.size()),&ref),data::DataError::kOk);
  ASSERT_EQ(registry.release_producer(h),data::DataError::kOk);
  for (int cycle=0; cycle<10000; ++cycle) {
    data::RemoteDataLease lease;
    ASSERT_EQ(registry.acquire(r.kind,r.content,r.representation,
                               data::DataLeaseKind::kRead,{703,static_cast<uint64_t>(cycle+1)},&lease),data::DataError::kOk);
    ASSERT_EQ(registry.release(lease.token),data::DataError::kOk);
  }
  EXPECT_EQ(registry.stats().active_leases,0u);
  EXPECT_EQ(registry.stats().bytes_used,bytes.size());
  ASSERT_EQ(registry.withdraw(ref),data::DataError::kOk);
  EXPECT_EQ(registry.stats().bytes_used,0u);
}

TEST(ContentRegistryTest, AcquireReplayIsIdempotent) {
  data::ContentRegistry registry(94,1024,8,8);const std::vector<uint8_t>b{4,2};
  data::DataReservation r;r.operation={94,1};r.kind=data::DataKind::kTensorBundle;r.content=data::ContentIdFromBytes(b.data(),b.size());r.representation=data::RepresentationIdFromString("replay");r.logical_bytes=b.size();
  data::AllocationHandle h;ASSERT_EQ(registry.reserve(r,&h),data::DataError::kOk);data::DataRef ref;ASSERT_EQ(registry.seal(h,b,data::DataChecksum(b.data(),b.size()),&ref),data::DataError::kOk);
  data::RemoteDataLease first,replayed;
  ASSERT_EQ(registry.acquire(r.kind,r.content,r.representation,data::DataLeaseKind::kRead,{800,9},&first),data::DataError::kOk);
  ASSERT_EQ(registry.acquire(r.kind,r.content,r.representation,data::DataLeaseKind::kRead,{800,9},&replayed),data::DataError::kOk);
  EXPECT_EQ(first.token,replayed.token);EXPECT_EQ(registry.stats().active_leases,1u);
  EXPECT_EQ(registry.release(first.token),data::DataError::kOk);
}

TEST(ContentRegistryTest, StaleServiceTokenCannotReleaseReusedLeaseNumber) {
  const std::vector<uint8_t>b{7};
  auto publish=[&](data::ContentRegistry* registry,uint64_t incarnation){data::DataReservation r;r.operation={incarnation,1};r.kind=data::DataKind::kKVPage;r.content=data::ContentIdFromBytes(b.data(),b.size());r.representation=data::RepresentationIdFromString("restart");r.logical_bytes=1;data::AllocationHandle h;EXPECT_EQ(registry->reserve(r,&h),data::DataError::kOk);data::DataRef ref;EXPECT_EQ(registry->seal(h,b,data::DataChecksum(b.data(),b.size()),&ref),data::DataError::kOk);return r;};
  data::ContentRegistry old_service(101,1024,8,8);auto old_r=publish(&old_service,101);data::RemoteDataLease stale;ASSERT_EQ(old_service.acquire(old_r.kind,old_r.content,old_r.representation,data::DataLeaseKind::kRead,{900,1},&stale),data::DataError::kOk);
  data::ContentRegistry new_service(102,1024,8,8);auto new_r=publish(&new_service,102);data::RemoteDataLease current;ASSERT_EQ(new_service.acquire(new_r.kind,new_r.content,new_r.representation,data::DataLeaseKind::kRead,{901,1},&current),data::DataError::kOk);
  ASSERT_EQ(stale.token.lease_id,current.token.lease_id);
  EXPECT_EQ(new_service.release(stale.token),data::DataError::kOwnerRestarted);
  EXPECT_EQ(new_service.stats().active_leases,1u);
  EXPECT_EQ(new_service.release(current.token),data::DataError::kOk);
}
