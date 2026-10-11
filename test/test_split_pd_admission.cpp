#include <gtest/gtest.h>
#include "serving/split_pd_admission.h"

TEST(SplitPDAdmissionTest, LongOutputsCannotConsumeShortSlotsOrTokens) {
  serving::SplitPDAdmission admission({2, 1}, {512, 4096});
  admission.acquire(1, 4096);
  EXPECT_FALSE(admission.can_acquire(1, 16));
  EXPECT_TRUE(admission.can_acquire(0, 256));
  admission.acquire(0, 256);
  admission.acquire(0, 256);
  EXPECT_FALSE(admission.can_acquire(0, 16));
  admission.release(0, 256);
  EXPECT_TRUE(admission.can_acquire(0, 256));
  EXPECT_EQ(admission.used(1), 4096);
}

TEST(SplitPDAdmissionTest, RoundFutureOutputToWholePages) {
  EXPECT_EQ(serving::SplitPDAdmission::charge(17, 16, 16), 48);
  serving::SplitPDAdmission admission({4, 2}, {32, 64});
  EXPECT_FALSE(admission.fits_request(0, 48));
  EXPECT_TRUE(admission.fits_request(1, 48));
}

TEST(SplitPDAdmissionTest, ReclaimAndRejectDoubleRelease) {
  serving::SplitPDAdmission admission({1, 1}, {128, 128});
  for (int round = 0; round < 10; ++round) {
    admission.acquire(0, 128);
    EXPECT_THROW(admission.acquire(0, 1), std::logic_error);
    admission.release(0, 128);
  }
  EXPECT_EQ(admission.active(0), 0);
  EXPECT_EQ(admission.used(0), 0);
  EXPECT_THROW(admission.release(0, 128), std::logic_error);
}
