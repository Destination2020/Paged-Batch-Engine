#ifndef PBE_CACHE_CORE_TEST_STUB_GLOG_LOGGING_H_
#define PBE_CACHE_CORE_TEST_STUB_GLOG_LOGGING_H_

// The cache core only uses base data types and Status. It never expands the
// logging-dependent STATUS_CHECK macro from base/base.h, so its CPU-only test
// target deliberately supplies this empty compatibility header.

#endif  // PBE_CACHE_CORE_TEST_STUB_GLOG_LOGGING_H_
