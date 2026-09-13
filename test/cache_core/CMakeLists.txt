cmake_minimum_required(VERSION 3.16)

project(PagedBatchCacheCoreTests LANGUAGES CXX)

enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(GTest CONFIG REQUIRED)

set(PBE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")

add_executable(
  test_cache_core
  "${PBE_ROOT}/infMain/source/base/base.cpp"
  "${PBE_ROOT}/infMain/source/cache/page_schema.cpp"
  "${PBE_ROOT}/infMain/source/cache/layout_codec.cpp"
  "${PBE_ROOT}/infMain/source/cache/transfer_plan.cpp"
  "${PBE_ROOT}/infMain/source/cache/host_store.cpp"
  "${PBE_ROOT}/infMain/source/cache/transfer_scheduler.cpp"
  "${PBE_ROOT}/infMain/source/data/protocol.cpp"
  "${PBE_ROOT}/infMain/source/data/data_runtime.cpp"
  "${PBE_ROOT}/infMain/source/data/tensor_bundle.cpp"
  "${PBE_ROOT}/test/test_host_store.cpp"
  "${PBE_ROOT}/test/test_cache_transfer_plan.cpp"
  "${PBE_ROOT}/test/test_logical_radix_probe.cpp")
target_sources(test_cache_core PRIVATE "${PBE_ROOT}/test/test_transfer_scheduler.cpp")
target_sources(test_cache_core PRIVATE "${PBE_ROOT}/test/data_core/test_data_protocol.cpp")
target_sources(test_cache_core PRIVATE "${PBE_ROOT}/test/data_core/test_data_runtime.cpp")
target_sources(test_cache_core PRIVATE "${PBE_ROOT}/test/data_core/test_tensor_bundle.cpp")

target_include_directories(
  test_cache_core
  BEFORE PRIVATE "${CMAKE_CURRENT_LIST_DIR}/stubs" "${PBE_ROOT}/infMain/include")
target_link_libraries(test_cache_core PRIVATE GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_cache_core)

option(KUIPER_TEST_CPU_OWNERSHIP "Build real CPU allocators, leases, directory and KV manager tests" OFF)
if(KUIPER_TEST_CPU_OWNERSHIP)
  find_package(glog CONFIG REQUIRED)
  find_package(Armadillo REQUIRED)
  add_executable(test_cache_ownership
    "${PBE_ROOT}/infMain/source/base/base.cpp"
    "${PBE_ROOT}/infMain/source/base/alloc.cpp"
    "${PBE_ROOT}/infMain/source/base/alloc_cpu.cpp"
    "${PBE_ROOT}/infMain/source/base/buffer.cpp"
    "${PBE_ROOT}/infMain/source/base/block_allocator.cpp"
    "${PBE_ROOT}/infMain/source/base/sequence_kv_manager.cpp"
    "${PBE_ROOT}/infMain/source/base/kv_cache_manager.cpp"
    "${PBE_ROOT}/infMain/source/tensor/tensor.cpp"
    "${PBE_ROOT}/infMain/source/cache/page_schema.cpp"
    "${PBE_ROOT}/infMain/source/cache/page_directory.cpp"
    "${PBE_ROOT}/infMain/source/cache/page_migration.cpp"
    "${PBE_ROOT}/infMain/source/cache/page_migration_flight_adapter.cpp"
    "${PBE_ROOT}/infMain/source/cache/transfer_plan.cpp"
    "${PBE_ROOT}/infMain/source/cache/transfer_scheduler.cpp"
    "${PBE_ROOT}/infMain/source/data/data_runtime.cpp"
    "${PBE_ROOT}/infMain/source/serving/request_checkpoint.cpp"
    "${PBE_ROOT}/test/test_page_migration.cpp"
    "${PBE_ROOT}/infMain/source/cache/layout_codec.cpp"
    "${PBE_ROOT}/infMain/source/cache/host_store.cpp"
    "${PBE_ROOT}/test/test_block_lease.cpp"
    "${PBE_ROOT}/test/test_page_directory.cpp"
    "${PBE_ROOT}/test/test_kv_cache_manager.cpp"
    "${PBE_ROOT}/test/test_request_checkpoint.cpp")
  target_sources(test_cache_ownership PRIVATE "${PBE_ROOT}/test/test_p5_ablation.cpp")
  target_compile_definitions(test_cache_ownership PRIVATE KUIPER_CPU_ONLY)
  target_include_directories(test_cache_ownership PRIVATE "${PBE_ROOT}/infMain/include" ${ARMADILLO_INCLUDE_DIRS})
  target_link_libraries(test_cache_ownership PRIVATE glog::glog GTest::gtest_main ${ARMADILLO_LIBRARIES})
  gtest_discover_tests(test_cache_ownership)
endif()
