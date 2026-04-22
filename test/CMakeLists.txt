# Updated on March 15, 2026
include(../cmake/cuda.cmake)
find_package(GTest REQUIRED)
find_package(glog REQUIRED)

set(link_ext_lib glog::glog GTest::gtest GTest::gtest_main pthread)
file(GLOB DIR_TEST CONFIGURE_DEPENDS "../test/*.cpp")
file(GLOB DIR_TEST_CUDA CONFIGURE_DEPENDS "../test/*.cu")
# 排除独立的基准测试程序（有自己的 main 函数）
list(FILTER DIR_TEST EXCLUDE REGEX ".*/test_cuda_graph_benchmark\\.cpp$")
list(FILTER DIR_TEST EXCLUDE REGEX ".*/test_paged_attention_fast_benchmark\\.cpp$")
list(FILTER DIR_TEST EXCLUDE REGEX ".*/test_paged_attention_fp8_benchmark\\.cpp$")

aux_source_directory(../test/test_cu DIR_TEST_CU)
aux_source_directory(../test/test_op DIR_TEST_OP)
aux_source_directory(../test/test_model DIR_TEST_MODEL)
aux_source_directory(../test/test_tensor DIR_TEST_TENSOR)
aux_source_directory(../test/optimized DIR_TEST_OPTIMIZED)

add_executable(test_llm ${DIR_TEST} ${DIR_TEST_CUDA} ${DIR_TEST_CU} ${DIR_TEST_OP} ${DIR_TEST_OPTIMIZED} ${DIR_TEST_TENSOR} ${DIR_TEST_MODEL})

# Enable debug symbols for CUDA (uncomment for CUDA debugging)
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")

get_target_property(GTEST_INCLUDE_DIRS GTest::gtest INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(test_llm SYSTEM BEFORE PRIVATE ${GTEST_INCLUDE_DIRS})
target_link_libraries(test_llm PRIVATE glog::glog GTest::gtest pthread)
target_include_directories(test_llm PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(test_llm PUBLIC ${GTest_INCLUDE_DIR})

target_include_directories(test_llm PUBLIC ../infMain/include)
target_link_directories(test_llm PUBLIC ${PROJECT_SOURCE_DIR}/lib)
if (LLAMA3_SUPPORT OR QWEN2_SUPPORT OR QWEN3_SUPPORT)
    message(STATUS "LINK LLAMA3 SUPPORT")
    target_link_libraries(test_llm PRIVATE absl::base re2::re2)
endif ()
target_link_libraries(test_llm PRIVATE llama)

set_target_properties(test_llm PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_target_properties(test_llm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(paged_attention_fast_bench ../test/test_paged_attention_fast_benchmark.cpp)
target_include_directories(paged_attention_fast_bench PUBLIC ../infMain/include)
target_link_directories(paged_attention_fast_bench PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(paged_attention_fast_bench PRIVATE glog::glog pthread llama cublas cublasLt cudart)
set_target_properties(paged_attention_fast_bench PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(paged_attention_fp8_bench ../test/test_paged_attention_fp8_benchmark.cpp)
target_include_directories(paged_attention_fp8_bench PUBLIC ../infMain/include)
target_link_directories(paged_attention_fp8_bench PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(paged_attention_fp8_bench PRIVATE glog::glog pthread llama cublas cublasLt cudart)
set_target_properties(paged_attention_fp8_bench PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
