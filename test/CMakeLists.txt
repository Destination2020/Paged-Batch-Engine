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
if(NOT QWEN2_SUPPORT)
  list(FILTER DIR_TEST_MODEL EXCLUDE REGEX ".*/test_qwen2_device_init\\.cpp$")
  list(FILTER DIR_TEST_MODEL EXCLUDE REGEX ".*/test_qwen25_vl_model\\.cpp$")
  list(FILTER DIR_TEST_MODEL EXCLUDE REGEX ".*/test_qwen25_vl_chunked\\.cpp$")
endif()
aux_source_directory(../test/test_tensor DIR_TEST_TENSOR)
aux_source_directory(../test/test_serving DIR_TEST_SERVING)
aux_source_directory(../test/data_core DIR_TEST_DATA)
aux_source_directory(../test/optimized DIR_TEST_OPTIMIZED)

add_executable(test_llm ${DIR_TEST} ${DIR_TEST_CUDA} ${DIR_TEST_CU} ${DIR_TEST_OP} ${DIR_TEST_OPTIMIZED} ${DIR_TEST_TENSOR} ${DIR_TEST_SERVING} ${DIR_TEST_MODEL} ${DIR_TEST_DATA})

# Enable debug symbols for CUDA (uncomment for CUDA debugging)
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")

get_target_property(GTEST_INCLUDE_DIRS GTest::gtest INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(test_llm SYSTEM BEFORE PRIVATE ${GTEST_INCLUDE_DIRS})
target_link_libraries(test_llm PRIVATE glog::glog GTest::gtest pthread)
target_include_directories(test_llm PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(test_llm PUBLIC ${GTest_INCLUDE_DIR})

target_include_directories(test_llm PUBLIC ../infMain/include)
if (LLAMA3_SUPPORT OR QWEN2_SUPPORT OR QWEN3_SUPPORT)
    message(STATUS "LINK LLAMA3 SUPPORT")
    target_link_libraries(test_llm PRIVATE absl::base re2::re2)
endif ()
target_link_libraries(test_llm PRIVATE llama)

set_target_properties(test_llm PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_target_properties(test_llm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(paged_attention_fast_bench ../test/test_paged_attention_fast_benchmark.cpp)
target_include_directories(paged_attention_fast_bench PUBLIC ../infMain/include)
target_link_libraries(paged_attention_fast_bench PRIVATE glog::glog pthread llama cublas cublasLt cudart)
set_target_properties(paged_attention_fast_bench PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(paged_attention_fp8_bench ../test/test_paged_attention_fp8_benchmark.cpp)
target_include_directories(paged_attention_fp8_bench PUBLIC ../infMain/include)
target_link_libraries(paged_attention_fp8_bench PRIVATE glog::glog pthread llama cublas cublasLt cudart)
set_target_properties(paged_attention_fp8_bench PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# CPU-only protocol checks do not initialize a CUDA context or load a model.
add_executable(test_http_protocol
  test_serving/test_http_protocol.cpp
  ../infMain/source/serving/serving_http_protocol.cpp)
target_include_directories(test_http_protocol PRIVATE ../infMain/include)
target_link_libraries(test_http_protocol PRIVATE GTest::gtest_main)
add_test(NAME pbe.http_protocol COMMAND test_http_protocol)
set_tests_properties(pbe.http_protocol PROPERTIES LABELS "cpu;serving")
add_test(NAME pbe.unit COMMAND test_llm "--gtest_filter=-test_*.*")
set_tests_properties(pbe.unit PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" LABELS "gpu;unit" RUN_SERIAL TRUE)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_test(NAME pbe.data_client_transport
    COMMAND ${Python3_EXECUTABLE} -m unittest discover
      -s ${CMAKE_SOURCE_DIR}/test/data_core -p test_client_transport.py)
  set_tests_properties(pbe.data_client_transport PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/python" LABELS "cpu;data" TIMEOUT 30)
  execute_process(COMMAND ${Python3_EXECUTABLE} -c "import zmq"
    RESULT_VARIABLE PBE_PYTHON_HAS_ZMQ OUTPUT_QUIET ERROR_QUIET)
  if(PBE_PYTHON_HAS_ZMQ EQUAL 0 AND KUIPER_BUILD_DEMOS)
    add_test(NAME pbe.http_integration
      COMMAND ${Python3_EXECUTABLE} -m unittest discover
        -s ${CMAKE_SOURCE_DIR}/tools/serving -p test_http_integration.py)
    set_tests_properties(pbe.http_integration PROPERTIES
      ENVIRONMENT "PBE_SERVING_BINARY=${CMAKE_BINARY_DIR}/demo/serving_qwen"
      LABELS "cpu;serving;integration" TIMEOUT 60)
    add_test(NAME pbe.deployment
      COMMAND ${Python3_EXECUTABLE} -m unittest discover
        -s ${CMAKE_SOURCE_DIR}/tools/serving -p test_deployment.py)
    set_tests_properties(pbe.deployment PROPERTIES LABELS "cpu;deployment" TIMEOUT 30)
  endif()
endif()

if(Python3_Interpreter_FOUND AND PBE_PYTHON_HAS_ZMQ EQUAL 0 AND KUIPER_BUILD_DEMOS AND KUIPER_HAS_ZMQ)
  add_test(NAME pbe.shared_kv_reuse
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/serving/validate_kv_reuse.py
      --build-dir ${CMAKE_BINARY_DIR} --rounds 3)
  set_tests_properties(pbe.shared_kv_reuse PROPERTIES LABELS "gpu;integration;kv"
    TIMEOUT 120 RUN_SERIAL TRUE)
endif()
