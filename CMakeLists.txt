# Updated on March 15, 2026
cmake_minimum_required(VERSION 3.16)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set the CUDA compiler path or name
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(HighPerInfFram CXX CUDA)
include(cmake/cuda.cmake)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

option(LLAMA3_SUPPORT OFF)
if (LLAMA3_SUPPORT)
  message(STATUS "LLAMA3 SUPPORT")
  add_definitions(-DLLAMA3_SUPPORT)
endif()
option(QWEN2_SUPPORT OFF)
if (QWEN2_SUPPORT)
  message(STATUS "QWEN2 SUPPORT")
  add_definitions(-DQWEN2_SUPPORT)
endif()

option(QWEN3_SUPPORT OFF)
if (QWEN3_SUPPORT)
  message(STATUS "QWEN3 SUPPORT")
  add_definitions(-DQWEN3_SUPPORT)
endif()

option(QWEN_MOE_SUPPORT OFF)
if (QWEN_MOE_SUPPORT)
  message(STATUS "QWEN_MOE SUPPORT")
  add_definitions(-DQWEN_MOE_SUPPORT)
endif()
# ---- Add dependencies via CPM ----
# see https://github.com/TheLartians/CPM.cmake for more info
option(USE_CPM "Use CPM for dependency management" OFF)

if(USE_CPM)
  # Use CPM to manage dependencies
  include(cmake/CPM.cmake)

  CPMAddPackage(
    NAME GTest
    GITHUB_REPOSITORY google/googletest
    VERSION 1.15.0
  )

  CPMAddPackage(
    NAME glog
    GITHUB_REPOSITORY google/glog
    VERSION 0.7.1
    OPTIONS "BUILD_TESTING Off"
  )

  CPMAddPackage(
    NAME Armadillo
    GITLAB_REPOSITORY conradsnicta/armadillo-code
    GIT_TAG 14.0.1
  )

  CPMAddPackage(
    NAME sentencepiece
    GITHUB_REPOSITORY google/sentencepiece
    VERSION 0.2.0
  )
  find_package(sentencepiece REQUIRED)

  if (LLAMA3_SUPPORT OR QWEN2_SUPPORT OR QWEN3_SUPPORT OR QWEN_MOE_SUPPORT)
    CPMAddPackage(
        NAME absl
        GITHUB_REPOSITORY abseil/abseil-cpp
        GIT_TAG 20240722.0
        OPTIONS "BUILD_TESTING Off" "ABSL_PROPAGATE_CXX_STD ON" "ABSL_ENABLE_INSTALL ON"
    )
    CPMAddPackage(
        NAME re2
        GITHUB_REPOSITORY google/re2
        GIT_TAG 2024-07-02
    )
    CPMAddPackage(
        NAME nlohmann_json
        GITHUB_REPOSITORY nlohmann/json
        VERSION 3.11.3
    )
  endif()
endif()

# ---------------------------------
# 优先使用系统路径，避免使用 Windows Anaconda 中的库
list(INSERT CMAKE_PREFIX_PATH 0 /usr/local /usr)
list(INSERT CMAKE_INCLUDE_PATH 0 /usr/local/include /usr/include)
list(INSERT CMAKE_LIBRARY_PATH 0 /usr/local/lib /usr/lib)

option(KUIPER_AUTO_ALIGN_CONDA_PACKAGES
       "When a Conda env is active, auto-align package configs that must stay ABI-compatible."
       ON)

set(KUIPER_CONDA_PREFIX "")
if(DEFINED ENV{CONDA_PREFIX} AND NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
  set(KUIPER_CONDA_PREFIX "$ENV{CONDA_PREFIX}")
endif()

function(kuiper_path_is_under_prefix out_var path prefix)
  if("${path}" STREQUAL "" OR "${prefix}" STREQUAL "")
    set(${out_var} FALSE PARENT_SCOPE)
    return()
  endif()

  string(LENGTH "${prefix}" _prefix_len)
  string(SUBSTRING "${path}" 0 ${_prefix_len} _path_prefix)
  if(_path_prefix STREQUAL "${prefix}")
    set(${out_var} TRUE PARENT_SCOPE)
  else()
    set(${out_var} FALSE PARENT_SCOPE)
  endif()
endfunction()

function(kuiper_maybe_align_package_dir package_name config_subdir)
  if(USE_CPM OR NOT KUIPER_AUTO_ALIGN_CONDA_PACKAGES OR "${KUIPER_CONDA_PREFIX}" STREQUAL "")
    return()
  endif()

  set(_candidate_dir "${KUIPER_CONDA_PREFIX}/${config_subdir}")
  set(_candidate_config "${_candidate_dir}/${package_name}Config.cmake")
  if(NOT EXISTS "${_candidate_config}")
    return()
  endif()

  set(_current_dir "")
  if(DEFINED ${package_name}_DIR)
    set(_current_dir "${${package_name}_DIR}")
  endif()

  kuiper_path_is_under_prefix(_current_is_conda "${_current_dir}" "${KUIPER_CONDA_PREFIX}")
  if("${_current_dir}" STREQUAL "" OR _current_is_conda OR _current_dir MATCHES "^/usr(/local)?/")
    set(${package_name}_DIR "${_candidate_dir}" CACHE PATH
        "Auto-aligned ${package_name} package directory" FORCE)
    message(STATUS "Auto-aligned ${package_name} to Conda package config: ${_candidate_dir}")
  else()
    message(STATUS "Keeping explicit ${package_name}_DIR=${_current_dir}")
  endif()
endfunction()

function(kuiper_report_package_origin package_name)
  if(NOT DEFINED ${package_name}_DIR OR "${${package_name}_DIR}" STREQUAL "")
    return()
  endif()

  kuiper_path_is_under_prefix(_is_conda "${${package_name}_DIR}" "${KUIPER_CONDA_PREFIX}")
  if(_is_conda)
    message(STATUS "${package_name} package root: CONDA (${${package_name}_DIR})")
  else()
    message(STATUS "${package_name} package root: ${${package_name}_DIR}")
  endif()
endfunction()

kuiper_maybe_align_package_dir(GTest "lib/cmake/GTest")
if(LLAMA3_SUPPORT OR QWEN2_SUPPORT OR QWEN3_SUPPORT OR QWEN_MOE_SUPPORT)
  kuiper_maybe_align_package_dir(nlohmann_json "share/cmake/nlohmann_json")
endif()

find_package(GTest CONFIG REQUIRED)
find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)
if (LLAMA3_SUPPORT OR QWEN2_SUPPORT OR QWEN3_SUPPORT OR QWEN_MOE_SUPPORT)
  find_package(absl REQUIRED)
  find_package(re2 REQUIRED)
  find_package(nlohmann_json CONFIG REQUIRED)
endif()

kuiper_report_package_origin(GTest)
if (LLAMA3_SUPPORT OR QWEN2_SUPPORT OR QWEN3_SUPPORT OR QWEN_MOE_SUPPORT)
  kuiper_report_package_origin(nlohmann_json)
  if(KUIPER_AUTO_ALIGN_CONDA_PACKAGES AND NOT USE_CPM AND
     NOT "${KUIPER_CONDA_PREFIX}" STREQUAL "")
    kuiper_path_is_under_prefix(_gtest_is_conda "${GTest_DIR}" "${KUIPER_CONDA_PREFIX}")
    kuiper_path_is_under_prefix(_json_is_conda "${nlohmann_json_DIR}" "${KUIPER_CONDA_PREFIX}")
    if(_gtest_is_conda AND NOT _json_is_conda)
      message(WARNING
              "GTest is using Conda while nlohmann_json is not. This is supported, "
              "but keep an eye on mixed package roots if test targets start failing.")
    elseif(_json_is_conda AND NOT _gtest_is_conda)
      message(WARNING
              "nlohmann_json is using Conda while GTest is not. This can reintroduce "
              "test_llm header/library mismatches; consider clearing the build cache.")
    endif()
  endif()
endif()
find_library(NVTOOLSEXT_LIBRARY nvToolsExt PATHS /usr/local/cuda/lib64 /usr/lib /usr/lib64)
if (NVTOOLSEXT_LIBRARY)
  message(STATUS "Found NVTX library: ${NVTOOLSEXT_LIBRARY}")
endif()

set(KUIPER_ENABLE_NCCL "AUTO" CACHE STRING
    "Enable NCCL-backed PD handoff support: AUTO, ON, or OFF")
set_property(CACHE KUIPER_ENABLE_NCCL PROPERTY STRINGS AUTO ON OFF)
string(TOUPPER "${KUIPER_ENABLE_NCCL}" KUIPER_ENABLE_NCCL_VALUE)
if(NOT KUIPER_ENABLE_NCCL_VALUE MATCHES "^(AUTO|ON|OFF)$")
  message(FATAL_ERROR "KUIPER_ENABLE_NCCL must be AUTO, ON, or OFF")
endif()
if(NOT KUIPER_ENABLE_NCCL_VALUE STREQUAL "OFF")
  find_path(NCCL_INCLUDE_DIR nccl.h
            PATHS /usr/local/cuda/include
                  /usr/local/cuda/targets/x86_64-linux/include
                  /usr/local/include
                  /usr/include)
  find_library(NCCL_LIBRARY nccl
               PATHS /usr/local/cuda/lib64
                     /usr/local/lib
                     /usr/lib
                     /usr/lib/x86_64-linux-gnu)
  if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
    set(KUIPER_HAS_NCCL ON)
    message(STATUS "Found NCCL: include=${NCCL_INCLUDE_DIR} library=${NCCL_LIBRARY}")
  elseif(KUIPER_ENABLE_NCCL_VALUE STREQUAL "ON")
    message(FATAL_ERROR
            "KUIPER_ENABLE_NCCL=ON but NCCL was not found. Install NCCL "
            "development headers/libraries or configure with -DKUIPER_ENABLE_NCCL=OFF.")
  else()
    set(KUIPER_HAS_NCCL OFF)
    message(STATUS "NCCL not found; building without NCCL-backed PD handoff")
  endif()
else()
  set(KUIPER_HAS_NCCL OFF)
  message(STATUS "NCCL support disabled")
endif()

set(KUIPER_ENABLE_ZMQ "AUTO" CACHE STRING
    "Enable ZeroMQ online API/engine process support: AUTO, ON, or OFF")
set_property(CACHE KUIPER_ENABLE_ZMQ PROPERTY STRINGS AUTO ON OFF)
string(TOUPPER "${KUIPER_ENABLE_ZMQ}" KUIPER_ENABLE_ZMQ_VALUE)
if(NOT KUIPER_ENABLE_ZMQ_VALUE MATCHES "^(AUTO|ON|OFF)$")
  message(FATAL_ERROR "KUIPER_ENABLE_ZMQ must be AUTO, ON, or OFF")
endif()
if(NOT KUIPER_ENABLE_ZMQ_VALUE STREQUAL "OFF")
  find_path(ZMQ_INCLUDE_DIR zmq.h
            PATHS /usr/local/include /usr/include)
  find_library(ZMQ_LIBRARY zmq
               PATHS /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu)
  if(ZMQ_INCLUDE_DIR AND ZMQ_LIBRARY)
    set(KUIPER_HAS_ZMQ ON)
    message(STATUS "Found ZeroMQ: include=${ZMQ_INCLUDE_DIR} library=${ZMQ_LIBRARY}")
  elseif(KUIPER_ENABLE_ZMQ_VALUE STREQUAL "ON")
    message(FATAL_ERROR
            "KUIPER_ENABLE_ZMQ=ON but ZeroMQ was not found. Install libzmq "
            "development headers/libraries or configure with -DKUIPER_ENABLE_ZMQ=OFF.")
  else()
    set(KUIPER_HAS_ZMQ OFF)
    message(STATUS "ZeroMQ not found; building ZMQ process roles as unavailable stubs")
  endif()
else()
  set(KUIPER_HAS_ZMQ OFF)
  message(STATUS "ZeroMQ support disabled")
endif()

aux_source_directory(infMain/source/tensor/ DIR_TENSOR)
aux_source_directory(infMain/source/base/ DIR_BASE)
aux_source_directory(infMain/source/op/ DIR_OP)
aux_source_directory(infMain/source/op/kernels/cpu DIR_KERNEL_CPU)
aux_source_directory(infMain/source/op/kernels/cuda DIR_KERNEL_CUDA)
aux_source_directory(infMain/source/op/kernels/ DIR_KERNEL)
aux_source_directory(infMain/source/sampler DIR_SAMPLE)
aux_source_directory(infMain/source/serving DIR_SERVING)
list(APPEND DIR_SERVING infMain/source/serving/serving_online_server.cpp)
list(APPEND DIR_SERVING infMain/source/serving/serving_http_server.cpp)
list(APPEND DIR_SERVING infMain/source/serving/serving_online_engine.cpp)
list(APPEND DIR_SERVING infMain/source/serving/serving_online_engine_pool.cpp)
list(APPEND DIR_SERVING infMain/source/serving/gpu_worker.cpp)
list(APPEND DIR_SERVING infMain/source/serving/pd_handoff.cpp)
list(APPEND DIR_SERVING infMain/source/serving/pd_coordinator.cpp)
list(APPEND DIR_SERVING infMain/source/serving/decode_kv_reservation.cpp)
list(APPEND DIR_SERVING infMain/source/serving/pd_worker.cpp)
list(APPEND DIR_SERVING infMain/source/serving/pd_engine.cpp)
list(APPEND DIR_SERVING infMain/source/serving/pd_handoff_builder.cpp)
list(REMOVE_DUPLICATES DIR_SERVING)

set(DIR_MODEL
    infMain/source/model/model.cpp
    infMain/source/model/raw_model_data.cpp
    infMain/source/model/llama3.cpp)

if (QWEN2_SUPPORT)
  list(APPEND DIR_MODEL
       infMain/source/model/qwen2.cpp
       infMain/source/model/qwen2_batch.cpp
       infMain/source/model/paged_kv_runtime.cpp
       infMain/source/model/serving_memory_planner.cpp)
endif()

if (QWEN3_SUPPORT)
  list(APPEND DIR_MODEL
       infMain/source/model/qwen3.cpp)
endif()

if (QWEN_MOE_SUPPORT)
  list(APPEND DIR_MODEL
       infMain/source/model/qwen_moe.cpp)
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(llama SHARED ${DIR_TENSOR} ${DIR_BASE} ${DIR_OP} ${DIR_KERNEL} ${DIR_MODEL} ${DIR_KERNEL_CPU} ${DIR_KERNEL_CUDA} ${DIR_KERNEL} ${DIR_SAMPLE} ${DIR_SERVING})
target_link_libraries(llama
  PUBLIC
    sentencepiece
    glog::glog
    pthread
    cudart
    cublas
    cublasLt
    armadillo)
if (NVTOOLSEXT_LIBRARY)
  target_link_libraries(llama PUBLIC ${NVTOOLSEXT_LIBRARY})
endif()
if (KUIPER_HAS_NCCL)
  target_compile_definitions(llama PUBLIC KUIPER_ENABLE_NCCL=1)
  target_include_directories(llama PUBLIC ${NCCL_INCLUDE_DIR})
  target_link_libraries(llama PUBLIC ${NCCL_LIBRARY})
endif()
if (KUIPER_HAS_ZMQ)
  target_compile_definitions(llama PUBLIC KUIPER_ENABLE_ZMQ=1)
  target_include_directories(llama PUBLIC ${ZMQ_INCLUDE_DIR})
  target_link_libraries(llama PUBLIC ${ZMQ_LIBRARY})
endif()
target_link_directories(llama PUBLIC ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64)

target_include_directories(llama PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(llama PUBLIC ${PROJECT_SOURCE_DIR}/infMain/include)
target_include_directories(llama PUBLIC ${Armadillo_INCLUDE_DIR})
target_include_directories(llama PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

if(USE_CPM)
  # Add sentencepiece include directory
  target_include_directories(llama PUBLIC ${sentencepiece_SOURCE_DIR}/src)
endif()

if (LLAMA3_SUPPORT OR QWEN2_SUPPORT OR QWEN3_SUPPORT OR QWEN_MOE_SUPPORT)
  target_link_libraries(llama PUBLIC absl::strings absl::str_format re2::re2
                                     nlohmann_json::nlohmann_json)
  if(USE_CPM)
    target_include_directories(llama BEFORE PUBLIC ${absl_SOURCE_DIR})
    target_include_directories(llama BEFORE PUBLIC ${nlohmann_json_SOURCE_DIR}/include)
  endif()
endif()

set_target_properties(llama PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_subdirectory(test)
add_subdirectory(demo)
