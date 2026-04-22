// Batched matmul kernel using cuBLAS gemm
#include "op/kernels/cuda/matmul_kernel_batch.cuh"
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <array>
#include <cstdint>
#include <unordered_map>
#include <glog/logging.h>

namespace kernel {

namespace {

struct LtMatmulKey {
  int32_t m = 0;
  int32_t n = 0;
  int32_t k = 0;
  size_t workspace_bytes = 0;

  bool operator==(const LtMatmulKey& other) const {
    return m == other.m &&
           n == other.n &&
           k == other.k &&
           workspace_bytes == other.workspace_bytes;
  }
};

struct LtMatmulKeyHash {
  size_t operator()(const LtMatmulKey& key) const {
    size_t h = static_cast<size_t>(key.m);
    h = h * 1315423911u + static_cast<size_t>(key.n);
    h = h * 1315423911u + static_cast<size_t>(key.k);
    h = h * 1315423911u + key.workspace_bytes;
    return h;
  }
};

bool run_bf16_cublas_lt_row_major(const tensor::Tensor& input,
                                  const tensor::Tensor& weight,
                                  const tensor::Tensor& output,
                                  int32_t m,
                                  int32_t n,
                                  int32_t k,
                                  const CudaConfig* config) {
  if (config == nullptr || config->cublas_lt_handle == nullptr) {
    return false;
  }

  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  constexpr cublasComputeType_t kComputeType = CUBLAS_COMPUTE_32F_FAST_16BF;
  constexpr cudaDataType_t kDataType = CUDA_R_16BF;

  cublasLtMatmulDesc_t operation_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;

  auto cleanup = [&]() {
    if (preference != nullptr) {
      cublasLtMatmulPreferenceDestroy(preference);
    }
    if (c_desc != nullptr) {
      cublasLtMatrixLayoutDestroy(c_desc);
    }
    if (b_desc != nullptr) {
      cublasLtMatrixLayoutDestroy(b_desc);
    }
    if (a_desc != nullptr) {
      cublasLtMatrixLayoutDestroy(a_desc);
    }
    if (operation_desc != nullptr) {
      cublasLtMatmulDescDestroy(operation_desc);
    }
  };

  cublasStatus_t status =
      cublasLtMatmulDescCreate(&operation_desc, kComputeType, CUDA_R_32F);
  if (status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return false;
  }

  cublasOperation_t trans_a = CUBLAS_OP_N;
  cublasOperation_t trans_b = CUBLAS_OP_T;
  status = cublasLtMatmulDescSetAttribute(
      operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a));
  if (status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return false;
  }
  status = cublasLtMatmulDescSetAttribute(
      operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b));
  if (status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return false;
  }

  status = cublasLtMatrixLayoutCreate(&a_desc, kDataType, m, k, k);
  if (status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return false;
  }
  status = cublasLtMatrixLayoutCreate(&b_desc, kDataType, n, k, k);
  if (status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return false;
  }
  status = cublasLtMatrixLayoutCreate(&c_desc, kDataType, m, n, n);
  if (status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return false;
  }

  constexpr cublasLtOrder_t kRowOrder = CUBLASLT_ORDER_ROW;
  status = cublasLtMatrixLayoutSetAttribute(
      a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &kRowOrder, sizeof(kRowOrder));
  if (status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return false;
  }
  status = cublasLtMatrixLayoutSetAttribute(
      b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &kRowOrder, sizeof(kRowOrder));
  if (status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return false;
  }
  status = cublasLtMatrixLayoutSetAttribute(
      c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &kRowOrder, sizeof(kRowOrder));
  if (status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return false;
  }

  status = cublasLtMatmulPreferenceCreate(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return false;
  }

  const size_t workspace_bytes = config->cublas_lt_workspace_bytes;
  status = cublasLtMatmulPreferenceSetAttribute(
      preference,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspace_bytes,
      sizeof(workspace_bytes));
  if (status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return false;
  }

  static std::unordered_map<LtMatmulKey, cublasLtMatmulHeuristicResult_t, LtMatmulKeyHash>
      heuristic_cache;
  const LtMatmulKey key{m, n, k, workspace_bytes};
  const auto cache_it = heuristic_cache.find(key);

  cublasLtMatmulHeuristicResult_t heuristic{};
  if (cache_it == heuristic_cache.end()) {
    int returned_count = 0;
    status = cublasLtMatmulAlgoGetHeuristic(
        config->cublas_lt_handle,
        operation_desc,
        a_desc,
        b_desc,
        c_desc,
        c_desc,
        preference,
        1,
        &heuristic,
        &returned_count);
    if (status != CUBLAS_STATUS_SUCCESS || returned_count == 0 ||
        heuristic.state != CUBLAS_STATUS_SUCCESS) {
      cleanup();
      return false;
    }
    heuristic_cache.emplace(key, heuristic);
  } else {
    heuristic = cache_it->second;
  }

  void* workspace = config->cublas_lt_workspace.get();
  if (heuristic.workspaceSize > workspace_bytes) {
    cleanup();
    return false;
  }

  status = cublasLtMatmul(
      config->cublas_lt_handle,
      operation_desc,
      &alpha,
      input.ptr<uint16_t>(),
      a_desc,
      weight.ptr<uint16_t>(),
      b_desc,
      &beta,
      const_cast<uint16_t*>(output.ptr<uint16_t>()),
      c_desc,
      const_cast<uint16_t*>(output.ptr<uint16_t>()),
      c_desc,
      &heuristic.algo,
      workspace,
      heuristic.workspaceSize,
      config->stream);
  cleanup();
  return status == CUBLAS_STATUS_SUCCESS;
}

}  // namespace

void matmul_batch_kernel_cu(const tensor::Tensor& input,
                            const tensor::Tensor& weight,
                            const tensor::Tensor& output,
                            int32_t batch_tokens,
                            const CudaConfig* config) {
  CHECK_NE(config, nullptr);
  CHECK_NE(config->cublas_handle, nullptr) << "cuBLAS handle not initialized";
  CHECK_GT(batch_tokens, 0);
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());
  CHECK_EQ(input.device_type(), base::DeviceType::kDeviceCUDA);
  CHECK_EQ(weight.device_type(), base::DeviceType::kDeviceCUDA);
  CHECK_EQ(output.device_type(), base::DeviceType::kDeviceCUDA);
  CHECK_EQ(input.dims_size(), 2);
  CHECK_EQ(weight.dims_size(), 2);
  CHECK_EQ(output.dims_size(), 2);
  CHECK_EQ(input.data_type(), weight.data_type());
  CHECK_EQ(input.data_type(), output.data_type());

  const int32_t n = weight.get_dim(0);  // out_dim
  const int32_t k = weight.get_dim(1);  // in_dim
  const int32_t m = batch_tokens;
  CHECK_EQ(input.get_dim(0), m);
  CHECK_EQ(input.get_dim(1), k);
  CHECK_EQ(output.get_dim(0), m);
  CHECK_EQ(output.get_dim(1), n);

  cublasHandle_t handle = config->cublas_handle;
  if (input.data_type() == base::DataType::kDataTypeFp32) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        weight.ptr<float>(),
        k,
        input.ptr<float>(),
        k,
        &beta,
        const_cast<float*>(output.ptr<float>()),
        n);
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << "cublasSgemm failed";
    return;
  }

  CHECK_EQ(input.data_type(), base::DataType::kDataTypeBf16);
  if (run_bf16_cublas_lt_row_major(input, weight, output, m, n, k, config)) {
    return;
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasStatus_t status = cublasGemmEx(
      handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      n,
      m,
      k,
      &alpha,
      weight.ptr<uint16_t>(),
      CUDA_R_16BF,
      k,
      input.ptr<uint16_t>(),
      CUDA_R_16BF,
      k,
      &beta,
      const_cast<uint16_t*>(output.ptr<uint16_t>()),
      CUDA_R_16BF,
      n,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT);
  CHECK_EQ(status, CUBLAS_STATUS_SUCCESS)
      << "cublasGemmEx BF16 failed, status=" << static_cast<int>(status)
      << ", batch_tokens=" << m << ", out_dim=" << n << ", in_dim=" << k;
}

}  // namespace kernel
