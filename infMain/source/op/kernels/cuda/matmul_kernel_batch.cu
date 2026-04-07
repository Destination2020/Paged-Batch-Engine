// Batched matmul kernel using cuBLAS gemm
#include "op/kernels/cuda/matmul_kernel_batch.cuh"
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <glog/logging.h>
#include "cuda_type_utils.cuh"

namespace kernel {

void matmul_batch_kernel_cu(const tensor::Tensor& input,
                            const tensor::Tensor& weight,
                            const tensor::Tensor& output,
                            int32_t batch_tokens,
                            const CudaConfig* config) {
  CHECK_NE(config, nullptr);
  CHECK_NE(config->cublas_handle, nullptr) << "cuBLAS handle not initialized";
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());
  CHECK_EQ(weight.dims_size(), 2);

  const int32_t N = weight.get_dim(0);  // out_dim
  const int32_t K = weight.get_dim(1);  // in_dim
  const int32_t M = batch_tokens;

  // Row-major: output[M,N] = input[M,K] × weight^T[K,N]
  // cuBLAS col-major trick:
  //   Treat row-major A[M,K] as col-major A^T[K,M]
  //   output^T[N,M] = weight[N,K] × input^T[K,M]
  //   cublasSgemm(CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, ...)
  // But weight is [N,K] row-major = [K,N] col-major, so we need OP_T on weight
  // input is [M,K] row-major = [K,M] col-major, so we use OP_N on input

  cublasHandle_t handle = config->cublas_handle;

  if (input.data_type() == base::DataType::kDataTypeFp32) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_T,   // weight: [N,K] row-major -> transpose for col-major
        CUBLAS_OP_N,   // input: [M,K] row-major -> [K,M] col-major, no transpose
        N, M, K,
        &alpha,
        weight.ptr<float>(), K,   // lda = K (row-major stride of weight)
        input.ptr<float>(), K,    // ldb = K (row-major stride of input)
        &beta,
        const_cast<float*>(output.ptr<float>()), N);  // ldc = N
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << "cublasSgemm failed";
  } else {
    CHECK_EQ(input.data_type(), base::DataType::kDataTypeBf16);
    // Use cublasGemmEx for BF16
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        N, M, K,
        &alpha,
        weight.ptr<uint16_t>(), CUDA_R_16BF, K,
        input.ptr<uint16_t>(), CUDA_R_16BF, K,
        &beta,
        const_cast<uint16_t*>(output.ptr<uint16_t>()), CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << "cublasGemmEx BF16 failed";
  }
}

}  // namespace kernel
