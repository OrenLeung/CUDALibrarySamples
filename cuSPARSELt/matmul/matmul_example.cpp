/******************************************************************************
 * Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 ******************************************************************************/

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparseLt.h>       // cusparseLt header
#include <cublas_v2.h>        // cublas header
#include <cstdio>             // printf
#include <cstdlib>            // std::rand
#include <iostream>   
#include <cuda_fp8.h>
#include <thread>
#include <chrono>


#define FP16 1000
#define INT8 1001
#define FP8  1002

/*
 * Choose your data type for matrices A and B
 */
#define AB_TYPE FP16
// #define AB_TYPE FP8
// #define AB_TYPE INT8

#if AB_TYPE == FP8
using AB_t         = __nv_fp8_e4m3;
using C_t          = __half;
using COMPUTE_t    = float;
#elif AB_TYPE == FP16
using AB_t         = __half;
using C_t          = __half;
using COMPUTE_t    = float;
#elif AB_TYPE == INT8
using AB_t         = int8_t;
using C_t          = int8_t; // can also be __half, __nv_bfloat16, int
using COMPUTE_t    = int;
#endif
                              
template <typename value_t>
struct cuda_type { };

template <>
struct cuda_type <__half> {
    static constexpr cudaDataType value = CUDA_R_16F;
};

template <>
struct cuda_type <__nv_fp8_e4m3> {
    static constexpr cudaDataType value = CUDA_R_8F_E4M3;
};

template <>
struct cuda_type <int8_t> {
    static constexpr cudaDataType value = CUDA_R_8I;
};

template <typename value_t>
struct cusparse_compute_type {  };

template <>
struct cusparse_compute_type<float> {
    static constexpr cusparseComputeType value = CUSPARSE_COMPUTE_32F;
};

template <>
struct cusparse_compute_type<int> {
    static constexpr cusparseComputeType value = CUSPARSE_COMPUTE_32I;
};

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUBLAS(func)                                                     \
{                                                                              \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        printf("cuBLAS API failed at line %d with error: %d\n",                \
               __LINE__, status);                                              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}


constexpr int EXIT_UNSUPPORTED = 2;

float generateNormalRandom() {
    static bool hasSpare = false;
    static double spare;
    
    if (hasSpare) {
        hasSpare = false;
        return spare;
    }

    hasSpare = true;

    double u, v, s;
    do {
        u = static_cast<double>(std::rand()) / RAND_MAX * 2.0 - 1.0;
        v = static_cast<double>(std::rand()) / RAND_MAX * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    s = std::sqrt(-2.0 * std::log(s) / s);
    spare = v * s;
    return u * s;
}

int run_fp16_gemm_sparse(cusparseLtHandle_t& handle, 
                          cusparseLtMatDescriptor_t& matA, 
                          cusparseLtMatDescriptor_t& matB, 
                          cusparseLtMatDescriptor_t& matC, 
                          cusparseLtMatmulPlan_t& plan, 
                          AB_t* dA_compressed, AB_t* dB, C_t* dC, C_t* dD, 
                          void* d_workspace, int m, int n, int k, 
                          const float alpha, const float beta) {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int i = 0; i < 5000; ++i) {
        CHECK_CUDA(cudaEventRecord(start, 0));

        CHECK_CUSPARSE(cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB, &beta, dC, dD, d_workspace, nullptr, 0));

        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

        double num_operations = 2.0 * m * n * k;
        double tflops = (num_operations / (milliseconds / 1000.0)) / 1e12;

        std::cout << "FP16 Sparse Run " << i + 1 << ": GEMM operation performance: " << tflops << " TFLOP/s" << std::endl;
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}

int run_fp16_gemm_dense(cublasHandle_t handle, int m, int n, int k, AB_t* dA, AB_t* dB, C_t* dC) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int i = 0; i < 5000; ++i) {
        CHECK_CUDA(cudaEventRecord(start, 0));

        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  m, n, k,
                                  &alpha,
                                  dA, CUDA_R_16F, m,
                                  dB, CUDA_R_16F, k,
                                  &beta,
                                  dC, CUDA_R_16F, m,
                                  CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

        double num_operations = 2.0 * m * n * k;
        double tflops = (num_operations / (milliseconds / 1000.0)) / 1e12;

        std::cout << "FP16 Dense Run " << i + 1 << ": GEMM operation performance: " << tflops << " TFLOP/s" << std::endl;
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}


int main(void) {
    int major_cc, minor_cc;
    CHECK_CUDA( cudaDeviceGetAttribute(&major_cc,
                                       cudaDevAttrComputeCapabilityMajor, 0) )
    CHECK_CUDA( cudaDeviceGetAttribute(&minor_cc,
                                       cudaDevAttrComputeCapabilityMinor, 0) )
    if (!(major_cc == 8 && minor_cc == 0) &&
        !(major_cc == 8 && minor_cc == 6) &&
        !(major_cc == 8 && minor_cc == 7) &&
        !(major_cc == 8 && minor_cc == 9) &&
        !(major_cc == 9 && minor_cc == 0)) {
        std::printf("\ncusparseLt is supported only on GPU devices with"
                    " compute capability == 8.0, 8.6, 8.7, 8.9, 9.0 current: %d.%d\n\n",
                     major_cc, minor_cc);
        return EXIT_UNSUPPORTED;
    }
    // Host problem definition, row-major order
    // bigger sizes may require dynamic allocations
    constexpr int m            = 8192;
    constexpr int n            = 8192;
    constexpr int k            = 8192;

    auto     order          = CUSPARSE_ORDER_ROW;
    auto     opA            = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto     opB            = CUSPARSE_OPERATION_TRANSPOSE;
    auto     type_AB        = cuda_type<AB_t>::value;
    auto     type_C         = cuda_type<C_t>::value;
    auto     compute_type   = cusparse_compute_type<COMPUTE_t>::value;
    bool     matmul_search  = true;
    bool     is_rowmajor    = (order == CUSPARSE_ORDER_ROW);
    bool     isA_transposed = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
    bool     isB_transposed = (opB != CUSPARSE_OPERATION_NON_TRANSPOSE);
    auto     num_A_rows     = (isA_transposed) ? k : m;
    auto     num_A_cols     = (isA_transposed) ? m : k;
    auto     num_B_rows     = (isB_transposed) ? n : k;
    auto     num_B_cols     = (isB_transposed) ? k : n;
    auto     num_C_rows     = m;
    auto     num_C_cols     = n;
    unsigned alignment      = 16;
    auto     lda            = (is_rowmajor) ? num_A_cols : num_A_rows;
    auto     ldb            = (is_rowmajor) ? num_B_cols : num_B_rows;
    auto     ldc            = (is_rowmajor) ? num_C_cols : num_C_rows;
    auto     A_height       = (is_rowmajor) ? num_A_rows : num_A_cols;
    auto     B_height       = (is_rowmajor) ? num_B_rows : num_B_cols;
    auto     C_height       = (is_rowmajor) ? num_C_rows : num_C_cols;
    
    auto     A_size         = A_height * lda * sizeof(AB_t);
    auto     B_size         = B_height * ldb * sizeof(AB_t);
    auto     C_size         = C_height * ldc * sizeof(C_t);

    auto     hA             = new AB_t[A_size / sizeof(AB_t)];
    auto     hB             = new AB_t[B_size / sizeof(AB_t)];
    auto     hC             = new C_t[C_size / sizeof(C_t)];

    // for (int i = 0; i < m * k; i++) 
    //     hA[i] = static_cast<AB_t>(static_cast<float>(std::rand() % 5 - 2)); // -2 ~ 2

    // for (int i = 0; i < k * n; i++)
    //     hB[i] = static_cast<AB_t>(static_cast<float>(std::rand() % 5 - 2));

    // for (int i = 0; i < m * n; i++)
    //     hC[i] = static_cast<C_t>(static_cast<float>(std::rand() % 5 - 2));

    for (int i = 0; i < m * k; i++) 
    hA[i] = static_cast<AB_t>(generateNormalRandom());

    for (int i = 0; i < k * n; i++)
        hB[i] = static_cast<AB_t>(generateNormalRandom());

    for (int i = 0; i < m * n; i++)
        hC[i] = static_cast<C_t>(generateNormalRandom());

    float alpha = 1.0f;
    float beta  = 1.0f;

    //--------------------------------------------------------------------------
    // Device memory management

    AB_t* dA, *dB, *dA_compressed;
    C_t* dC, *dD;
    int    *d_valid;

    CHECK_CUDA( cudaMalloc((void**) &dA, A_size) )
    CHECK_CUDA( cudaMalloc((void**) &dB, B_size) )
    CHECK_CUDA( cudaMalloc((void**) &dC, C_size) )
    CHECK_CUDA( cudaMalloc((void**) &d_valid, sizeof(int)) )
    dD = dC;

    CHECK_CUDA( cudaMemcpy(dA, hA, A_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size, cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    cusparseLtHandle_t             handle;
    cusparseLtMatDescriptor_t      matA, matB, matC;
    cusparseLtMatmulDescriptor_t   matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t         plan;
    cudaStream_t                   stream = nullptr;

    CHECK_CUSPARSE( cusparseLtInit(&handle) )

    // matrix descriptor initialization
    CHECK_CUSPARSE( cusparseLtStructuredDescriptorInit(
                                            &handle, &matA, num_A_rows,
                                            num_A_cols, lda, alignment,
                                            type_AB, order,
                                            CUSPARSELT_SPARSITY_50_PERCENT) )

    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                            &handle, &matB, num_B_rows,
                                            num_B_cols, ldb, alignment,
                                            type_AB, order) )
    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                            &handle, &matC, num_C_rows,
                                            num_C_cols, ldc, alignment,
                                            type_C, order) )

    // matmul, algorithm selection, and plan initialization
    CHECK_CUSPARSE( cusparseLtMatmulDescriptorInit(
                                            &handle, &matmul, opA, opB,
                                            &matA, &matB, &matC, &matC,
                                            compute_type) )

    CHECK_CUSPARSE( cusparseLtMatmulAlgSelectionInit(
                                            &handle, &alg_sel, &matmul,
                                            CUSPARSELT_MATMUL_ALG_DEFAULT) )

    CHECK_CUSPARSE( cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel))

    CHECK_CUSPARSE(cusparseLtMatmulDescSetAttribute(&handle,
                                                    &matmul,
                                                    CUSPARSELT_MATMUL_SPARSE_MAT_POINTER,
                                                    &dA,
                                                    sizeof(dA)));

    //--------------------------------------------------------------------------
    // Prune the A matrix (in-place) and check the correctness
    CHECK_CUSPARSE( cusparseLtSpMMAPrune(&handle, &matmul, dA, dA,
                                         CUSPARSELT_PRUNE_SPMMA_TILE, stream) )
    CHECK_CUSPARSE( cusparseLtSpMMAPruneCheck(&handle, &matmul, dA,
                                              d_valid, stream) )
    int is_valid;
    CHECK_CUDA( cudaMemcpyAsync(&is_valid, d_valid, sizeof(int),
                                cudaMemcpyDeviceToHost, stream) )
    CHECK_CUDA( cudaStreamSynchronize(stream) )
    if (is_valid != 0) {
        std::printf("!!!! The matrix has been pruned in a wrong way. "
                    "cusparseLtMatmul will not provide correct results\n");
        return EXIT_FAILURE;
    }
    //--------------------------------------------------------------------------
    // Compress the A matrix
    size_t compressed_size, compressed_buffer_size;
    void*  dA_compressedBuffer;
    CHECK_CUSPARSE( cusparseLtSpMMACompressedSize(&handle, &plan,
                                                  &compressed_size,
                                                  &compressed_buffer_size) )
    CHECK_CUDA( cudaMalloc((void**) &dA_compressed, compressed_size) )
    CHECK_CUDA( cudaMalloc((void**) &dA_compressedBuffer,
                           compressed_buffer_size) )

    CHECK_CUSPARSE( cusparseLtSpMMACompress(&handle, &plan, dA, dA_compressed,
                                            dA_compressedBuffer,stream) )

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Search the best kernel
    int           num_streams = 0;
    cudaStream_t* streams     = nullptr;

    if (matmul_search) {
        CHECK_CUSPARSE( cusparseLtMatmulSearch(&handle, &plan, &alpha,
                                               dA_compressed, dB, &beta,
                                               dC, dD, nullptr,
                                               streams, num_streams) )
        // dC accumulates so reset dC for correctness check
        CHECK_CUDA( cudaMemcpy(dC, hC, C_size, cudaMemcpyHostToDevice) )
    } else {
    // otherwise, it is possible to set it directly:
        int alg = 0;
        CHECK_CUSPARSE( cusparseLtMatmulAlgSetAttribute(
                                               &handle, &alg_sel,
                                               CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                               &alg, sizeof(alg)))
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    size_t workspace_size;
    CHECK_CUSPARSE( cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel))

    CHECK_CUSPARSE( cusparseLtMatmulGetWorkspace(&handle, &plan,
                                                 &workspace_size))
    void* d_workspace;
    CHECK_CUDA( cudaMalloc((void**) &d_workspace, workspace_size) )

    // Perform the matrix multiplication and benchmark
    run_fp16_gemm_sparse(handle, matA, matB, matC, plan, dA_compressed, dB, dC, dD, d_workspace, m, n, k, alpha, beta);

    // time.sleep(60) to allow the GPU to cool down to prevent power & thermal throttling
    std::this_thread::sleep_for(std::chrono::seconds(60));

    // Initialize cuBLAS handle for dense GEMM
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));

    // Perform the dense matrix multiplication and benchmark
    run_fp16_gemm_dense(cublas_handle, m, n, k, dA, dB, dC);

    // Destroy cuBLAS handle
    CHECK_CUBLAS(cublasDestroy(cublas_handle));

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // destroy plan and handle
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matA) )
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matB) )
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matC) )
    CHECK_CUSPARSE( cusparseLtMatmulPlanDestroy(&plan) )
    CHECK_CUSPARSE( cusparseLtDestroy(&handle) )
    //--------------------------------------------------------------------------
    // host memory deallocation
    delete[] hA;
    delete[] hB;
    delete[] hC;
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dA_compressed) )
    CHECK_CUDA( cudaFree(dA) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    CHECK_CUDA( cudaFree(d_valid) )
    CHECK_CUDA( cudaFree(d_workspace) )
    CHECK_CUDA( cudaFree(dA_compressedBuffer) )

    return EXIT_SUCCESS;
}
