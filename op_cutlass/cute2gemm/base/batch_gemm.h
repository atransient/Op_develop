#include <cute/tensor.hpp>
#include <iostream>

//fp16fp16fp16
void
gemm_tt(int b, int m, int n, int k,
        cute::half_t alpha,
        cute::half_t const* A,
        cute::half_t const* B,
        cute::half_t beta,
        cute::half_t      * C,
        cudaStream_t stream = 0);


//fp16fp16fp32
void
gemm_tt(int b, int m, int n, int k,
    float alpha,
    cute::half_t const* A,
    cute::half_t const* B,
    float beta,
    float      * C,
        cudaStream_t stream = 0);

//bf16bf16fp32
void
gemm_tt(int b, int m, int n, int k,
    float alpha,
    cute::bfloat16_t const* A,
    cute::bfloat16_t const* B,
    float beta,
    float      * C,
        cudaStream_t stream = 0);

//bf16bf16bf16
void
gemm_tt(int b, int m, int n, int k,
    float alpha,
    cute::bfloat16_t const* A,
    cute::bfloat16_t const* B,
    float beta,
    cute::bfloat16_t      * C,
        cudaStream_t stream = 0);


//fp16fp16fp16
void
gemm_nt(int b, int m, int n, int k,
        cute::half_t alpha,
        cute::half_t const* A,
        cute::half_t const* B,
        cute::half_t beta,
        cute::half_t      * C,
        cudaStream_t stream = 0);


//fp16fp16fp32
void
gemm_nt(int b, int m, int n, int k,
    float alpha,
    cute::half_t const* A,
    cute::half_t const* B,
    float beta,
    float      * C,
        cudaStream_t stream = 0);

//bf16bf16fp32
void
gemm_nt(int b, int m, int n, int k,
    float alpha,
    cute::bfloat16_t const* A,
    cute::bfloat16_t const* B,
    float beta,
    float      * C,
        cudaStream_t stream = 0);

//bf16bf16bf16
void
gemm_nt(int b, int m, int n, int k,
    float alpha,
    cute::bfloat16_t const* A,
    cute::bfloat16_t const* B,
    float beta,
    cute::bfloat16_t      * C,
        cudaStream_t stream = 0);