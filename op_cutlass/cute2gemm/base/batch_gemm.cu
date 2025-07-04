#include "gemm_kernel.h"
#include "batch_gemm.h"

#include "cutlass/pipeline/sm90_pipeline.hpp"

#include "cutlass/arch/mma_sm90.h"
#include "cutlass/device_kernel.h"
#include <cute/tensor.hpp>


//fp16fp16fp16
void
gemm_nt(int b, int m, int n, int k,
    cute::half_t alpha,
    cute::half_t const* A,
    cute::half_t const* B,
    cute::half_t beta,
    cute::half_t      * C,
        cudaStream_t stream)
{
  using TA = cute::half_t;
  using TB = cute::half_t;
  // Define shapes (dynamic)
  auto Batch = int(b);
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(Batch, M, N, K);                     // (M, N, K)

  // Define TN strides (mixed)
  auto dA = make_stride(Int<1>{}, M, M * K);                      // (dM, dK)
  auto dB = make_stride(Int<1>{}, N, N * K);                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, M, M * N);                      // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int< 64>{};
  auto bBatch = Int< 1>{};
  auto cta_tiler = make_shape(bM, bN, bK, bBatch);                   // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<  3>{};  // Pipeline

  // Define the smem layouts (static)
  auto sA = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TA>{}, make_shape(bM,bK,bP));
  auto sB = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TB>{}, make_shape(bN,bK,bP));

  // Define the MMA
  TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN,GMMA::Major::MN>{});

  // Define the TMAs
  // Create Global memory tensors for TMA inspection
  Tensor mA = make_tensor(A, make_shape(M,K,Batch), dA);
  Tensor mB = make_tensor(B, make_shape(N,K,Batch), dB);

  // Create TMA Atoms with the desired copy operation on the source and destination
  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM,bK));
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN,bK));

  //
  // Setup and Launch
  //

  // Launch parameter setup
  int smem_size = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>));
  dim3 dimBlock(size(tiled_mma));
  dim3 dimCluster(1, 1, 1);
  dim3 dimGrid(round_up(size(ceil_div(m, bM)), dimCluster.x),
               round_up(size(ceil_div(n, bN)), dimCluster.y),
               size(ceil_div(Batch, bBatch)));
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

  void const* kernel_ptr = reinterpret_cast<void const*>(
                              &gemm_device<decltype(prob_shape), decltype(cta_tiler),
                              cute::half_t, decltype(sA), decltype(tmaA),
                              cute::half_t, decltype(sB), decltype(tmaB),
                              cute::half_t, decltype(dC), decltype(tiled_mma),
                                           decltype(alpha), decltype(beta)>);

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
    kernel_ptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size));

  // Kernel Launch
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr,
                                                             prob_shape, cta_tiler,
                                                             A, tmaA,
                                                             B, tmaB,
                                                             C, dC, tiled_mma,
                                                             alpha, beta);
  CUTE_CHECK_LAST();

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: Failed at kernel Launch" << std::endl;
  }
}

//fp16fp16fp32
void
gemm_nt(int b, int m, int n, int k,
    float alpha,
    cute::half_t const* A,
    cute::half_t const* B,
    float beta,
    float      * C,
        cudaStream_t stream)
{
  using TA = cute::half_t;
  using TB = cute::half_t;
  // Define shapes (dynamic)
  auto Batch = int(b);
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(Batch, M, N, K);                     // (M, N, K)

  // Define TN strides (mixed)
  auto dA = make_stride(Int<1>{}, M, M * K);                      // (dM, dK)
  auto dB = make_stride(Int<1>{}, N, N * K);                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, M, M * N);                      // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int< 64>{};
  auto bBatch = Int< 1>{};
  auto cta_tiler = make_shape(bM, bN, bK, bBatch);                   // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<  3>{};  // Pipeline

  // Define the smem layouts (static)
  auto sA = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TA>{}, make_shape(bM,bK,bP));
  auto sB = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TB>{}, make_shape(bN,bK,bP));

  // Define the MMA
  TiledMMA tiled_mma = make_tiled_mma(SM90_64x32x16_F32F16F16_SS<GMMA::Major::MN,GMMA::Major::MN>{});

  // Define the TMAs
  // Create Global memory tensors for TMA inspection
  Tensor mA = make_tensor(A, make_shape(M,K,Batch), dA);
  Tensor mB = make_tensor(B, make_shape(N,K,Batch), dB);

  // Create TMA Atoms with the desired copy operation on the source and destination
  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM,bK));
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN,bK));

  //
  // Setup and Launch
  //

  // Launch parameter setup
  int smem_size = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>));
  dim3 dimBlock(size(tiled_mma));
  dim3 dimCluster(1, 1, 1);
  dim3 dimGrid(round_up(size(ceil_div(m, bM)), dimCluster.x),
               round_up(size(ceil_div(n, bN)), dimCluster.y),
               size(ceil_div(Batch, bBatch)));
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

  void const* kernel_ptr = reinterpret_cast<void const*>(
                              &gemm_device<decltype(prob_shape), decltype(cta_tiler),
                              cute::half_t, decltype(sA), decltype(tmaA),
                              cute::half_t, decltype(sB), decltype(tmaB),
                              float, decltype(dC), decltype(tiled_mma),
                                           decltype(alpha), decltype(beta)>);

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
    kernel_ptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size));

  // Kernel Launch
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr,
                                                             prob_shape, cta_tiler,
                                                             A, tmaA,
                                                             B, tmaB,
                                                             C, dC, tiled_mma,
                                                             alpha, beta);
  CUTE_CHECK_LAST();

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: Failed at kernel Launch" << std::endl;
  }
}

//bf16bf16fp32
void
gemm_nt(int b, int m, int n, int k,
    float alpha,
    cute::bfloat16_t const* A,
    cute::bfloat16_t const* B,
    float beta,
    float      * C,
        cudaStream_t stream)
{
  using TA = cute::bfloat16_t;
  using TB = cute::bfloat16_t;
  // Define shapes (dynamic)
  auto Batch = int(b);
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(Batch, M, N, K);                     // (M, N, K)

  // Define TN strides (mixed)
  auto dA = make_stride(Int<1>{}, M, M * K);                      // (dM, dK)
  auto dB = make_stride(Int<1>{}, N, N * K);                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, M, M * N);                      // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int< 64>{};
  auto bBatch = Int< 1>{};
  auto cta_tiler = make_shape(bM, bN, bK, bBatch);                   // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<  3>{};  // Pipeline

  // Define the smem layouts (static)
  auto sA = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TA>{}, make_shape(bM,bK,bP));
  auto sB = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TB>{}, make_shape(bN,bK,bP));

  // Define the MMA
  TiledMMA tiled_mma = make_tiled_mma(SM90_64x32x16_F32BF16BF16_SS<GMMA::Major::MN,GMMA::Major::MN>{});

  // Define the TMAs
  // Create Global memory tensors for TMA inspection
  Tensor mA = make_tensor(A, make_shape(M,K,Batch), dA);
  Tensor mB = make_tensor(B, make_shape(N,K,Batch), dB);

  // Create TMA Atoms with the desired copy operation on the source and destination
  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM,bK));
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN,bK));

  //
  // Setup and Launch
  //

  // Launch parameter setup
  int smem_size = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>));
  dim3 dimBlock(size(tiled_mma));
  dim3 dimCluster(1, 1, 1);
  dim3 dimGrid(round_up(size(ceil_div(m, bM)), dimCluster.x),
               round_up(size(ceil_div(n, bN)), dimCluster.y),
               size(ceil_div(Batch, bBatch)));
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

  void const* kernel_ptr = reinterpret_cast<void const*>(
                              &gemm_device<decltype(prob_shape), decltype(cta_tiler),
                              cute::bfloat16_t, decltype(sA), decltype(tmaA),
                              cute::bfloat16_t, decltype(sB), decltype(tmaB),
                                           float, decltype(dC), decltype(tiled_mma),
                                           decltype(alpha), decltype(beta)>);

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
    kernel_ptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size));

  // Kernel Launch
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr,
                                                             prob_shape, cta_tiler,
                                                             A, tmaA,
                                                             B, tmaB,
                                                             C, dC, tiled_mma,
                                                             alpha, beta);
  CUTE_CHECK_LAST();

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: Failed at kernel Launch" << std::endl;
  }
}


//bf16bf16bf16
void
gemm_nt(int b, int m, int n, int k,
    float alpha,
    cute::bfloat16_t const* A,
    cute::bfloat16_t const* B,
    float beta,
    cute::bfloat16_t      * C,
        cudaStream_t stream)
{
  using TA = cute::bfloat16_t;
  using TB = cute::bfloat16_t;
  // Define shapes (dynamic)
  auto Batch = int(b);
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(Batch, M, N, K);                     // (M, N, K)

  // Define TN strides (mixed)
  auto dA = make_stride(Int<1>{}, M, M * K);                      // (dM, dK)
  auto dB = make_stride(Int<1>{}, N, N * K);                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, M, M * N);                      // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int< 64>{};
  auto bBatch = Int< 1>{};
  auto cta_tiler = make_shape(bM, bN, bK, bBatch);                   // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<  3>{};  // Pipeline

  // Define the smem layouts (static)
  auto sA = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TA>{}, make_shape(bM,bK,bP));
  auto sB = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TB>{}, make_shape(bN,bK,bP));

  // Define the MMA
  TiledMMA tiled_mma = make_tiled_mma(SM90_64x32x16_F32BF16BF16_SS<GMMA::Major::MN,GMMA::Major::MN>{});

  // Define the TMAs
  // Create Global memory tensors for TMA inspection
  Tensor mA = make_tensor(A, make_shape(M,K,Batch), dA);
  Tensor mB = make_tensor(B, make_shape(N,K,Batch), dB);

  // Create TMA Atoms with the desired copy operation on the source and destination
  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM,bK));
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN,bK));

  //
  // Setup and Launch
  //

  // Launch parameter setup
  int smem_size = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>));
  dim3 dimBlock(size(tiled_mma));
  dim3 dimCluster(1, 1, 1);
  dim3 dimGrid(round_up(size(ceil_div(m, bM)), dimCluster.x),
               round_up(size(ceil_div(n, bN)), dimCluster.y),
               size(ceil_div(Batch, bBatch)));
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

  void const* kernel_ptr = reinterpret_cast<void const*>(
                              &gemm_device<decltype(prob_shape), decltype(cta_tiler),
                              cute::bfloat16_t, decltype(sA), decltype(tmaA),
                              cute::bfloat16_t, decltype(sB), decltype(tmaB),
                              cute::bfloat16_t, decltype(dC), decltype(tiled_mma),
                                           decltype(alpha), decltype(beta)>);

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
    kernel_ptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size));

  // Kernel Launch
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr,
                                                             prob_shape, cta_tiler,
                                                             A, tmaA,
                                                             B, tmaB,
                                                             C, dC, tiled_mma,
                                                             alpha, beta);
  CUTE_CHECK_LAST();

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: Failed at kernel Launch" << std::endl;
  }
}