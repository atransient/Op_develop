#include <iostream>

#include "batch_gemm.h"
#include "data_generate.h"



int main()
{
  int b = 8;
  int m = 1024;

  int n = 1024;

  int k = 512;

  using TA = cute::half_t;
  using TB = cute::half_t;
  using TC = cute::half_t;
  using TI = cute::half_t;

  TI alpha = TI(1.0f);
  TI beta  = TI(0.0f);

  Surface<half> A_gpu(b*m*k, false);
  // Surface<__nv_bfloat16> B_gpu(b*n*k, false);
  Surface<half> B_gpu(b*n*k, false);
  Surface<half> C_gpu(b*m*n, false);


  gemm_nt(b, m, n, k,
    alpha,
    reinterpret_cast<TA*>(A_gpu.devPtr),
    reinterpret_cast<TB*>(B_gpu.devPtr),
    beta,
    reinterpret_cast<TC*>(C_gpu.devPtr));

  printf();

  // thrust::host_vector<TC> cute_result = d_C;

 return 0;

}
