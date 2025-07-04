#include <iostream>

#include "batch_gemm.h"
#include "data_generate.h"



int main()
{
  int b = 64;
  int m = 512;

  int n = 256;

  int k = 512;

  using TA = cute::bfloat16_t;
  using TB = cute::bfloat16_t;
  using TC = float;
  using TI = float;

  TI alpha = TI(1.0f);
  TI beta  = TI(0.0f);

  Surface<__nv_bfloat16> A_gpu(b*m*k, false);
  Surface<__nv_bfloat16> B_gpu(b*n*k, false);
  Surface<float> C_gpu(b*m*n, false);


  gemm_nt(b, m, n, k,
    alpha,
    reinterpret_cast<TA*>(A_gpu.devPtr),
    reinterpret_cast<TB*>(B_gpu.devPtr),
    beta,
    reinterpret_cast<TC*>(C_gpu.devPtr));

  // thrust::host_vector<TC> cute_result = d_C;

 return 0;

}
