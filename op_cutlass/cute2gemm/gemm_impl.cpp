#include <iostream>
#include <vector>

#include <cute/tensor.hpp>

#include "gemm_impl.h"
#include "batch_gemm.h"
#include "tensor_impl.h"

void cutlass_matmul_tensor(MyTensor A_tensor, MyTensor B_tensor, MyTensor C_tensor, TensorDataType_t caltype)
{
    auto A_shape = A_tensor.get_dims();
    auto B_shape = B_tensor.get_dims();
    int64_t b = A_shape[0], m = A_shape[1], n = B_shape[2], k = A_shape[2];
    switch (A_tensor.data_type())
    {
    case TensorDataType_t::BFLOAT16:
        {
            if (C_tensor.data_type() == TensorDataType_t::BFLOAT16)
            {
                float alpha = float(1.0f);
                float beta = float(0.0f);
                gemm_tt(b, m, n, k,
                    alpha,
                    reinterpret_cast<cute::bfloat16_t*>(A_tensor.data_ptr()),
                    reinterpret_cast<cute::bfloat16_t*>(B_tensor.data_ptr()),
                    beta,
                    reinterpret_cast<cute::bfloat16_t*>(C_tensor.data_ptr())
                );
            }
            else
            {  
                float alpha = float(1.0f);
                float beta = float(0.0f);
                gemm_tt(b, m, n, k,
                    alpha,
                    reinterpret_cast<cute::bfloat16_t*>(A_tensor.data_ptr()),
                    reinterpret_cast<cute::bfloat16_t*>(B_tensor.data_ptr()),
                    beta,
                    reinterpret_cast<float*>(C_tensor.data_ptr())
                );
            }
            break;
        }
    
    case TensorDataType_t::HALF:
        {
            if (C_tensor.data_type() == TensorDataType_t::HALF)
            {
                cute::half_t alpha = cute::half_t(1.0f);
                cute::half_t beta = cute::half_t(0.0f);
                gemm_tt(b, m, n, k,
                    alpha,
                    reinterpret_cast<cute::half_t*>(A_tensor.data_ptr()),
                    reinterpret_cast<cute::half_t*>(B_tensor.data_ptr()),
                    beta,
                    reinterpret_cast<cute::half_t*>(C_tensor.data_ptr())
                );
            }
            else
            {  
                float alpha = float(1.0f);
                float beta = float(0.0f);
                gemm_tt(b, m, n, k,
                    alpha,
                    reinterpret_cast<cute::half_t*>(A_tensor.data_ptr()),
                    reinterpret_cast<cute::half_t*>(B_tensor.data_ptr()),
                    beta,
                    reinterpret_cast<float*>(C_tensor.data_ptr())
                );
            }
            break;
        }
        
    default:
        break;
    }
}