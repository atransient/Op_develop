#include <iostream>

#include "tensor_impl.h"

void my_matmul(uint64_t A_ptr, uint64_t B_ptr, uint64_t C_ptr, int64_t b, int64_t m, int64_t n, int64_t k);
void my_matmul_tensor(MyTensor A_tensor, MyTensor B_tensor, MyTensor C_tensor, TensorDataType_t caltype = TensorDataType_t::FLOAT);