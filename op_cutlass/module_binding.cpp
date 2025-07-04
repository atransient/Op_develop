#include <iostream>

#include <pybind11/pybind11.h>

#include "gemm_impl.h"

PYBIND11_MODULE(cutlass_op, m)
{
    m.doc() = "pybind11 cutlass operator implement";
    m.def("cutlass_matmul_tensor", &cutlass_matmul_tensor, "matmul implement",
        pybind11::arg("A_tensor"), pybind11::arg("B_tensor"), 
        pybind11::arg("C_tensor"), pybind11::arg("caltype") = TensorDataType_t::FLOAT);
}