#include <iostream>

#include <pybind11/pybind11.h>

#include "fe_header.h"
#include "be_header.h"

PYBIND11_MODULE(cudnn_op, m)
{
    m.doc() = "pybind11 cudnn frontend operator implement";
    m.def("my_matmul", &my_matmul, "matmul implement");
    m.def("my_matmul_tensor", &my_matmul_tensor, "matmul implement",
        pybind11::arg("A_tensor"), pybind11::arg("B_tensor"), 
        pybind11::arg("C_tensor"), pybind11::arg("caltype") = TensorDataType_t::FLOAT);

    m.def("conv2d_forward", &conv2d_forward, "conv2d_forward implement");
    m.def("conv2d_bpa", &conv2d_forward, "conv2d_bpa implement");
    m.def("conv2d_bpw", &conv2d_forward, "conv2d_bpw implement");
}