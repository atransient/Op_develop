#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tensor_impl.h"

using std::vector;
namespace py = pybind11;

PYBIND11_MODULE(MyTensor, m)
{
    m.doc() = "pybind11 tensor implement";

    pybind11::enum_<TensorDataType_t>(m, "TensorDataType_t")
        .value("FLOAT", TensorDataType_t::FLOAT)
        .value("DOUBLE", TensorDataType_t::DOUBLE)
        .value("HALF", TensorDataType_t::HALF)
        .value("INT8", TensorDataType_t::INT8)
        .value("INT32", TensorDataType_t::INT32)
        .value("INT8x4", TensorDataType_t::INT8x4)
        .value("UINT8", TensorDataType_t::UINT8)
        .value("UINT8x4", TensorDataType_t::UINT8x4)
        .value("INT8x32", TensorDataType_t::INT8x32)
        .value("BFLOAT16", TensorDataType_t::BFLOAT16)
        .value("INT64", TensorDataType_t::INT64)
        .value("BOOLEAN", TensorDataType_t::BOOLEAN)
        .value("FP8_E4M3", TensorDataType_t::FP8_E4M3)
        .value("FP8_E5M2", TensorDataType_t::FP8_E5M2)
        .value("FAST_FLOAT_FOR_FP8", TensorDataType_t::FAST_FLOAT_FOR_FP8);

    py::class_<MyTensor>(m, "MyTensor")
        .def(py::init<uint64_t,vector<uint64_t>,TensorDataType_t>(),
            py::arg("valaddr"),
            py::arg("valdim"),
            py::arg("valtype") = TensorDataType_t::FLOAT)
        .def("topn_val", &MyTensor::topn_val);
}