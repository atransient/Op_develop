#include <cudnn_frontend.h>
#include <iostream>
#include "tensor_impl.h"
namespace fe = cudnn_frontend;

struct gemm_info
{
    int64_t b;
    int64_t m;
    int64_t n;
    int64_t k;
};

std::tuple<std::shared_ptr<cudnn_frontend::graph::Graph>, std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>
matmul_graph(cudnnHandle_t handle, const gemm_info& shape_info, TensorDataType_t iodtype = TensorDataType_t::FLOAT, TensorDataType_t cdtype = TensorDataType_t::FLOAT);