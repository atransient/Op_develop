#include <cudnn_frontend.h>
#include <iostream>

#include "graph_build.h"

namespace fe = cudnn_frontend;


fe::DataType_t type_switch(TensorDataType_t type)
{
    switch (type)
    {
        case TensorDataType_t::FLOAT:
            return fe::DataType_t::FLOAT;
        case TensorDataType_t::DOUBLE:
            return fe::DataType_t::DOUBLE;
        case TensorDataType_t::HALF:
            return fe::DataType_t::HALF;
        case TensorDataType_t::INT32:
            return fe::DataType_t::INT32;
        case TensorDataType_t::BFLOAT16:
            return fe::DataType_t::BFLOAT16;
        default:
            return fe::DataType_t::FLOAT;
    }
}

std::tuple<std::shared_ptr<cudnn_frontend::graph::Graph>, std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>
matmul_graph(cudnnHandle_t handle, const gemm_info& shape_info, TensorDataType_t iodtype, TensorDataType_t cdtype)
{
    auto graph = std::make_shared<fe::graph::Graph>();
    auto A_attributes = fe::graph::Tensor_attributes()
                                .set_name("A")
                                .set_dim({shape_info.b, shape_info.m, shape_info.k})
                                .set_stride({shape_info.m * shape_info.k, shape_info.k, 1})
                                .set_data_type(type_switch(iodtype));
    auto A = graph->tensor(A_attributes);
    auto B_attributes = fe::graph::Tensor_attributes()
                                .set_name("B")
                                .set_dim({shape_info.b, shape_info.k, shape_info.n})
                                .set_stride({shape_info.k * shape_info.n, shape_info.n, 1})
                                .set_data_type(type_switch(iodtype));
    auto B = graph->tensor(B_attributes);
    auto matmul_attributes = fe::graph::Matmul_attributes().set_compute_data_type(type_switch(cdtype));
    
    std::shared_ptr<fe::graph::Tensor_attributes> C;
    C = graph->matmul(A, B, matmul_attributes);
    C->set_output(true).set_data_type(type_switch(iodtype));
    auto status = graph->validate();
    status = graph->build_operation_graph(handle);
    graph->create_execution_plans({fe::HeurMode_t::A});
    graph->check_support(handle);
    graph->build_plans(handle, fe::BuildPlanPolicy_t::ALL);
    return std::make_tuple(graph, A, B, C);
}