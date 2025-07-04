#include <cudnn_frontend.h>
#include <vector>

#include "fe_header.h"
#include "data_generate.h"
#include "graph_build.h"
#include "tensor_impl.h"

using std::vector;
namespace fe = cudnn_frontend;

void my_matmul_tensor(MyTensor A_tensor, MyTensor B_tensor, MyTensor C_tensor, TensorDataType_t caltype)
{
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    auto A_shape = A_tensor.get_dims();
    auto B_shape = B_tensor.get_dims();
    gemm_info shape_info = {A_shape[0], A_shape[1], B_shape[2], A_shape[2]};
    auto [graph, A, B, C] = matmul_graph(handle, shape_info, A_tensor.data_type(), caltype);
    int64_t workspace_size = graph->get_workspace_size();
    Surface<int8_t> workspace(workspace_size, false);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack;
    variant_pack = {
        {A, reinterpret_cast<void*>(A_tensor.data_ptr())},
        {B, reinterpret_cast<void*>(B_tensor.data_ptr())},
        {C, reinterpret_cast<void*>(C_tensor.data_ptr())}
    };
    graph->execute(handle, variant_pack, workspace.devPtr);
    CUDNN_CHECK(cudnnDestroy(handle));
}


// int main()
// {
//     gemm_info shape_info = {16,   32,   32,  128};
//     int64_t a_vol = shape_info.b * shape_info.m * shape_info.k;
//     int64_t b_vol = shape_info.b * shape_info.n * shape_info.k;
//     int64_t c_vol = shape_info.b * shape_info.m * shape_info.n;
//     Surface<half> A_gpu(a_vol, false);
//     Surface<half> B_gpu(b_vol, false);
//     Surface<float> C_gpu(c_vol, false);

//     vector<uint64_t> A_dim = {shape_info.b, shape_info.m, shape_info.k};
//     vector<uint64_t> B_dim = {shape_info.b, shape_info.n, shape_info.k};
//     vector<uint64_t> C_dim = {shape_info.b, shape_info.m, shape_info.n};
//     MyTensor A_tensor(reinterpret_cast<uint64_t>(A_gpu.devPtr), A_dim);
//     MyTensor B_tensor(reinterpret_cast<uint64_t>(B_gpu.devPtr), B_dim);
//     MyTensor C_tensor(reinterpret_cast<uint64_t>(C_gpu.devPtr), C_dim);

//     my_matmul_tensor(A_tensor, B_tensor, C_tensor);
// }