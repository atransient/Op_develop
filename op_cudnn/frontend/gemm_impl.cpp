#include <cudnn_frontend.h>

#include "fe_header.h"
#include "data_generate.h"
#include "graph_build.h"

namespace fe = cudnn_frontend;

void test_run()
{
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));
    gemm_info shape_info = {16,   32,   32,  128};
    auto [graph, A, B, C] = matmul_graph(handle, shape_info);
    
    int64_t a_vol = shape_info.b * shape_info.m * shape_info.k;
    int64_t b_vol = shape_info.b * shape_info.n * shape_info.k;
    int64_t c_vol = shape_info.b * shape_info.m * shape_info.n;
    int64_t workspace_size = graph->get_workspace_size();

    Surface<half> A_gpu(a_vol, false);
    Surface<half> B_gpu(b_vol, false);
    Surface<float> C_gpu(c_vol, false);
    Surface<int8_t> workspace(workspace_size, false);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack;
    variant_pack = {
        {A, A_gpu.devPtr},
        {B, B_gpu.devPtr},
        {C, C_gpu.devPtr}
    };
    graph->execute(handle, variant_pack, workspace.devPtr);
    CUDNN_CHECK(cudnnDestroy(handle));
}

void my_matmul(uint64_t A_ptr, uint64_t B_ptr, uint64_t C_ptr, int64_t b, int64_t m, int64_t n, int64_t k)
{
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));
    gemm_info shape_info = {b,   m,   n,  k};
    auto [graph, A, B, C] = matmul_graph(handle, shape_info);
    int64_t workspace_size = graph->get_workspace_size();
    Surface<int8_t> workspace(workspace_size, false);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack;
    variant_pack = {
        {A, reinterpret_cast<void*>(A_ptr)},
        {B, reinterpret_cast<void*>(B_ptr)},
        {C, reinterpret_cast<void*>(C_ptr)}
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
//     my_matmul(reinterpret_cast<uint64_t>(A_gpu.devPtr), reinterpret_cast<uint64_t>(B_gpu.devPtr), reinterpret_cast<uint64_t>(C_gpu.devPtr), shape_info.b, shape_info.m, shape_info.n, shape_info.k);
// }