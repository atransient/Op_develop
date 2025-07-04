#include <cudnn_frontend.h>
#include "data_generate.h"
namespace fe = cudnn_frontend;
void
matmul_dynamic_shapes(bool use_abs = false, bool use_bias = false) {

    // clang-format off
    struct {
        int64_t b,    m,    n,    k;
    } matmul_shapes[] = {
        {      16,   32,   32,  128},
        // {      16,   64,   64,  128},
        // {      16,   80,   80,  128},
        // {      32,  128,  128,  256},
        // {      32,   64,   64,  256},
    };
    // clang-format on

    constexpr int matmul_shapes_count = sizeof(matmul_shapes) / sizeof(matmul_shapes[0]);
    int64_t max_a_volume = 0, max_b_volume = 0, max_c_volume = 0, max_bias_volume = 0;
    for (int idx_shape = 0; idx_shape < matmul_shapes_count; ++idx_shape) {
        const auto& matmul_shape = matmul_shapes[idx_shape];
        max_a_volume             = std::max(max_a_volume, matmul_shape.b * matmul_shape.m * matmul_shape.k);
        max_b_volume             = std::max(max_b_volume, matmul_shape.b * matmul_shape.k * matmul_shape.n);
        max_c_volume             = std::max(max_c_volume, matmul_shape.b * matmul_shape.m * matmul_shape.n);
        max_bias_volume          = std::max(max_bias_volume, matmul_shape.b * matmul_shape.m);
    }

    auto kernel_cache = std::make_shared<fe::KernelCache>();

    const auto build_new_graph = [&matmul_shapes, &kernel_cache, &use_abs, &use_bias](cudnnHandle_t handle,
                                                                                      int idx_shape) {
        const auto& matmul_shape = matmul_shapes[idx_shape];

        // Make cudnn graph
        fe::graph::Graph graph{};

        // graph.set_dynamic_shape_enabled(true).set_kernel_cache(kernel_cache);

        // Create the two non-virtual input tensors A and B.
        // There are read from global memory.
        auto A_attributes = fe::graph::Tensor_attributes()
                                .set_name("A")
                                .set_dim({matmul_shape.b, matmul_shape.m, matmul_shape.k})
                                .set_stride({matmul_shape.m * matmul_shape.k, matmul_shape.k, 1})
                                .set_data_type(fe::DataType_t::BFLOAT16);
        auto A = graph.tensor(A_attributes);

        auto B_attributes = fe::graph::Tensor_attributes()
                                .set_name("B")
                                .set_dim({matmul_shape.b, matmul_shape.k, matmul_shape.n})
                                .set_stride({matmul_shape.k * matmul_shape.n, matmul_shape.n, 1})
                                .set_data_type(fe::DataType_t::BFLOAT16);
        auto B = graph.tensor(B_attributes);

        auto matmul_attributes = fe::graph::Matmul_attributes().set_compute_data_type(fe::DataType_t::FLOAT);

        std::shared_ptr<fe::graph::Tensor_attributes> C;
        std::shared_ptr<fe::graph::Tensor_attributes> Bias;

        if (use_abs) {
            // Add abs operation
            auto pw_0_attributes = fe::graph::Pointwise_attributes()
                                       .set_name("pw0_Abs")
                                       .set_mode(fe::PointwiseMode_t::ABS)
                                       .set_compute_data_type(fe::DataType_t::FLOAT);

            auto A_after_pw_0 = graph.pointwise(A, pw_0_attributes);
            A_after_pw_0->set_data_type(fe::DataType_t::BFLOAT16);

            C = graph.matmul(A_after_pw_0, B, matmul_attributes);
        } else if (use_bias) {
            // Create Bias vector
            auto Bias_attributes = fe::graph::Tensor_attributes()
                                       .set_name("Bias")
                                       .set_dim({matmul_shape.b, matmul_shape.m, 1})
                                       .set_stride({matmul_shape.m, 1, 1})
                                       .set_data_type(fe::DataType_t::BFLOAT16);
            Bias = graph.tensor(Bias_attributes);

            // Add ADD operation
            auto pw_0_attributes = fe::graph::Pointwise_attributes()
                                       .set_name("pw0_Add")
                                       .set_mode(fe::PointwiseMode_t::ADD)
                                       .set_compute_data_type(fe::DataType_t::FLOAT);

            auto A_after_pw_0 = graph.pointwise(A, Bias, pw_0_attributes);
            A_after_pw_0->set_data_type(fe::DataType_t::BFLOAT16);

            C = graph.matmul(A_after_pw_0, B, matmul_attributes);
        } else {
            C = graph.matmul(A, B, matmul_attributes);
        }
        C->set_output(true).set_data_type(fe::DataType_t::FLOAT);

        std::cout << graph << std::endl;
        auto status = graph.validate();
        

        status = graph.build_operation_graph(handle);

        graph.create_execution_plans({fe::HeurMode_t::A}).is_good();

        graph.check_support(handle).is_good();

        graph.build_plans(handle, fe::BuildPlanPolicy_t::ALL).is_good();

        return std::make_tuple(graph, A, B, C, Bias);
    };

    // Run cudnn graph
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    for (int idx_shape = 0; idx_shape < matmul_shapes_count; idx_shape++) {
        auto [graph, A, B, C, Bias] = build_new_graph(handle, idx_shape);
        int64_t wpsize = graph.get_workspace_size();
        // Initialize input tensors
        Surface<half> A_gpu(max_a_volume, false);
        Surface<half> B_gpu(max_b_volume, false);
        Surface<float> C_gpu(max_c_volume, false);
        Surface<half> Bias_gpu(max_bias_volume, false);
        Surface<int8_t> workspace(wpsize, false);

        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack;
        if (use_bias) {
            variant_pack = {{A, A_gpu.devPtr}, {B, B_gpu.devPtr}, {C, C_gpu.devPtr}, {Bias, Bias_gpu.devPtr}};
        } else {
            variant_pack = {{A, A_gpu.devPtr}, {B, B_gpu.devPtr}, {C, C_gpu.devPtr}};
        }
        graph.execute(handle, variant_pack, workspace.devPtr).is_good();
    }

    CUDNN_CHECK(cudnnDestroy(handle));
}
int main()
{


    matmul_dynamic_shapes(false, false);
}