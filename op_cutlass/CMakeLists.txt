add_subdirectory(cute2gemm)

pybind11_add_module(cutlass_op module_binding.cpp)

target_link_libraries(
    cutlass_op PRIVATE
    cutlass_gemm_impl
)

target_compile_options(cutlass_op PRIVATE -g)