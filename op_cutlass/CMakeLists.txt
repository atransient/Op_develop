add_subdirectory(cute2gemm)

pybind11_add_module(cutlass_op module_binding.cpp)

target_link_libraries(
    cutlass_op PRIVATE
    cute2gemm
)