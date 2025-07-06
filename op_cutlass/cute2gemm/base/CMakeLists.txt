
add_library(
    batch_gemm STATIC
    batch_gemm.cu
)

target_include_directories(
    batch_gemm PUBLIC
    ${cutlass_path}
    ${CUDA_INCLUDE_DIRS}
)

target_link_directories(
    batch_gemm PUBLIC
    ${cu_path}
)

target_link_libraries(
    batch_gemm PUBLIC
    ${CUDA_LIBRARIES}
    Tensor
)

target_include_directories(
    batch_gemm INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(batch_gemm PRIVATE --generate-code=arch=compute_90a,code=[sm_90a] --expt-relaxed-constexpr -g)