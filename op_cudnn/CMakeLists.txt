# cmake_minimum_required(VERSION 3.17)

# project(cudnn_fe LANGUAGES CXX CUDA)

# add_compile_options(-fPIC)

set(CURRENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
cmake_path(GET CURRENT_DIR PARENT_PATH PARENT_DIR)


# set(PYBIND11_SOURCE_DIR "${PARENT_DIR}/third_party/pybind11")
# set(PYBIND11_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/pybind11_build")

# add_subdirectory(${PYBIND11_SOURCE_DIR} ${PYBIND11_BINARY_DIR})

add_subdirectory(frontend)
add_subdirectory(backend)

pybind11_add_module(cudnn_op module_binding.cpp)


target_link_libraries(
    cudnn_op PUBLIC
    gemm_impl
    CONV_BE
)
