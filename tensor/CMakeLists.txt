cmake_minimum_required(VERSION 3.17)

project(MyTensor LANGUAGES CXX CUDA)

add_compile_options(-fPIC)

set(CURRENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
cmake_path(GET CURRENT_DIR PARENT_PATH PARENT_DIR)

# set(PYBIND11_SOURCE_DIR "${PARENT_DIR}/third_party/pybind11")
# set(PYBIND11_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/pybind11_build")

# add_subdirectory(${PYBIND11_SOURCE_DIR} ${PYBIND11_BINARY_DIR})



add_subdirectory(impl)

pybind11_add_module(MyTensor module_binding.cpp)

target_link_libraries(
    MyTensor PUBLIC
    Tensor
)

target_include_directories(
    MyTensor PUBLIC
    ${utils_path}
)