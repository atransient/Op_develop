# cmake_minimum_required(VERSION 3.17)

# project(Tensor LANGUAGES CXX CUDA)

# add_compile_options(-fPIC)

# find_package(CUDA REQUIRED)

# set(cu_path "${CUDA_TOOLKIT_ROOT_DIR}/include" "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
find_package(CUDA REQUIRED)

set(cu_path "${CUDA_TOOLKIT_ROOT_DIR}/include" "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
set(utils_path "/home/wenyi.huang/Project/myself/Op_develop/utils")

add_library(Tensor SHARED 
    tensor_impl.cpp
)

target_include_directories(
    Tensor PUBLIC
    ${CUDA_INCLUDE_DIRS}
)

target_link_directories(
    Tensor PUBLIC
    ${cu_path}
)

target_link_libraries(
    Tensor PUBLIC
    ${CUDA_LIBRARIES}
)

target_include_directories(
    Tensor INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_directories(
    Tensor INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)