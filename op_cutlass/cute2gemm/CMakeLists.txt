# cmake_minimum_required(VERSION 3.17)

# project(cudnn_fe LANGUAGES CXX CUDA)

# add_compile_options(-fPIC)

# find_package(CUDA REQUIRED)
# set(cu_path "${CUDA_TOOLKIT_ROOT_DIR}/include" "${CUDA_TOOLKIT_ROOT_DIR}/lib64")

# include(FetchContent)


FetchContent_Declare(
  cutlass # 内部引用名
  GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
#   GIT_BRANCH main
  GIT_TAG v3.9.0
  GIT_PROGRESS TRUE
  # GIT_SUBMODULES ALL # 如果这个库自身也有 submodule，加上这个
)

message(STATUS "start git clone cutlass library     ")
FetchContent_GetProperties(cutlass)
if(NOT cutlass_POPULATED)
  FetchContent_Populate(cutlass)
endif()
set(cutlass_path "${cutlass_SOURCE_DIR}/include")


add_subdirectory(base)

add_library(cutlass_gemm_impl SHARED 
    gemm_impl.cpp
)

target_link_directories(
    cutlass_gemm_impl PUBLIC
    ${cu_path}
)

target_link_libraries(
    cutlass_gemm_impl PUBLIC
    batch_gemm
    ${CUDA_LIBRARIES}
    Tensor
)

target_include_directories(
    cutlass_gemm_impl INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(cutlass_gemm_impl PRIVATE -g)