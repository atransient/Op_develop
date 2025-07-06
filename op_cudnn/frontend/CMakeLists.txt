# cmake_minimum_required(VERSION 3.17)

# project(cudnn_fe LANGUAGES CXX CUDA)

# add_compile_options(-fPIC)


message(STATUS "Found CUDA at: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "CUDA include directories: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA link libraries: ${CUDA_LIBRARIES}")

message(STATUS "fe_utils path: ${CMAKE_CURRENT_SOURCE_DIR}")


set(pubutils_path "/home/wenyi.huang/Project/myself/Op_develop/utils")

set(tensor_path "/home/wenyi.huang/Project/myself/Op_develop/utils")

FetchContent_Declare(
  cudnnfe # 内部引用名
  GIT_REPOSITORY https://github.com/NVIDIA/cudnn-frontend.git
#   GIT_BRANCH main
  GIT_TAG v1.10.0
  GIT_PROGRESS TRUE
  # GIT_SUBMODULES ALL # 如果这个库自身也有 submodule，加上这个
)

FetchContent_GetProperties(cudnnfe)
if(NOT cudnnfe_POPULATED)
  FetchContent_Populate(cudnnfe)
endif()
# message(STATUS "cudnnfe_SOURCE_DIR:  ${cudnnfe_SOURCE_DIR}")
set(fesrc_path "${cudnnfe_SOURCE_DIR}/include")
message(STATUS "fesrc_path  tag:  ${fesrc_path}")
# FetchContent_MakeAvailable(cudnn_fe)


add_subdirectory(fe_utils)
add_library(gemm_impl SHARED 
    gemm_impl.cpp
    gemm_tensor_impl.cpp
)

target_include_directories(
    gemm_impl PUBLIC
    ${fesrc_path}
    ${pubutils_path}
    ${CUDA_INCLUDE_DIRS}
)

target_link_directories(
    gemm_impl PUBLIC
    ${cu_path}
)

target_link_libraries(
    gemm_impl PUBLIC
    fe_utils
    ${CUDA_LIBRARIES}
    Tensor
)

target_include_directories(
    gemm_impl INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(gemm_impl PRIVATE -g)


# add_executable(gemm_tensor_impl gemm_tensor_impl.cpp)
# target_include_directories(
#     gemm_tensor_impl PUBLIC
#     ${fesrc_path}
#     ${pubutils_path}
#     ${CUDA_INCLUDE_DIRS}
# )
# target_link_directories(
#     gemm_tensor_impl PUBLIC
#     ${cu_path}
# )
# target_link_libraries(
#     gemm_tensor_impl PUBLIC
#     fe_utils
#     ${CUDA_LIBRARIES}
#     Tensor
# )
# target_compile_options(gemm_tensor_impl PRIVATE -g)