# cmake_minimum_required(VERSION 3.12)

# project(Op_dev)
# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

cmake_minimum_required(VERSION 3.17)

project(Op_dev LANGUAGES CXX CUDA)

add_compile_options(-fPIC)

set(TOP_LEVEL_BUILD_DIR ${CMAKE_BINARY_DIR})
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TOP_LEVEL_BUILD_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${TOP_LEVEL_BUILD_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${TOP_LEVEL_BUILD_DIR}/lib)


find_package(CUDA REQUIRED)
set(cu_path "${CUDA_TOOLKIT_ROOT_DIR}/include" "${CUDA_TOOLKIT_ROOT_DIR}/lib64")

include(FetchContent)
set(FETCHCONTENT_QUIET FALSE)
# set(FETCHCONTENT_BASE_DIR ${CMAKE_SOURCE_DIR}/third_party)
set(FETCHCONTENT_BASE_DIR "${CMAKE_BINARY_DIR}/third_party")

set(FETCHCONTENT_UPDATES_DISCONNECTED TRUE CACHE BOOL "Do not attempt to update FetchContent content if disconnected from network.")

FetchContent_Declare(
  pybind11 # 内部引用名
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  # GIT_BRANCH master
  GIT_TAG v2.13.2
  GIT_PROGRESS TRUE
  # GIT_SUBMODULES ALL # 如果这个库自身也有 submodule，加上这个
)
FetchContent_MakeAvailable(pybind11)



add_subdirectory(pybind_test)
add_subdirectory(tensor)
add_subdirectory(op_cudnn)
add_subdirectory(op_cutlass)
add_subdirectory(op_cuda)