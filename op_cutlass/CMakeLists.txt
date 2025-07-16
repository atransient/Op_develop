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
set(cutlass_utils_path "${cutlass_SOURCE_DIR}/tools/util/include")


add_subdirectory(cute2gemm)


pybind11_add_module(cutlass_op module_binding.cpp)

target_link_libraries(
    cutlass_op PRIVATE
    cutlass_gemm_impl
)

target_compile_options(cutlass_op PRIVATE -g)

add_subdirectory(layout_test)
add_subdirectory(search)