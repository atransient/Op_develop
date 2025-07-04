add_library(
    fe_utils STATIC
    graph_build.cpp
)

message(STATUS "fesrc_path link libraries: ${fesrc_path}")
message(STATUS "CUDA_LIBRARIES link libraries: ${CUDA_LIBRARIES}")
target_include_directories(
    fe_utils PUBLIC
    ${fesrc_path}
    ${CUDA_INCLUDE_DIRS}
)

target_link_directories(
    fe_utils PUBLIC
    ${cu_path}
)

target_link_libraries(
    fe_utils PUBLIC
    ${CUDA_LIBRARIES}
    cudnn
    Tensor
)

target_include_directories(
    fe_utils INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)