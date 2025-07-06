add_library(
    be_utils STATIC
    configer.cpp
)


target_include_directories(
    be_utils INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)


target_link_libraries(
    be_utils PUBLIC 
    be_build_config 
    Tensor
    ${CUDA_LIBRARIES}
)


target_compile_options(be_utils PRIVATE -g)