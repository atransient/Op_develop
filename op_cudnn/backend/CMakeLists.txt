
add_library(be_build_config INTERFACE)
target_compile_features(be_build_config INTERFACE cxx_std_17)

target_compile_options(be_build_config INTERFACE
  "-g"
)

target_link_libraries(
    be_build_config INTERFACE
    cudnn
    cudart
)

target_include_directories(
    be_build_config INTERFACE
    ${CUDA_INCLUDE_DIRS}
)


add_subdirectory(be_utils)

add_library(
        CONV_BE STATIC
        be_fwd.cpp
        be_bpw.cpp
        be_bpa.cpp
    )

target_compile_options(
  CONV_BE PRIVATE -g
)

target_link_libraries(
  CONV_BE PUBLIC
  be_utils
  be_build_config
)


target_include_directories(
    CONV_BE INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
