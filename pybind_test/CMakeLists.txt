cmake_minimum_required(VERSION 3.12)
project(math_functions)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CURRENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
cmake_path(GET CURRENT_DIR PARENT_PATH PARENT_DIR)

# # 添加本地 pybind11 子目录
# set(PYBIND11_SOURCE_DIR "${PARENT_DIR}/third_party/pybind11")
# set(PYBIND11_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/pybind11_build")

# add_subdirectory(${PYBIND11_SOURCE_DIR} ${PYBIND11_BINARY_DIR})

# # 查找 Python
# find_package(Python COMPONENTS Interpreter Development REQUIRED)

# 创建 pybind11 模块
pybind11_add_module(math_functions math_functions.cpp)

# 编译器特定设置
target_compile_definitions(math_functions PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

# 设置输出目录（可选）
set_target_properties(math_functions PROPERTIES
    OUTPUT_NAME "math_functions"
)


# 创建 pybind11 模块
pybind11_add_module(type_binding_demo type_binding_demo.cpp)

# 编译器特定设置
target_compile_definitions(type_binding_demo PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

# 设置输出目录（可选）
set_target_properties(type_binding_demo PROPERTIES
    OUTPUT_NAME "type_binding_demo"
)