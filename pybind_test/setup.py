import os
import sys


PYBIND11_DIR = os.path.join(os.path.dirname(os.getcwd()), "third_party", "pybind11")

if PYBIND11_DIR not in sys.path: # 避免重复添加
    sys.path.append(PYBIND11_DIR)

from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11

file_list = ["type_binding_demo.cpp", "math_functions.cpp"]
# , "math_functions.cpp"

# 定义扩展模块
ext_modules = [
    Pybind11Extension(
        name.split(".")[0],
        sources=[name],
        include_dirs=[
            # 包含 pybind11 头文件路径
            pybind11.get_include(),
        ],
        language='c++'
    ) for name in file_list
]

setup(
    name="binding_demo",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
)