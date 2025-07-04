import os
import sys
# PYBIND11_DIR = os.path.join(os.getcwd(), "python_build")
PYBIND11_DIR = os.path.join(os.getcwd(), "cmake_build")

if PYBIND11_DIR not in sys.path: # 避免重复添加
    sys.path.append(PYBIND11_DIR)
    
    
import type_binding_demo as tbd

def test_basic_types():
    print("=== 基本类型推导测试 ===")
    
    # 基本类型自动推导
    print(f"add_int(5, 3) = {tbd.add_int(5, 3)}")
    a = 5.5
    print(f"add_double(5.5, 3.2) = {tbd.add_double(a, 3.2)}")
    print(f"add_float(5.5, 3.2) = {tbd.add_float(float(5.5), float(3.2))}")
    
    # Python 到 C++ 的自动类型转换
    # print(f"add_int(5.9, 3.1) = {tbd.add_int(5.9, 3.1)}")  # 浮点数会被截断为整数
    print(f"add_double(5, 3) = {tbd.add_double(5, 3)}")     # 整数会被转为浮点数

def test_overloaded_functions():
    print("\n=== 重载函数类型指定测试 ===")
    
    print(f"multiply(5, 3) = {tbd.multiply(5, 3)}")
    # print(f"multiply_double(5.5, 3.2) = {tbd.multiply_double(5.5, 3.2)}")
    print(f"multiply(5.5, 3.2) = {tbd.multiply(5.5, 3.2)}")

def test_template_functions():
    print("\n=== 模板函数实例化测试 ===")
    
    print(f"generic_add_int(10, 20) = {tbd.generic_add_int(10, 20)}")
    print(f"generic_add_double(10.5, 20.3) = {tbd.generic_add_double(10.5, 20.3)}")

def test_custom_types():
    print("\n=== 自定义类型测试 ===")
    
    p1 = tbd.Point(1.0, 2.0)
    p2 = tbd.Point(3.0, 4.0)
    print(f"p1 = {p1}")
    print(f"p2 = {p2}")
    
    result = tbd.add_points(p1, p2)
    print(f"add_points(p1, p2) = {result}")

def test_stl_containers():
    print("\n=== STL 容器类型转换测试 ===")
    
    # list -> vector<int> -> list
    input_list = [1, 2, 3, 4, 5]
    result_list = tbd.process_vector(input_list)
    print(f"process_vector({input_list}) = {result_list}")
    
    # dict -> map<string, int> -> dict
    input_dict = {"apple": 10, "banana": 20, "orange": 30}
    result_dict = tbd.process_map(input_dict)
    print(f"process_map({input_dict}) = {result_dict}")

def test_default_parameters():
    print("\n=== 默认参数测试 ===")
    
    print(f"greet('World') = {tbd.greet('World')}")
    print(f"greet('World', 'Hi') = {tbd.greet('World', 'Hi')}")

def test_return_policies():
    print("\n=== 返回值策略测试 ===")
    
    data = [1, 2, 3, 4, 5]
    holder = tbd.DataHolder(data)
    
    # 不同的返回策略
    ref_data = holder.get_data_ref()
    copy_data = holder.get_data_copy()
    ptr_data = holder.get_data_ptr()
    
    print(f"原始数据: {data}")
    print(f"引用返回: {ref_data}")
    print(f"拷贝返回: {copy_data}")
    print(f"指针返回: {ptr_data}")
    
    # 验证引用和拷贝的区别
    print(f"引用和原始是同一对象: {ref_data is ptr_data}")
    print(f"拷贝和原始是不同对象: {copy_data is not ref_data}")

def test_type_conversion():
    print("\n=== 类型转换测试 ===")
    
    # 测试不同 Python 类型传递给 C++ 函数
    print("=== 传递不同 Python 类型 ===")
    
    # 整数
    print(f"add_double(整数5, 整数3) = {tbd.add_double(5, 3)}")
    
    # 浮点数
    # print(f"add_int(浮点5.7, 浮点3.2) = {tbd.add_int(5.7, 3.2)}")
    
    # 字符串
    # try:
    #     result = tbd.add_int("hello", "world")
    #     print(f"add_int('hello', 'world') = {result}")
    # except TypeError as e:
    #     print(f"add_int('hello', 'world') 类型错误: {e}")
    
    # # 显式类型检查
    # try:
    #     result = tbd.explicit_types(5.7, 3.2, "test")  # 第一个参数不允许转换
    #     print(f"explicit_types 结果: {result}")
    # except TypeError as e:
    #     print(f"explicit_types 类型错误: {e}")

def test_lambda_functions():
    print("\n=== Lambda 函数类型推导测试 ===")
    
    result = tbd.lambda_demo(5, 3.14)
    print(f"lambda_demo(5, 3.14) = {result}")

def show_type_info():
    print("\n=== 类型信息查看 ===")
    
    # 查看函数签名
    print(f"add_int 函数: {tbd.add_int}")
    print(f"add_double 函数: {tbd.add_double}")
    print(f"Point 类: {tbd.Point}")
    
    # 查看帮助信息
    print("\n=== 函数帮助信息 ===")
    help(tbd.add_int)

if __name__ == "__main__":
    try:
        test_basic_types()
        test_overloaded_functions()
        test_template_functions()
        test_custom_types()
        test_stl_containers()
        test_default_parameters()
        test_return_policies()
        test_type_conversion()
        test_lambda_functions()
        show_type_info()
        
        print("\n✅ 所有类型绑定测试完成!")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请先编译模块: python setup.py build_ext --inplace")
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        import traceback
        traceback.print_exc()