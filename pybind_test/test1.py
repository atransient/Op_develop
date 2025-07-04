#!/usr/bin/env python3
import math_functions

def test_functions():
    print("=== 测试基本函数 ===")
    
    # 测试加法
    result = math_functions.add(5, 3)
    print(f"add(5, 3) = {result}")
    
    # 测试乘法
    result = math_functions.multiply_t(2.5, 4.0)
    print(f"multiply(2.5, 4.0) = {result}")
    
    # 测试字符串处理
    greeting = math_functions.greet_t("World")
    print(f"greet('World') = {greeting}")
    
    # 测试列表处理
    input_list = [1, 2, 3, 4, 5]
    result_list = math_functions.process_list(input_list)
    print(f"process_list({input_list}) = {result_list}")

def test_calculator():
    print("\n=== 测试 Calculator 类 ===")
    
    # 创建计算器实例
    calc = math_functions.Calculator(10.0)
    print(f"初始值: {calc.get_value()}")
    
    # 执行一些操作
    calc.add(5.0)
    print(f"加 5 后: {calc.get_value()}")
    
    calc.multiply(2.0)
    print(f"乘以 2 后: {calc.get_value()}")
    
    calc.reset()
    print(f"重置后: {calc.get_value()}")

if __name__ == "__main__":
    try:
        test_functions()
        test_calculator()
        print("\n✅ 所有测试通过!")
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请先编译模块: python setup.py build_ext --inplace")
    except Exception as e:
        print(f"❌ 运行错误: {e}")