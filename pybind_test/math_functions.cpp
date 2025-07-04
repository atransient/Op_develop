#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>

// 简单的数学函数
int add(int a, int b) {
    return a + b;
}

double multiply_t(double a, double b) {
    return a * b;
}

// 处理字符串
std::string greet(const std::string& name) {
    return "Hello, " + name + "!";
}

// 处理向量
std::vector<int> process_list(const std::vector<int>& input) {
    std::vector<int> result;
    for (int val : input) {
        result.push_back(val * 2);
    }
    return result;
}

// 简单的类示例
class Calculator {
public:
    Calculator(double initial_value) : value_(initial_value) {}
    
    void add(double x) { value_ += x; }
    void multiply(double x) { value_ *= x; }
    double get_value() const { return value_; }
    void reset() { value_ = 0.0; }
    
private:
    double value_;
};

// pybind11 模块定义
PYBIND11_MODULE(math_functions, m) {
    m.doc() = "pybind11 example plugin";
    
    // 导出函数
    m.def("add", &add, "A function that adds two numbers");
    m.def("multiply_t", &multiply_t, "A function that multiplies two numbers");
    m.def("greet_t", &greet, "A function that greets someone");
    m.def("process_list", &process_list, "A function that doubles all elements in a list");
    
    // 导出类
    pybind11::class_<Calculator>(m, "Calculator")
        .def(pybind11::init<double>())
        .def("add", &Calculator::add)
        .def("multiply", &Calculator::multiply)
        .def("get_value", &Calculator::get_value)
        .def("reset", &Calculator::reset);
}