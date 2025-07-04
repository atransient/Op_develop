#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <map>

// 1. 基本类型推导
int add_int(int a, int b) { return a + b; }
double add_double(double a, double b) {return a + b; }
float add_float(float a, float b) { return a + b; }

// 2. 重载函数 - 需要显式指定类型
int multiply(int a, int b) { return a * b; }
double multiply(double a, double b) { return a * b; }

// 3. 模板函数 - 需要显式实例化
template<typename T>
T generic_add(T a, T b) {
    return a + b;
}

// 4. 复杂类型参数
struct Point {
    double x, y;
    Point(double x = 0, double y = 0) : x(x), y(y) {}
};

Point add_points(const Point& p1, const Point& p2) {
    return Point(p1.x + p2.x, p1.y + p2.y);
}

// 5. STL 容器类型
// std::vector<int> process_vector(const std::vector<int>& input) {
//     std::vector<int> result;
//     for (int val : input) {
//         result.push_back(val * 2);
//     }
//     input[0] += 10;
//     return result;
// }
std::vector<int> process_vector(const std::vector<int>& input) {
    std::vector<int> result;
    for (int val : input) {
        result.push_back(val * 2);
    }
    return result;
}

std::map<std::string, int> process_map(const std::map<std::string, int>& input) {
    std::map<std::string, int> result;
    for (const auto& pair : input) {
        result[pair.first] = pair.second * 2;
    }
    return result;
}

// 6. 可选参数和默认值
std::string greet(const std::string& name, const std::string& prefix = "Hello") {
    return prefix + ", " + name + "!";
}

// 7. 返回值策略演示
class DataHolder {
private:
    std::vector<int> data_;
public:
    DataHolder(const std::vector<int>& data) : data_(data) {}
    
    // 返回引用 - 需要指定返回值策略
    const std::vector<int>& get_data_ref() const { return data_; }
    
    // 返回拷贝
    std::vector<int> get_data_copy() const { return data_; }
    
    // 返回指针
    const std::vector<int>* get_data_ptr() const { return &data_; }
};

// 8. 函数指针类型
typedef int (*BinaryOp)(int, int);

int apply_operation(int a, int b, BinaryOp op) {
    return op(a, b);
}

int subtract(int a, int b) { return a - b; }

PYBIND11_MODULE(type_binding_demo, m) {
    m.doc() = "pybind11 类型绑定机制演示";
    
    // 1. 基本类型 - 自动推导
    m.def("add_int", &add_int, "自动推导 int 类型");
    m.def("add_double", &add_double, "自动推导 double 类型");
    m.def("add_float", &add_float, "自动推导 float 类型");
    
    // 2. 重载函数 - 需要显式类型转换
    m.def("multiply", static_cast<int(*)(int, int)>(&multiply), "int 版本的 multiply");
    // m.def("multiply_double", static_cast<double(*)(double, double)>(&multiply), "double 版本的 multiply");
    m.def("multiply", static_cast<double(*)(double, double)>(&multiply), "double 版本的 multiply");
    
    // 3. 模板函数 - 显式实例化
    m.def("generic_add_int", &generic_add<int>, "模板函数 int 实例");
    m.def("generic_add_double", &generic_add<double>, "模板函数 double 实例");
    
    // 4. 自定义类型
    pybind11::class_<Point>(m, "Point")
        .def(pybind11::init<double, double>())
        .def(pybind11::init<>())  // 默认构造函数
        .def_readwrite("x", &Point::x)
        .def_readwrite("y", &Point::y)
        .def("__repr__", [](const Point& p) {
            return "Point(" + std::to_string(p.x) + ", " + std::to_string(p.y) + ")";
        });
    
    m.def("add_points", &add_points, "Point 类型推导");
    
    // 5. STL 容器 - 需要包含 pybind11/stl.h
    m.def("process_vector", &process_vector, "vector<int> 自动转换");
    m.def("process_map", &process_map, "map<string, int> 自动转换");
    
    // 6. 默认参数
    m.def("greet", &greet, "带默认参数的函数",
          pybind11::arg("name"), pybind11::arg("prefix") = "Hello");
    
    // 7. 返回值策略
    pybind11::class_<DataHolder>(m, "DataHolder")
        .def(pybind11::init<const std::vector<int>&>())
        .def("get_data_ref", &DataHolder::get_data_ref, 
             pybind11::return_value_policy::reference_internal)  // 返回引用
        .def("get_data_copy", &DataHolder::get_data_copy)       // 返回拷贝（默认）
        .def("get_data_ptr", &DataHolder::get_data_ptr,
             pybind11::return_value_policy::reference_internal); // 返回指针
    
    // 8. 函数指针 - 复杂类型推导
    m.def("apply_operation", &apply_operation, "函数指针参数");
    m.def("subtract", &subtract, "减法函数");
    
    // 9. Lambda 函数 - 编译时类型推导
    m.def("lambda_demo", [](int x, double y) -> double {
        return x * y;
    }, "Lambda 函数类型推导");
    
    // 10. 显式参数类型说明（文档用）
    m.def("explicit_types", [](int a, double b, const std::string& c) -> std::string {
        return "a=" + std::to_string(a) + ", b=" + std::to_string(b) + ", c=" + c;
    }, "显式类型演示", 
       pybind11::arg("a").noconvert(),  // 不允许类型转换
       pybind11::arg("b"), 
       pybind11::arg("c"));
}