#ifndef TENSOR_IMPL_H
#define TENSOR_IMPL_H

#include <iostream>
#include <vector>

using std::vector;

enum class TensorDataType_t {
    FLOAT,
    DOUBLE,
    HALF,
    INT8,
    INT32,
    INT8x4,
    UINT8,
    UINT8x4,
    INT8x32,
    BFLOAT16,
    INT64,
    BOOLEAN,
    FP8_E4M3,
    FP8_E5M2,
    FAST_FLOAT_FOR_FP8
};

class MyTensor
{
public:
    MyTensor(uint64_t valaddr, vector<uint64_t>valdim, TensorDataType_t valtype=TensorDataType_t::FLOAT);

    void topn_val(int num);

    vector<uint64_t> get_dims();

    uint64_t data_ptr();

    TensorDataType_t data_type();

private:
    vector<uint64_t> dims;
    uint64_t addr;
    TensorDataType_t dtype;
};

#endif