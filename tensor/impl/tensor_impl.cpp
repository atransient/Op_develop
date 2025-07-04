#include <iostream>
#include <vector>
#include <typeinfo>

#include "tensor_impl.h"
// #include "data_generate.h"

using std::vector;


MyTensor::MyTensor(uint64_t valaddr, vector<uint64_t>valdim, TensorDataType_t valtype):dims(valdim),addr(valaddr),dtype(valtype)
{}



void MyTensor::topn_val(int num)
{
    switch (dtype)
    {
        case TensorDataType_t::FLOAT:
        {
            float * fval_addr = reinterpret_cast<float*>(addr);
            for (int i = 0; i < num; ++i)
            {
                printf("%f\t",fval_addr[i]);
            }
            printf("\n");
            break;
        }

        default:
            break;
    }
}

vector<uint64_t> MyTensor::get_dims()
{
    return dims;
}

uint64_t MyTensor::data_ptr()
{
    return addr;
}

TensorDataType_t MyTensor::data_type()
{
    return dtype;
}