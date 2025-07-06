from MyTensor import TensorDataType_t, MyTensor

import torch

def tensor_type_mapping(dtype):
    if dtype == torch.float32:
        return TensorDataType_t.FLOAT
    if dtype == torch.half:
        return TensorDataType_t.HALF
    if dtype == torch.bfloat16:
        return TensorDataType_t.BFLOAT16
    if dtype == torch.double:
        return TensorDataType_t.DOUBLE
    return TensorDataType_t.FLOAT 

def torch2self(pytensor):
    dim_val = list(pytensor.shape)
    dtype = tensor_type_mapping(pytensor.dtype)
    return MyTensor(pytensor.data_ptr(), dim_val, dtype)
