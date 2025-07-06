from py_utils.add_so import add_sopath

add_sopath()

import torch
import torch.nn as nn
from py_utils.tensor_exchange import torch2self

from MyTensor import MyTensor
from cudnn_op import my_matmul_tensor

dtype = torch.bfloat16
device = "cuda"

A_matrix = torch.randn(8,16,32,dtype=dtype,device=device)
B_matrix = torch.randn(8,32,64,dtype=dtype,device=device)

torch_C = torch.matmul(A_matrix, B_matrix)

cudnn_res = torch.zeros_like(torch_C)
mytensor_A = torch2self(A_matrix)
mytensor_B = torch2self(B_matrix)
mytensor_C = torch2self(cudnn_res)

my_matmul_tensor(mytensor_A, mytensor_B, mytensor_C)

print(torch.allclose(torch_C, cudnn_res))