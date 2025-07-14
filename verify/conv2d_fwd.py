from py_utils.add_so import add_sopath

add_sopath()

import torch
import torch.nn as nn
from py_utils.tensor_exchange import torch2self

from MyTensor import MyTensor
from cudnn_op import conv2d_forward

N, C, H, W = 1,64, 16, 16
K, C_, R, S = 64, 1, 3, 3
padding = 0      # 填充
stride = 1                   # 步幅
dilation = 1                 # 膨胀系数
groups = 64 
dtype = torch.bfloat16
device = "cuda"

input = torch.randn((N, C, H, W), dtype=dtype, device=device)
weight = torch.randn((K, C_, R, S), dtype=dtype, device=device)

conv2d_layer = nn.Conv2d(in_channels=C, out_channels=K, kernel_size=(R, S), 
                             padding=padding, stride=stride, 
                             dilation=dilation, groups=groups, bias=False, dtype=dtype, device=device)

conv2d_layer.weight.data = weight
torchres = conv2d_layer(input)

cudnnres = torch.zeros_like(torchres)

mytensor_res = torch2self(cudnnres)
mytensor_activation = torch2self(input)
mytensor_weight = torch2self(weight)
conv2d_forward(mytensor_activation, mytensor_weight, mytensor_res, padding, stride, dilation, groups)

print(torch.allclose(torchres,cudnnres,1e-2))