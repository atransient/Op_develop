from cudnn_op import my_matmul

import torch
import torch.nn as nn

device = "cuda"

mA = torch.randn(3,4,5, device=device)
mB = torch.randn(3,5,10, device=device)

mC = torch.matmul(mA, mB)

cudnn_C = torch.zeros_like(mC)

my_matmul(mA.data_ptr(), mB.data_ptr(), cudnn_C.data_ptr(), 3, 4, 10, 5)
print(torch.allclose(mC, cudnn_C,1e-1))