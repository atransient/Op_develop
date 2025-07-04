from MyTensor import TensorDataType_t, MyTensor
from cudnn_op import my_matmul_tensor, my_matmul
import torch

device = "cuda"

A_dtype = TensorDataType_t.FLOAT
A_dim = [8,20,10]
A_val = torch.randn(8,20,10,device=device)
A_tensor = MyTensor(A_val.data_ptr(), A_dim, A_dtype)

B_dtype = TensorDataType_t.FLOAT
B_dim = [8,10,5]
B_val = torch.randn(8,10,5,device=device)
B_tensor = MyTensor(B_val.data_ptr(), B_dim, B_dtype)

# C_dtype = TensorDataType_t.FLOAT
# C_dim = [8,20,5]
# C_val = torch.randn(8,20,5,device=device)
# C_tensor = MyTensor(C_val.data_ptr(), C_dim, C_dtype)

C_torch = torch.matmul(A_val, B_val)

C_cudnn = torch.zeros_like(C_torch)
C_dim = [8,20,5]
C_dtype = TensorDataType_t.FLOAT
C_tensor = MyTensor(C_cudnn.data_ptr(), C_dim, C_dtype)

my_matmul_tensor(A_tensor, B_tensor, C_tensor)

C_cudnn1 = torch.zeros_like(C_torch)
my_matmul(A_val.data_ptr(), B_val.data_ptr(), C_cudnn1.data_ptr(), 8, 20, 5, 10)
# (A_val.data_ptr(), B_val.data_ptr(), C_cudnn1.data_ptr(), 8, 20, 5 10)
print(C_torch[0,0])