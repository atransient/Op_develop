## about project
A operator develop architecture, support develop operator by cudnn backend, cudnn frontend, cutlass. 
We also can compare the accuracy of our operator with pytorch, because this architecture can pack our c++, cuda operator into a python api.
It also implement a base tensor class to pass data info through python and c++, and we can extract the pytorch tensor info into the self implement tensor.

## project file structure
  1. cudnn_op  : develop operator by cudnn
     - cmakelists.txt
     - module_binding.cpp  : binding cpp function into python through pybind11
     - backend : operator that develop by cudnn backend
       - be_utils : use to set cudnn api info, such as tensor, convolution
       - cmakelists.txt
       - be_fwd.cpp : implement convolution forward
       - be_bpw.cpp : implement convolution backward weight
       - be_bpa.cpp : implement convolution backward activation
       - be_header.h : header file
      - frontend
       - fe_utils: the same as be_utils
       - cmakelists.txt
       - gemm_tensor_impl.cpp : implement batch gemm by cudnn frontend
       - fe_header.h : header_file
      
  2. cutlass_op : develop operator by cutlass
     - cmakelists.txt
     - module_binding.cpp
     - cute2gemm  : implement batch gemm using cutlass cute
       - gemm_impl.cpp: switch python data type into specify gemm api
       - base: implement batch gemm by cute, supporting half, bfloat16, fused precision
      
  3. tensor : contain data info, transfer data info through python and c++
     - impl: implement base tensor class, use for containing data_ptr, tensor shape info, data_type
    
  4. utils
     - data_generate.h  : use to generate rand data, support bfloat16, half, float, int8 and so on, this class contain host and device value address
    
  5. verify : compare implement operator with pytorch
     - py_utils : contain .so file path setting, and switch pytorch tensor into our base tensor
     - matmul_cutlass.py : compare batch gemm accuracy with pytorch
    
## usage
```
git clone
cd Op_develop
mkdir build && cd build && cmake .. && make -j
cd ../verify   and run verify demo
