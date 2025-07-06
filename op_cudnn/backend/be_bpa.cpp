#include <iostream>
#include <cudnn.h>
#include <vector>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "configer.h"
#include "tensor_impl.h"

using std::vector;


void bpa_be_devicec(cudnnHandle_t handle, vector<uint>v_info, uint64_t dY_ptr, uint64_t W_ptr, uint64_t dX_ptr, cudnnDataType_t iodtype, cudnnDataType_t cdtype) {
    uint n = v_info[0], c = v_info[1], h = v_info[2], w = v_info[3];
    uint k = v_info[4], r = v_info[5], s = v_info[6];
    uint stride = v_info[7], dilation = v_info[8], padding = v_info[9], groups = v_info[10];

    // 1. Initialize tensor descriptors for input, filter, and output
    cudnnTensorDescriptor_t dinput_desc, grad_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    
    auto [oup_size, workspace_size, algo] = be_bpa_desc(handle, filter_desc, grad_desc, dinput_desc, conv_desc, {n,c,h,w,k,r,s,stride,dilation,padding,groups}, iodtype, cdtype);
   

    void *d_W = reinterpret_cast<void *>(W_ptr);
    void *d_dY = reinterpret_cast<void *>(dY_ptr);
    void *d_dX = reinterpret_cast<void *>(dX_ptr);


    int8_t *d_workspace = nullptr;
    cudaMalloc((void**)&d_workspace, workspace_size);
    // 5. Perform convolution (depthwise)
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN_ERR(cudnnConvolutionBackwardData(handle, &alpha, filter_desc, d_W, grad_desc, d_dY, conv_desc, algo, d_workspace, workspace_size, &beta, dinput_desc, d_dX));


    // 6. Free allocated memory
    cudaFree(d_workspace);

    // 7. Destroy cuDNN descriptors
    cudnnDestroyTensorDescriptor(dinput_desc);
    cudnnDestroyTensorDescriptor(grad_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
}

void conv2d_bpa(MyTensor dY, MyTensor Weight, MyTensor dX, uint padding, uint stride, uint dilation, uint groups) {
    auto Activation_shape = dX.get_dims();
    auto Weight_shape = Weight.get_dims();
    uint n = static_cast<uint>(Activation_shape[0]);
    uint c = static_cast<uint>(Activation_shape[1]);
    uint h = static_cast<uint>(Activation_shape[2]);
    uint w = static_cast<uint>(Activation_shape[3]);
    uint k = static_cast<uint>(Weight_shape[0]);
    uint r = static_cast<uint>(Weight_shape[2]);
    uint s = static_cast<uint>(Weight_shape[3]);
    
    cudnnHandle_t handle;
    CHECK_CUDNN_ERR(cudnnCreate(&handle));
    bpa_be_devicec(handle, {n,c,h,w,k,r,s,stride,dilation,padding,groups}, dY.data_ptr(), Weight.data_ptr(), dX.data_ptr(), be_type_switch(dX.data_type()), be_type_switch(dY.data_type()));

    cudnnDestroy(handle);
}