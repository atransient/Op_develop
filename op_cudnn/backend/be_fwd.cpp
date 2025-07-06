#include <iostream>
#include <cudnn.h>

#include "configer.h"
#include "tensor_impl.h"

void fwd_be_devicec(cudnnHandle_t handle, vector<uint>v_info, uint64_t X_ptr, uint64_t W_ptr, uint64_t Y_ptr, cudnnDataType_t iodtype, cudnnDataType_t cdtype) {
    uint n = v_info[0], c = v_info[1], h = v_info[2], w = v_info[3];
    uint k = v_info[4], r = v_info[5], s = v_info[6];
    uint stride = v_info[7], dilation = v_info[8], padding = v_info[9], groups = v_info[10];

    // 1. Initialize tensor descriptors for input, filter, and output
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    
    auto [oup_size, workspace_size, algo] = be_fwd_desc(handle, input_desc, filter_desc, output_desc, conv_desc, {n,c,h,w,k,r,s,stride,dilation,padding,groups},iodtype,cdtype);
   
    void *d_X = reinterpret_cast<void *>(X_ptr);
    void *d_W = reinterpret_cast<void *>(W_ptr);
    void *d_Y = reinterpret_cast<void *>(Y_ptr);

    int8_t *d_workspace = nullptr;
    cudaMalloc((void**)&d_workspace, workspace_size);
    // 5. Perform convolution (depthwise)
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN_ERR(cudnnConvolutionForward(handle, &alpha, input_desc, d_X, filter_desc, d_W, conv_desc, algo, d_workspace, workspace_size, &beta, output_desc, d_Y));

    cudaDeviceSynchronize();
    // 6. Free allocated memory
    cudaFree(d_workspace);

    // 7. Destroy cuDNN descriptors
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
}


void conv2d_forward(MyTensor Activation, MyTensor Weight, MyTensor Result, uint padding, uint stride, uint dilation, uint groups) {
    auto Activation_shape = Activation.get_dims();
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
    fwd_be_devicec(handle, {n,c,h,w,k,r,s,stride,dilation,padding,groups}, Activation.data_ptr(), Weight.data_ptr(), Result.data_ptr(), be_type_switch(Activation.data_type()), be_type_switch(Result.data_type()));
    cudnnDestroy(handle);
}