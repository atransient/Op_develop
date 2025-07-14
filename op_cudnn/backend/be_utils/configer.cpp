#include <iostream>
#include <cudnn.h>
#include <vector>
#include <tuple>

#include "configer.h"
#include "tensor_impl.h"

using std::vector;

cudnnDataType_t be_type_switch(TensorDataType_t type)
{
    switch (type)
    {
        case TensorDataType_t::FLOAT:
            return cudnnDataType_t::CUDNN_DATA_FLOAT;
        case TensorDataType_t::DOUBLE:
            return cudnnDataType_t::CUDNN_DATA_DOUBLE;
        case TensorDataType_t::HALF:
            return cudnnDataType_t::CUDNN_DATA_HALF;
        case TensorDataType_t::INT32:
            return cudnnDataType_t::CUDNN_DATA_INT32;
        case TensorDataType_t::BFLOAT16:
            return cudnnDataType_t::CUDNN_DATA_BFLOAT16;
        default:
            return cudnnDataType_t::CUDNN_DATA_FLOAT;
    }
}

std::tuple<int, size_t, cudnnConvolutionFwdAlgo_t> be_fwd_desc(cudnnHandle_t &handle, cudnnTensorDescriptor_t &input_desc, cudnnFilterDescriptor_t &filter_desc, cudnnTensorDescriptor_t &output_desc, cudnnConvolutionDescriptor_t &conv_desc, be_para_info info, cudnnDataType_t iodtype, cudnnDataType_t cdtype)
{
    // Input tensor descriptor
    CHECK_CUDNN_ERR(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN_ERR(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, iodtype, info.n, info.c, info.h, info.w));

    // Filter tensor descriptor (depthwise: one filter per input channel)
    CHECK_CUDNN_ERR(cudnnCreateFilterDescriptor(&filter_desc));
    CHECK_CUDNN_ERR(cudnnSetFilter4dDescriptor(filter_desc, iodtype, CUDNN_TENSOR_NCHW, info.k, int(info.c / info.groups), info.r, info.s));
    // CHECK_CUDNN_ERR(cudnnSetFilter4dDescriptor(filter_desc, iodtype, CUDNN_TENSOR_NCHW, info.k, info.c, info.r, info.s));

    // Convolution descriptor (with groups set to the number of input channels)
    CHECK_CUDNN_ERR(cudnnCreateConvolutionDescriptor(&conv_desc));
    CHECK_CUDNN_ERR(cudnnSetConvolution2dDescriptor(conv_desc, info.padding, info.padding, info.stride, info.stride, info.dilation, info.dilation, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    CHECK_CUDNN_ERR(cudnnSetConvolutionGroupCount(conv_desc, info.groups));

    int On = 0, Oc = 0, Oh = 0, Ow = 0;
    cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc, &On, &Oc, &Oh, &Ow);

    // Output tensor descriptor
    CHECK_CUDNN_ERR(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN_ERR(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, iodtype, On, Oc, Oh, Ow));


    int algocount = 0;  // Use a well-supported algorithm
    int maxcount = 0;
    cudnnConvolutionFwdAlgoPerf_t algorithms[8];
    cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &maxcount);
    
    CHECK_CUDNN_ERR(cudnnGetConvolutionForwardAlgorithm_v7(handle, input_desc, filter_desc, conv_desc, output_desc, maxcount, &algocount,algorithms));

    // 3. Get workspace size
    size_t workspace_size = 0;
    CHECK_CUDNN_ERR(cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc, filter_desc, conv_desc, output_desc, algorithms[0].algo, &workspace_size));

    // return On * Oc * Oh * Ow;
    return std::make_tuple(On * Oc * Oh * Ow, workspace_size, algorithms[0].algo);
}

vector<int> be_get_yshape(be_para_info info)
{
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    CHECK_CUDNN_ERR(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN_ERR(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, info.n, info.c, info.h, info.w));

    // Filter tensor descriptor (depthwise: one filter per input channel)
    CHECK_CUDNN_ERR(cudnnCreateFilterDescriptor(&filter_desc));
    CHECK_CUDNN_ERR(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, info.k, int(info.c/info.groups), info.r, info.s));
    
    CHECK_CUDNN_ERR(cudnnCreateConvolutionDescriptor(&conv_desc));
    CHECK_CUDNN_ERR(cudnnSetConvolution2dDescriptor(conv_desc, info.padding, info.padding, info.stride, info.stride, info.dilation, info.dilation, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    CHECK_CUDNN_ERR(cudnnSetConvolutionGroupCount(conv_desc, info.groups));
    int On = 0, Oc = 0, Oh = 0, Ow = 0;
    cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc, &On, &Oc, &Oh, &Ow);
    
    return {On, Oc, Oh, Ow};
}


std::tuple<int, size_t, cudnnConvolutionBwdFilterAlgo_t> be_bpw_desc(cudnnHandle_t &handle, cudnnTensorDescriptor_t &input_desc, cudnnTensorDescriptor_t &grad_desc, cudnnFilterDescriptor_t &filter_desc, cudnnConvolutionDescriptor_t &conv_desc, be_para_info info, cudnnDataType_t iodtype, cudnnDataType_t cdtype)
{
    // Input tensor descriptor
    CHECK_CUDNN_ERR(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN_ERR(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, iodtype, info.n, info.c, info.h, info.w));

    vector<int>grad_shape = be_get_yshape(info);

    // grad tensor descriptor
    CHECK_CUDNN_ERR(cudnnCreateTensorDescriptor(&grad_desc));
    CHECK_CUDNN_ERR(cudnnSetTensor4dDescriptor(grad_desc, CUDNN_TENSOR_NCHW, iodtype, grad_shape[0], grad_shape[1], grad_shape[2], grad_shape[3]));

    CHECK_CUDNN_ERR(cudnnCreateFilterDescriptor(&filter_desc));
    CHECK_CUDNN_ERR(cudnnSetFilter4dDescriptor(filter_desc, iodtype, CUDNN_TENSOR_NCHW, info.k, int(info.c / info.groups), info.r, info.s));

    // Convolution descriptor (with groups set to the number of input channels)
    CHECK_CUDNN_ERR(cudnnCreateConvolutionDescriptor(&conv_desc));
    CHECK_CUDNN_ERR(cudnnSetConvolution2dDescriptor(conv_desc, info.padding, info.padding, info.stride, info.stride, info.dilation, info.dilation, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));



    int algocount = 0;  // Use a well-supported algorithm
    int maxcount = 0;
    cudnnConvolutionBwdFilterAlgoPerf_t algorithms[10];
    cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle, &maxcount);
    
    CHECK_CUDNN_ERR(cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle, input_desc, grad_desc, conv_desc, filter_desc, maxcount, &algocount,algorithms));

    // 3. Get workspace size
    size_t workspace_size = 0;
    CHECK_CUDNN_ERR(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, input_desc, grad_desc, conv_desc, filter_desc, algorithms[0].algo, &workspace_size));

    // return On * Oc * Oh * Ow;
    return std::make_tuple(grad_shape[0] * grad_shape[1] * grad_shape[2] * grad_shape[3], workspace_size, algorithms[0].algo);
}

std::tuple<int, size_t, cudnnConvolutionBwdDataAlgo_t> be_bpa_desc(cudnnHandle_t &handle, cudnnFilterDescriptor_t &filter_desc, cudnnTensorDescriptor_t &grad_desc, cudnnTensorDescriptor_t &dinput_desc, cudnnConvolutionDescriptor_t &conv_desc, be_para_info info, cudnnDataType_t iodtype, cudnnDataType_t cdtype)
{
    // Input tensor descriptor
    CHECK_CUDNN_ERR(cudnnCreateTensorDescriptor(&dinput_desc));
    CHECK_CUDNN_ERR(cudnnSetTensor4dDescriptor(dinput_desc, CUDNN_TENSOR_NCHW, iodtype, info.n, info.c, info.h, info.w));

    vector<int>grad_shape = be_get_yshape(info);

    // grad tensor descriptor
    CHECK_CUDNN_ERR(cudnnCreateTensorDescriptor(&grad_desc));
    CHECK_CUDNN_ERR(cudnnSetTensor4dDescriptor(grad_desc, CUDNN_TENSOR_NCHW, iodtype, grad_shape[0], grad_shape[1], grad_shape[2], grad_shape[3]));

    CHECK_CUDNN_ERR(cudnnCreateFilterDescriptor(&filter_desc));
    CHECK_CUDNN_ERR(cudnnSetFilter4dDescriptor(filter_desc, iodtype, CUDNN_TENSOR_NCHW, info.k, int(info.c/info.groups), info.r, info.s));

    // Convolution descriptor (with groups set to the number of input channels)
    CHECK_CUDNN_ERR(cudnnCreateConvolutionDescriptor(&conv_desc));
    CHECK_CUDNN_ERR(cudnnSetConvolution2dDescriptor(conv_desc, info.padding, info.padding, info.stride, info.stride, info.dilation, info.dilation, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));



    int algocount = 0;  // Use a well-supported algorithm
    int maxcount = 0;
    cudnnConvolutionBwdDataAlgoPerf_t algorithms[10];
    cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle, &maxcount);
    
    CHECK_CUDNN_ERR(cudnnGetConvolutionBackwardDataAlgorithm_v7(handle, filter_desc, grad_desc, conv_desc, dinput_desc, maxcount, &algocount,algorithms));

    // 3. Get workspace size
    size_t workspace_size = 0;
    CHECK_CUDNN_ERR(cudnnGetConvolutionBackwardDataWorkspaceSize(handle, filter_desc, grad_desc, conv_desc, dinput_desc, algorithms[0].algo, &workspace_size));

    // return On * Oc * Oh * Ow;
    return std::make_tuple(grad_shape[0] * grad_shape[1] * grad_shape[2] * grad_shape[3], workspace_size, algorithms[0].algo);
}