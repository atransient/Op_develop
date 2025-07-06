#include <iostream>
#include <cudnn.h>
#include <tuple>
#include <vector>
#include "tensor_impl.h"

using std::vector;


#define CHECK_CUDNN_ERR(status)                             \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
        std::cerr << "CUDNN error: " << cudnnGetErrorString(status) << std::endl; \
        exit(EXIT_FAILURE);                                 \
    }

struct be_para_info
{
    uint n;
    uint c;
    uint h;
    uint w;
    uint k;
    uint r;
    uint s;
    uint stride = 1;
    uint dilation = 1;
    uint padding = 0;
    uint groups = 1;
};

vector<int> be_get_yshape(be_para_info info);

std::tuple<int, size_t, cudnnConvolutionFwdAlgo_t> be_fwd_desc(cudnnHandle_t &handle, cudnnTensorDescriptor_t &input_desc, cudnnFilterDescriptor_t &filter_desc, cudnnTensorDescriptor_t &ouput_desc, cudnnConvolutionDescriptor_t &conv_desc, be_para_info info, cudnnDataType_t iodtype = CUDNN_DATA_FLOAT, cudnnDataType_t cdtype = CUDNN_DATA_FLOAT);
std::tuple<int, size_t, cudnnConvolutionBwdFilterAlgo_t> be_bpw_desc(cudnnHandle_t &handle, cudnnTensorDescriptor_t &input_desc, cudnnTensorDescriptor_t &grad_desc, cudnnFilterDescriptor_t &filter_desc, cudnnConvolutionDescriptor_t &conv_desc, be_para_info info, cudnnDataType_t iodtype = CUDNN_DATA_FLOAT, cudnnDataType_t cdtype = CUDNN_DATA_FLOAT);
std::tuple<int, size_t, cudnnConvolutionBwdDataAlgo_t> be_bpa_desc(cudnnHandle_t &handle, cudnnFilterDescriptor_t &filter_desc, cudnnTensorDescriptor_t &grad_desc, cudnnTensorDescriptor_t &dinput_desc, cudnnConvolutionDescriptor_t &conv_desc, be_para_info info, cudnnDataType_t iodtype = CUDNN_DATA_FLOAT, cudnnDataType_t cdtype = CUDNN_DATA_FLOAT);
cudnnDataType_t be_type_switch(TensorDataType_t type);