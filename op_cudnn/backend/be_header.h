#include <iostream>

#include "tensor_impl.h"

void conv2d_forward(MyTensor Activation, MyTensor Weight, MyTensor Result, uint padding, uint stride, uint dilation, uint groups);
void conv2d_bpa(MyTensor dY, MyTensor Weight, MyTensor dX, uint padding, uint stride, uint dilation, uint groups);
void conv2d_bpw(MyTensor Activation, MyTensor dY, MyTensor dWeight, uint padding, uint stride, uint dilation, uint groups);