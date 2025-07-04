#include <iostream>
#include <vector>

#include "data_generate.h"
#include "tensor_impl.h"

using std::vector;

int main()
{
    Surface<float> test(20, false);
    vector<uint64_t> dims = {10,2};
    MyTensor tensor(reinterpret_cast<uint64_t>(test.hostPtr), dims);
    tensor.topn_val(10);
    
    float* tptr = test.hostPtr;
    for (int i = 0; i < 10; ++i)
        printf("%f\t", tptr[i]);
    printf("\n");
}