#include "cute/tensor.hpp"

using namespace cute;

int main()
{
    print("=============== logical_product 1D===============\n");

    Layout layout_A1 = make_layout(make_shape(Int<2>{},Int<2>{}),make_stride(Int<4>{},Int<1>{}));
    Layout layout_B1 = make_layout(make_shape(Int<6>{}),make_stride(Int<1>{}));
    print("layout_A1:  ");print(layout_A1);print("\n");
    print("layout_B1:  ");print(layout_B1);print("\n");
    print("\tlogical_product:   ");print(logical_product(layout_A1,layout_B1));print("\n");print("\n");
}