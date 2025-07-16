#include "cute/tensor.hpp"

using namespace cute;

int main()
{
    print("=============== blocked_product ===============\n");

    Layout layout_A1 = make_layout(make_shape(Int<2>{},Int<2>{}),make_stride(Int<4>{},Int<1>{}));
    Layout layout_B1 = make_layout(make_shape(Int<6>{}),make_stride(Int<1>{}));
    print("layout_A:  ");print(layout_A1);print("\n");
    print("layout_B:  ");print(layout_B1);print("\n");
    print("\tlogical_product:   ");print(logical_product(layout_A1,layout_B1));print("\n");
    print("\tblocked_product:   ");print(blocked_product(layout_A1,layout_B1));print("\n");
    print("\traked_product  :   ");print(raked_product(layout_A1,layout_B1));print("\n");print("\n");

    Layout layout_A2 = make_layout(make_shape(Int<2>{},Int<5>{}),make_stride(Int<5>{},Int<1>{}));
    Layout layout_B2 = make_layout(make_shape(Int<3>{},Int<4>{}),make_stride(Int<1>{},Int<3>{}));
    print("layout_A:  ");print(layout_A2);print("\n");
    print("layout_B:  ");print(layout_B2);print("\n");
    print("\tlogical_product:   ");print(logical_product(layout_A2,layout_B2));print("\n");
    print("\tblocked_product:   ");print(blocked_product(layout_A2,layout_B2));print("\n");
    print("\traked_product  :   ");print(raked_product(layout_A2,layout_B2));print("\n");print("\n");

}