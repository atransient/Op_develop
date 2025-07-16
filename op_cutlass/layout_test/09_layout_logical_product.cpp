#include "cute/tensor.hpp"

using namespace cute;

int main()
{
    print("=============== logical_product 1D===============\n");

    Layout layout_A1 = make_layout(make_shape(Int<2>{},Int<2>{}),make_stride(Int<4>{},Int<1>{}));
    Layout layout_B1 = make_layout(make_shape(Int<6>{}),make_stride(Int<1>{}));
    print("layout_A:  ");print(layout_A1);print("\n");
    print("layout_B:  ");print(layout_B1);print("\n");
    print("\tlogical_product:   ");print(logical_product(layout_A1,layout_B1));print("\n");print("\n");

    Layout layout_A2 = make_layout(make_shape(Int<2>{},Int<2>{}),make_stride(Int<4>{},Int<1>{}));
    Layout layout_B2 = make_layout(make_shape(Int<6>{}),make_stride(Int<2>{}));
    print("layout_A:  ");print(layout_A2);print("\n");
    print("layout_B:  ");print(layout_B2);print("\n");
    print("\tlogical_product:   ");print(logical_product(layout_A2,layout_B2));print("\n");print("\n");

    Layout layout_A3 = make_layout(make_shape(Int<2>{},Int<2>{}),make_stride(Int<4>{},Int<1>{}));
    Layout layout_B3 = make_layout(make_shape(Int<4>{},Int<2>{}),make_stride(Int<2>{},Int<1>{}));
    print("layout_A:  ");print(layout_A3);print("\n");
    print("layout_B:  ");print(layout_B3);print("\n");
    print("\tlogical_product:   ");print(logical_product(layout_A3,layout_B3));print("\n");print("\n");

    Layout layout_A4 = make_layout(make_shape(Int<4>{},Int<2>{},Int<3>{}),make_stride(Int<2>{},Int<1>{},Int<8>{}));
    Layout layout_B4 = make_layout(make_shape(Int<4>{}),make_stride(Int<2>{}));
    print("layout_A:  ");print(layout_A4);print("\n");
    print("layout_B:  ");print(layout_B4);print("\n");
    print("\tcosize B:  ");print(cosize(layout_B4));print("\n");
    print("\tlayout_A complement:  ");print(complement(layout_A4, size(layout_A4) * cosize(layout_B4)));print("\n");
    print("\tlogical_divide:   ");print(logical_product(layout_A4,layout_B4));print("\n");print("\n");
}