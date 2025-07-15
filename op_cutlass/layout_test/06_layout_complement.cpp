#include "cute/tensor.hpp"

using namespace cute;

int main()
{
    Layout layout_A1 = make_layout(make_shape(Int<8>{}),make_stride(Int<2>{}));
    Shape shape_B1 = make_shape(Int<9>{});
    print("layout A:  ");print(layout_A1);print("\n");
    print("shape B :  ");print(shape_B1);print("\n");
    print("\tcomplement:   ");print(complement(layout_A1,shape_B1));print("\n");print("\n");

    Layout layout_A2 = make_layout(make_shape(Int<8>{}),make_stride(Int<2>{}));
    Shape shape_B2 = make_shape(Int<10>{});
    print("layout A:  ");print(layout_A2);print("\n");
    print("shape B :  ");print(shape_B2);print("\n");
    print("\tcomplement:   ");print(complement(layout_A2,shape_B2));print("\n");print("\n");

    Layout layout_A3 = make_layout(make_shape(Int<8>{}),make_stride(Int<2>{}));
    Shape shape_B3 = make_shape(Int<16>{});
    print("layout A:  ");print(layout_A3);print("\n");
    print("shape B :  ");print(shape_B3);print("\n");
    print("\tcomplement:   ");print(complement(layout_A3,shape_B3));print("\n");print("\n");

    Layout layout_A4 = make_layout(make_shape(Int<8>{}),make_stride(Int<2>{}));
    Shape shape_B4 = make_shape(Int<18>{});
    print("layout A:  ");print(layout_A4);print("\n");
    print("shape B :  ");print(shape_B4);print("\n");
    print("\tcomplement:   ");print(complement(layout_A4,shape_B4));print("\n");print("\n");

    Layout layout_A5 = make_layout(make_shape(Int<8>{}),make_stride(Int<2>{}));
    Shape shape_B5 = make_shape(Int<17>{},Int<4>{});
    print("layout A:  ");print(layout_A5);print("\n");
    print("shape B :  ");print(shape_B5);print("\n");
    print("\tcomplement:   ");print(complement(layout_A5,shape_B5));print("\n");print("\n");

    Layout layout_A6 = make_layout(make_shape(Int<8>{}),make_stride(Int<2>{}));
    Shape shape_B6 = make_shape(Int<68>{});
    print("layout A:  ");print(layout_A6);print("\n");
    print("shape B :  ");print(shape_B6);print("\n");
    print("\tcomplement:   ");print(complement(layout_A6,shape_B6));print("\n");print("\n");
}