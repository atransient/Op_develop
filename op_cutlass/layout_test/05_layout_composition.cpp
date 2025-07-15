#include "cute/tensor.hpp"

using namespace cute;

int main()
{
    Layout layout_A0 = make_layout(make_shape(Int<8>{}),make_stride(Int<1>{}));
    Layout layout_B0 = make_layout(make_shape(Int<4>{}),make_stride(Int<2>{}));
    print("layout A:  ");print(layout_A0);print("\n");
    print("layout B:  ");print(layout_B0);print("\n");
    print("\tcomposition:   ");print(composition(layout_A0,layout_B0));print("\n");print("\n");

    Layout layout_A1 = make_layout(make_shape(Int<8>{}),make_stride(Int<3>{}));
    Layout layout_B1 = make_layout(make_shape(Int<4>{}),make_stride(Int<2>{}));
    print("layout A:  ");print(layout_A1);print("\n");
    print("layout B:  ");print(layout_B1);print("\n");
    print("\tcomposition:   ");print(composition(layout_A1,layout_B1));print("\n");print("\n");

    Layout layout_A2 = make_layout(make_shape(Int<8>{},Int<2>{}),make_stride(Int<2>{},Int<1>{}));
    Layout layout_B2 = make_layout(make_shape(Int<4>{}),make_stride(Int<2>{}));
    print("layout A:  ");print(layout_A2);print("\n");
    print("layout B:  ");print(layout_B2);print("\n");
    print("\tcomposition:   ");print(composition(layout_A2,layout_B2));print("\n");print("\n");

    Layout layout_A3 = make_layout(make_shape(Int<2>{},Int<8>{}),make_stride(Int<8>{},Int<1>{}));
    Layout layout_B3 = make_layout(make_shape(Int<4>{}),make_stride(Int<2>{}));
    print("layout A:  ");print(layout_A3);print("\n");
    print("layout B:  ");print(layout_B3);print("\n");
    print("\tcomposition:   ");print(composition(layout_A3,layout_B3));print("\n");print("\n");

    Layout layout_A4 = make_layout(make_shape(Int<16>{}),make_stride(Int<2>{}));
    Layout layout_B4 = make_layout(make_shape(Int<2>{},Int<8>{}),make_stride(Int<8>{},Int<1>{}));
    print("layout A:  ");print(layout_A4);print("\n");
    print("layout B:  ");print(layout_B4);print("\n");
    print("\tcomposition:   ");print(composition(layout_A4,layout_B4));print("\n");print("\n");

    Layout layout_A5 = make_layout(make_shape(Int<4>{},Int<6>{}),make_stride(Int<6>{},Int<1>{}));
    Layout layout_B5 = make_layout(make_shape(Int<4>{},Int<8>{}),make_stride(Int<8>{},Int<1>{}));
    print("layout A:  ");print(layout_A5);print("\n");
    print("layout B:  ");print(layout_B5);print("\n");
    print("\tcomposition:   ");print(composition(layout_A5,layout_B5));print("\n");print("\n");

    Layout layout_A6 = make_layout(make_shape (Int<2>{},make_shape (Int<3>{},Int<4>{})),
                            make_stride(Int<12>{},make_stride(Int<4>{},Int<1>{})));
    Layout layout_B6 = make_layout(make_shape (Int<2>{},make_shape (Int<2>{},Int<6>{})),
                            make_stride(Int<12>{},make_stride(Int<1>{},Int<2>{})));
    print("layout A:  ");print(layout_A6);print("\n");
    print("layout B:  ");print(layout_B6);print("\n");
    print("\tcomposition:   ");print(composition(layout_A6,layout_B6));print("\n");print("\n");

    print("layout A:  ");print(layout_A6);print("\n");
    print("layout B:  ");print(coalesce(layout_B6));print("\n");
    print("\tcomposition:   ");print(composition(layout_A6,coalesce(layout_B6)));print("\n");print("\n");


    // Layout layout_A7 = make_layout(make_shape (Int<8>{},Int<4>{},Int<3>{}),
    //                         make_stride(Int<12>{},Int<3>{},Int<1>{}));
    Layout layout_A7 = make_layout(make_shape (Int<8>{},Int<4>{}),
                            make_stride(Int<12>{},Int<3>{}));
    auto tiler = make_tile(Layout<_4,_2>{},Layout<_2,_1>{});
    print("layout A:  ");print(layout_A7);print("\n");
    print("tiler   :  ");print(tiler);print("\n");
    print("\tcomposition:   ");print(composition(layout_A7,tiler));print("\n");
}