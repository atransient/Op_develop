#include "cute/tensor.hpp"

using namespace cute;

int main()
{
    print("=============== logical_divide 1D===============\n");

    Layout layout1 = make_layout(make_shape(Int<16>{}),make_stride(Int<1>{}));
    Layout tiler1 = make_layout(make_shape(Int<4>{}),make_stride(Int<2>{}));
    print("layout:  ");print(layout1);print("\n");
    print("tiler :  ");print(tiler1);print("\n");
    print("\tlogical_divide:   ");print(logical_divide(layout1,tiler1));print("\n");print("\n");

    Layout layout2 = make_layout(make_shape(Int<4>{},Int<2>{},Int<3>{}),make_stride(Int<2>{},Int<1>{},Int<8>{}));
    Layout tiler2 = make_layout(make_shape(Int<4>{}),make_stride(Int<2>{}));
    print("layout:  ");print(layout2);print("\n");
    print("tiler :  ");print(tiler2);print("\n");
    // print("tiler complement:  ");print(complement(tiler1, size(layout1)));print("\n");
    print("\tlogical_divide:   ");print(logical_divide(layout2,tiler2));print("\n");print("\n");

    Layout layout3 = make_layout(make_shape(Int<32>{}),make_stride(Int<1>{}));
    Layout tiler3 = make_layout(make_shape(Int<4>{},Int<4>{}),make_stride(Int<8>{},Int<2>{}));
    print("layout:  ");print(layout3);print("\n");
    print("tiler :  ");print(tiler3);print("\n");
    print("\tlogical_divide:   ");print(logical_divide(layout3,tiler3));print("\n");print("\n");

    Layout layout4 = make_layout(make_shape(Int<32>{},Int<4>{}),make_stride(Int<4>{},Int<1>{}));
    Layout tiler4 = make_layout(make_shape(Int<4>{},Int<4>{}),make_stride(Int<8>{},Int<2>{}));
    print("layout:  ");print(layout4);print("\n");
    print("tiler :  ");print(tiler4);print("\n");
    print("\tlogical_divide:   ");print(logical_divide(layout4,tiler4));print("\n");print("\n");

    Layout layout5 = make_layout(make_shape(Int<32>{},Int<4>{}),make_stride(Int<1>{},Int<32>{}));
    Layout tiler5 = make_layout(make_shape(Int<4>{},Int<4>{}),make_stride(Int<8>{},Int<2>{}));
    print("layout:  ");print(layout5);print("\n");
    print("tiler :  ");print(tiler5);print("\n");
    print("\tlogical_divide:   ");print(logical_divide(layout5,tiler5));print("\n");print("\n");

}