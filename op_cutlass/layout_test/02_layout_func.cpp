#include "cute/tensor.hpp"

using namespace cute;

int main()
{
    Layout test_layout = make_layout(make_shape (Int<2>{},make_shape (Int<3>{},Int<4>{})),
                            make_stride(Int<20>{},make_stride(Int<5>{},Int<1>{})));

    print("layout:  ");print(test_layout);print("\n");
    print("\trank:   ");print(rank(test_layout));print("\n");
    print("\tdepth:   ");print(depth(test_layout));print("\n");
    print("\tsize:   ");print(size(test_layout));print("\n");
    print("\tcosize:   ");print(cosize(test_layout));print("\n");


    Layout test_layout0 = make_layout(make_shape (make_shape (Int<2>{},Int<1>{}),make_shape (Int<3>{},Int<4>{})),
                            make_stride(make_stride(Int<20>{},Int<15>{}),make_stride(Int<5>{},Int<1>{})));
    print("layout:  ");print(test_layout0);print("\n");
    print("\trank:   ");print(rank(test_layout0));print("\n");
    print("\tdepth:   ");print(depth(test_layout0));print("\n");

    Layout test_layout1 = make_layout(make_shape (Int<2>{},make_shape (Int<3>{},Int<4>{},Int<2>{})),
                            make_stride(Int<24>{},make_stride(Int<8>{},Int<2>{},Int<1>{})));
    print("layout:  ");print(test_layout1);print("\n");
    print("\trank:   ");print(rank(test_layout1));print("\n");
    print("\tdepth:   ");print(depth(test_layout1));print("\n");


    Layout test_layout2 = make_layout(make_shape (Int<2>{},make_shape (Int<3>{},make_shape (Int<4>{},Int<2>{}))),
                            make_stride(Int<24>{},make_stride(Int<8>{},make_stride(Int<2>{},Int<1>{}))));
    print("layout:  ");print(test_layout2);print("\n");
    print("\trank:   ");print(rank(test_layout2));print("\n");
    print("\tdepth:   ");print(depth(test_layout2));print("\n");
    print("\tget<0>:   ");print(get<0>(test_layout2));print("\n");
    print("\tget<1>:   ");print(get<1>(test_layout2));print("\n");
    print("\tget<1,0>:   ");print(get<1,0>(test_layout2));print("\n");
}