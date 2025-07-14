#include "cute/tensor.hpp"

using namespace cute;

int main()
{
    Layout test_layout = make_layout(make_shape (Int<2>{},make_shape (Int<3>{},Int<4>{})),
                            make_stride(Int<12>{},make_stride(Int<4>{},Int<1>{})));

    print("layout:  ");print(test_layout);print("\n");
    print("\tcoalesce:   ");print(coalesce(test_layout));print("\n");
    print("\tcoalesce coalesce:   ");print(coalesce(coalesce(test_layout)));print("\n");

    Layout test_layout0 = make_layout(make_shape (Int<2>{},Int<3>{},Int<4>{},Int<2>{}),
                            make_stride(Int<24>{},Int<8>{},Int<2>{},Int<1>{}));
    print("layout:  ");print(test_layout0);print("\n");
    print("\tcoalesce:   ");print(coalesce(test_layout0));print("\n");

    auto test_layout1 = Layout<Shape <_2,Shape <_1,_6>>,
                        Stride<_1,Stride<_6,_2>>>{};
    print("layout:  ");print(test_layout1);print("\n");
    print("\tcoalesce:   ");print(coalesce(test_layout1));print("\n");

    Layout test_layout2 = make_layout(make_shape (Int<2>{},Int<3>{},Int<4>{},Int<2>{}),
                            make_stride(Int<1>{},Int<2>{},Int<6>{},Int<24>{}));
    print("layout:  ");print(test_layout2);print("\n");
    print("\tcoalesce:   ");print(coalesce(test_layout2));print("\n");

    Layout test_layout3 = make_layout(make_shape (Int<2>{},make_shape (Int<3>{},Int<4>{})),
                            make_stride(Int<12>{},make_stride(Int<1>{},Int<3>{})));

    print("layout:  ");print(test_layout3);print("\n");
    print("\tcoalesce:   ");print(coalesce(test_layout3));print("\n");

    Layout test_layout4 = make_layout(make_shape (Int<2>{},make_shape (Int<3>{},Int<1>{}, Int<4>{})),
                            make_stride(Int<12>{},make_stride(Int<1>{},Int<10>{},Int<3>{})));

    print("layout:  ");print(test_layout4);print("\n");
    print("\tcoalesce:   ");print(coalesce(test_layout4));print("\n");
    print("\tcoalesce layout<0>:   ");print(coalesce(layout<0>(test_layout4)));print("\n");
    print("\tcoalesce:   ");print(make_layout(coalesce(layout<0>(test_layout4)),coalesce(layout<1>(test_layout4))));print("\n");
}