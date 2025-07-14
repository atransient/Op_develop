#include "cute/tensor.hpp"

using namespace cute;

int main()
{
    Layout test_layout = make_layout(make_shape (Int<2>{},make_shape (Int<3>{},Int<4>{})),
                            make_stride(Int<20>{},make_stride(Int<5>{},Int<1>{})));

    print("layout:  ");print(test_layout);print("\n");
    print("\tlayout<0>:   ");print(layout<0>(test_layout));print("\n");
    print("\tlayout<1>:   ");print(layout<1>(test_layout));print("\n");
    print("\tlayout<1,0>:   ");print(layout<1,0>(test_layout));print("\n");
    print("\tlayout<1,1>:   ");print(layout<1,1>(test_layout));print("\n");


    Layout test_layout2 = make_layout(make_shape (Int<2>{},Int<3>{},Int<4>{},Int<2>{}),
                            make_stride(Int<24>{},Int<8>{},Int<2>{},Int<1>{}));
    print("layout:  ");print(test_layout2);print("\n");
    print("\tselect<0,1>:   ");print(select<0,1>(test_layout2));print("\n");
    print("\tselect<1,0>:   ");print(select<1,0>(test_layout2));print("\n");
    print("\ttake<0,1>:   ");print(take<0,1>(test_layout2));print("\n");
    print("\ttake<1,4>:   ");print(take<1,4>(test_layout2));print("\n");
    print("\tgroup<0,1>:   ");print(group<0,1>(test_layout2));print("\n");
    print("\tgroup<1,4>:   ");print(group<1,4>(test_layout2));print("\n");
    Layout t0 = group<0,1>(test_layout2);
    Layout t1 = group<1,4>(test_layout2);
    print("\tflatten group<0,1>:   ");print(flatten(t0));print("\n");
    print("\tflatten group<1,4>:   ");print(flatten(t1));print("\n");
}