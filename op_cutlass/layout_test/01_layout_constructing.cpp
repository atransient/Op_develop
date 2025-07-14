#include "cute/tensor.hpp"

using namespace cute;

int main()
{
    Layout s8 = make_layout(Int<8>{});
    print("make_layout(Int<8>{}):    ");print(s8);print("\n");
    Layout d8 = make_layout(8);
    print("make_layout(8):    ");print(d8);print("\n");
    Layout s2xs4 = make_layout(make_shape(Int<2>{},Int<4>{}));
    print("make_layout(make_shape(Int<2>{},Int<4>{})):    ");print(s2xs4);print("\n");
    Layout s2xd4 = make_layout(make_shape(Int<2>{},4));
    print("make_layout(make_shape(Int<2>{},4)):    ");print(s2xd4);print("\n");
    Layout ts2xd4 = make_layout(make_shape(2,Int<4>{}));
    print("make_layout(make_shape(2,Int<4>{})):    ");print(ts2xd4);print("\n");
    Layout s2xd4_a = make_layout(make_shape (Int< 2>{},4),
                                make_stride(Int<12>{},Int<1>{}));
    print("make_layout(make_shape (Int< 2>{},4),make_stride(Int<12>{},Int<1>{})):    ");print(s2xd4_a);print("\n");
    Layout s2xd4_col = make_layout(make_shape(Int<2>{},4),
                                LayoutLeft{});
    print("make_layout(make_shape(Int<2>{},4),LayoutLeft{}):    ");print(s2xd4_col);print("\n");
    Layout s2xd4_row = make_layout(make_shape(Int<2>{},4),
                                LayoutRight{});
    print("make_layout(make_shape(Int<2>{},4),LayoutRight{}):    ");print(s2xd4_row);print("\n");
    Layout s2xh4 = make_layout(make_shape (2,make_shape (2,2)),
                            make_stride(4,make_stride(2,1)));
    print("make_layout(make_shape (2,make_shape (2,2)),make_stride(4,make_stride(2,1))):    ");print(s2xh4);print("\n");
    Layout s2xh4_col = make_layout(shape(s2xh4),
                                LayoutLeft{});
    print("make_layout(shape(s2xh4),LayoutLeft{}):    ");print(s2xh4_col);print("\n");
    Layout ts2xh4 = make_layout(make_shape (Int<2>{},make_shape (Int<2>{},Int<2>{})),
                            make_stride(Int<4>{},make_stride(Int<2>{},Int<1>{})));
    print("make_layout(make_shape (Int<2>{},make_shape (Int<2>{},Int<2>{})),make_stride(Int<4>{},make_stride(Int<2>{},Int<1>{}))):    ");print(ts2xh4);print("\n");
    Layout ts2xh4_col = make_layout(shape(ts2xh4),
                                LayoutLeft{});
    print("make_layout(shape(ts2xh4),LayoutLeft{}):    ");print(ts2xh4_col);print("\n");
    return 0;
}

