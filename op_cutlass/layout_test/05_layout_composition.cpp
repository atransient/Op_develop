#include "cute/tensor.hpp"

using namespace cute;

int main()
{
    Layout layout_A = make_layout(make_shape (Int<2>{},make_shape (Int<3>{},Int<4>{})),
                            make_stride(Int<12>{},make_stride(Int<4>{},Int<1>{})));

    
    Layout layout_B = make_layout(make_shape (Int<2>{},make_shape (Int<2>{},Int<6>{})),
                            make_stride(Int<12>{},make_stride(Int<1>{},Int<2>{})));
    print("layout A:  ");print(layout_A);print("\n");
    print("layout B:  ");print(layout_B);print("\n");
    print("coalesce layout B:  ");print(coalesce(layout_B));print("\n");
    print("\tcomposition:   ");print(composition(layout_A,coalesce(layout_B)));print("\n");
    print("\tcomposition:   ");print(composition(layout_A,layout_B));print("\n");

}