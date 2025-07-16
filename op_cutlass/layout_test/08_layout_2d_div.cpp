#include "cute/tensor.hpp"

using namespace cute;

int main()
{
    print("=============== logical_divide 2D ===============\n");   
    Layout layout1 = make_layout(make_shape(Int<8>{},Int<4>{},Int<3>{}),make_stride(Int<4>{},Int<1>{},Int<32>{}));
    Tile tiler1 = make_tile(Layout<_4,_2>{},Layout<_2,_2>{});
    print("layout:  ");print(layout1);print("\n");
    print("tiler :  ");print(tiler1);print("\n");
    print("\tlogical_divide:   ");print(logical_divide(layout1,tiler1));print("\n");print("\n");

    print("=============== zipped_divide 2D ===============\n");   
    Layout layout2 = make_layout(make_shape(Int<8>{},Int<4>{},Int<3>{}),make_stride(Int<4>{},Int<1>{},Int<32>{}));
    Tile tiler2 = make_tile(Layout<_4,_2>{},Layout<_2,_2>{});
    print("layout:  ");print(layout2);print("\n");
    print("tiler :  ");print(tiler2);print("\n");
    print("\tzipped_divide:   ");print(zipped_divide(layout2,tiler2));print("\n");print("\n");

    print("=============== tiled_divide 2D ===============\n");   
    Layout layout3 = make_layout(make_shape(Int<8>{},Int<4>{},Int<3>{}),make_stride(Int<4>{},Int<1>{},Int<32>{}));
    Tile tiler3 = make_tile(Layout<_4,_2>{},Layout<_2,_2>{});
    print("layout:  ");print(layout3);print("\n");
    print("tiler :  ");print(tiler3);print("\n");
    print("\ttiled_divide:   ");print(tiled_divide(layout3,tiler3));print("\n");print("\n");

    print("=============== flat_divide 2D ===============\n");   
    Layout layout4 = make_layout(make_shape(Int<8>{},Int<4>{},Int<3>{}),make_stride(Int<4>{},Int<1>{},Int<32>{}));
    Tile tiler4 = make_tile(Layout<_4,_2>{},Layout<_2,_2>{});
    print("layout:  ");print(layout4);print("\n");
    print("tiler :  ");print(tiler4);print("\n");
    print("\tflat_divide:   ");print(flat_divide(layout4,tiler4));print("\n");print("\n");
}