#include "cute/tensor.hpp"

using namespace cute;

int main()
{
    print("=============== logical_divide 2D ===============\n");   
    Layout layout8 = make_layout(make_shape(Int<8>{},Int<4>{},Int<3>{}),make_stride(Int<4>{},Int<1>{},Int<32>{}));
    Tile tiler8 = make_tile(Layout<_4,_2>{},Layout<_2,_2>{});
    print("layout:  ");print(layout8);print("\n");
    print("tiler :  ");print(tiler8);print("\n");
    print("\tlogical_divide:   ");print(logical_divide(layout8,tiler8));print("\n");print("\n");

    print("=============== zipped_divide 2D ===============\n");   
    Layout layout8 = make_layout(make_shape(Int<8>{},Int<4>{},Int<3>{}),make_stride(Int<4>{},Int<1>{},Int<32>{}));
    Tile tiler8 = make_tile(Layout<_4,_2>{},Layout<_2,_2>{});
    print("layout:  ");print(layout8);print("\n");
    print("tiler :  ");print(tiler8);print("\n");
    print("\tzipped_divide:   ");print(zipped_divide(layout8,tiler8));print("\n");print("\n");

    print("=============== tiled_divide 2D ===============\n");   
    Layout layout8 = make_layout(make_shape(Int<8>{},Int<4>{},Int<3>{}),make_stride(Int<4>{},Int<1>{},Int<32>{}));
    Tile tiler8 = make_tile(Layout<_4,_2>{},Layout<_2,_2>{});
    print("layout:  ");print(layout8);print("\n");
    print("tiler :  ");print(tiler8);print("\n");
    print("\ttiled_divide:   ");print(tiled_divide(layout8,tiler8));print("\n");print("\n");

    print("=============== flat_divide 2D ===============\n");   
    Layout layout8 = make_layout(make_shape(Int<8>{},Int<4>{},Int<3>{}),make_stride(Int<4>{},Int<1>{},Int<32>{}));
    Tile tiler8 = make_tile(Layout<_4,_2>{},Layout<_2,_2>{});
    print("layout:  ");print(layout8);print("\n");
    print("tiler :  ");print(tiler8);print("\n");
    print("\tflat_divide:   ");print(flat_divide(layout8,tiler8));print("\n");print("\n");
}