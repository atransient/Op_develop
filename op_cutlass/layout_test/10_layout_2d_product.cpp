#include "cute/tensor.hpp"

using namespace cute;

int main()
{
    print("=============== logical_product 2D===============\n");
    Layout layout_A1 = Layout<_2,_5>{};
    Layout layout_B1 = Layout<_3,_5>{};
    print("layout_A:  ");print(layout_A1);print("\n");
    print("layout_B:  ");print(layout_B1);print("\n");
    print("\tlogical_product:   ");print(logical_product(layout_A1,layout_B1));print("\n");print("\n");

    Layout layout_A2 = Layout<_5,_1>{};
    Layout layout_B2 = Layout<_4,_6>{};
    print("layout_A:  ");print(layout_A2);print("\n");
    print("layout_B:  ");print(layout_B2);print("\n");
    print("\tlogical_product:   ");print(logical_product(layout_A2,layout_B2));print("\n");print("\n");

    Layout layout1 = make_layout(make_shape(Int<2>{},Int<5>{}),make_stride(Int<5>{},Int<1>{}));
    Tile tiler1 = make_tile(Layout<_3,_5>{},Layout<_4,_6>{});
    print("layout:  ");print(layout1);print("\n");
    print("tiler :  ");print(tiler1);print("\n");
    print("\tlogical_product:   ");print(logical_product(layout1,tiler1));print("\n");print("\n");

    Layout layout2 = make_layout(make_shape(Int<2>{},Int<5>{},Int<4>{}),make_stride(Int<5>{},Int<1>{},Int<120>{}));
    Tile tiler2 = make_tile(Layout<_3,_5>{},Layout<_4,_6>{});
    print("layout:  ");print(layout2);print("\n");
    print("tiler :  ");print(tiler2);print("\n");
    print("\tlogical_product:   ");print(logical_product(layout2,tiler2));print("\n");print("\n");

    print("=============== zipped_product 2D===============\n");
    Layout layout3 = make_layout(make_shape(Int<2>{},Int<5>{}),make_stride(Int<5>{},Int<1>{}));
    Tile tiler3 = make_tile(Layout<_3,_5>{},Layout<_4,_6>{});
    print("layout:  ");print(layout3);print("\n");
    print("tiler :  ");print(tiler3);print("\n");
    print("\tzipped_product:   ");print(zipped_product(layout3,tiler3));print("\n");print("\n");

    Layout layout4 = make_layout(make_shape(Int<2>{},Int<5>{},Int<4>{}),make_stride(Int<5>{},Int<1>{},Int<120>{}));
    Tile tiler4 = make_tile(Layout<_3,_5>{},Layout<_4,_6>{});
    print("layout:  ");print(layout4);print("\n");
    print("tiler :  ");print(tiler4);print("\n");
    print("\tzipped_product:   ");print(zipped_product(layout4,tiler4));print("\n");print("\n");

    print("=============== tiled_product 2D===============\n");
    Layout layout5 = make_layout(make_shape(Int<2>{},Int<5>{}),make_stride(Int<5>{},Int<1>{}));
    Tile tiler5 = make_tile(Layout<_3,_5>{},Layout<_4,_6>{});
    print("layout:  ");print(layout5);print("\n");
    print("tiler :  ");print(tiler5);print("\n");
    print("\ttiled_product:   ");print(tiled_product(layout5,tiler5));print("\n");print("\n");

    Layout layout6 = make_layout(make_shape(Int<2>{},Int<5>{},Int<4>{}),make_stride(Int<5>{},Int<1>{},Int<120>{}));
    Tile tiler6 = make_tile(Layout<_3,_5>{},Layout<_4,_6>{});
    print("layout:  ");print(layout6);print("\n");
    print("tiler :  ");print(tiler6);print("\n");
    print("\ttiled_product:   ");print(tiled_product(layout6,tiler6));print("\n");print("\n");

    print("=============== flat_product 2D===============\n");
    Layout layout7 = make_layout(make_shape(Int<2>{},Int<5>{}),make_stride(Int<5>{},Int<1>{}));
    Tile tiler7 = make_tile(Layout<_3,_5>{},Layout<_4,_6>{});
    print("layout:  ");print(layout7);print("\n");
    print("tiler :  ");print(tiler7);print("\n");
    print("\tflat_product:   ");print(flat_product(layout7,tiler7));print("\n");print("\n");

    Layout layout8 = make_layout(make_shape(Int<2>{},Int<5>{},Int<4>{}),make_stride(Int<5>{},Int<1>{},Int<120>{}));
    Tile tiler8 = make_tile(Layout<_3,_5>{},Layout<_4,_6>{});
    print("layout:  ");print(layout8);print("\n");
    print("tiler :  ");print(tiler8);print("\n");
    print("\tflat_product:   ");print(flat_product(layout8,tiler8));print("\n");
}