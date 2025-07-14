#include "cute/tensor.hpp"

#include <iostream>

using namespace cute;

template <class Shape, class Stride>
void print1D(Layout<Shape,Stride> const& layout)
{
  for (int i = 0; i < size(layout); ++i) {
    printf("%3d  ", layout(i));
  }
}

int main()
{
    Layout<Shape<_2, _3>, Stride<_1, _2>>layout0;
    print_layout(layout0);
    print("printf 1D:    ");print1D(layout0);print("\n");

    Layout<Shape<_2, _3>, Stride<_3, _1>>layout1;
    print_layout(layout1);
    print("printf 1D:    ");print1D(layout1);print("\n");
    return 0;
}