#include "tensor.hh"
#include "Eigen/Dense"
#include <vector>
#include <iostream>

int main()
{
    std::vector<size_t> shape = {2, 3, 4};
    std::vector<size_t> flatten_shape = {24};
    Tensor<float> t(shape);
    t[{0, 0, 2}] = 5;

    assert(t.shape() == shape);
    assert(t.size() == 24);
    assert(t.ndim() == 3);
    t.flatten();
    assert(t.shape() == flatten_shape);
    assert(t.size() == 24);
    assert(t.ndim() == 1);

    assert(t[{2}] == 5);

    return 0;
}