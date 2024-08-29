#pragma once
#include "omp.h"
#include <vector>
#include <cassert>

template <typename T>
class Tensor
{
public:
    // Constructors and destructors
    Tensor(const std::vector<size_t> &shape);

    // Accessors
    const std::vector<T> &arr() const;
    std::vector<size_t> shape() const;
    size_t size() const;
    size_t ndim() const;

    // Modifiers
    void reshape(const std::vector<size_t> &shape);
    void flatten();
    void setValue(size_t index, T val);

    // Element access
    T &operator[](const std::vector<size_t> &indices);

    // Arithmetic operations
    Tensor<T> operator+(const Tensor<T> &other);
    Tensor<T> operator-(const Tensor<T> &other);
    void operator+=(const Tensor<T> &other);
    void operator-=(const Tensor<T> &other);

private:
    // Private members
    std::vector<T> _arr;
    std::vector<size_t> _shape;
    size_t _ndim;
    size_t _size;
};