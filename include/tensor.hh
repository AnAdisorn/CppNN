#pragma once
#include <Eigen/Dense>
#include <vector>
#include <iterator>

template <typename T>
class Tensor
{
public:
    Tensor(const std::vector<size_t> &shape) : _shape(shape)
    {
        assert(!std::empty(shape));
        _ndim = _shape.size();

        size_t size = 1;
        for (size_t i = 0; i < _ndim; i++)
            size *= _shape[i];

        _size = size;
        _arr.resize(size);
    }

    const std::vector<T> &arr() const
    {
        return _arr;
    }

    std::vector<size_t> shape() const
    {
        return _shape;
    }

    size_t size() const
    {
        return _size;
    }

    size_t ndim() const
    {
        return _ndim;
    }

    void reshape(const std::vector<size_t> &shape)
    {
        assert(!std::empty(shape));
        size_t size = 1;

        for (auto it = shape.begin(); it != shape.end(); ++it)
        {
            size *= *it;
        }
        assert(size == _size);

        _shape = shape;
        _ndim = _shape.size();
    }

    void flatten()
    {
        reshape({_size});
    }

    T &operator[](const std::vector<size_t> &indices)
    {
        assert(indices.size() == _ndim);

        size_t index = 0;
        size_t stride = 1;

        size_t i = 0;
        for (auto it = indices.rbegin(); it != indices.rend(); ++it)
        {
            index += stride * (*it);
            stride *= _shape[i];
            ++i;
        }

        assert(index < _size);
        return _arr[index];
    }

    void setArrIndex(size_t index, T val)
    {
        _arr[i] = val;
    }

    Tensor<T> operator+(const Tensor<T> &other)
    {
        assert(_shape == other.shape());
        Tensor<T> result({_shape});

        auto &other_arr = other.arr();
#pragma omp parallel for
        for (size_t i = 0; i < _size; i++)
        {
            result.setArr(i, _arr[i] + other_arr[i]);
        }

        return result;
    }

    void operator+=(const Tensor<T> &other)
    {
        assert(_shape == other.shape());

        auto &other_arr = other.arr();
#pragma omp parallel for
        for (size_t i = 0; i < _size; i++)
        {
            _arr[i] += other_arr[i];
        }
    }

private:
    std::vector<T> _arr;
    std::vector<size_t> _shape;
    size_t _ndim;
    size_t _size;
};