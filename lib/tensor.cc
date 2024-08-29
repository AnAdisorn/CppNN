#include "tensor.hh"

template <typename T>
Tensor<T>::Tensor(const std::vector<size_t> &shape) : _shape(shape)
{
    assert(!std::empty(shape));
    _ndim = _shape.size();

    size_t size = 1;
    for (size_t i = 0; i < _ndim; i++)
        size *= _shape[i];

    _size = size;
    _arr.resize(size);
}

template <typename T>
const std::vector<T> &Tensor<T>::arr() const
{
    return _arr;
}

template <typename T>
std::vector<size_t> Tensor<T>::shape() const
{
    return _shape;
}

template <typename T>
size_t Tensor<T>::size() const
{
    return _size;
}

template <typename T>
size_t Tensor<T>::ndim() const
{
    return _ndim;
}

template <typename T>
void Tensor<T>::reshape(const std::vector<size_t> &shape)
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

template <typename T>
void Tensor<T>::flatten()
{
    reshape({_size});
}

template <typename T>
void Tensor<T>::setValue(size_t index, T val)
{
    assert(index <= _size);
    _arr[index] = val;
}

template <typename T>
T &Tensor<T>::operator[](const std::vector<size_t> &indices)
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

template <typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T> &other)
{
    assert(_shape == other.shape());
    Tensor<T> result({_shape});

    auto &other_arr = other.arr();
#pragma omp parallel for
    for (size_t i = 0; i < _size; i++)
    {
        result.setValue(i, _arr[i] + other_arr[i]);
    }

    return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator-(const Tensor<T> &other)
{
    assert(_shape == other.shape());
    Tensor<T> result({_shape});

    auto &other_arr = other.arr();
#pragma omp parallel for
    for (size_t i = 0; i < _size; i++)
    {
        result.setValue(i, _arr[i] - other_arr[i]);
    }

    return result;
}

template <typename T>
void Tensor<T>::operator+=(const Tensor<T> &other)
{
    assert(_shape == other.shape());

    auto &other_arr = other.arr();
#pragma omp parallel for
    for (size_t i = 0; i < _size; i++)
    {
        _arr[i] += other_arr[i];
    }
}

template <typename T>
void Tensor<T>::operator-=(const Tensor<T> &other)
{
    assert(_shape == other.shape());

    auto &other_arr = other.arr();
#pragma omp parallel for
    for (size_t i = 0; i < _size; i++)
    {
        _arr[i] -= other_arr[i];
    }
}

template class Tensor<int>;
template class Tensor<float>;
template class Tensor<double>;