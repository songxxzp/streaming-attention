/**
 * @file tensor.h
 * @brief PyTorch-style Tensor class declaration in C++
 *
 * Provides a lightweight tensor library with common deep learning operations.
 * Backed by std::vector for simplicity and portability.
 */

#ifndef TENSOR_LIB_TENSOR_H
#define TENSOR_LIB_TENSOR_H

#include <vector>
#include <memory>
#include <functional>
#include <initializer_list>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <type_traits>

namespace tensor_lib {

/**
 * @brief Shape class for tensor dimensions
 */
class Shape {
public:
    std::vector<size_t> dims;

    Shape() = default;
    Shape(const std::vector<size_t>& d);
    Shape(std::initializer_list<size_t> init);

    size_t ndim() const;
    size_t operator[](size_t i) const;
    size_t& operator[](size_t i);

    size_t total() const;

    bool operator==(const Shape& other) const;
    bool operator!=(const Shape& other) const;

    std::string to_string() const;
};

/**
 * @brief Main Tensor class
 *
 * @tparam T Data type (float, int, etc.)
 */
template <typename T = float>
class Tensor {
private:
    std::vector<T> data_;
    Shape shape_;

    // Helper function for index computation
    size_t flat_index(const std::vector<size_t>& indices) const;

public:
    // ========== Constructors ==========

    Tensor() = default;

    explicit Tensor(const Shape& shape);
    Tensor(const Shape& shape, const T& value);
    Tensor(const std::vector<T>& data, const Shape& shape);
    Tensor(std::vector<T>&& data, const Shape& shape) noexcept;

    // Copy constructor
    Tensor(const Tensor& other) = default;

    // Move assignment
    Tensor& operator=(Tensor&& other) noexcept;

    // Copy assignment
    Tensor& operator=(const Tensor& other) = default;

    // ========== Accessors ==========

    const Shape& shape() const;
    size_t ndim() const;
    size_t size() const;
    size_t total() const;

    T* data();
    const T* data() const;

    // Element access
    T& operator[](size_t index);
    const T& operator[](size_t index) const;

    // Multi-dimensional access
    T& operator()(const std::vector<size_t>& indices);
    const T& operator()(const std::vector<size_t>& indices) const;

    // ========== Element-wise Operations ==========

    Tensor<T> operator+(const Tensor<T>& other) const;
    Tensor<T> operator+(const T& scalar) const;
    Tensor<T> operator-(const Tensor<T>& other) const;
    Tensor<T> operator*(const Tensor<T>& other) const;
    Tensor<T> operator*(const T& scalar) const;
    Tensor<T> operator/(const Tensor<T>& other) const;
    Tensor<T> operator/(const T& scalar) const;

    // In-place operations
    Tensor<T>& operator+=(const Tensor<T>& other);
    Tensor<T>& operator*=(const T& scalar);

    // ========== Reduction Operations ==========

    T sum() const;
    T mean() const;
    T max() const;
    T min() const;

    Tensor<T> sum(int dim) const;
    Tensor<T> mean(int dim) const;
    Tensor<T> max(int dim) const;

    // ========== Matrix Operations ==========

    Tensor<T> matmul(const Tensor<T>& other) const;
    Tensor<T> transpose() const;

    // ========== Shape Manipulations ==========

    Tensor<T> reshape(const Shape& new_shape) const;
    Tensor<T> squeeze() const;
    Tensor<T> unsqueeze(size_t dim) const;

    // ========== Mathematical Functions ==========

    Tensor<T> exp() const;
    Tensor<T> log() const;
    Tensor<T> sqrt() const;
    Tensor<T> square() const;
    Tensor<T> abs() const;

    // ========== Utility Functions ==========

    void fill(const T& value);
    void zero();

    std::string to_string() const;

    // ========== Static Factory Methods ==========

    static Tensor<T> zeros(const Shape& shape);
    static Tensor<T> ones(const Shape& shape);
    static Tensor<T> randn(const Shape& shape);
    static Tensor<T> uniform(const Shape& shape, T low = 0, T high = 1);
    static Tensor<T> eye(size_t n, size_t m = 0);
};

// Type aliases
using TensorF = Tensor<float>;
using TensorD = Tensor<double>;
using TensorI = Tensor<int>;
using TensorL = Tensor<long>;

} // namespace tensor_lib

// Include template implementations
#include "tensor_impl.tpp"

#endif // TENSOR_LIB_TENSOR_H
