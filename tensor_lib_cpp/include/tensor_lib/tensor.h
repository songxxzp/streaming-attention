/**
 * @file tensor.h
 * @brief PyTorch-style Tensor class implementation in C++
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
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <type_traits>
#include <random>

namespace tensor_lib {

/**
 * @brief Shape class for tensor dimensions
 */
class Shape {
public:
    std::vector<size_t> dims;

    Shape() = default;
    Shape(const std::vector<size_t>& d) : dims(d) {}
    Shape(std::initializer_list<size_t> init) : dims(init) {}

    size_t ndim() const { return dims.size(); }
    size_t operator[](size_t i) const { return dims[i]; }
    size_t& operator[](size_t i) { return dims[i]; }

    size_t total() const {
        if (dims.empty()) return 0;
        size_t total = 1;
        for (auto d : dims) total *= d;
        return total;
    }

    bool operator==(const Shape& other) const { return dims == other.dims; }
    bool operator!=(const Shape& other) const { return dims != other.dims; }

    std::string to_string() const {
        std::string s = "(";
        for (size_t i = 0; i < dims.size(); ++i) {
            if (i > 0) s += ", ";
            s += std::to_string(dims[i]);
        }
        s += ")";
        return s;
    }
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

    // Helper functions for index computation
    size_t flat_index(const std::vector<size_t>& indices) const {
        size_t index = 0;
        size_t stride = 1;
        for (int i = static_cast<int>(shape_.dims.size()) - 1; i >= 0; --i) {
            index += indices[i] * stride;
            stride *= shape_.dims[i];
        }
        return index;
    }

public:
    // ========== Constructors ==========

    Tensor() = default;

    Tensor(const Shape& shape) : shape_(shape) {
        data_.resize(shape_.total());
    }

    Tensor(const Shape& shape, const T& value) : shape_(shape) {
        data_.resize(shape_.total(), value);
    }

    Tensor(const std::vector<T>& data, const Shape& shape)
        : data_(data), shape_(shape) {
        if (data.size() != shape.total()) {
            throw std::invalid_argument("Data size does not match shape");
        }
    }

    // Move constructor
    Tensor(std::vector<T>&& data, const Shape& shape) noexcept
        : data_(std::move(data)), shape_(shape) {}

    // Copy constructor
    Tensor(const Tensor& other) = default;

    // Move assignment
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
            shape_ = other.shape_;
        }
        return *this;
    }

    // Copy assignment
    Tensor& operator=(const Tensor& other) = default;

    // ========== Accessors ==========

    const Shape& shape() const { return shape_; }
    size_t ndim() const { return shape_.ndim(); }
    size_t size() const { return data_.size(); }
    size_t total() const { return shape_.total(); }

    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

    // Element access
    T& operator[](size_t index) { return data_[index]; }
    const T& operator[](size_t index) const { return data_[index]; }

    // Multi-dimensional access
    T& operator()(const std::vector<size_t>& indices) {
        return data_[flat_index(indices)];
    }

    const T& operator()(const std::vector<size_t>& indices) const {
        return data_[flat_index(indices)];
    }

    // ========== Element-wise Operations ==========

    Tensor<T> operator+(const Tensor<T>& other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Shape mismatch for addition");
        }
        std::vector<T> result(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) {
            result[i] = data_[i] + other.data_[i];
        }
        return Tensor(std::move(result), shape_);
    }

    Tensor<T> operator+(const T& scalar) const {
        std::vector<T> result(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) {
            result[i] = data_[i] + scalar;
        }
        return Tensor(std::move(result), shape_);
    }

    Tensor<T> operator-(const Tensor<T>& other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Shape mismatch for subtraction");
        }
        std::vector<T> result(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) {
            result[i] = data_[i] - other.data_[i];
        }
        return Tensor(std::move(result), shape_);
    }

    Tensor<T> operator*(const Tensor<T>& other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Shape mismatch for multiplication");
        }
        std::vector<T> result(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) {
            result[i] = data_[i] * other.data_[i];
        }
        return Tensor(std::move(result), shape_);
    }

    Tensor<T> operator*(const T& scalar) const {
        std::vector<T> result(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) {
            result[i] = data_[i] * scalar;
        }
        return Tensor(std::move(result), shape_);
    }

    Tensor<T> operator/(const Tensor<T>& other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Shape mismatch for division");
        }
        std::vector<T> result(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) {
            result[i] = data_[i] / other.data_[i];
        }
        return Tensor(std::move(result), shape_);
    }

    Tensor<T> operator/(const T& scalar) const {
        std::vector<T> result(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) {
            result[i] = data_[i] / scalar;
        }
        return Tensor(std::move(result), shape_);
    }

    // In-place operations
    Tensor<T>& operator+=(const Tensor<T>& other) {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Shape mismatch for in-place addition");
        }
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] += other.data_[i];
        }
        return *this;
    }

    Tensor<T>& operator*=(const T& scalar) {
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] *= scalar;
        }
        return *this;
    }

    // ========== Reduction Operations ==========

    T sum() const {
        return std::accumulate(data_.begin(), data_.end(), static_cast<T>(0));
    }

    T mean() const {
        if (data_.empty()) return static_cast<T>(0);
        return sum() / static_cast<T>(data_.size());
    }

    T max() const {
        if (data_.empty()) {
            throw std::runtime_error("Cannot compute max of empty tensor");
        }
        return *std::max_element(data_.begin(), data_.end());
    }

    T min() const {
        if (data_.empty()) {
            throw std::runtime_error("Cannot compute min of empty tensor");
        }
        return *std::min_element(data_.begin(), data_.end());
    }

    Tensor<T> sum(int dim) const;
    Tensor<T> mean(int dim) const;
    Tensor<T> max(int dim) const;

    // ========== Matrix Operations ==========

    Tensor<T> matmul(const Tensor<T>& other) const {
        // Only support 2D matrix multiplication for now
        if (ndim() != 2 || other.ndim() != 2) {
            throw std::invalid_argument("Matrix multiplication requires 2D tensors");
        }
        if (shape_[1] != other.shape_[0]) {
            throw std::invalid_argument("Shape mismatch for matrix multiplication");
        }

        size_t M = shape_[0];
        size_t K = shape_[1];
        size_t N = other.shape_[1];

        std::vector<T> result(M * N, static_cast<T>(0));

        // Naive matrix multiplication: O(M*K*N)
        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; ++n) {
                for (size_t k = 0; k < K; ++k) {
                    result[m * N + n] += data_[m * K + k] * other.data_[k * N + n];
                }
            }
        }

        return Tensor(std::move(result), Shape({M, N}));
    }

    // Matrix transpose (2D only)
    Tensor<T> transpose() const {
        if (ndim() != 2) {
            throw std::invalid_argument("Transpose only supports 2D tensors");
        }

        size_t rows = shape_[0];
        size_t cols = shape_[1];

        std::vector<T> result(data_.size());
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result[j * rows + i] = data_[i * cols + j];
            }
        }

        return Tensor(std::move(result), Shape({cols, rows}));
    }

    // ========== Shape Manipulations ==========

    Tensor<T> reshape(const Shape& new_shape) const {
        if (new_shape.total() != data_.size()) {
            throw std::invalid_argument("New shape must have same total size");
        }
        return Tensor(data_, new_shape);
    }

    Tensor<T> squeeze() const {
        std::vector<size_t> new_dims;
        for (auto d : shape_.dims) {
            if (d != 1) new_dims.push_back(d);
        }
        if (new_dims.empty()) new_dims.push_back(1);
        return reshape(Shape(new_dims));
    }

    Tensor<T> unsqueeze(size_t dim) const {
        std::vector<size_t> new_dims = shape_.dims;
        new_dims.insert(new_dims.begin() + dim, 1);
        return reshape(Shape(new_dims));
    }

    // ========== Mathematical Functions ==========

    Tensor<T> exp() const {
        std::vector<T> result(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) {
            result[i] = std::exp(data_[i]);
        }
        return Tensor(std::move(result), shape_);
    }

    Tensor<T> log() const {
        std::vector<T> result(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) {
            result[i] = std::log(data_[i]);
        }
        return Tensor(std::move(result), shape_);
    }

    Tensor<T> sqrt() const {
        std::vector<T> result(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) {
            result[i] = std::sqrt(data_[i]);
        }
        return Tensor(std::move(result), shape_);
    }

    Tensor<T> square() const {
        std::vector<T> result(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) {
            result[i] = data_[i] * data_[i];
        }
        return Tensor(std::move(result), shape_);
    }

    Tensor<T> abs() const {
        std::vector<T> result(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) {
            result[i] = std::abs(data_[i]);
        }
        return Tensor(std::move(result), shape_);
    }

    // ========== Utility Functions ==========

    void fill(const T& value) {
        std::fill(data_.begin(), data_.end(), value);
    }

    void zero() {
        fill(static_cast<T>(0));
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "Tensor(shape=" << shape_.to_string() << ", dtype=" << typeid(T).name() << ")\n";
        oss << "data:\n";

        if (data_.empty()) {
            oss << "  []\n";
            return oss.str();
        }

        // Simple printing for 1D, 2D, and 3D tensors
        if (ndim() == 1) {
            oss << "  [";
            for (size_t i = 0; i < data_.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << std::fixed << std::setprecision(4) << data_[i];
            }
            oss << "]";
        } else if (ndim() == 2) {
            size_t rows = shape_[0];
            size_t cols = shape_[1];
            oss << "  [[";
            for (size_t i = 0; i < rows; ++i) {
                if (i > 0) oss << "   [";
                for (size_t j = 0; j < cols; ++j) {
                    if (j > 0) oss << ", ";
                    oss << std::fixed << std::setprecision(4) << data_[i * cols + j];
                }
                oss << "]";
                if (i < rows - 1) oss << ",\n";
            }
            oss << "]";
        }

        return oss.str();
    }

    // ========== Static Factory Methods ==========

    static Tensor<T> zeros(const Shape& shape) {
        return Tensor(shape, static_cast<T>(0));
    }

    static Tensor<T> ones(const Shape& shape) {
        return Tensor(shape, static_cast<T>(1));
    }

    static Tensor<T> randn(const Shape& shape) {
        Tensor result(shape);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<T> dist(0, 1);

        for (size_t i = 0; i < result.data_.size(); ++i) {
            result.data_[i] = dist(gen);
        }
        return result;
    }

    static Tensor<T> uniform(const Shape& shape, T low = 0, T high = 1) {
        Tensor result(shape);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(low, high);

        for (size_t i = 0; i < result.data_.size(); ++i) {
            result.data_[i] = dist(gen);
        }
        return result;
    }

    static Tensor<T> eye(size_t n, size_t m = 0) {
        if (m == 0) m = n;
        Tensor result(Shape({n, m}));
        result.zero();
        size_t min_dim = std::min(n, m);
        for (size_t i = 0; i < min_dim; ++i) {
            result.data_[i * m + i] = static_cast<T>(1);
        }
        return result;
    }
};

// Type aliases
using TensorF = Tensor<float>;
using TensorD = Tensor<double>;
using TensorI = Tensor<int>;
using TensorL = Tensor<long>;

} // namespace tensor_lib

#endif // TENSOR_LIB_TENSOR_H
