/**
 * @file tensor.h
 * @brief PyTorch-style Tensor class (non-template, float-only)
 *
 * Designed for simplicity and performance, inspired by llama.cpp
 */

#ifndef TENSOR_CPP_TENSOR_H
#define TENSOR_CPP_TENSOR_H

#include <vector>
#include <memory>
#include <initializer_list>
#include <stdexcept>
#include <iostream>
#include <iomanip>

namespace tensor_cpp {

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
 * @brief Main Tensor class (float-only, non-template)
 *
 * Designed for performance and simplicity, similar to llama.cpp approach
 */
class Tensor {
private:
    std::vector<float> data_;
    Shape shape_;

    size_t flat_index(const std::vector<size_t>& indices) const;

public:
    // ========== Constructors ==========

    Tensor() = default;
    explicit Tensor(const Shape& shape);
    Tensor(const Shape& shape, float value);
    Tensor(const std::vector<float>& data, const Shape& shape);
    Tensor(std::vector<float>&& data, const Shape& shape) noexcept;

    // Copy/move
    Tensor(const Tensor& other) = default;
    Tensor& operator=(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other) = default;

    // ========== Accessors ==========

    const Shape& shape() const { return shape_; }
    size_t ndim() const { return shape_.ndim(); }
    size_t size() const { return data_.size(); }
    size_t total() const { return shape_.total(); }

    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }

    // Element access
    float& operator[](size_t index) { return data_[index]; }
    const float& operator[](size_t index) const { return data_[index]; }

    float& operator()(const std::vector<size_t>& indices);
    const float& operator()(const std::vector<size_t>& indices) const;

    // ========== Element-wise Operations ==========

    Tensor operator+(const Tensor& other) const;
    Tensor operator+(float scalar) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator*(float scalar) const;
    Tensor operator/(const Tensor& other) const;
    Tensor operator/(float scalar) const;

    Tensor& operator+=(const Tensor& other);
    Tensor& operator*=(float scalar);

    // ========== Reduction Operations ==========

    float sum() const;
    float mean() const;
    float max() const;
    float min() const;

    // ========== Matrix Operations ==========

    Tensor matmul(const Tensor& other) const;
    Tensor transpose() const;

    // ========== Shape Manipulations ==========

    Tensor reshape(const Shape& new_shape) const;
    Tensor squeeze() const;
    Tensor unsqueeze(size_t dim) const;

    // ========== Mathematical Functions ==========

    Tensor exp() const;
    Tensor log() const;
    Tensor sqrt() const;
    Tensor square() const;
    Tensor abs() const;

    // ========== Utility Functions ==========

    void fill(float value);
    void zero();

    std::string to_string() const;

    // ========== Static Factory Methods ==========

    static Tensor zeros(const Shape& shape);
    static Tensor ones(const Shape& shape);
    static Tensor randn(const Shape& shape);
    static Tensor uniform(const Shape& shape, float low = 0, float high = 1);
    static Tensor eye(size_t n, size_t m = 0);
};

// Type alias for compatibility
using TensorF = Tensor;

} // namespace tensor_cpp

#endif // TENSOR_CPP_TENSOR_H
