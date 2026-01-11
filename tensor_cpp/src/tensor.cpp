/**
 * @file tensor.cpp
 * @brief Tensor class implementation
 */

#include "tensor_cpp/tensor.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>

namespace tensor_cpp {

// ========== Constructors ==========

Tensor::Tensor(const Shape& shape) : shape_(shape) {
    data_.resize(shape_.total());
}

Tensor::Tensor(const Shape& shape, float value) : shape_(shape) {
    data_.resize(shape_.total(), value);
}

Tensor::Tensor(const std::vector<float>& data, const Shape& shape)
    : data_(data), shape_(shape) {
    if (data.size() != shape.total()) {
        throw std::invalid_argument("Data size does not match shape");
    }
}

Tensor::Tensor(std::vector<float>&& data, const Shape& shape) noexcept
    : data_(std::move(data)), shape_(shape) {
    if (data_.size() != shape_.total()) {
        // Note: can't throw in noexcept, but this shouldn't happen
    }
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        data_ = std::move(other.data_);
        shape_ = other.shape_;
    }
    return *this;
}

// ========== Multi-dimensional Access ==========

size_t Tensor::flat_index(const std::vector<size_t>& indices) const {
    size_t index = 0;
    size_t stride = 1;
    for (int i = static_cast<int>(shape_.dims.size()) - 1; i >= 0; --i) {
        index += indices[i] * stride;
        stride *= shape_.dims[i];
    }
    return index;
}

float& Tensor::operator()(const std::vector<size_t>& indices) {
    return data_[flat_index(indices)];
}

const float& Tensor::operator()(const std::vector<size_t>& indices) const {
    return data_[flat_index(indices)];
}

// ========== Element-wise Operations ==========

Tensor Tensor::operator+(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Shape mismatch for addition");
    }
    std::vector<float> result(data_.size());
    for (size_t i = 0; i < data_.size(); ++i) {
        result[i] = data_[i] + other.data_[i];
    }
    return Tensor(std::move(result), shape_);
}

Tensor Tensor::operator+(float scalar) const {
    std::vector<float> result(data_.size());
    for (size_t i = 0; i < data_.size(); ++i) {
        result[i] = data_[i] + scalar;
    }
    return Tensor(std::move(result), shape_);
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Shape mismatch for subtraction");
    }
    std::vector<float> result(data_.size());
    for (size_t i = 0; i < data_.size(); ++i) {
        result[i] = data_[i] - other.data_[i];
    }
    return Tensor(std::move(result), shape_);
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Shape mismatch for multiplication");
    }
    std::vector<float> result(data_.size());
    for (size_t i = 0; i < data_.size(); ++i) {
        result[i] = data_[i] * other.data_[i];
    }
    return Tensor(std::move(result), shape_);
}

Tensor Tensor::operator*(float scalar) const {
    std::vector<float> result(data_.size());
    for (size_t i = 0; i < data_.size(); ++i) {
        result[i] = data_[i] * scalar;
    }
    return Tensor(std::move(result), shape_);
}

Tensor Tensor::operator/(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Shape mismatch for division");
    }
    std::vector<float> result(data_.size());
    for (size_t i = 0; i < data_.size(); ++i) {
        result[i] = data_[i] / other.data_[i];
    }
    return Tensor(std::move(result), shape_);
}

Tensor Tensor::operator/(float scalar) const {
    std::vector<float> result(data_.size());
    for (size_t i = 0; i < data_.size(); ++i) {
        result[i] = data_[i] / scalar;
    }
    return Tensor(std::move(result), shape_);
}

Tensor& Tensor::operator+=(const Tensor& other) {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Shape mismatch for in-place addition");
    }
    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] += other.data_[i];
    }
    return *this;
}

Tensor& Tensor::operator*=(float scalar) {
    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] *= scalar;
    }
    return *this;
}

// ========== Reduction Operations ==========

float Tensor::sum() const {
    return std::accumulate(data_.begin(), data_.end(), 0.0f);
}

float Tensor::mean() const {
    if (data_.empty()) return 0.0f;
    return sum() / static_cast<float>(data_.size());
}

float Tensor::max() const {
    if (data_.empty()) {
        throw std::runtime_error("Cannot compute max of empty tensor");
    }
    return *std::max_element(data_.begin(), data_.end());
}

float Tensor::min() const {
    if (data_.empty()) {
        throw std::runtime_error("Cannot compute min of empty tensor");
    }
    return *std::min_element(data_.begin(), data_.end());
}

// ========== Matrix Operations ==========

Tensor Tensor::matmul(const Tensor& other) const {
    if (ndim() != 2 || other.ndim() != 2) {
        throw std::invalid_argument("Matrix multiplication requires 2D tensors");
    }
    if (shape_[1] != other.shape_[0]) {
        throw std::invalid_argument("Shape mismatch for matrix multiplication");
    }

    size_t M = shape_[0];
    size_t K = shape_[1];
    size_t N = other.shape_[1];

    std::vector<float> result(M * N, 0.0f);

    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            for (size_t k = 0; k < K; ++k) {
                result[m * N + n] += data_[m * K + k] * other.data_[k * N + n];
            }
        }
    }

    return Tensor(std::move(result), Shape({M, N}));
}

Tensor Tensor::transpose() const {
    if (ndim() != 2) {
        throw std::invalid_argument("Transpose only supports 2D tensors");
    }

    size_t rows = shape_[0];
    size_t cols = shape_[1];

    std::vector<float> result(data_.size());
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[j * rows + i] = data_[i * cols + j];
        }
    }

    return Tensor(std::move(result), Shape({cols, rows}));
}

// ========== Shape Manipulations ==========

Tensor Tensor::reshape(const Shape& new_shape) const {
    if (new_shape.total() != data_.size()) {
        throw std::invalid_argument("New shape must have same total size");
    }
    return Tensor(data_, new_shape);
}

Tensor Tensor::squeeze() const {
    std::vector<size_t> new_dims;
    for (auto d : shape_.dims) {
        if (d != 1) new_dims.push_back(d);
    }
    if (new_dims.empty()) new_dims.push_back(1);
    return reshape(Shape(new_dims));
}

Tensor Tensor::unsqueeze(size_t dim) const {
    std::vector<size_t> new_dims = shape_.dims;
    new_dims.insert(new_dims.begin() + dim, 1);
    return reshape(Shape(new_dims));
}

// ========== Mathematical Functions ==========

Tensor Tensor::exp() const {
    std::vector<float> result(data_.size());
    for (size_t i = 0; i < data_.size(); ++i) {
        result[i] = std::exp(data_[i]);
    }
    return Tensor(std::move(result), shape_);
}

Tensor Tensor::log() const {
    std::vector<float> result(data_.size());
    for (size_t i = 0; i < data_.size(); ++i) {
        result[i] = std::log(data_[i]);
    }
    return Tensor(std::move(result), shape_);
}

Tensor Tensor::sqrt() const {
    std::vector<float> result(data_.size());
    for (size_t i = 0; i < data_.size(); ++i) {
        result[i] = std::sqrt(data_[i]);
    }
    return Tensor(std::move(result), shape_);
}

Tensor Tensor::square() const {
    std::vector<float> result(data_.size());
    for (size_t i = 0; i < data_.size(); ++i) {
        result[i] = data_[i] * data_[i];
    }
    return Tensor(std::move(result), shape_);
}

Tensor Tensor::abs() const {
    std::vector<float> result(data_.size());
    for (size_t i = 0; i < data_.size(); ++i) {
        result[i] = std::abs(data_[i]);
    }
    return Tensor(std::move(result), shape_);
}

// ========== Utility Functions ==========

void Tensor::fill(float value) {
    std::fill(data_.begin(), data_.end(), value);
}

void Tensor::zero() {
    fill(0.0f);
}

std::string Tensor::to_string() const {
    std::ostringstream oss;
    oss << "Tensor(shape=" << shape_.to_string() << ")\n";
    oss << "data:\n";

    if (data_.empty()) {
        oss << "  []\n";
        return oss.str();
    }

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

Tensor Tensor::zeros(const Shape& shape) {
    return Tensor(shape, 0.0f);
}

Tensor Tensor::ones(const Shape& shape) {
    return Tensor(shape, 1.0f);
}

Tensor Tensor::randn(const Shape& shape) {
    Tensor result(shape);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < result.data_.size(); ++i) {
        result.data_[i] = dist(gen);
    }
    return result;
}

Tensor Tensor::uniform(const Shape& shape, float low, float high) {
    Tensor result(shape);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(low, high);

    for (size_t i = 0; i < result.data_.size(); ++i) {
        result.data_[i] = dist(gen);
    }
    return result;
}

Tensor Tensor::eye(size_t n, size_t m) {
    if (m == 0) m = n;
    Tensor result(Shape({n, m}));
    result.zero();
    size_t min_dim = std::min(n, m);
    for (size_t i = 0; i < min_dim; ++i) {
        result.data_[i * m + i] = 1.0f;
    }
    return result;
}

} // namespace tensor_cpp
