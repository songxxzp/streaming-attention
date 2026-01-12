/**
 * @file debug_qkv_weights.cpp
 * @brief 检查qkv_projs和分开的q_proj, k_proj, v_proj是否一致
 */

#include "tensor_cpp/qwen3_loader.h"
#include <iostream>
#include <cmath>

using namespace tensor_cpp;
using namespace tensor_cpp::qwen3;

int main() {
    std::string model_path = "/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors";
    std::cout << "Loading model...\n";
    Qwen3Weights weights = load_qwen3_weights(model_path);

    const auto& layer = weights.layers[0];

    std::cout << "\nLayer 0 QKV weights:\n";
    std::cout << "  q_proj shape: [" << layer.q_proj.shape()[0] << ", " << layer.q_proj.shape()[1] << "]\n";
    std::cout << "  k_proj shape: [" << layer.k_proj.shape()[0] << ", " << layer.k_proj.shape()[1] << "]\n";
    std::cout << "  v_proj shape: [" << layer.v_proj.shape()[0] << ", " << layer.v_proj.shape()[1] << "]\n";
    std::cout << "  qkv_projs shape: [" << layer.qkv_projs.shape()[0] << ", " << layer.qkv_projs.shape()[1] << "]\n\n";

    // 从qkv_projs提取q_proj
    size_t q_size = layer.q_proj.shape()[0];
    size_t k_size = layer.k_proj.shape()[0];
    size_t hidden_size = layer.q_proj.shape()[1];

    std::cout << "Extracting from qkv_projs:\n";
    std::cout << "  q_size: " << q_size << "\n";
    std::cout << "  k_size: " << k_size << "\n";
    std::cout << "  hidden_size: " << hidden_size << "\n\n";

    // 提取q_proj
    std::vector<float> q_extracted(q_size * hidden_size);
    for (size_t row = 0; row < q_size; ++row) {
        for (size_t col = 0; col < hidden_size; ++col) {
            q_extracted[row * hidden_size + col] = layer.qkv_projs.data()[row * hidden_size + col];
        }
    }

    // 提取k_proj
    std::vector<float> k_extracted(k_size * hidden_size);
    for (size_t row = 0; row < k_size; ++row) {
        for (size_t col = 0; col < hidden_size; ++col) {
            k_extracted[row * hidden_size + col] = layer.qkv_projs.data()[(q_size + row) * hidden_size + col];
        }
    }

    // 提取v_proj
    size_t v_size = layer.v_proj.shape()[0];
    std::vector<float> v_extracted(v_size * hidden_size);
    for (size_t row = 0; row < v_size; ++row) {
        for (size_t col = 0; col < hidden_size; ++col) {
            v_extracted[row * hidden_size + col] = layer.qkv_projs.data()[(q_size + k_size + row) * hidden_size + col];
        }
    }

    // 比较q_proj
    std::cout << "Comparing q_proj:\n";
    float max_diff_q = 0.0f;
    for (size_t i = 0; i < layer.q_proj.size(); ++i) {
        float diff = std::abs(layer.q_proj.data()[i] - q_extracted[i]);
        max_diff_q = std::max(max_diff_q, diff);
    }
    std::cout << "  Max diff: " << max_diff_q << "\n";

    // 比较k_proj
    std::cout << "Comparing k_proj:\n";
    float max_diff_k = 0.0f;
    for (size_t i = 0; i < layer.k_proj.size(); ++i) {
        float diff = std::abs(layer.k_proj.data()[i] - k_extracted[i]);
        max_diff_k = std::max(max_diff_k, diff);
    }
    std::cout << "  Max diff: " << max_diff_k << "\n";

    // 比较v_proj
    std::cout << "Comparing v_proj:\n";
    float max_diff_v = 0.0f;
    for (size_t i = 0; i < layer.v_proj.size(); ++i) {
        float diff = std::abs(layer.v_proj.data()[i] - v_extracted[i]);
        max_diff_v = std::max(max_diff_v, diff);
    }
    std::cout << "  Max diff: " << max_diff_v << "\n";

    if (max_diff_q < 1e-6f && max_diff_k < 1e-6f && max_diff_v < 1e-6f) {
        std::cout << "\n*** OK: Weights match! ***\n";
    } else {
        std::cout << "\n*** ERROR: Weights don't match! ***\n";
    }

    return 0;
}
