/**
 * @file debug_qkv_projection.cpp
 * @brief 比较baseline和AVX2的QKV projection输出
 */

#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/attention_avx.h"
#include <iostream>
#include <cmath>

using namespace tensor_cpp;
using namespace tensor_cpp::ops;
using namespace tensor_cpp::ops::avx2;
using namespace tensor_cpp::qwen3;

int main() {
    std::string model_path = "/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors";
    std::cout << "Loading model...\n";
    Qwen3Weights weights = load_qwen3_weights(model_path);

    const auto& layer = weights.layers[0];

    // 创建测试输入 - [1, 2, 1024]
    std::vector<float> input_data(1 * 2 * 1024);
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = static_cast<float>(i % 100) / 100.0f;
    }
    Tensor hidden_states(std::move(input_data), Shape({1, 2, 1024}));

    std::cout << "\nInput shape: [" << hidden_states.shape()[0] << ", "
              << hidden_states.shape()[1] << ", " << hidden_states.shape()[2] << "]\n";
    std::cout << "qkv_projs shape: [" << layer.qkv_projs.shape()[0] << ", "
              << layer.qkv_projs.shape()[1] << "]\n\n";

    size_t batch = 1;
    size_t seq_len = 2;
    size_t hidden_size = 1024;
    size_t q_size = 2048;
    size_t k_size = 1024;
    size_t v_size = 1024;

    // ========== Baseline: 从qkv_projs提取并计算 ==========
    std::cout << "Baseline: Extract QKV from qkv_projs and project...\n";

    // 提取q_proj, k_proj, v_proj
    std::vector<float> q_data_baseline(q_size * hidden_size);
    std::vector<float> k_data_baseline(k_size * hidden_size);
    std::vector<float> v_data_baseline(v_size * hidden_size);

    const float* qkv_data = layer.qkv_projs.data();
    for (size_t row = 0; row < q_size; ++row) {
        for (size_t col = 0; col < hidden_size; ++col) {
            q_data_baseline[row * hidden_size + col] = qkv_data[row * hidden_size + col];
        }
    }
    for (size_t row = 0; row < k_size; ++row) {
        for (size_t col = 0; col < hidden_size; ++col) {
            k_data_baseline[row * hidden_size + col] = qkv_data[(q_size + row) * hidden_size + col];
        }
    }
    for (size_t row = 0; row < v_size; ++row) {
        for (size_t col = 0; col < hidden_size; ++col) {
            v_data_baseline[row * hidden_size + col] = qkv_data[(q_size + k_size + row) * hidden_size + col];
        }
    }

    Tensor q_proj_baseline(std::move(q_data_baseline), Shape({static_cast<long>(q_size), static_cast<long>(hidden_size)}));
    Tensor k_proj_baseline(std::move(k_data_baseline), Shape({static_cast<long>(k_size), static_cast<long>(hidden_size)}));
    Tensor v_proj_baseline(std::move(v_data_baseline), Shape({static_cast<long>(v_size), static_cast<long>(hidden_size)}));

    // Reshape to [batch*seq_len, hidden_size] and project
    Tensor hidden_reshaped = hidden_states.view({static_cast<long>(batch * seq_len), static_cast<long>(hidden_size)});
    Tensor q_out_baseline = linear(hidden_reshaped, q_proj_baseline, nullptr);
    Tensor k_out_baseline = linear(hidden_reshaped, k_proj_baseline, nullptr);
    Tensor v_out_baseline = linear(hidden_reshaped, v_proj_baseline, nullptr);

    std::cout << "  Q output shape: [" << q_out_baseline.shape()[0] << ", " << q_out_baseline.shape()[1] << "]\n";
    std::cout << "  K output shape: [" << k_out_baseline.shape()[0] << ", " << k_out_baseline.shape()[1] << "]\n";
    std::cout << "  V output shape: [" << v_out_baseline.shape()[0] << ", " << v_out_baseline.shape()[1] << "]\n\n";

    // ========== AVX2: 直接使用分开的q_proj, k_proj, v_proj ==========
    std::cout << "AVX2: Use separate q_proj, k_proj, v_proj...\n";

    Tensor q_out_avx2 = linear_avx2(hidden_reshaped, layer.q_proj, nullptr);
    Tensor k_out_avx2 = linear_avx2(hidden_reshaped, layer.k_proj, nullptr);
    Tensor v_out_avx2 = linear_avx2(hidden_reshaped, layer.v_proj, nullptr);

    std::cout << "  Q output shape: [" << q_out_avx2.shape()[0] << ", " << q_out_avx2.shape()[1] << "]\n";
    std::cout << "  K output shape: [" << k_out_avx2.shape()[0] << ", " << k_out_avx2.shape()[1] << "]\n";
    std::cout << "  V output shape: [" << v_out_avx2.shape()[0] << ", " << v_out_avx2.shape()[1] << "]\n\n";

    // ========== 比较 ==========
    std::cout << "============================================================\n";
    std::cout << "Comparison\n";
    std::cout << "============================================================\n\n";

    auto compare = [](const std::string& name, const Tensor& t1, const Tensor& t2) {
        if (t1.size() != t2.size()) {
            std::cout << name << ": SIZE MISMATCH\n";
            return;
        }

        float max_diff = 0.0f;
        float max_rel_diff = 0.0f;
        for (size_t i = 0; i < t1.size(); ++i) {
            float diff = std::abs(t1.data()[i] - t2.data()[i]);
            max_diff = std::max(max_diff, diff);

            float abs_val = std::max(std::abs(t1.data()[i]), std::abs(t2.data()[i]));
            float rel_diff = (abs_val > 1e-6f) ? (diff / abs_val) : diff;
            max_rel_diff = std::max(max_rel_diff, rel_diff);
        }

        std::cout << name << ":\n";
        std::cout << "  Max abs diff: " << max_diff << "\n";
        std::cout << "  Max rel diff: " << (max_rel_diff * 100.0f) << "%\n";

        if (max_diff > 0.1f || max_rel_diff > 0.01f) {
            std::cout << "  *** ERROR ***\n";
        } else {
            std::cout << "  *** OK ***\n";
        }
        std::cout << "\n";
    };

    compare("Q projection", q_out_baseline, q_out_avx2);
    compare("K projection", k_out_baseline, k_out_avx2);
    compare("V projection", v_out_baseline, v_out_avx2);

    return 0;
}
