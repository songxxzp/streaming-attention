/**
 * @file debug_qk_norm.cpp
 * @brief Compare QKNorm outputs between baseline and AVX2
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

    // Create test input - [1, 2, 1024]
    std::vector<float> input_data(1 * 2 * 1024);
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = static_cast<float>(i % 100) / 100.0f;
    }
    Tensor hidden_states(std::move(input_data), Shape({1, 2, 1024}));

    size_t batch = 1;
    size_t seq_len = 2;
    size_t hidden_size = 1024;
    size_t num_attention_heads = 16;
    size_t num_key_value_heads = 8;
    size_t head_dim = 128;

    // Compute QKV projections using baseline
    Tensor hidden_reshaped = hidden_states.view({static_cast<long>(batch * seq_len), static_cast<long>(hidden_size)});

    // Extract QKV from qkv_projs
    size_t q_size = num_attention_heads * head_dim;
    size_t k_size = num_key_value_heads * head_dim;
    size_t v_size = num_key_value_heads * head_dim;

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

    Tensor q_out_baseline = linear(hidden_reshaped, q_proj_baseline, nullptr);
    Tensor k_out_baseline = linear(hidden_reshaped, k_proj_baseline, nullptr);
    Tensor v_out_baseline = linear(hidden_reshaped, v_proj_baseline, nullptr);

    // Reshape and transpose
    Tensor q_reshaped_baseline = q_out_baseline.view({batch, seq_len, q_size / head_dim, head_dim});
    Tensor k_reshaped_baseline = k_out_baseline.view({batch, seq_len, k_size / head_dim, head_dim});
    Tensor q_baseline = q_reshaped_baseline.transpose(1, 2);
    Tensor k_baseline = k_reshaped_baseline.transpose(1, 2);

    std::cout << "Baseline Q shape: [" << q_baseline.shape()[0] << ", " << q_baseline.shape()[1] << ", " << q_baseline.shape()[2] << ", " << q_baseline.shape()[3] << "]\n";
    std::cout << "Baseline K shape: [" << k_baseline.shape()[0] << ", " << k_baseline.shape()[1] << ", " << k_baseline.shape()[2] << ", " << k_baseline.shape()[3] << "]\n\n";

    // Apply QKNorm manually (baseline implementation)
    size_t q_total_elements = batch * num_attention_heads * seq_len * head_dim;
    std::vector<float> q_normed_baseline_data(q_total_elements);
    const float* q_norm_weight_data = layer.q_norm_weight.data();
    const float* q_baseline_data = q_baseline.data();

    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < num_attention_heads; ++h) {
            for (size_t s = 0; s < seq_len; ++s) {
                size_t base_idx = ((b * num_attention_heads + h) * seq_len + s) * head_dim;
                float sum_sq = 0.0f;
                for (size_t i = 0; i < head_dim; ++i) {
                    float val = q_baseline_data[base_idx + i];
                    sum_sq += val * val;
                }
                float rms = std::sqrt(sum_sq / static_cast<float>(head_dim)) + 1e-6f;
                for (size_t i = 0; i < head_dim; ++i) {
                    q_normed_baseline_data[base_idx + i] = (q_baseline_data[base_idx + i] / rms) * q_norm_weight_data[i];
                }
            }
        }
    }
    Tensor q_normed_baseline(std::move(q_normed_baseline_data), q_baseline.shape());

    // Now AVX2 version
    Tensor q_out_avx2 = linear_avx2(hidden_reshaped, layer.q_proj, nullptr);
    Tensor k_out_avx2 = linear_avx2(hidden_reshaped, layer.k_proj, nullptr);

    Tensor q_reshaped_avx2 = q_out_avx2.view({batch, seq_len, num_attention_heads, head_dim});
    Tensor k_reshaped_avx2 = k_out_avx2.view({batch, seq_len, num_key_value_heads, head_dim});
    Tensor q_avx2 = q_reshaped_avx2.transpose(1, 2);
    Tensor k_avx2 = k_reshaped_avx2.transpose(1, 2);

    std::cout << "AVX2 Q shape: [" << q_avx2.shape()[0] << ", " << q_avx2.shape()[1] << ", " << q_avx2.shape()[2] << ", " << q_avx2.shape()[3] << "]\n";
    std::cout << "AVX2 K shape: [" << k_avx2.shape()[0] << ", " << k_avx2.shape()[1] << ", " << k_avx2.shape()[2] << ", " << k_avx2.shape()[3] << "]\n\n";

    // Apply QKNorm manually (AVX2 implementation)
    std::vector<float> q_normed_avx2_data(q_total_elements);
    const float* q_avx2_data = q_avx2.data();

    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < num_attention_heads; ++h) {
            for (size_t s = 0; s < seq_len; ++s) {
                size_t base_idx = ((b * num_attention_heads + h) * seq_len + s) * head_dim;
                float sum_sq = 0.0f;
                for (size_t i = 0; i < head_dim; ++i) {
                    float val = q_avx2_data[base_idx + i];
                    sum_sq += val * val;
                }
                float rms = std::sqrt(sum_sq / static_cast<float>(head_dim)) + 1e-6f;
                for (size_t i = 0; i < head_dim; ++i) {
                    q_normed_avx2_data[base_idx + i] = (q_avx2_data[base_idx + i] / rms) * q_norm_weight_data[i];
                }
            }
        }
    }
    Tensor q_normed_avx2(std::move(q_normed_avx2_data), q_avx2.shape());

    // Compare
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

    std::cout << "============================================================\n";
    std::cout << "Comparison\n";
    std::cout << "============================================================\n\n";

    compare("Q projection (before QKNorm)", q_baseline, q_avx2);
    compare("Q after QKNorm", q_normed_baseline, q_normed_avx2);

    return 0;
}
