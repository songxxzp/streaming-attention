/**
 * @file debug_avx2_layers.cpp
 * @brief 逐层比较baseline和AVX2的输出来定位bug
 */

#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/qwen3_ops_avx.h"
#include "tensor_cpp/ops.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace tensor_cpp;
using namespace tensor_cpp::qwen3;
using namespace tensor_cpp::ops;

// 比较两个tensor的输出
void compare_tensors(const std::string& name, const Tensor& t1, const Tensor& t2, float threshold = 0.001f) {
    if (t1.size() != t2.size()) {
        std::cout << "  " << name << ": SIZE MISMATCH (" << t1.size() << " vs " << t2.size() << ")\n";
        return;
    }

    float max_diff = 0.0f;
    float max_rel_diff = 0.0f;
    int mismatch_count = 0;

    for (size_t i = 0; i < t1.size(); ++i) {
        float v1 = t1.data()[i];
        float v2 = t2.data()[i];
        float diff = std::abs(v1 - v2);

        if (diff > max_diff) {
            max_diff = diff;
        }

        // 相对误差
        float abs_val = std::max(std::abs(v1), std::abs(v2));
        float rel_diff = (abs_val > 1e-6f) ? (diff / abs_val) : diff;

        if (rel_diff > max_rel_diff) {
            max_rel_diff = rel_diff;
        }

        if (diff > threshold) {
            mismatch_count++;
        }
    }

    std::cout << "  " << name << ":\n";
    std::cout << "    Max abs diff: " << max_diff << "\n";
    std::cout << "    Max rel diff: " << (max_rel_diff * 100.0f) << "%\n";
    std::cout << "    Mismatch count (>" << threshold << "): " << mismatch_count << " / " << t1.size();

    if (max_diff > 1.0f || max_rel_diff > 0.1f) {
        std::cout << " *** ERROR ***";
    }
    std::cout << "\n";
}

int main() {
    std::string model_path = "/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors";
    std::cout << "Loading model...\n";
    Qwen3Weights weights = load_qwen3_weights(model_path);
    std::cout << "Model loaded!\n\n";

    // 简单的测试输入 - 只用2个token
    std::vector<long> input_ids = {100, 200};
    Shape input_shape({1, 2});
    TensorL input(input_ids, input_shape);

    std::cout << "Test input: 2 tokens\n";
    std::cout << "Running baseline and AVX2...\n\n";

    // 运行baseline
    Tensor output_baseline = qwen3::qwen3_forward(
        input,
        weights.embed_tokens,
        weights.layers,
        weights.norm_weight,
        weights.num_layers,
        weights.num_attention_heads,
        weights.num_key_value_heads,
        weights.head_dim,
        1e-6f
    );

    // 运行AVX2
    Tensor output_avx2 = avx2::qwen3_forward_avx(
        input,
        weights.embed_tokens,
        weights.layers,
        weights.norm_weight,
        weights.num_layers,
        weights.num_attention_heads,
        weights.num_key_value_heads,
        weights.head_dim,
        1e-6f
    );

    std::cout << "\n============================================================\n";
    std::cout << "FULL OUTPUT COMPARISON\n";
    std::cout << "============================================================\n\n";

    compare_tensors("Final Output", output_baseline, output_avx2, 0.001f);

    std::cout << "\n============================================================\n";
    std::cout << "LAYER-BY-LAYER COMPARISON\n";
    std::cout << "============================================================\n\n";

    // Precompute RoPE
    size_t seq_len = 2;
    auto [cos, sin] = compute_rope_freqs(seq_len, weights.head_dim, 1000000.0f);

    // 逐层比较
    Tensor hidden_baseline = embedding(input, weights.embed_tokens);
    Tensor hidden_avx2 = embedding(input, weights.embed_tokens);
    compare_tensors("Layer 0: Embedding", hidden_baseline, hidden_avx2);

    for (size_t layer_idx = 0; layer_idx < weights.num_layers; ++layer_idx) {
        const auto& layer = weights.layers[layer_idx];

        // Baseline forward (使用qkv_projs)
        Tensor out_baseline = qwen3_decoder_layer(
            hidden_baseline,
            weights.num_attention_heads,
            weights.num_key_value_heads,
            weights.head_dim,
            1e-6f,
            layer.input_layernorm_weight,
            layer.qkv_projs,  // 注意：使用合并的QKV权重
            layer.o_proj,
            layer.q_norm_weight,
            layer.k_norm_weight,
            layer.post_attention_layernorm_weight,
            layer.gate_proj,
            layer.up_proj,
            layer.down_proj,
            cos,
            sin
        );

        // AVX2 forward (使用分开的q_proj, k_proj, v_proj)
        Tensor out_avx2 = avx2::qwen3_decoder_layer_avx(
            hidden_avx2,
            weights.num_attention_heads,
            weights.num_key_value_heads,
            weights.head_dim,
            1e-6f,
            layer.input_layernorm_weight,
            layer.q_proj,  // 使用分开的QKV权重
            layer.k_proj,
            layer.v_proj,
            layer.o_proj,
            layer.q_norm_weight,
            layer.k_norm_weight,
            layer.post_attention_layernorm_weight,
            layer.gate_proj,
            layer.up_proj,
            layer.down_proj,
            cos,
            sin
        );

        compare_tensors("Layer " + std::to_string(layer_idx + 1) + " output", out_baseline, out_avx2, 0.001f);

        hidden_baseline = out_baseline;
        hidden_avx2 = out_avx2;

        // 如果找到错误，可以提前退出
        if (layer_idx >= 2) {  // 只测试前3层
            std::cout << "\n... (stopping after layer 3 for debugging)\n";
            break;
        }
    }

    return 0;
}
