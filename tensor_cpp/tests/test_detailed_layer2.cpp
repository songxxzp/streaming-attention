#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/ops.h"
#include <iostream>
#include <cmath>
#include <limits>

using namespace tensor_cpp;
using namespace tensor_cpp::qwen3;

void detailed_stats(const Tensor& t, const char* name) {
    float min_val = t[0], max_val = t[0];
    bool has_inf = false, has_nan = false;
    size_t num_large = 0;

    for (size_t i = 0; i < t.size(); ++i) {
        float val = t[i];
        if (std::isinf(val)) has_inf = true;
        if (std::isnan(val)) has_nan = true;
        if (std::abs(val) > 1000.0f) num_large++;
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
    }

    std::cout << "  " << name << ": mean=" << t.mean() << ", range=[" << min_val << ", " << max_val << "]";
    if (num_large > 0) std::cout << " (" << num_large << " > 1000)";
    if (has_inf) std::cout << " ⚠️  INF";
    if (has_nan) std::cout << " ⚠️  NAN";
    std::cout << "\n";
}

int main() {
    try {
        std::cout << "Detailed Layer 2 Analysis to Find Numerical Explosion\n";
        std::cout << "========================================================\n\n";

        Qwen3Weights weights = load_qwen3_weights("/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors");

        // Create input
        std::vector<long> input_ids_data = {9707};
        Shape input_shape({1, 1});
        TensorL input_ids(input_ids_data, input_shape);

        Tensor hidden = ops::embedding(input_ids, weights.embed_tokens);
        detailed_stats(hidden, "Initial embedding");

        // Run through layers 0 and 1 first
        for (int layer_idx = 0; layer_idx < 2; ++layer_idx) {
            const Qwen3LayerWeights& layer = weights.layers[layer_idx];
            auto [cos, sin] = qwen3::compute_rope_freqs(1, 128, 1000000.0f);

            hidden = qwen3::qwen3_decoder_layer(
                hidden, 16, 8, 128, 1e-6f,
                layer.input_layernorm_weight, layer.qkv_projs, layer.o_proj,
                layer.q_norm_weight, layer.k_norm_weight,
                layer.post_attention_layernorm_weight,
                layer.gate_proj, layer.up_proj, layer.down_proj,
                cos, sin
            );

            detailed_stats(hidden, ("After layer " + std::to_string(layer_idx)).c_str());
        }

        // Now detailed analysis of Layer 2
        std::cout << "\n=== DETAILED LAYER 2 ANALYSIS ===\n\n";
        const Qwen3LayerWeights& layer = weights.layers[2];
        auto [cos, sin] = qwen3::compute_rope_freqs(1, 128, 1000000.0f);

        // Step 1: Input layernorm
        std::cout << "Step 1: Input LayerNorm\n";
        Tensor hidden_ln = ops::rms_norm(hidden, &layer.input_layernorm_weight, 1e-6f);
        detailed_stats(hidden_ln, "  After input_layernorm");

        // Step 2: QKV projections
        std::cout << "\nStep 2: QKV Projections\n";
        Tensor qkv = qwen3::qwen3_qkv_linear(hidden_ln, layer.qkv_projs);
        detailed_stats(qkv, "  QKV output");

        // Step 3: Split and reshape
        std::cout << "\nStep 3: Split Q, K, V\n";
        size_t num_heads = 16, kv_heads = 8, head_dim = 128;
        size_t q_size = num_heads * head_dim;
        size_t kv_size = kv_heads * head_dim;

        Tensor q = qkv.slice(0, 0, q_size);
        Tensor k = qkv.slice(0, q_size, q_size + kv_size);
        Tensor v = qkv.slice(0, q_size + kv_size, q_size + 2 * kv_size);

        q = q.view({1, num_heads, 1, head_dim});
        k = k.view({1, kv_heads, 1, head_dim});
        v = v.view({1, kv_heads, 1, head_dim});

        detailed_stats(q, "  Q (reshaped)");
        detailed_stats(k, "  K (reshaped)");
        detailed_stats(v, "  V (reshaped)");

        // Step 4: QKNorm
        std::cout << "\nStep 4: QKNorm\n";
        q = qwen3::apply_qk_norm(q, layer.q_norm_weight);
        k = qwen3::apply_qk_norm(k, layer.k_norm_weight);
        detailed_stats(q, "  Q after norm");
        detailed_stats(k, "  K after norm");

        // Step 5: RoPE
        std::cout << "\nStep 5: RoPE\n";
        qwen3::apply_rope_inplace(q, cos, sin);
        qwen3::apply_rope_inplace(k, cos, sin);
        detailed_stats(q, "  Q after RoPE");
        detailed_stats(k, "  K after RoPE");

        // Step 6: Repeat KV for GQA
        std::cout << "\nStep 6: Repeat KV (GQA)\n";
        k = qwen3::repeat_kv(k, num_heads / kv_heads);
        v = qwen3::repeat_kv(v, num_heads / kv_heads);
        detailed_stats(k, "  K after repeat");
        detailed_stats(v, "  V after repeat");

        // Step 7: Attention scores
        std::cout << "\nStep 7: Attention Scores\n";
        Tensor attn_scores = ops::matmul_transpose_b(q, k);
        detailed_stats(attn_scores, "  Attention scores (before scaling)");

        attn_scores = attn_scores * (1.0f / std::sqrt(static_cast<float>(head_dim)));
        detailed_stats(attn_scores, "  Attention scores (after scaling)");

        // Step 8: Softmax
        std::cout << "\nStep 8: Softmax\n";
        Tensor attn_weights = ops::softmax(attn_scores, -1);
        detailed_stats(attn_weights, "  Attention weights");

        // Step 9: Apply to V
        std::cout << "\nStep 9: Attention Output\n";
        Tensor attn_output = ops::matmul(attn_weights, v);
        detailed_stats(attn_output, "  After matmul with V");

        // Step 10: Reshape and transpose
        std::cout << "\nStep 10: Reshape for o_proj\n";
        attn_output = attn_output.view({1, 1, num_heads * head_dim});
        detailed_stats(attn_output, "  After view");

        // Step 11: o_proj
        std::cout << "\nStep 11: o_proj (output projection)\n";
        Tensor attn_out = ops::linear(attn_output, layer.o_proj, nullptr);
        detailed_stats(attn_out, "  After o_proj");

        // Step 12: Residual 1
        std::cout << "\nStep 12: Residual Connection 1\n";
        Tensor residual = hidden;
        hidden = residual + attn_out;
        detailed_stats(hidden, "  After residual + attention");

        // Step 13: Post attention layernorm
        std::cout << "\nStep 13: Post Attention LayerNorm\n";
        hidden_ln = ops::rms_norm(hidden, &layer.post_attention_layernorm_weight, 1e-6f);
        detailed_stats(hidden_ln, "  After post_layernorm");

        // Step 14: MLP - Gate projection
        std::cout << "\nStep 14: MLP - Gate Projection\n";
        Tensor gate = ops::linear(hidden_ln, layer.gate_proj, nullptr);
        detailed_stats(gate, "  Gate output");

        // Step 15: MLP - Up projection
        std::cout << "\nStep 15: MLP - Up Projection\n";
        Tensor up = ops::linear(hidden_ln, layer.up_proj, nullptr);
        detailed_stats(up, "  Up output");

        // Step 16: MLP - SwiGLU activation
        std::cout << "\nStep 16: MLP - SwiGLU Activation\n";
        const Shape& gate_shape = gate.shape();
        std::vector<float> gated_data(gate.size());
        #pragma omp parallel for if(gate.size() > 1000)
        for (size_t i = 0; i < gate.size(); ++i) {
            float silu_gate = gate[i] * (1.0f / (1.0f + std::exp(-gate[i])));
            gated_data[i] = silu_gate * up[i];
        }
        Tensor gated(Tensor::vector_type(gated_data.begin(), gated_data.end()), gate_shape);
        detailed_stats(gated, "  After SwiGLU");

        // Step 17: MLP - Down projection
        std::cout << "\nStep 17: MLP - Down Projection\n";
        Tensor mlp_out = ops::linear(gated, layer.down_proj, nullptr);
        detailed_stats(mlp_out, "  After down_proj");

        // Step 18: Residual 2
        std::cout << "\nStep 18: Residual Connection 2\n";
        hidden = hidden + mlp_out;
        detailed_stats(hidden, "  Final layer output");

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
