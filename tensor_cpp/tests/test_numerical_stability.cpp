#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/ops.h"
#include <iostream>
#include <cmath>
#include <limits>

using namespace tensor_cpp;
using namespace tensor_cpp::qwen3;

void check_tensor_stats(const Tensor& t, const char* name) {
    float min_val = std::numeric_limits<float>::max();
    float max_val = -std::numeric_limits<float>::max();
    bool has_inf = false;
    bool has_nan = false;
    size_t num_large = 0;
    
    for (size_t i = 0; i < t.size(); ++i) {
        float val = t[i];
        if (std::isinf(val)) has_inf = true;
        if (std::isnan(val)) has_nan = true;
        if (std::abs(val) > 1000.0f) num_large++;
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
    }
    
    std::cout << name << ":\n";
    std::cout << "  Shape: [";
    for (size_t i = 0; i < t.shape().ndim(); ++i) {
        std::cout << t.shape()[i] << (i < t.shape().ndim() - 1 ? ", " : "");
    }
    std::cout << "]\n";
    std::cout << "  mean=" << t.mean() << ", range=[" << min_val << ", " << max_val << "]\n";
    std::cout << "  Elements > 1000: " << num_large << "/" << t.size();
    if (has_inf) std::cout << " ⚠️  HAS INF!";
    if (has_nan) std::cout << " ⚠️  HAS NAN!";
    std::cout << "\n";
}

int main() {
    try {
        std::cout << "Checking numerical stability...\n\n";
        
        Qwen3Weights weights = load_qwen3_weights("/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors");
        
        // Create input
        std::vector<long> input_ids_data = {9707};
        Shape input_shape({1, 1});
        TensorL input_ids(input_ids_data, input_shape);
        
        Tensor hidden = ops::embedding(input_ids, weights.embed_tokens);
        check_tensor_stats(hidden, "1. Input embedding");
        
        // Process first few layers with detailed stats
        for (int layer_idx = 0; layer_idx < 3; ++layer_idx) {
            std::cout << "\n============================================================\n";
            std::cout << "Layer " << layer_idx << "\n";
            std::cout << "============================================================\n";
            
            const Qwen3LayerWeights& layer = weights.layers[layer_idx];
            size_t num_heads = weights.num_attention_heads;
            size_t kv_heads = weights.num_key_value_heads;
            size_t head_dim = weights.head_dim;
            
            auto [cos, sin] = qwen3::compute_rope_freqs(hidden.shape()[1], head_dim, 1000000.0f);
            
            // Input layernorm
            Tensor hidden_ln = ops::rms_norm(hidden, &layer.input_layernorm_weight, 1e-6f);
            check_tensor_stats(hidden_ln, "2. After input_layernorm");
            
            // Attention
            Tensor attn_out = qwen3::qwen3_attention(
                hidden_ln, num_heads, kv_heads, head_dim,
                layer.qkv_projs, layer.o_proj,
                layer.q_norm_weight, layer.k_norm_weight,
                cos, sin, false
            );
            check_tensor_stats(attn_out, "3. After attention");
            
            // Residual
            Tensor residual = hidden;
            hidden = residual + attn_out;
            check_tensor_stats(hidden, "4. After residual 1");
            
            // Post layernorm
            hidden_ln = ops::rms_norm(hidden, &layer.post_attention_layernorm_weight, 1e-6f);
            check_tensor_stats(hidden_ln, "5. After post_layernorm");
            
            // MLP
            Tensor mlp_out = qwen3::qwen3_mlp(hidden_ln, layer.gate_proj, layer.up_proj, layer.down_proj);
            check_tensor_stats(mlp_out, "6. MLP output");
            
            // Residual
            hidden = hidden + mlp_out;
            check_tensor_stats(hidden, "7. Layer output");
        }
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
