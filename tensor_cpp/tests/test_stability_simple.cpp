#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/ops.h"
#include <iostream>
#include <cmath>
#include <limits>

using namespace tensor_cpp;
using namespace tensor_cpp::qwen3;

void check_stats(const Tensor& t, const char* name) {
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
    
    std::cout << name << ": mean=" << t.mean() << ", range=[" << min_val << ", " << max_val << "]";
    if (num_large > 0) std::cout << " (" << num_large << " > 1000)";
    if (has_inf) std::cout << " ⚠️  INF";
    if (has_nan) std::cout << " ⚠️  NAN";
    std::cout << "\n";
}

int main() {
    try {
        Qwen3Weights weights = load_qwen3_weights("/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors");
        
        std::vector<long> input_ids_data = {9707};
        Shape input_shape({1, 1});
        TensorL input_ids(input_ids_data, input_shape);
        
        Tensor hidden = ops::embedding(input_ids, weights.embed_tokens);
        check_stats(hidden, "Input");
        
        for (int layer_idx = 0; layer_idx < 3; ++layer_idx) {
            std::cout << "\n--- Layer " << layer_idx << " ---\n";
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
            
            check_stats(hidden, "Output");
        }
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
