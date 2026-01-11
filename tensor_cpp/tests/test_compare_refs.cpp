#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/ops.h"
#include <iostream>
#include <fstream>
#include <cmath>

using namespace tensor_cpp;
using namespace tensor_cpp::qwen3;

void compare_tensor(const Tensor& cpp, const char* filename, const char* name) {
    std::ifstream ref(filename, std::ios::binary);
    if (!ref) {
        std::cout << "  " << name << ": FILE NOT FOUND\n";
        return;
    }
    
    std::vector<float> pytorch(cpp.size());
    ref.read(reinterpret_cast<char*>(pytorch.data()), cpp.size() * sizeof(float));
    
    float max_diff = 0;
    for (size_t i = 0; i < cpp.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(cpp[i] - pytorch[i]));
    }
    
    float pytorch_mean = 0;
    for (size_t i = 0; i < pytorch.size(); ++i) pytorch_mean += pytorch[i];
    pytorch_mean /= pytorch.size();
    
    std::cout << "  " << name << ":\n";
    std::cout << "    C++:      mean=" << cpp.mean() << ", range=[" << cpp.min() << ", " << cpp.max() << "]\n";
    std::cout << "    PyTorch: mean=" << pytorch_mean << ", range=[";
    float pytorch_min = pytorch[0], pytorch_max = pytorch[0];
    for (size_t i = 1; i < pytorch.size(); ++i) {
        pytorch_min = std::min(pytorch_min, pytorch[i]);
        pytorch_max = std::max(pytorch_max, pytorch[i]);
    }
    std::cout << pytorch_min << ", " << pytorch_max << "]\n";
    std::cout << "    Max diff: " << max_diff;
    
    if (max_diff < 0.01) {
        std::cout << " ✓\n";
    } else if (max_diff < 0.1) {
        std::cout << " ⚠\n";
    } else {
        std::cout << " ✗\n";
    }
}

int main() {
    try {
        Qwen3Weights weights = load_qwen3_weights("/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors");
        
        // Create input
        std::vector<long> input_ids_data = {9707};
        Shape input_shape({1, 1});
        TensorL input_ids(input_ids_data, input_shape);
        
        Tensor hidden = ops::embedding(input_ids, weights.embed_tokens);
        compare_tensor(hidden, "/tmp/pytorch_input_ref.bin", "Input embedding");
        
        // Run layer 0
        const Qwen3LayerWeights& layer = weights.layers[0];
        size_t num_heads = weights.num_attention_heads;
        size_t kv_heads = weights.num_key_value_heads;
        size_t head_dim = weights.head_dim;
        
        auto [cos, sin] = qwen3::compute_rope_freqs(hidden.shape()[1], head_dim, 1000000.0f);
        
        Tensor output = qwen3::qwen3_decoder_layer(
            hidden, num_heads, kv_heads, head_dim, 1e-6f,
            layer.input_layernorm_weight, layer.qkv_projs, layer.o_proj,
            layer.q_norm_weight, layer.k_norm_weight,
            layer.post_attention_layernorm_weight,
            layer.gate_proj, layer.up_proj, layer.down_proj,
            cos, sin
        );
        
        compare_tensor(output, "/tmp/pytorch_layer0_output_ref.bin", "Layer 0 output");
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
