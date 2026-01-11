#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/ops.h"
#include <iostream>

using namespace tensor_cpp;
using namespace tensor_cpp::qwen3;

int main() {
    try {
        std::cout << "Checking attention output dimensions...\n\n";

        // Load weights
        Qwen3Weights weights = load_qwen3_weights("/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors");
        const Qwen3LayerWeights& layer = weights.layers[0];

        // Create input
        std::vector<float> input_data(1024);
        for (size_t i = 0; i < input_data.size(); ++i) input_data[i] = 0.1f;
        Tensor hidden(input_data, Shape({1, 1, 1024}));  // [batch, seq, hidden]

        std::cout << "Input shape: [" << hidden.shape()[0] << ", " << hidden.shape()[1] << ", " << hidden.shape()[2] << "]\n\n";

        // Call qwen3_attention
        size_t num_heads = weights.num_attention_heads;
        size_t kv_heads = weights.num_key_value_heads;
        size_t head_dim = weights.head_dim;

        auto [cos, sin] = compute_rope_freqs(1, head_dim, 1000000.0f);

        Tensor attn_out = qwen3_attention(
            hidden,
            num_heads,
            kv_heads,
            head_dim,
            layer.qkv_projs,
            layer.o_proj,
            layer.q_norm_weight,
            layer.k_norm_weight,
            cos,
            sin,
            false
        );

        std::cout << "Attention output shape: [" << attn_out.shape()[0] << ", " << attn_out.shape()[1] << ", " << attn_out.shape()[2] << "]\n";
        std::cout << "Expected: [1, 1, 1024]\n\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
