/**
 * @file qwen3_loader.h
 * @brief Qwen3 weight loading from safetensors format
 */

#ifndef TENSOR_CPP_QWEN3_LOADER_H
#define TENSOR_CPP_QWEN3_LOADER_H

#include "tensor_cpp/tensor.h"
#include "tensor_cpp/qwen3_ops.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <cstring>  // for memcpy

namespace tensor_cpp {
namespace qwen3 {

/**
 * @brief BF16 to float32 conversion
 *
 * BF16 (bfloat16) format:
 * - 1 sign bit
 * - 8 exponent bits (same as FP32)
 * - 7 mantissa bits (truncated from FP32)
 */
inline float bf16_to_float(uint16_t bf16) {
    // Convert BF16 to FP32
    // BF16: S[15] E[14:7] M[6:0]
    // FP32: S[31] E[30:23] M[22:0]

    uint32_t bits = static_cast<uint32_t>(bf16) << 16;  // Shift to align exponent
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

/**
 * @brief Tensor metadata from safetensors header
 */
struct TensorInfo {
    std::string dtype;
    std::vector<size_t> shape;
    size_t offset_start;
    size_t offset_end;

    size_t total_elements() const {
        size_t total = 1;
        for (auto s : shape) total *= s;
        return total;
    }
};

/**
 * @brief Qwen3 model weights container
 */
struct Qwen3Weights {
    // Embeddings
    Tensor embed_tokens;  // [vocab_size, hidden_size]

    // Final norm
    Tensor norm_weight;   // [hidden_size]

    // LM head (often tied with embed_tokens)
    Tensor lm_head;       // [vocab_size, hidden_size]

    // All layers
    std::vector<Qwen3LayerWeights> layers;

    // Config
    size_t num_layers;
    size_t num_attention_heads;
    size_t num_key_value_heads;
    size_t head_dim;
    size_t hidden_size;
    size_t vocab_size;
};

/**
 * @brief Load Qwen3 weights from safetensors file
 *
 * @param path Path to model.safetensors
 * @return Loaded weights
 */
Qwen3Weights load_qwen3_weights(const std::string& path);

/**
 * @brief Simple test: load weights and run forward pass
 *
 * @param safetensors_path Path to model.safetensors
 * @param input_ids Input token IDs [batch, seq_len]
 * @return Output hidden states
 */
Tensor test_qwen3_forward(
    const std::string& safetensors_path,
    const std::vector<long>& input_ids_data,
    const std::vector<size_t>& input_shape
);

} // namespace qwen3
} // namespace tensor_cpp

#endif // TENSOR_CPP_QWEN3_LOADER_H
