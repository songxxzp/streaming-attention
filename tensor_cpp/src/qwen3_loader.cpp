/**
 * @file qwen3_loader.cpp
 * @brief Qwen3 weight loading implementation
 */

#include "tensor_cpp/qwen3_loader.h"
#include <fstream>
#include <cstring>
#include <iostream>

namespace tensor_cpp {
namespace qwen3 {

// Simple JSON value extraction (for demo purposes)
static std::string extract_json_value(const std::string& json, const std::string& key) {
    // Very simple JSON parser - just for finding "key": "value" or "key": [values]
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";

    pos = json.find(':', pos);
    if (pos == std::string::npos) return "";

    // Skip whitespace
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n')) {
        pos++;
    }

    if (pos >= json.size()) return "";

    if (json[pos] == '"') {
        // String value
        size_t end = json.find('"', pos + 1);
        if (end == std::string::npos) return "";
        return json.substr(pos + 1, end - pos - 1);
    } else if (json[pos] == '[') {
        // Array value - find matching bracket
        int depth = 1;
        size_t end = pos + 1;
        while (end < json.size() && depth > 0) {
            if (json[end] == '[') depth++;
            else if (json[end] == ']') depth--;
            end++;
        }
        return json.substr(pos, end - pos);
    } else if (json[pos] == '{') {
        // Object value - find matching brace
        int depth = 1;
        size_t end = pos + 1;
        while (end < json.size() && depth > 0) {
            if (json[end] == '{') depth++;
            else if (json[end] == '}') depth--;
            end++;
        }
        return json.substr(pos, end - pos);
    } else {
        // Number value (simple case)
        size_t end = json.find_first_of(",}\n]", pos);
        if (end == std::string::npos) return "";
        return json.substr(pos, end - pos);
    }
}

static std::vector<size_t> parse_shape(const std::string& shape_str) {
    std::vector<size_t> result;
    std::string str = shape_str;

    // Remove brackets and split by comma
    size_t start = str.find('[');
    if (start == std::string::npos) return result;
    str = str.substr(start + 1);

    size_t end = str.rfind(']');
    if (end != std::string::npos) {
        str = str.substr(0, end);
    }

    // Parse comma-separated values
    size_t pos = 0;
    while (pos < str.size()) {
        // Skip whitespace
        while (pos < str.size() && (str[pos] == ' ' || str[pos] == '\t' || str[pos] == '\n')) {
            pos++;
        }
        if (pos >= str.size()) break;

        // Find next comma or end
        size_t next = str.find(',', pos);
        if (next == std::string::npos) next = str.size();

        std::string val_str = str.substr(pos, next - pos);
        if (!val_str.empty()) {
            result.push_back(std::stoull(val_str));
        }

        pos = next + 1;
    }

    return result;
}

static std::vector<size_t> parse_data_offsets(const std::string& offsets_str) {
    std::vector<size_t> result;
    std::string str = offsets_str;

    // Remove brackets
    size_t start = str.find('[');
    if (start == std::string::npos) return result;
    str = str.substr(start + 1);

    size_t end = str.rfind(']');
    if (end != std::string::npos) {
        str = str.substr(0, end);
    }

    // Parse comma-separated values
    size_t pos = 0;
    while (pos < str.size()) {
        while (pos < str.size() && (str[pos] == ' ' || str[pos] == '\t' || str[pos] == '\n')) {
            pos++;
        }
        if (pos >= str.size()) break;

        size_t next = str.find(',', pos);
        if (next == std::string::npos) next = str.size();

        std::string val_str = str.substr(pos, next - pos);
        if (!val_str.empty()) {
            result.push_back(std::stoull(val_str));
        }

        pos = next + 1;
    }

    return result;
}

// Helper function to load a single tensor from safetensors file
static Tensor load_tensor_from_file(
    std::ifstream& file,
    const std::string& header_json,
    const std::string& tensor_name,
    size_t header_len
) {
    std::string tensor_json = extract_json_value(header_json, tensor_name);
    if (tensor_json.empty()) {
        throw std::runtime_error("Tensor not found: " + tensor_name);
    }

    std::vector<size_t> shape = parse_shape(extract_json_value(tensor_json, "shape"));
    std::vector<size_t> offsets = parse_data_offsets(extract_json_value(tensor_json, "data_offsets"));

    size_t num_elements = 1;
    for (auto s : shape) num_elements *= s;

    std::vector<uint16_t> bf16_data(num_elements);
    file.seekg(offsets[0] + 8 + header_len, std::ios::beg);
    file.read(reinterpret_cast<char*>(bf16_data.data()), num_elements * sizeof(uint16_t));

    std::vector<float> float_data(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
        float_data[i] = bf16_to_float(bf16_data[i]);
    }

    return Tensor(std::move(float_data), Shape(shape));
}

Qwen3Weights load_qwen3_weights(const std::string& path) {
    Qwen3Weights weights;
    std::ifstream file(path, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open safetensors file: " + path);
    }

    // Read header length (first 8 bytes, little-endian)
    uint64_t header_len;
    file.read(reinterpret_cast<char*>(&header_len), sizeof(uint64_t));

    // Read JSON header
    std::vector<char> header_data(header_len + 1);
    file.read(header_data.data(), header_len);
    header_data[header_len] = '\0';
    std::string header_json(header_data.data());

    std::cout << "Loading weights from: " << path << std::endl;
    std::cout << "Header size: " << header_len << " bytes" << std::endl;

    // Load base weights using helper function
    weights.embed_tokens = load_tensor_from_file(file, header_json, "model.embed_tokens.weight", header_len);
    weights.vocab_size = weights.embed_tokens.shape()[0];
    weights.hidden_size = weights.embed_tokens.shape()[1];
    std::cout << "Loaded embed_tokens: [" << weights.vocab_size << ", " << weights.hidden_size << "]" << std::endl;

    weights.norm_weight = load_tensor_from_file(file, header_json, "model.norm.weight", header_len);
    std::cout << "Loaded norm" << std::endl;

    weights.lm_head = load_tensor_from_file(file, header_json, "lm_head.weight", header_len);
    std::cout << "Loaded lm_head" << std::endl;

    // Load all layers
    weights.num_layers = 28;
    weights.num_attention_heads = 16;
    weights.num_key_value_heads = 8;
    weights.head_dim = 128;

    weights.layers.resize(weights.num_layers);

    std::cout << "\nLoading " << weights.num_layers << " decoder layers..." << std::endl;

    for (size_t i = 0; i < weights.num_layers; ++i) {
        Qwen3LayerWeights layer;

        // Layer norms
        layer.input_layernorm_weight = load_tensor_from_file(
            file, header_json, "model.layers." + std::to_string(i) + ".input_layernorm.weight", header_len);

        layer.post_attention_layernorm_weight = load_tensor_from_file(
            file, header_json, "model.layers." + std::to_string(i) + ".post_attention_layernorm.weight", header_len);

        // Attention projections - for Qwen3 we have separate q, k, v projections
        // We need to combine them into qkv for our implementation
        Tensor q_proj = load_tensor_from_file(
            file, header_json, "model.layers." + std::to_string(i) + ".self_attn.q_proj.weight", header_len);

        Tensor k_proj = load_tensor_from_file(
            file, header_json, "model.layers." + std::to_string(i) + ".self_attn.k_proj.weight", header_len);

        Tensor v_proj = load_tensor_from_file(
            file, header_json, "model.layers." + std::to_string(i) + ".self_attn.v_proj.weight", header_len);

        // QKNorm weights (Qwen3-specific)
        layer.q_norm_weight = load_tensor_from_file(
            file, header_json, "model.layers." + std::to_string(i) + ".self_attn.q_norm.weight", header_len);

        layer.k_norm_weight = load_tensor_from_file(
            file, header_json, "model.layers." + std::to_string(i) + ".self_attn.k_norm.weight", header_len);

        layer.o_proj = load_tensor_from_file(
            file, header_json, "model.layers." + std::to_string(i) + ".self_attn.o_proj.weight", header_len);

        // MLP projections
        layer.gate_proj = load_tensor_from_file(
            file, header_json, "model.layers." + std::to_string(i) + ".mlp.gate_proj.weight", header_len);

        layer.up_proj = load_tensor_from_file(
            file, header_json, "model.layers." + std::to_string(i) + ".mlp.up_proj.weight", header_len);

        layer.down_proj = load_tensor_from_file(
            file, header_json, "model.layers." + std::to_string(i) + ".mlp.down_proj.weight", header_len);

        // Combine QKV projections into a single tensor for convenience
        // q_proj: [q_out, hidden], k_proj: [k_out, hidden], v_proj: [v_out, hidden]
        // qkv: [q_out + k_out + v_out, hidden]
        size_t q_out = q_proj.shape()[0];
        size_t k_out = k_proj.shape()[0];
        size_t v_out = v_proj.shape()[0];
        size_t hidden = q_proj.shape()[1];

        std::vector<float> qkv_data((q_out + k_out + v_out) * hidden);

        // Copy q_proj, then k_proj, then v_proj
        size_t offset = 0;
        for (size_t i = 0; i < q_out * hidden; ++i) {
            qkv_data[offset++] = q_proj[i];
        }
        for (size_t i = 0; i < k_out * hidden; ++i) {
            qkv_data[offset++] = k_proj[i];
        }
        for (size_t i = 0; i < v_out * hidden; ++i) {
            qkv_data[offset++] = v_proj[i];
        }

        layer.qkv_projs = Tensor(std::move(qkv_data), Shape({static_cast<long>(q_out + k_out + v_out), static_cast<long>(hidden)}));

        weights.layers[i] = layer;

        if ((i + 1) % 5 == 0 || i == weights.num_layers - 1) {
            std::cout << "  Loaded layer " << (i + 1) << "/" << weights.num_layers << "\r" << std::flush;
        }
    }

    std::cout << "\n\nConfig: layers=" << weights.num_layers
              << ", attn_heads=" << weights.num_attention_heads
              << ", kv_heads=" << weights.num_key_value_heads
              << ", head_dim=" << weights.head_dim
              << ", hidden_size=" << weights.hidden_size
              << ", vocab_size=" << weights.vocab_size << std::endl;

    return weights;
}

Tensor test_qwen3_forward(
    const std::string& safetensors_path,
    const std::vector<long>& input_ids_data,
    const std::vector<size_t>& input_shape
) {
    std::cout << "\n========== Qwen3 Forward Test ==========" << std::endl;

    // Load weights
    Qwen3Weights weights = load_qwen3_weights(safetensors_path);

    // Create input_ids tensor
    TensorL input_ids(input_ids_data, Shape(input_shape));

    std::cout << "Input shape: [" << input_ids.shape()[0] << ", " << input_ids.shape()[1] << "]" << std::endl;

    // For a complete test, we would need to load all layer weights
    // For now, this is a placeholder showing the structure

    std::cout << "\nNote: Full forward pass requires loading all " << weights.num_layers << " layer weights" << std::endl;
    std::cout << "This is a structural test - actual inference coming soon!" << std::endl;

    // Return a dummy output for now
    std::vector<float> dummy_output(input_ids_data.size() * weights.hidden_size, 0.0f);
    return Tensor(std::move(dummy_output), Shape({static_cast<long>(input_shape[0]), static_cast<long>(input_shape[1]), static_cast<long>(weights.hidden_size)}));
}

} // namespace qwen3
} // namespace tensor_cpp
