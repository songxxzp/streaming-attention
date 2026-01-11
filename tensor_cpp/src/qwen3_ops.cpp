/**
 * @file qwen3_ops.cpp
 * @brief Qwen3-specific operators for complete model inference
 *
 * Implements:
 * - Rotary Position Embedding (RoPE) application
 * - KV cache repetition for GQA (Grouped Query Attention)
 * - Causal mask generation
 * - Complete Qwen3 decoder layer
 */

#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/ops.h"
#include <algorithm>
#include <limits>
#include <cmath>

namespace tensor_cpp {
namespace qwen3 {

// ============================================================================
// Helper: rotate_half for RoPE
// ============================================================================

/**
 * Rotates half the hidden dims of the input
 * Input shape: [batch, heads, seq_len, head_dim]
 * Output: x with first half and second half rotated
 */
static Tensor rotate_half(const Tensor& x) {
    const Shape& x_shape = x.shape();
    size_t batch = x_shape[0];
    size_t heads = x_shape[1];
    size_t seq_len = x_shape[2];
    size_t head_dim = x_shape[3];

    if (head_dim % 2 != 0) {
        throw std::invalid_argument("head_dim must be even for rotate_half");
    }

    size_t half_dim = head_dim / 2;
    std::vector<float> result(x.size());

    #pragma omp parallel for if(batch * heads * seq_len * head_dim > 1000)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < heads; ++h) {
            for (size_t s = 0; s < seq_len; ++s) {
                size_t base_idx = ((b * heads + h) * seq_len + s) * head_dim;

                // x1 = x[..., :half_dim]
                // x2 = x[..., half_dim:]
                // output = cat([-x2, x1], dim=-1)
                for (size_t i = 0; i < half_dim; ++i) {
                    // First half: -x2 (negated second half of input)
                    result[base_idx + i] = -x[base_idx + half_dim + i];
                    // Second half: x1 (first half of input)
                    result[base_idx + half_dim + i] = x[base_idx + i];
                }
            }
        }
    }

    return Tensor(std::move(result), x_shape);
}

// ============================================================================
// RoPE: Compute Rotary Embedding Frequencies
// ============================================================================

std::pair<Tensor, Tensor> compute_rope_freqs(
    size_t seq_len,
    size_t head_dim,
    float theta
) {
    if (head_dim % 2 != 0) {
        throw std::invalid_argument("head_dim must be even for RoPE");
    }

    size_t half_dim = head_dim / 2;

    // Compute inverse frequencies: 1 / (theta ^ (i / dim))
    std::vector<float> inv_freq(half_dim);
    for (size_t i = 0; i < half_dim; ++i) {
        inv_freq[i] = 1.0f / std::pow(theta, static_cast<float>(i) / static_cast<float>(head_dim));
    }

    // Compute cos and sin for each position
    std::vector<float> cos_data(seq_len * half_dim);
    std::vector<float> sin_data(seq_len * half_dim);

    #pragma omp parallel for if(seq_len * half_dim > 1000)
    for (size_t pos = 0; pos < seq_len; ++pos) {
        for (size_t i = 0; i < half_dim; ++i) {
            float angle = static_cast<float>(pos) * inv_freq[i];
            cos_data[pos * half_dim + i] = std::cos(angle);
            sin_data[pos * half_dim + i] = std::sin(angle);
        }
    }

    Tensor cos_tensor(std::move(cos_data), Shape({static_cast<long>(seq_len), static_cast<long>(half_dim)}));
    Tensor sin_tensor(std::move(sin_data), Shape({static_cast<long>(seq_len), static_cast<long>(half_dim)}));

    return {cos_tensor, sin_tensor};
}

// ============================================================================
// RoPE: Apply Rotary Position Embedding
// ============================================================================

std::pair<Tensor, Tensor> apply_rotary_pos_emb(
    const Tensor& q,
    const Tensor& k,
    const Tensor& cos,
    const Tensor& sin
) {
    // q, k shape: [batch, num_heads, seq_len, head_dim]
    // cos, sin shape: [seq_len, head_dim//2] or [batch, 1, seq_len, head_dim//2]

    const Shape& q_shape = q.shape();
    size_t batch = q_shape[0];
    size_t num_heads = q_shape[1];
    size_t seq_len = q_shape[2];
    size_t head_dim = q_shape[3];

    if (head_dim % 2 != 0) {
        throw std::invalid_argument("head_dim must be even for RoPE");
    }

    size_t half_dim = head_dim / 2;

    // Apply RoPE: q_embed = (q * cos) + (rotate_half(q) * sin)
    // For each position, we modify both first and second half of head_dim

    // Rotate queries and keys
    Tensor q_rotated = rotate_half(q);
    Tensor k_rotated = rotate_half(k);

    std::vector<float> q_embed_data(q.size());
    std::vector<float> k_embed_data(k.size());

    // Get number of KV heads (K and V have fewer heads than Q for GQA)
    size_t num_kv_heads = k.shape()[1];

    // Compute RoPE for queries
    #pragma omp parallel for if(batch * num_heads * seq_len * head_dim > 1000)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t s = 0; s < seq_len; ++s) {
                size_t base_idx = ((b * num_heads + h) * seq_len + s) * head_dim;

                for (size_t i = 0; i < half_dim; ++i) {
                    // Get cos and sin for this position
                    float cos_val = cos[s * half_dim + i];
                    float sin_val = sin[s * half_dim + i];

                    // Apply RoPE to query (both halves)
                    q_embed_data[base_idx + i] = q[base_idx + i] * cos_val + q_rotated[base_idx + i] * sin_val;
                    q_embed_data[base_idx + half_dim + i] = q[base_idx + half_dim + i] * cos_val + q_rotated[base_idx + half_dim + i] * sin_val;
                }
            }
        }
    }

    // Compute RoPE for keys (separate loop because K has fewer heads)
    #pragma omp parallel for if(batch * num_kv_heads * seq_len * head_dim > 1000)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < num_kv_heads; ++h) {
            for (size_t s = 0; s < seq_len; ++s) {
                size_t base_idx = ((b * num_kv_heads + h) * seq_len + s) * head_dim;

                for (size_t i = 0; i < half_dim; ++i) {
                    // Get cos and sin for this position
                    float cos_val = cos[s * half_dim + i];
                    float sin_val = sin[s * half_dim + i];

                    // Apply RoPE to key (both halves)
                    k_embed_data[base_idx + i] = k[base_idx + i] * cos_val + k_rotated[base_idx + i] * sin_val;
                    k_embed_data[base_idx + half_dim + i] = k[base_idx + half_dim + i] * cos_val + k_rotated[base_idx + half_dim + i] * sin_val;
                }
            }
        }
    }

    Tensor q_embed(std::move(q_embed_data), q_shape);
    Tensor k_embed(std::move(k_embed_data), k.shape());

    return {q_embed, k_embed};
}

// ============================================================================
// GQA: Repeat KV heads
// ============================================================================

Tensor repeat_kv(
    const Tensor& hidden_states,
    int n_rep
) {
    // hidden_states shape: [batch, num_kv_heads, seq_len, head_dim]
    // Output shape: [batch, num_attention_heads, seq_len, head_dim]
    // where num_attention_heads = num_kv_heads * n_rep

    if (n_rep == 1) {
        return hidden_states;
    }

    const Shape& shape = hidden_states.shape();
    size_t batch = shape[0];
    size_t num_kv_heads = shape[1];
    size_t seq_len = shape[2];
    size_t head_dim = shape[3];

    size_t num_heads_out = num_kv_heads * n_rep;

    std::vector<float> result(batch * num_heads_out * seq_len * head_dim);

    #pragma omp parallel for if(batch * num_kv_heads * n_rep * seq_len * head_dim > 1000)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < num_kv_heads; ++h) {
            for (size_t r = 0; r < static_cast<size_t>(n_rep); ++r) {
                size_t h_out = h * n_rep + r;
                for (size_t s = 0; s < seq_len; ++s) {
                    for (size_t d = 0; d < head_dim; ++d) {
                        size_t in_idx = ((b * num_kv_heads + h) * seq_len + s) * head_dim + d;
                        size_t out_idx = ((b * num_heads_out + h_out) * seq_len + s) * head_dim + d;
                        result[out_idx] = hidden_states[in_idx];
                    }
                }
            }
        }
    }

    return Tensor(std::move(result), Shape({static_cast<long>(batch), static_cast<long>(num_heads_out),
                                            static_cast<long>(seq_len), static_cast<long>(head_dim)}));
}

// ============================================================================
// Causal Mask Generation
// ============================================================================

Tensor create_causal_mask(
    size_t seq_len,
    float dtype
) {
    // Create causal mask where mask[i, j] = 0 if j <= i else -inf
    std::vector<float> mask_data(seq_len * seq_len);

    #pragma omp parallel for if(seq_len * seq_len > 1000)
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
            if (j <= i) {
                mask_data[i * seq_len + j] = 0.0f;
            } else {
                mask_data[i * seq_len + j] = -std::numeric_limits<float>::infinity();
            }
        }
    }

    return Tensor(std::move(mask_data), Shape({static_cast<long>(seq_len), static_cast<long>(seq_len)}));
}

// ============================================================================
// Qwen3 MLP (SwiGLU)
// ============================================================================

Tensor qwen3_mlp(
    const Tensor& hidden_states,
    const Tensor& gate_proj,
    const Tensor& up_proj,
    const Tensor& down_proj
) {
    // hidden_states: [batch, seq_len, hidden_size]
    // gate_proj: [intermediate_size, hidden_size]
    // up_proj: [intermediate_size, hidden_size]
    // down_proj: [hidden_size, intermediate_size]

    // Compute gate = gate_proj(x)
    Tensor gate = ops::linear(hidden_states, gate_proj, nullptr);

    // Compute up = up_proj(x)
    Tensor up = ops::linear(hidden_states, up_proj, nullptr);

    // SwiGLU activation: SiLU(gate) * up
    const Shape& gate_shape = gate.shape();
    std::vector<float> gated_data(gate.size());
    #pragma omp parallel for if(gate.size() > 1000)
    for (size_t i = 0; i < gate.size(); ++i) {
        float silu_gate = gate[i] * (1.0f / (1.0f + std::exp(-gate[i])));
        gated_data[i] = silu_gate * up[i];
    }
    Tensor gated(std::move(gated_data), gate_shape);

    // Compute output: down_proj(gated)
    Tensor output = ops::linear(gated, down_proj, nullptr);

    return output;
}

// ============================================================================
// Qwen3 Attention with GQA
// ============================================================================

Tensor qwen3_attention(
    const Tensor& hidden_states,
    size_t num_attention_heads,
    size_t num_key_value_heads,
    size_t head_dim,
    const Tensor& q_proj,
    const Tensor& k_proj,
    const Tensor& v_proj,
    const Tensor& o_proj,
    const Tensor& q_norm_weight,
    const Tensor& k_norm_weight,
    const Tensor& cos,
    const Tensor& sin,
    bool has_cache
) {
    // hidden_states: [batch, seq_len, hidden_size]
    const Shape& hidden_shape = hidden_states.shape();
    size_t batch = hidden_shape[0];
    size_t seq_len = hidden_shape[1];
    size_t hidden_size = hidden_shape[2];

    // Compute Q, K, V projections
    Tensor q_proj_out = ops::linear(hidden_states, q_proj, nullptr);
    Tensor k_proj_out = ops::linear(hidden_states, k_proj, nullptr);
    Tensor v_proj_out = ops::linear(hidden_states, v_proj, nullptr);

    // Reshape to [batch, num_heads, seq_len, head_dim]
    size_t q_total_heads = num_attention_heads;
    size_t kv_total_heads = num_key_value_heads;

    Tensor q_reshaped = q_proj_out.view({batch, seq_len, q_total_heads, head_dim});
    Tensor k_reshaped = k_proj_out.view({batch, seq_len, kv_total_heads, head_dim});
    Tensor v_reshaped = v_proj_out.view({batch, seq_len, kv_total_heads, head_dim});

    // Transpose to [batch, num_heads, seq_len, head_dim]
    Tensor q = q_reshaped.transpose(1, 2);  // [batch, num_heads, seq_len, head_dim]
    Tensor k = k_reshaped.transpose(1, 2);  // [batch, num_kv_heads, seq_len, head_dim]
    Tensor v = v_reshaped.transpose(1, 2);  // [batch, num_kv_heads, seq_len, head_dim]

    // ========== QKNorm: Apply RMS normalization to Q and K per-head ==========
    // This is Qwen3-specific: normalize Q and K along the head_dim dimension
    // q_norm_weight and k_norm_weight have shape [head_dim]
    const float* q_norm_data = q_norm_weight.data();
    const float* k_norm_data = k_norm_weight.data();

    // Apply QKNorm to Q: [batch, num_heads, seq_len, head_dim]
    // Normalize along the last dimension (head_dim)
    size_t q_total_elements = batch * q_total_heads * seq_len * head_dim;
    std::vector<float> q_normed_data(q_total_elements);
    #pragma omp parallel for if(batch * q_total_heads * seq_len * head_dim > 1000)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < q_total_heads; ++h) {
            for (size_t s = 0; s < seq_len; ++s) {
                size_t base_idx = ((b * q_total_heads + h) * seq_len + s) * head_dim;

                // Compute variance
                float sum_sq = 0.0f;
                for (size_t i = 0; i < head_dim; ++i) {
                    float val = q[base_idx + i];
                    sum_sq += val * val;
                }
                float variance = sum_sq / head_dim;
                float rms = std::sqrt(variance + 1e-6f);

                // Normalize and scale
                for (size_t i = 0; i < head_dim; ++i) {
                    q_normed_data[base_idx + i] = (q[base_idx + i] / rms) * q_norm_data[i];
                }
            }
        }
    }
    Tensor q_normed(std::move(q_normed_data), q.shape());

    // Apply QKNorm to K: [batch, num_kv_heads, seq_len, head_dim]
    size_t k_total_elements = batch * kv_total_heads * seq_len * head_dim;
    std::vector<float> k_normed_data(k_total_elements);
    #pragma omp parallel for if(batch * kv_total_heads * seq_len * head_dim > 1000)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < kv_total_heads; ++h) {
            for (size_t s = 0; s < seq_len; ++s) {
                size_t base_idx = ((b * kv_total_heads + h) * seq_len + s) * head_dim;

                // Compute variance
                float sum_sq = 0.0f;
                for (size_t i = 0; i < head_dim; ++i) {
                    float val = k[base_idx + i];
                    sum_sq += val * val;
                }
                float variance = sum_sq / head_dim;
                float rms = std::sqrt(variance + 1e-6f);

                // Normalize and scale
                for (size_t i = 0; i < head_dim; ++i) {
                    k_normed_data[base_idx + i] = (k[base_idx + i] / rms) * k_norm_data[i];
                }
            }
        }
    }
    Tensor k_normed(std::move(k_normed_data), k.shape());
    // ========== End QKNorm ==========

    // Apply RoPE (now to normalized Q and K)
    auto [q_rope, k_rope] = apply_rotary_pos_emb(q_normed, k_normed, cos, sin);

    // Repeat KV for GQA
    int n_rep = static_cast<int>(num_attention_heads / num_key_value_heads);
    Tensor k_repeated = repeat_kv(k_rope, n_rep);
    Tensor v_repeated = repeat_kv(v, n_rep);

    // Compute attention with causal mask
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Create causal mask
    Tensor causal_mask = create_causal_mask(seq_len);
    Tensor mask = causal_mask.view({1, 1, seq_len, seq_len});

    // Use self_attention from ops
    Tensor attn_output = ops::self_attention(q_rope, k_repeated, v_repeated, &mask, scale);

    // Transpose back: [batch, seq_len, num_heads, head_dim]
    Tensor attn_output_t = attn_output.transpose(1, 2);

    // Reshape to [batch, seq_len, hidden_size]
    Tensor attn_output_reshaped = attn_output_t.contiguous().view({batch, seq_len, q_total_heads * head_dim});

    // Apply output projection
    Tensor output = ops::linear(attn_output_reshaped, o_proj, nullptr);

    return output;
}

// ============================================================================
// Qwen3 Decoder Layer
// ============================================================================

Tensor qwen3_decoder_layer(
    const Tensor& hidden_states,
    size_t num_attention_heads,
    size_t num_key_value_heads,
    size_t head_dim,
    float rms_norm_eps,
    const Tensor& input_layernorm_weight,
    const Tensor& qkv_projs,
    const Tensor& o_proj,
    const Tensor& q_norm_weight,
    const Tensor& k_norm_weight,
    const Tensor& post_attention_layernorm_weight,
    const Tensor& gate_mlp,
    const Tensor& up_mlp,
    const Tensor& down_mlp,
    const Tensor& cos,
    const Tensor& sin
) {
    // Input layernorm
    Tensor residual = hidden_states;
    Tensor normed = ops::rms_norm(hidden_states, &input_layernorm_weight, rms_norm_eps);

    // Split QKV projections
    // qkv_projs shape: [q_out + k_out + v_out, hidden_size]
    // For Qwen3-0.6B: [2048 + 1024 + 1024, 1024] = [4096, 1024]
    // Rows 0 to q_size-1 are q_proj, rows q_size to q_size+k_size-1 are k_proj, etc.
    size_t hidden_size = hidden_states.shape()[2];
    size_t q_size = num_attention_heads * head_dim;  // 16 * 128 = 2048
    size_t k_size = num_key_value_heads * head_dim;  // 8 * 128 = 1024
    size_t v_size = num_key_value_heads * head_dim;  // 8 * 128 = 1024

    // Extract Q, K, V from combined QKV
    // Each projection is stored as [out_features, hidden_size] which is what we need
    std::vector<float> q_data(q_size * hidden_size);
    std::vector<float> k_data(k_size * hidden_size);
    std::vector<float> v_data(v_size * hidden_size);

    // q_proj: rows 0 to q_size-1 of qkv_projs
    for (size_t row = 0; row < q_size; ++row) {
        for (size_t col = 0; col < hidden_size; ++col) {
            q_data[row * hidden_size + col] = qkv_projs[row * hidden_size + col];
        }
    }

    // k_proj: rows q_size to q_size+k_size-1 of qkv_projs
    for (size_t row = 0; row < k_size; ++row) {
        for (size_t col = 0; col < hidden_size; ++col) {
            k_data[row * hidden_size + col] = qkv_projs[(q_size + row) * hidden_size + col];
        }
    }

    // v_proj: rows q_size+k_size to q_size+k_size+v_size-1 of qkv_projs
    for (size_t row = 0; row < v_size; ++row) {
        for (size_t col = 0; col < hidden_size; ++col) {
            v_data[row * hidden_size + col] = qkv_projs[(q_size + k_size + row) * hidden_size + col];
        }
    }

    // Each proj has shape [proj_size, hidden_size] for linear layer
    Tensor q_proj(std::move(q_data), Shape({static_cast<long>(q_size), static_cast<long>(hidden_size)}));
    Tensor k_proj(std::move(k_data), Shape({static_cast<long>(k_size), static_cast<long>(hidden_size)}));
    Tensor v_proj(std::move(v_data), Shape({static_cast<long>(v_size), static_cast<long>(hidden_size)}));

    // Self-attention
    Tensor attn_output = qwen3_attention(
        normed,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_norm_weight,
        k_norm_weight,
        cos,
        sin,
        false
    );

    // Residual connection
    Tensor hidden_states_after_attn = residual + attn_output;

    // Post-attention layernorm
    residual = hidden_states_after_attn;
    Tensor post_normed = ops::rms_norm(hidden_states_after_attn, &post_attention_layernorm_weight, rms_norm_eps);

    // MLP
    Tensor mlp_output = qwen3_mlp(post_normed, gate_mlp, up_mlp, down_mlp);

    // Residual connection
    Tensor output = residual + mlp_output;

    return output;
}

// ============================================================================
// Qwen3 Model (Full Forward Pass)
// ============================================================================

Tensor qwen3_forward(
    const TensorL& input_ids,
    const Tensor& token_embedding,
    const std::vector<Qwen3LayerWeights>& layers,
    const Tensor& norm_weight,
    size_t num_layers,
    size_t num_attention_heads,
    size_t num_key_value_heads,
    size_t head_dim,
    float rms_norm_eps
) {
    // input_ids: [batch_size, seq_len]
    // token_embedding: [vocab_size, hidden_size]

    size_t batch_size = input_ids.shape()[0];
    size_t seq_len = input_ids.shape()[1];
    size_t hidden_size = token_embedding.shape()[1];

    // Embed tokens
    Tensor hidden_states = ops::embedding(input_ids, token_embedding);

    // Precompute RoPE frequencies
    auto [cos, sin] = compute_rope_freqs(seq_len, head_dim, 1000000.0f);

    // Pass through decoder layers
    for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        const Qwen3LayerWeights& layer = layers[layer_idx];

        hidden_states = qwen3_decoder_layer(
            hidden_states,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            layer.input_layernorm_weight,
            layer.qkv_projs,
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
    }

    // Final layer norm
    Tensor output = ops::rms_norm(hidden_states, &norm_weight, rms_norm_eps);

    return output;
}

} // namespace qwen3
} // namespace tensor_cpp
