/**
 * @file qwen3_ops.h
 * @brief Qwen3-specific operators for complete model inference
 *
 * Implements:
 * - Rotary Position Embedding (RoPE) application
 * - KV cache repetition for GQA (Grouped Query Attention)
 * - Causal mask generation
 * - Complete Qwen3 decoder layer
 */

#ifndef TENSOR_CPP_QWEN3_OPS_H
#define TENSOR_CPP_QWEN3_OPS_H

#include "tensor_cpp/tensor.h"
#include "tensor_cpp/ops.h"
#include <vector>
#include <cmath>
#include <complex>

namespace tensor_cpp {
namespace qwen3 {

// ============================================================================
// Layer Weights Structure (must be declared first)
// ============================================================================

struct Qwen3LayerWeights {
    // Attention weights
    Tensor qkv_projs;      // Combined QKV projection [hidden_size, 3 * num_heads * head_dim]
    Tensor o_proj;         // Output projection [hidden_size, hidden_size]

    // QKNorm weights (Qwen3-specific: normalize Q and K per-head)
    Tensor q_norm_weight;  // [head_dim]
    Tensor k_norm_weight;  // [head_dim]

    // Layer norms
    Tensor input_layernorm_weight;
    Tensor post_attention_layernorm_weight;

    // MLP weights
    Tensor gate_proj;
    Tensor up_proj;
    Tensor down_proj;
};

// ============================================================================
// RoPE: Apply Rotary Position Embedding to Query and Key
// ============================================================================

/**
 * Apply rotary position embedding to query and key tensors
 *
 * @param q   Query tensor [batch, num_heads, seq_len, head_dim]
 * @param k   Key tensor [batch, num_kv_heads, seq_len, head_dim]
 * @param cos Cosine values [seq_len, head_dim//2] or [batch, 1, seq_len, head_dim//2]
 * @param sin Sine values [seq_len, head_dim//2] or [batch, 1, seq_len, head_dim//2]
 * @return Pair of rotated query and key
 */
std::pair<Tensor, Tensor> apply_rotary_pos_emb(
    const Tensor& q,
    const Tensor& k,
    const Tensor& cos,
    const Tensor& sin
);

/**
 * Precompute rotary embedding cos/sin values
 *
 * @param seq_len Sequence length
 * @param head_dim Head dimension (must be even)
 * @param theta RoPE theta (default 10000.0)
 * @return Pair of (cos, sin) tensors [seq_len, head_dim//2]
 */
std::pair<Tensor, Tensor> compute_rope_freqs(
    size_t seq_len,
    size_t head_dim,
    float theta = 10000.0f
);

// ============================================================================
// GQA: Repeat Key/Value for Grouped Query Attention
// ============================================================================

/**
 * Repeat KV heads for GQA
 * Transform from [batch, num_kv_heads, seq_len, head_dim]
 * to [batch, num_attention_heads, seq_len, head_dim]
 *
 * @param hidden_states KV states [batch, num_kv_heads, seq_len, head_dim]
 * @param n_rep Number of repetitions (num_attention_heads / num_kv_heads)
 * @return Repeated hidden states
 */
Tensor repeat_kv(
    const Tensor& hidden_states,
    int n_rep
);

// ============================================================================
// Causal Mask Generation
// ============================================================================

/**
 * Create causal attention mask for decoder-only models
 *
 * @param seq_len Sequence length
 * @param dtype Data type for mask (default float)
 * @return Causal mask [seq_len, seq_len] where mask[i, j] = 0 if j <= i else -inf
 */
Tensor create_causal_mask(
    size_t seq_len,
    float dtype = 0.0f  // Use float (0.0f) for mask
);

// ============================================================================
// Qwen3 Attention with GQA
// ============================================================================

/**
 * Qwen3 Attention with Grouped Query Attention
 *
 * @param hidden_states Input [batch, seq_len, hidden_size]
 * @param num_attention_heads Number of attention heads (16 for Qwen3-0.6B)
 * @param num_key_value_heads Number of KV heads (8 for Qwen3-0.6B)
 * @param head_dim Head dimension (128 for Qwen3-0.6B)
 * @param q_proj Query projection weight
 * @param k_proj Key projection weight
 * @param v_proj Value projection weight
 * @param o_proj Output projection weight
 * @param cos RoPE cosine values
 * @param sin RoPE sine values
 * @param has_cache Whether to use KV cache
 * @return Output tensor [batch, seq_len, hidden_size]
 */
Tensor qwen3_attention(
    const Tensor& hidden_states,
    size_t num_attention_heads,
    size_t num_key_value_heads,
    size_t head_dim,
    const Tensor& q_proj,
    const Tensor& k_proj,
    const Tensor& v_proj,
    const Tensor& o_proj,
    const Tensor& q_norm_weight,  // QKNorm for Q (per-head normalization)
    const Tensor& k_norm_weight,  // QKNorm for K (per-head normalization)
    const Tensor& cos,
    const Tensor& sin,
    bool has_cache = false
);

// ============================================================================
// Qwen3 MLP (SwiGLU)
// ============================================================================

/**
 * Qwen3 MLP layer with SwiGLU activation
 *
 * @param hidden_states Input [batch, seq_len, hidden_size]
 * @param gate_proj Gate projection weight [intermediate_size, hidden_size]
 * @param up_proj Up projection weight [intermediate_size, hidden_size]
 * @param down_proj Down projection weight [hidden_size, intermediate_size]
 * @return Output [batch, seq_len, hidden_size]
 */
Tensor qwen3_mlp(
    const Tensor& hidden_states,
    const Tensor& gate_proj,
    const Tensor& up_proj,
    const Tensor& down_proj
);

// ============================================================================
// Qwen3 Decoder Layer
// ============================================================================

/**
 * Single Qwen3 decoder layer
 *
 * @param hidden_states Input [batch, seq_len, hidden_size]
 * @param num_attention_heads Number of attention heads
 * @param num_key_value_heads Number of KV heads
 * @param head_dim Head dimension
 * @param rms_norm_eps RMS norm epsilon
 * @param qkv_projs Combined QKV projection weights
 * @param o_proj Output projection weight
 * @param gate_mlp MLP gate projection weight
 * @param up_mlp MLP up projection weight
 * @param down_mlp MLP down projection weight
 * @param cos RoPE cosine values
 * @param sin RoPE sine values
 * @return Output [batch, seq_len, hidden_size]
 */
Tensor qwen3_decoder_layer(
    const Tensor& hidden_states,
    size_t num_attention_heads,
    size_t num_key_value_heads,
    size_t head_dim,
    float rms_norm_eps,
    const Tensor& input_layernorm_weight,
    const Tensor& qkv_projs,  // Combined QKV weights
    const Tensor& o_proj,
    const Tensor& q_norm_weight,  // QKNorm for Q (per-head normalization)
    const Tensor& k_norm_weight,  // QKNorm for K (per-head normalization)
    const Tensor& post_attention_layernorm_weight,
    const Tensor& gate_mlp,
    const Tensor& up_mlp,
    const Tensor& down_mlp,
    const Tensor& cos,
    const Tensor& sin
);

// ============================================================================
// Qwen3 Model (Full Forward Pass)
// ============================================================================

/**
 * Complete Qwen3 model forward pass
 *
 * @param input_ids Input token IDs [batch_size, seq_len]
 * @param token_embedding Word embedding weight [vocab_size, hidden_size]
 * @param layers List of decoder layer weights
 * @param norm_weight Final layer norm weight
 * @param num_layers Number of layers (28 for Qwen3-0.6B)
 * @param num_attention_heads Number of attention heads (16)
 * @param num_key_value_heads Number of KV heads (8)
 * @param head_dim Head dimension (128)
 * @param rms_norm_eps RMS norm epsilon (1e-6)
 * @return Hidden states [batch_size, seq_len, hidden_size]
 */
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
);

} // namespace qwen3
} // namespace tensor_cpp

#endif // TENSOR_CPP_QWEN3_OPS_H
