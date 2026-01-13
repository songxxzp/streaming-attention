/**
 * @file qwen3_ops_mpi.cpp
 * @brief Implementation of MPI+OpenMP parallelized Qwen3 operators
 */

#include "tensor_cpp/qwen3_ops_mpi.h"
#include "tensor_cpp/ops.h"
#include <algorithm>

using namespace tensor_cpp::ops;

namespace tensor_cpp {
namespace qwen3 {
namespace mpi {

#ifdef MPI_VERSION

// ============================================================================
// Qwen3 MLP with MPI+OpenMP
// ============================================================================

Tensor qwen3_mlp_mpi_omp(
    const Tensor& hidden_states,
    const Tensor& gate_proj,
    const Tensor& up_proj,
    const Tensor& down_proj,
    MPI_Comm comm
) {
    // hidden_states: [batch, seq_len, hidden_size]
    // gate_proj, up_proj: [intermediate_size, hidden_size]
    // down_proj: [hidden_size, intermediate_size]

    size_t batch_size = hidden_states.shape()[0];
    size_t seq_len = hidden_states.shape()[1];
    size_t hidden_size = hidden_states.shape()[2];

    // Reshape to [seq_len, hidden_size] for linear layer
    Tensor hidden_reshaped = hidden_states.view({static_cast<long>(batch_size * seq_len), static_cast<long>(hidden_size)});

    // Gate projection (with MPI)
    Tensor gate = ops::mpi::linear_mpi_omp(hidden_reshaped, gate_proj, nullptr, comm);

    // Up projection (with MPI)
    Tensor up = ops::mpi::linear_mpi_omp(hidden_reshaped, up_proj, nullptr, comm);

    // SwiGLU activation (with MPI)
    Tensor activated = ops::mpi::swiglu_mpi_omp(gate, up, comm);

    // Down projection (with MPI)
    Tensor output = ops::mpi::linear_mpi_omp(activated, down_proj, nullptr, comm);

    // Reshape back
    output = output.view({static_cast<long>(batch_size), static_cast<long>(seq_len), static_cast<long>(hidden_size)});

    return output;
}

// ============================================================================
// Qwen3 Attention with MPI+OpenMP
// ============================================================================

Tensor qwen3_attention_mpi_omp(
    const Tensor& hidden_states,
    size_t num_attention_heads,
    size_t num_key_value_heads,
    size_t head_dim,
    const Tensor& qkv_projs,
    const Tensor& o_proj,
    const Tensor& q_norm_weight,
    const Tensor& k_norm_weight,
    const Tensor& cos,
    const Tensor& sin,
    MPI_Comm comm
) {
    // hidden_states: [batch, seq_len, hidden_size]
    size_t batch_size = hidden_states.shape()[0];
    size_t seq_len = hidden_states.shape()[1];
    size_t hidden_size = hidden_states.shape()[2];

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Compute QKV projection
    // qkv_projs: [3 * num_heads * head_dim, hidden_size] for Q, K, V combined
    // For Qwen3: Q = num_attention_heads * head_dim, K = num_kv_heads * head_dim, V = num_kv_heads * head_dim

    size_t q_dim = num_attention_heads * head_dim;
    size_t kv_dim = num_key_value_heads * head_dim;

    // Split qkv_projs into Q, K, V projections by extracting submatrices
    // qkv_projs shape: [q_dim + kv_dim + kv_dim, hidden_size]
    size_t total_qkv_dim = q_dim + 2 * kv_dim;
    size_t weight_hidden_size = qkv_projs.shape()[1];

    // Extract Q projection: [0:q_dim, :]
    std::vector<float> q_proj_data(q_dim * weight_hidden_size);
    for (size_t i = 0; i < q_dim; ++i) {
        for (size_t j = 0; j < weight_hidden_size; ++j) {
            q_proj_data[i * weight_hidden_size + j] = qkv_projs[i * weight_hidden_size + j];
        }
    }
    Tensor q_proj(std::move(q_proj_data), Shape({static_cast<long>(q_dim), static_cast<long>(weight_hidden_size)}));

    // Extract K projection: [q_dim:q_dim+kv_dim, :]
    std::vector<float> k_proj_data(kv_dim * weight_hidden_size);
    for (size_t i = 0; i < kv_dim; ++i) {
        for (size_t j = 0; j < weight_hidden_size; ++j) {
            k_proj_data[i * weight_hidden_size + j] = qkv_projs[(q_dim + i) * weight_hidden_size + j];
        }
    }
    Tensor k_proj(std::move(k_proj_data), Shape({static_cast<long>(kv_dim), static_cast<long>(weight_hidden_size)}));

    // Extract V projection: [q_dim+kv_dim:q_dim+2*kv_dim, :]
    std::vector<float> v_proj_data(kv_dim * weight_hidden_size);
    for (size_t i = 0; i < kv_dim; ++i) {
        for (size_t j = 0; j < weight_hidden_size; ++j) {
            v_proj_data[i * weight_hidden_size + j] = qkv_projs[(q_dim + kv_dim + i) * weight_hidden_size + j];
        }
    }
    Tensor v_proj(std::move(v_proj_data), Shape({static_cast<long>(kv_dim), static_cast<long>(weight_hidden_size)}));

    // Compute Q, K, V (reshape for linear layer)
    Tensor hidden_reshaped = hidden_states.view({static_cast<long>(batch_size * seq_len), static_cast<long>(hidden_size)});

    Tensor q = ops::mpi::linear_mpi_omp(hidden_reshaped, q_proj, nullptr, comm);
    Tensor k = ops::mpi::linear_mpi_omp(hidden_reshaped, k_proj, nullptr, comm);
    Tensor v = ops::mpi::linear_mpi_omp(hidden_reshaped, v_proj, nullptr, comm);

    // Reshape Q, K, V to [batch, num_heads, seq_len, head_dim]
    q = q.view({static_cast<long>(batch_size), static_cast<long>(num_attention_heads),
                static_cast<long>(seq_len), static_cast<long>(head_dim)});
    k = k.view({static_cast<long>(batch_size), static_cast<long>(num_key_value_heads),
                static_cast<long>(seq_len), static_cast<long>(head_dim)});
    v = v.view({static_cast<long>(batch_size), static_cast<long>(num_key_value_heads),
                static_cast<long>(seq_len), static_cast<long>(head_dim)});

    // Apply QKNorm (Qwen3-specific: normalize Q and K per-head)
    // For now, skip the actual QKNorm computation (would require per-head RMS norm)

    // Apply RoPE
    auto qk_rope = apply_rotary_pos_emb(q, k, cos, sin);
    q = qk_rope.first;
    k = qk_rope.second;

    // Compute attention with MPI (distributes heads across ranks)
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    Tensor attn_output = ops::mpi::self_attention_mpi_omp(
        q, k, v, nullptr, scale,
        num_attention_heads, num_key_value_heads, comm
    );

    // Reshape output for projection
    attn_output = attn_output.view({static_cast<long>(batch_size * seq_len), static_cast<long>(num_attention_heads * head_dim)});

    // Output projection
    Tensor output = ops::mpi::linear_mpi_omp(attn_output, o_proj, nullptr, comm);

    // Reshape back
    output = output.view({static_cast<long>(batch_size), static_cast<long>(seq_len), static_cast<long>(hidden_size)});

    return output;
}

// ============================================================================
// Qwen3 Decoder Layer with MPI+OpenMP
// ============================================================================

Tensor qwen3_decoder_layer_mpi_omp(
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
    const Tensor& sin,
    MPI_Comm comm
) {
    // Input layernorm
    Tensor residual = hidden_states;
    Tensor hidden = rms_norm(hidden_states, &input_layernorm_weight, rms_norm_eps);

    // Self-attention with MPI
    Tensor attn_output = qwen3_attention_mpi_omp(
        hidden, num_attention_heads, num_key_value_heads, head_dim,
        qkv_projs, o_proj, q_norm_weight, k_norm_weight, cos, sin, comm
    );

    // Residual connection
    hidden = residual + attn_output;

    // Post-attention layernorm
    residual = hidden;
    hidden = rms_norm(hidden, &post_attention_layernorm_weight, rms_norm_eps);

    // MLP with MPI
    Tensor mlp_output = qwen3_mlp_mpi_omp(hidden, gate_mlp, up_mlp, down_mlp, comm);

    // Residual connection
    hidden = residual + mlp_output;

    return hidden;
}

// ============================================================================
// Complete Qwen3 Forward Pass with MPI+OpenMP
// ============================================================================

Tensor qwen3_forward_mpi_omp(
    const TensorL& input_ids,
    const Tensor& token_embedding,
    const std::vector<Qwen3LayerWeights>& layers,
    const Tensor& norm_weight,
    const Tensor& lm_head,
    size_t num_layers,
    size_t num_attention_heads,
    size_t num_key_value_heads,
    size_t head_dim,
    float rms_norm_eps,
    MPI_Comm comm
) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Embed tokens
    Tensor hidden_states = embedding(input_ids, token_embedding);

    // Compute RoPE frequencies (all ranks compute the same)
    size_t seq_len = input_ids.shape()[1];
    auto rope_freqs = compute_rope_freqs(seq_len, head_dim);
    Tensor cos = rope_freqs.first;
    Tensor sin = rope_freqs.second;

    // Process through decoder layers
    for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        const auto& layer = layers[layer_idx];

        hidden_states = qwen3_decoder_layer_mpi_omp(
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
            sin,
            comm
        );
    }

    // Final layernorm
    Tensor hidden_normed = rms_norm(hidden_states, &norm_weight, rms_norm_eps);

    // LM head projection to get logits
    size_t batch_size = input_ids.shape()[0];
    size_t hidden_size = hidden_states.shape()[2];
    size_t vocab_size = lm_head.shape()[0];
    size_t num_samples = batch_size * seq_len;

    std::vector<float> logits_data(num_samples * vocab_size);

    #pragma omp parallel for if(num_samples * vocab_size > 1000)
    for (size_t s = 0; s < num_samples; ++s) {
        for (size_t v = 0; v < vocab_size; ++v) {
            float sum = 0.0f;
            for (size_t h = 0; h < hidden_size; ++h) {
                sum += hidden_normed.data()[s * hidden_size + h] * lm_head.data()[v * hidden_size + h];
            }
            logits_data[s * vocab_size + v] = sum;
        }
    }

    return Tensor(std::move(logits_data), Shape({static_cast<long>(batch_size), static_cast<long>(seq_len), static_cast<long>(vocab_size)}));
}

// ============================================================================
// MPI+OpenMP Decoder Layer with KV Cache Support
// ============================================================================

Tensor qwen3_decoder_layer_mpi_omp_with_cache(
    const Tensor& hidden_states,
    KVCache* kv_cache,
    size_t layer_idx,
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
    const Tensor& sin,
    MPI_Comm comm
) {
    // For now, delegate to the baseline implementation with KV cache
    // TODO: Optimize with MPI data parallelism for MLP and attention
    return qwen3::qwen3_decoder_layer_with_cache(
        hidden_states,
        kv_cache,
        layer_idx,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        rms_norm_eps,
        input_layernorm_weight,
        qkv_projs,
        o_proj,
        q_norm_weight,
        k_norm_weight,
        post_attention_layernorm_weight,
        gate_mlp,
        up_mlp,
        down_mlp,
        cos,
        sin
    );
}

// ============================================================================
// MPI+OpenMP Forward Pass with KV Cache Support
// ============================================================================

Tensor qwen3_forward_mpi_omp_with_cache(
    const TensorL& input_ids,
    KVCache* kv_cache,
    const Tensor& token_embedding,
    const std::vector<Qwen3LayerWeights>& layers,
    const Tensor& norm_weight,
    const Tensor& lm_head,
    size_t num_layers,
    size_t num_attention_heads,
    size_t num_key_value_heads,
    size_t head_dim,
    float rms_norm_eps,
    MPI_Comm comm
) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // For now, delegate to baseline implementation with KV cache
    // TODO: Optimize with MPI data parallelism
    // All ranks compute the same result (data parallelism not yet implemented for cache)

    Tensor result = qwen3::qwen3_forward_with_cache(
        input_ids,
        kv_cache,
        token_embedding,
        layers,
        norm_weight,
        lm_head,
        num_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        rms_norm_eps
    );

    return result;
}

#endif // MPI_VERSION

} // namespace mpi
} // namespace qwen3
} // namespace tensor_cpp
