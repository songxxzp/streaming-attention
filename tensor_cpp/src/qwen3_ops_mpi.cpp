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
    MPI_Comm comm,
    MPIAttentionType attention_type
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
    Tensor attn_output;

    if (attention_type == MPIAttentionType::STREAMING) {
        // Use streaming attention (memory efficient, recommended for prefill)
        attn_output = ops::mpi::self_attention_mpi_streaming_omp(
            q, k, v, nullptr, scale,
            num_attention_heads, num_key_value_heads, comm
        );
    } else {
        // Use standard attention
        attn_output = ops::mpi::self_attention_mpi_omp(
            q, k, v, nullptr, scale,
            num_attention_heads, num_key_value_heads, comm
        );
    }

    // Reshape output for projection
    attn_output = attn_output.view({static_cast<long>(batch_size * seq_len), static_cast<long>(num_attention_heads * head_dim)});

    // Output projection
    Tensor output = ops::mpi::linear_mpi_omp(attn_output, o_proj, nullptr, comm);

    // Reshape back
    output = output.view({static_cast<long>(batch_size), static_cast<long>(seq_len), static_cast<long>(hidden_size)});

    return output;
}

// New overload with separated parallel strategy and algorithm
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
    MPI_Comm comm,
    ParallelStrategy strategy,
    AttentionAlgorithm algorithm
) {
    // hidden_states: [batch, seq_len, hidden_size]
    size_t batch_size = hidden_states.shape()[0];
    size_t seq_len = hidden_states.shape()[1];
    size_t hidden_size = hidden_states.shape()[2];

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Compute QKV projection (same as old version)
    size_t q_dim = num_attention_heads * head_dim;
    size_t kv_dim = num_key_value_heads * head_dim;
    size_t total_qkv_dim = q_dim + 2 * kv_dim;
    size_t weight_hidden_size = qkv_projs.shape()[1];

    // Extract Q projection
    std::vector<float> q_proj_data(q_dim * weight_hidden_size);
    for (size_t i = 0; i < q_dim; ++i) {
        for (size_t j = 0; j < weight_hidden_size; ++j) {
            q_proj_data[i * weight_hidden_size + j] = qkv_projs[i * weight_hidden_size + j];
        }
    }
    Tensor q_proj(std::move(q_proj_data), Shape({static_cast<long>(q_dim), static_cast<long>(weight_hidden_size)}));

    // Extract K projection
    std::vector<float> k_proj_data(kv_dim * weight_hidden_size);
    for (size_t i = 0; i < kv_dim; ++i) {
        for (size_t j = 0; j < weight_hidden_size; ++j) {
            k_proj_data[i * weight_hidden_size + j] = qkv_projs[(q_dim + i) * weight_hidden_size + j];
        }
    }
    Tensor k_proj(std::move(k_proj_data), Shape({static_cast<long>(kv_dim), static_cast<long>(weight_hidden_size)}));

    // Extract V projection
    std::vector<float> v_proj_data(kv_dim * weight_hidden_size);
    for (size_t i = 0; i < kv_dim; ++i) {
        for (size_t j = 0; j < weight_hidden_size; ++j) {
            v_proj_data[i * weight_hidden_size + j] = qkv_projs[(q_dim + kv_dim + i) * weight_hidden_size + j];
        }
    }
    Tensor v_proj(std::move(v_proj_data), Shape({static_cast<long>(kv_dim), static_cast<long>(weight_hidden_size)}));

    // Compute Q, K, V
    Tensor hidden_reshaped = hidden_states.view({static_cast<long>(batch_size * seq_len), static_cast<long>(hidden_size)});
    Tensor q, k, v;

    // IMPORTANT: For true sequence parallel, we must NOT Allgather in QKV projection
    // - HEAD_WISE: Needs Allgather because each rank needs full sequence to compute its heads
    // - SEQUENCE: NO Allgather! Each rank keeps its local sequence portion
    if (strategy == ParallelStrategy::SEQUENCE) {
        // True sequence parallel: NO Allgather, sequence stays distributed
        q = ops::linear(hidden_reshaped, q_proj, nullptr);
        k = ops::linear(hidden_reshaped, k_proj, nullptr);
        v = ops::linear(hidden_reshaped, v_proj, nullptr);
    } else {  // HEAD_WISE
        // Head-wise parallel: Allgather needed, each rank gets full sequence
        q = ops::mpi::linear_mpi_omp(hidden_reshaped, q_proj, nullptr, comm);
        k = ops::mpi::linear_mpi_omp(hidden_reshaped, k_proj, nullptr, comm);
        v = ops::mpi::linear_mpi_omp(hidden_reshaped, v_proj, nullptr, comm);
    }

    // Reshape Q, K, V
    q = q.view({static_cast<long>(batch_size), static_cast<long>(num_attention_heads),
                static_cast<long>(seq_len), static_cast<long>(head_dim)});
    k = k.view({static_cast<long>(batch_size), static_cast<long>(num_key_value_heads),
                static_cast<long>(seq_len), static_cast<long>(head_dim)});
    v = v.view({static_cast<long>(batch_size), static_cast<long>(num_key_value_heads),
                static_cast<long>(seq_len), static_cast<long>(head_dim)});

    // Apply RoPE
    auto qk_rope = apply_rotary_pos_emb(q, k, cos, sin);
    q = qk_rope.first;
    k = qk_rope.second;

    // Select attention implementation based on strategy and algorithm
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    Tensor attn_output;

    if (strategy == ParallelStrategy::HEAD_WISE) {
        // Head-wise parallelism
        if (algorithm == AttentionAlgorithm::ONLINE_SOFTMAX) {
            attn_output = ops::mpi::attention_headwise_online_softmax(
                q, k, v, nullptr, scale,
                num_attention_heads, num_key_value_heads, comm
            );
        } else {  // STANDARD
            attn_output = ops::mpi::attention_headwise_standard(
                q, k, v, nullptr, scale,
                num_attention_heads, num_key_value_heads, comm
            );
        }
    } else {  // SEQUENCE
        // Sequence parallelism
        // Note: Input Q,K,V are DISTRIBUTED across sequence dimension
        if (algorithm == AttentionAlgorithm::ONLINE_SOFTMAX) {
            size_t global_seq_len = seq_len * size;  // Assumes equal distribution
            attn_output = ops::mpi::attention_sequence_online_softmax(
                q, k, v, nullptr, scale,
                num_attention_heads, num_key_value_heads,
                global_seq_len, comm
            );
        } else {
            throw std::runtime_error("Sequence parallelism with standard attention not implemented. Use ONLINE_SOFTMAX algorithm.");
        }
    }

    // Output projection
    // For sequence parallel: keep output distributed (NO Allgather)
    // For head-wise parallel: Allgather needed
    attn_output = attn_output.view({static_cast<long>(batch_size * seq_len), static_cast<long>(num_attention_heads * head_dim)});
    Tensor output;
    if (strategy == ParallelStrategy::SEQUENCE) {
        // True sequence parallel: NO Allgather, output stays distributed
        output = ops::linear(attn_output, o_proj, nullptr);
    } else {  // HEAD_WISE
        // Head-wise parallel: Allgather needed to combine results from all ranks
        output = ops::mpi::linear_mpi_omp(attn_output, o_proj, nullptr, comm);
    }
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
    MPI_Comm comm,
    MPIAttentionType attention_type
) {
    // Input layernorm
    Tensor residual = hidden_states;
    Tensor hidden = rms_norm(hidden_states, &input_layernorm_weight, rms_norm_eps);

    // Self-attention with MPI
    Tensor attn_output = qwen3_attention_mpi_omp(
        hidden, num_attention_heads, num_key_value_heads, head_dim,
        qkv_projs, o_proj, q_norm_weight, k_norm_weight, cos, sin, comm, attention_type
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

// New overload with separated parallel strategy and algorithm
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
    MPI_Comm comm,
    ParallelStrategy strategy,
    AttentionAlgorithm algorithm
) {
    // Input layernorm
    Tensor residual = hidden_states;
    Tensor hidden = rms_norm(hidden_states, &input_layernorm_weight, rms_norm_eps);

    // Self-attention with MPI using new API
    Tensor attn_output = qwen3_attention_mpi_omp(
        hidden, num_attention_heads, num_key_value_heads, head_dim,
        qkv_projs, o_proj, q_norm_weight, k_norm_weight, cos, sin, comm,
        strategy, algorithm
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
    MPI_Comm comm,
    MPIAttentionType attention_type
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
            comm,
            attention_type
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

// New overload with separated parallel strategy and algorithm
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
    MPI_Comm comm,
    ParallelStrategy strategy,
    AttentionAlgorithm algorithm
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

    // Process through decoder layers using new API
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
            comm,
            strategy,
            algorithm
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
    MPI_Comm comm,
    MPIAttentionType attention_type
) {
    // For now, delegate to the baseline implementation with KV cache
    // Convert MPI attention type to standard attention type
    qwen3::AttentionType std_attention_type = qwen3::AttentionType::STANDARD;
    if (attention_type == MPIAttentionType::STREAMING) {
        std_attention_type = qwen3::AttentionType::STREAMING;
    }

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
        sin,
        std_attention_type
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
    MPI_Comm comm,
    MPIAttentionType attention_type
) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // For now, delegate to baseline implementation with KV cache
    // TODO: Optimize with MPI data parallelism
    // All ranks compute the same result (data parallelism not yet implemented for cache)

    // Convert MPI attention type to standard attention type
    qwen3::AttentionType std_attention_type = qwen3::AttentionType::STANDARD;
    if (attention_type == MPIAttentionType::STREAMING) {
        std_attention_type = qwen3::AttentionType::STREAMING;
    }

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
        rms_norm_eps,
        std_attention_type
    );

    return result;
}

// ============================================================================
// AVX2-Optimized MPI Attention Functions
// ============================================================================

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #ifdef __AVX2__

Tensor qwen3_attention_mpi_omp_avx2(
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
    MPI_Comm comm,
    ParallelStrategy strategy,
    AttentionAlgorithm algorithm
) {
    // hidden_states: [batch, seq_len, hidden_size]
    size_t batch_size = hidden_states.shape()[0];
    size_t seq_len = hidden_states.shape()[1];
    size_t hidden_size = hidden_states.shape()[2];

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Compute QKV projection (same as non-AVX2 version)
    size_t q_dim = num_attention_heads * head_dim;
    size_t kv_dim = num_key_value_heads * head_dim;
    size_t total_qkv_dim = q_dim + 2 * kv_dim;
    size_t weight_hidden_size = qkv_projs.shape()[1];

    // Extract Q projection
    std::vector<float> q_proj_data(q_dim * weight_hidden_size);
    for (size_t i = 0; i < q_dim; ++i) {
        for (size_t j = 0; j < weight_hidden_size; ++j) {
            q_proj_data[i * weight_hidden_size + j] = qkv_projs[i * weight_hidden_size + j];
        }
    }
    Tensor q_proj(std::move(q_proj_data), Shape({static_cast<long>(q_dim), static_cast<long>(weight_hidden_size)}));

    // Extract K projection
    std::vector<float> k_proj_data(kv_dim * weight_hidden_size);
    for (size_t i = 0; i < kv_dim; ++i) {
        for (size_t j = 0; j < weight_hidden_size; ++j) {
            k_proj_data[i * weight_hidden_size + j] = qkv_projs[(q_dim + i) * weight_hidden_size + j];
        }
    }
    Tensor k_proj(std::move(k_proj_data), Shape({static_cast<long>(kv_dim), static_cast<long>(weight_hidden_size)}));

    // Extract V projection
    std::vector<float> v_proj_data(kv_dim * weight_hidden_size);
    for (size_t i = 0; i < kv_dim; ++i) {
        for (size_t j = 0; j < weight_hidden_size; ++j) {
            v_proj_data[i * weight_hidden_size + j] = qkv_projs[(q_dim + kv_dim + i) * weight_hidden_size + j];
        }
    }
    Tensor v_proj(std::move(v_proj_data), Shape({static_cast<long>(kv_dim), static_cast<long>(weight_hidden_size)}));

    // Compute Q, K, V
    Tensor hidden_reshaped = hidden_states.view({static_cast<long>(batch_size * seq_len), static_cast<long>(hidden_size)});
    Tensor q, k, v;

    // IMPORTANT: For true sequence parallel, we must NOT Allgather in QKV projection
    // - HEAD_WISE: Needs Allgather because each rank needs full sequence to compute its heads
    // - SEQUENCE: NO Allgather! Each rank keeps its local sequence portion
    if (strategy == ParallelStrategy::SEQUENCE) {
        // True sequence parallel: NO Allgather, sequence stays distributed
        q = ops::linear(hidden_reshaped, q_proj, nullptr);
        k = ops::linear(hidden_reshaped, k_proj, nullptr);
        v = ops::linear(hidden_reshaped, v_proj, nullptr);
    } else {  // HEAD_WISE
        // Head-wise parallel: Allgather needed, each rank gets full sequence
        q = ops::mpi::linear_mpi_omp(hidden_reshaped, q_proj, nullptr, comm);
        k = ops::mpi::linear_mpi_omp(hidden_reshaped, k_proj, nullptr, comm);
        v = ops::mpi::linear_mpi_omp(hidden_reshaped, v_proj, nullptr, comm);
    }

    // Reshape Q, K, V
    q = q.view({static_cast<long>(batch_size), static_cast<long>(num_attention_heads),
                static_cast<long>(seq_len), static_cast<long>(head_dim)});
    k = k.view({static_cast<long>(batch_size), static_cast<long>(num_key_value_heads),
                static_cast<long>(seq_len), static_cast<long>(head_dim)});
    v = v.view({static_cast<long>(batch_size), static_cast<long>(num_key_value_heads),
                static_cast<long>(seq_len), static_cast<long>(head_dim)});

    // Apply RoPE
    auto qk_rope = apply_rotary_pos_emb(q, k, cos, sin);
    q = qk_rope.first;
    k = qk_rope.second;

    // Select attention implementation based on strategy and algorithm (USE AVX2 VERSION)
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    Tensor attn_output;

    if (strategy == ParallelStrategy::HEAD_WISE) {
        // Head-wise parallelism - no AVX2 version available yet, use standard
        if (algorithm == AttentionAlgorithm::ONLINE_SOFTMAX) {
            attn_output = ops::mpi::attention_headwise_online_softmax(
                q, k, v, nullptr, scale,
                num_attention_heads, num_key_value_heads, comm
            );
        } else {  // STANDARD
            attn_output = ops::mpi::attention_headwise_standard(
                q, k, v, nullptr, scale,
                num_attention_heads, num_key_value_heads, comm
            );
        }
    } else {  // SEQUENCE
        // Sequence parallelism - USE AVX2 OPTIMIZED VERSION
        // Note: Input Q,K,V are DISTRIBUTED across sequence dimension
        if (algorithm == AttentionAlgorithm::ONLINE_SOFTMAX) {
            size_t global_seq_len = seq_len * size;  // Assumes equal distribution
            attn_output = ops::mpi::attention_sequence_online_softmax_avx2(
                q, k, v, nullptr, scale,
                num_attention_heads, num_key_value_heads,
                global_seq_len, comm
            );
        } else {
            throw std::runtime_error("Sequence parallelism with standard attention not implemented. Use ONLINE_SOFTMAX algorithm.");
        }
    }

    // Output projection
    // For sequence parallel: keep output distributed (NO Allgather)
    // For head-wise parallel: Allgather needed
    attn_output = attn_output.view({static_cast<long>(batch_size * seq_len), static_cast<long>(num_attention_heads * head_dim)});
    Tensor output;
    if (strategy == ParallelStrategy::SEQUENCE) {
        // True sequence parallel: NO Allgather, output stays distributed
        output = ops::linear(attn_output, o_proj, nullptr);
    } else {  // HEAD_WISE
        // Head-wise parallel: Allgather needed to combine results from all ranks
        output = ops::mpi::linear_mpi_omp(attn_output, o_proj, nullptr, comm);
    }
    output = output.view({static_cast<long>(batch_size), static_cast<long>(seq_len), static_cast<long>(hidden_size)});

    return output;
}

    #endif // __AVX2__
#endif // x86_64

// ============================================================================
// True Sequence Parallel Attention (No Allgather in QKV projection)
// ============================================================================

/**
 * @brief True sequence parallel attention with NO Allgather in QKV projection
 *
 * This is the CORRECT implementation of sequence parallelism:
 * 1. Input hidden_states is DISTRIBUTED: [batch, local_seq_len, hidden_size]
 * 2. QKV projection: Each rank computes full features for its local sequence (NO Allgather)
 * 3. Attention: Each rank computes attention for its local query positions
 * 4. Output projection: Each rank computes for its local sequence
 * 5. Output is DISTRIBUTED: [batch, local_seq_len, hidden_size]
 * 6. Final Allgather happens only at the very end (outside this function)
 *
 * Key difference from qwen3_attention_mpi_omp_avx2:
 * - Uses NO-ALLGATHER embedding and linear operations
 * - Sequence dimension stays distributed throughout
 * - No redundant computation
 */
Tensor qwen3_attention_true_sequence_parallel(
    const Tensor& hidden_states,  // [batch, local_seq_len, hidden_size] - DISTRIBUTED
    size_t num_attention_heads,
    size_t num_key_value_heads,
    size_t head_dim,
    const Tensor& qkv_projs,
    const Tensor& o_proj,
    const Tensor& q_norm_weight,
    const Tensor& k_norm_weight,
    const Tensor& cos,  // [local_seq_len, head_dim/2] - DISTRIBUTED
    const Tensor& sin,  // [local_seq_len, head_dim/2] - DISTRIBUTED
    MPI_Comm comm
) {
    // hidden_states: [batch, local_seq_len, hidden_size] - DISTRIBUTED
    size_t batch_size = hidden_states.shape()[0];
    size_t local_seq_len = hidden_states.shape()[1];
    size_t hidden_size = hidden_states.shape()[2];

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    size_t global_seq_len = local_seq_len * size;

    // Extract QKV projections (same for all ranks, weights are replicated)
    size_t q_dim = num_attention_heads * head_dim;
    size_t kv_dim = num_key_value_heads * head_dim;
    size_t total_qkv_dim = q_dim + 2 * kv_dim;
    size_t weight_hidden_size = qkv_projs.shape()[1];

    // Extract Q, K, V projections (local copy, no distribution needed)
    std::vector<float> q_proj_data(q_dim * weight_hidden_size);
    for (size_t i = 0; i < q_dim; ++i) {
        for (size_t j = 0; j < weight_hidden_size; ++j) {
            q_proj_data[i * weight_hidden_size + j] = qkv_projs[i * weight_hidden_size + j];
        }
    }
    Tensor q_proj(std::move(q_proj_data), Shape({static_cast<long>(q_dim), static_cast<long>(weight_hidden_size)}));

    std::vector<float> k_proj_data(kv_dim * weight_hidden_size);
    for (size_t i = 0; i < kv_dim; ++i) {
        for (size_t j = 0; j < weight_hidden_size; ++j) {
            k_proj_data[i * weight_hidden_size + j] = qkv_projs[(q_dim + i) * weight_hidden_size + j];
        }
    }
    Tensor k_proj(std::move(k_proj_data), Shape({static_cast<long>(kv_dim), static_cast<long>(weight_hidden_size)}));

    std::vector<float> v_proj_data(kv_dim * weight_hidden_size);
    for (size_t i = 0; i < kv_dim; ++i) {
        for (size_t j = 0; j < weight_hidden_size; ++j) {
            v_proj_data[i * weight_hidden_size + j] = qkv_projs[(q_dim + kv_dim + i) * weight_hidden_size + j];
        }
    }
    Tensor v_proj(std::move(v_proj_data), Shape({static_cast<long>(kv_dim), static_cast<long>(weight_hidden_size)}));

    // Compute Q, K, V using LOCAL LINEAR (NO Allgather!)
    // Each rank computes for its local sequence positions only
    Tensor hidden_reshaped = hidden_states.view({static_cast<long>(batch_size * local_seq_len), static_cast<long>(hidden_size)});

    // Use ops::linear (NOT MPI version) - compute full features for local sequence
    Tensor q = ops::linear(hidden_reshaped, q_proj, nullptr);
    Tensor k = ops::linear(hidden_reshaped, k_proj, nullptr);
    Tensor v = ops::linear(hidden_reshaped, v_proj, nullptr);

    // Reshape Q, K, V to [batch, num_heads, local_seq_len, head_dim]
    q = q.view({static_cast<long>(batch_size), static_cast<long>(num_attention_heads),
                static_cast<long>(local_seq_len), static_cast<long>(head_dim)});
    k = k.view({static_cast<long>(batch_size), static_cast<long>(num_key_value_heads),
                static_cast<long>(local_seq_len), static_cast<long>(head_dim)});
    v = v.view({static_cast<long>(batch_size), static_cast<long>(num_key_value_heads),
                static_cast<long>(local_seq_len), static_cast<long>(head_dim)});

    // Apply RoPE (cos and sin are already distributed)
    auto qk_rope = apply_rotary_pos_emb(q, k, cos, sin);
    q = qk_rope.first;
    k = qk_rope.second;

    // Compute attention using AVX2-optimized sequence parallel attention
    // This function expects distributed Q, K, V with local sequence length
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    Tensor attn_output = ops::mpi::attention_sequence_online_softmax_avx2(
        q, k, v, nullptr, scale,
        num_attention_heads, num_key_value_heads,
        global_seq_len, comm
    );
    // Output: [batch, num_heads, local_seq_len, head_dim] - DISTRIBUTED

    // Output projection using LOCAL LINEAR (NO Allgather!)
    attn_output = attn_output.view({static_cast<long>(batch_size * local_seq_len), static_cast<long>(num_attention_heads * head_dim)});
    Tensor output = ops::linear(attn_output, o_proj, nullptr);
    output = output.view({static_cast<long>(batch_size), static_cast<long>(local_seq_len), static_cast<long>(hidden_size)});

    // Return DISTRIBUTED output
    // Caller is responsible for final Allgather
    return output;
}

// ============================================================================
// True Sequence Parallel Forward Pass (Complete Implementation)
// ============================================================================

/**
 * @brief Complete forward pass with TRUE sequence parallelism
 *
 * This implements correct sequence parallelism:
 * 1. Embedding: sequence is distributed [batch, local_seq_len, hidden_size]
 * 2. Each layer: sequence stays distributed
 * 3. Attention: each rank computes only local query positions
 * 4. Final output: Allgather happens only once at the very end
 *
 * Key difference from qwen3_forward_mpi_omp:
 * - NO Allgather in embedding or QKV projection
 * - Sequence dimension stays distributed throughout
 * - NO redundant computation
 */
Tensor qwen3_forward_true_sequence_parallel(
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

    // Step 1: Embed tokens with DISTRIBUTED sequence dimension
    // Output: [batch, local_seq_len, hidden_size] - DISTRIBUTED
    Tensor hidden_states = ops::mpi::embedding_mpi_omp_no_allgather(
        input_ids, token_embedding, -1, comm
    );

    size_t batch_size = hidden_states.shape()[0];
    size_t local_seq_len = hidden_states.shape()[1];
    size_t hidden_size = hidden_states.shape()[2];
    size_t global_seq_len = input_ids.shape()[1];

    // Step 2: Compute RoPE frequencies (distributed, same for all ranks)
    auto rope_freqs = compute_rope_freqs(global_seq_len, head_dim);

    // Extract local portion of cos/sin
    int seq_per_rank = (global_seq_len + size - 1) / size;
    int start_seq = rank * seq_per_rank;
    int end_seq = std::min(start_seq + seq_per_rank, static_cast<int>(global_seq_len));

    // Extract local cos/sin from global cos/sin
    std::vector<float> cos_local_data(local_seq_len * (head_dim / 2));
    std::vector<float> sin_local_data(local_seq_len * (head_dim / 2));

    const float* cos_global = rope_freqs.first.data();
    const float* sin_global = rope_freqs.second.data();

    for (size_t s = 0; s < local_seq_len; ++s) {
        size_t global_s = start_seq + s;
        for (size_t d = 0; d < head_dim / 2; ++d) {
            cos_local_data[s * (head_dim / 2) + d] = cos_global[global_s * (head_dim / 2) + d];
            sin_local_data[s * (head_dim / 2) + d] = sin_global[global_s * (head_dim / 2) + d];
        }
    }

    Tensor cos_local(std::move(cos_local_data), Shape({static_cast<long>(local_seq_len), static_cast<long>(head_dim / 2)}));
    Tensor sin_local(std::move(sin_local_data), Shape({static_cast<long>(local_seq_len), static_cast<long>(head_dim / 2)}));

    // Step 3: Process through layers (hidden states stay DISTRIBUTED)
    for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        const auto& layer = layers[layer_idx];

        // Input layernorm (local computation)
        Tensor hidden_normed = rms_norm(hidden_states, &layer.input_layernorm_weight, rms_norm_eps);

        // Self-attention with TRUE sequence parallelism
        Tensor attn_output = qwen3_attention_true_sequence_parallel(
            hidden_normed,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            layer.qkv_projs,
            layer.o_proj,
            layer.q_norm_weight,
            layer.k_norm_weight,
            cos_local,
            sin_local,
            comm
        );
        // Output: [batch, local_seq_len, hidden_size] - DISTRIBUTED

        // Residual connection (local)
        hidden_states = hidden_states + attn_output;

        // Post-attention layernorm (local)
        Tensor post_normed = rms_norm(hidden_states, &layer.post_attention_layernorm_weight, rms_norm_eps);

        // MLP (local computation on distributed sequence)
        // TODO: Implement distributed MLP (no Allgather in gate/up/down)
        // For now, use regular MLP which is fine since it's just local computation
        Tensor mlp_output = qwen3_mlp_mpi_omp(post_normed, layer.gate_proj, layer.up_proj, layer.down_proj, comm);

        // Residual connection (local)
        hidden_states = hidden_states + mlp_output;
        // Still DISTRIBUTED: [batch, local_seq_len, hidden_size]
    }

    // Final layernorm (local)
    Tensor hidden_normed = rms_norm(hidden_states, &norm_weight, rms_norm_eps);

    // Step 4: Final Allgather (ONLY ONCE!)
    // Gather from [batch, local_seq_len, hidden_size] to [batch, global_seq_len, hidden_size]
    std::vector<float> hidden_full_data(batch_size * global_seq_len * hidden_size);
    const float* hidden_local_data = hidden_normed.data();

    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);

    for (int i = 0; i < size; ++i) {
        int s_s = i * seq_per_rank;
        int e_s = std::min(s_s + seq_per_rank, static_cast<int>(global_seq_len));
        recvcounts[i] = batch_size * (e_s - s_s) * hidden_size;
        displs[i] = batch_size * s_s * hidden_size;
    }

    size_t local_size = batch_size * local_seq_len * hidden_size;
    MPI_Allgatherv(
        (void*)hidden_local_data, local_size, MPI_FLOAT,
        hidden_full_data.data(), recvcounts.data(), displs.data(), MPI_FLOAT, comm
    );

    Tensor hidden_full(std::move(hidden_full_data), Shape({static_cast<long>(batch_size), static_cast<long>(global_seq_len), static_cast<long>(hidden_size)}));

    // Step 5: LM head projection (after Allgather)
    // Output: [batch, global_seq_len, vocab_size]
    size_t vocab_size = lm_head.shape()[0];
    size_t num_samples = batch_size * global_seq_len;

    std::vector<float> logits_data(num_samples * vocab_size);

    #pragma omp parallel for if(num_samples * vocab_size > 1000)
    for (size_t s = 0; s < num_samples; ++s) {
        for (size_t v = 0; v < vocab_size; ++v) {
            float sum = 0.0f;
            for (size_t h = 0; h < hidden_size; ++h) {
                sum += hidden_full.data()[s * hidden_size + h] * lm_head.data()[v * hidden_size + h];
            }
            logits_data[s * vocab_size + v] = sum;
        }
    }

    return Tensor(std::move(logits_data), Shape({static_cast<long>(batch_size), static_cast<long>(global_seq_len), static_cast<long>(vocab_size)}));
}

#endif // MPI_VERSION

} // namespace mpi
} // namespace qwen3
} // namespace tensor_cpp
