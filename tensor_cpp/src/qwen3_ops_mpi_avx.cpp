/**
 * @file qwen3_ops_mpi_avx.cpp
 * @brief MPI+AVX2 hybrid Qwen3 operator implementations
 */

#include "tensor_cpp/qwen3_ops_mpi_avx.h"
#include "tensor_cpp/ops.h"
#include "tensor_cpp/ops_mpi.h"
#include "tensor_cpp/ops_avx.h"
#include "tensor_cpp/attention_avx.h"
#include "tensor_cpp/qwen3_ops_mpi.h"
#include "tensor_cpp/qwen3_ops_avx.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>

using namespace tensor_cpp::ops;
using namespace tensor_cpp::ops::avx2;

namespace tensor_cpp {
namespace qwen3 {
namespace mpi_avx {

#ifdef MPI_VERSION

// ============================================================================
// Qwen3 MLP (SwiGLU) with MPI+AVX2
// ============================================================================

Tensor qwen3_mlp_mpi_avx(
    const Tensor& hidden_states,
    const Tensor& gate_proj,
    const Tensor& up_proj,
    const Tensor& down_proj,
    MPI_Comm comm
) {
    // Aligned with qwen3_mlp_mpi_omp implementation
    // Uses same structure and communication pattern, only adds AVX2 to matmul

    // hidden_states: [batch, seq_len, hidden_size]
    // gate_proj, up_proj: [intermediate_size, hidden_size]
    // down_proj: [hidden_size, intermediate_size]

    size_t batch_size = hidden_states.shape()[0];
    size_t seq_len = hidden_states.shape()[1];
    size_t hidden_size = hidden_states.shape()[2];

    // Reshape to [batch * seq_len, hidden_size] for linear layer
    Tensor hidden_reshaped = hidden_states.view({static_cast<long>(batch_size * seq_len), static_cast<long>(hidden_size)});

    // Gate projection (with MPI+AVX2)
    Tensor gate = ops::mpi::linear_mpi_omp_avx2(hidden_reshaped, gate_proj, nullptr, comm);

    // Up projection (with MPI+AVX2)
    Tensor up = ops::mpi::linear_mpi_omp_avx2(hidden_reshaped, up_proj, nullptr, comm);

    // SwiGLU activation (with MPI)
    Tensor activated = ops::mpi::swiglu_mpi_omp(gate, up, comm);

    // Down projection (with MPI+AVX2)
    Tensor output = ops::mpi::linear_mpi_omp_avx2(activated, down_proj, nullptr, comm);

    // Reshape back to [batch, seq_len, hidden_size]
    output = output.view({static_cast<long>(batch_size), static_cast<long>(seq_len), static_cast<long>(hidden_size)});

    return output;
}

// ============================================================================
// Qwen3 Decoder Layer with MPI+AVX2
// ============================================================================

Tensor qwen3_decoder_layer_mpi_avx(
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
    // Convert MPIAttentionType to mpi::MPIAttentionType for the call
    mpi::MPIAttentionType mpi_attn_type = mpi::MPIAttentionType::STANDARD;
    if (attention_type == MPIAttentionType::STREAMING) {
        mpi_attn_type = mpi::MPIAttentionType::STREAMING;
    }

    Tensor attn_output = qwen3::mpi::qwen3_attention_mpi_omp(
        hidden, num_attention_heads, num_key_value_heads, head_dim,
        qkv_projs, o_proj, q_norm_weight, k_norm_weight, cos, sin, comm,
        mpi_attn_type
    );

    // Residual
    hidden = residual + attn_output;

    // Post-attention layernorm
    residual = hidden;
    hidden = rms_norm(hidden, &post_attention_layernorm_weight, rms_norm_eps);

    // MLP with MPI+AVX2
    Tensor mlp_output = qwen3_mlp_mpi_avx(hidden, gate_mlp, up_mlp, down_mlp, comm);

    // Residual
    hidden = residual + mlp_output;

    return hidden;
}

// New overload with separated parallel strategy and algorithm
Tensor qwen3_decoder_layer_mpi_avx(
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

    // Self-attention with MPI+AVX2
    // Convert mpi_avx enums to mpi enums for the call
    mpi::ParallelStrategy mpi_strategy = static_cast<mpi::ParallelStrategy>(strategy);
    mpi::AttentionAlgorithm mpi_algorithm = static_cast<mpi::AttentionAlgorithm>(algorithm);

    Tensor attn_output = qwen3::mpi::qwen3_attention_mpi_omp_avx2(
        hidden, num_attention_heads, num_key_value_heads, head_dim,
        qkv_projs, o_proj, q_norm_weight, k_norm_weight, cos, sin, comm,
        mpi_strategy, mpi_algorithm
    );

    // Residual
    hidden = residual + attn_output;

    // Post-attention layernorm
    residual = hidden;
    hidden = rms_norm(hidden, &post_attention_layernorm_weight, rms_norm_eps);

    // MLP with MPI+AVX2
    Tensor mlp_output = qwen3_mlp_mpi_avx(hidden, gate_mlp, up_mlp, down_mlp, comm);

    // Residual
    hidden = residual + mlp_output;

    return hidden;
}

// ============================================================================
// Qwen3 Model (Full Forward Pass) with MPI+AVX2
// ============================================================================

Tensor qwen3_forward_mpi_avx(
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
    // Embed tokens
    Tensor hidden_states = embedding(input_ids, token_embedding);

    // Compute RoPE frequencies
    size_t seq_len = input_ids.shape()[1];
    auto rope_freqs = compute_rope_freqs(seq_len, head_dim);
    Tensor cos = rope_freqs.first;
    Tensor sin = rope_freqs.second;

    // Process through layers
    for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        const auto& layer = layers[layer_idx];

        hidden_states = qwen3_decoder_layer_mpi_avx(
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
Tensor qwen3_forward_mpi_avx(
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

    size_t batch_size = input_ids.shape()[0];
    size_t seq_len = input_ids.shape()[1];

    // Embed tokens - USE CORRECT METHOD BASED ON STRATEGY
    Tensor hidden_states;
    if (strategy == ParallelStrategy::SEQUENCE) {
        // True sequence parallel: distribute sequence positions, NO Allgather
        // Each rank computes embedding for its local sequence portion
        hidden_states = ops::mpi::embedding_mpi_omp_no_allgather(
            input_ids, token_embedding, -1, comm
        );  // [batch, local_seq_len, hidden_size] - DISTRIBUTED
    } else {  // HEAD_WISE
        // Head-wise parallel: all ranks need full sequence
        hidden_states = embedding(input_ids, token_embedding);
        // [batch, seq_len, hidden_size] - FULL (all ranks have copy)
    }

    // Compute RoPE frequencies
    auto rope_freqs = compute_rope_freqs(seq_len, head_dim);
    Tensor cos, sin;

    if (strategy == ParallelStrategy::SEQUENCE) {
        // Extract local cos/sin for DISTRIBUTED sequence
        size_t local_seq_len = hidden_states.shape()[1];
        size_t rope_dim = head_dim / 2;

        std::vector<float> cos_local_data(batch_size * local_seq_len * rope_dim);
        std::vector<float> sin_local_data(batch_size * local_seq_len * rope_dim);

        const float* cos_full_data = rope_freqs.first.data();
        const float* sin_full_data = rope_freqs.second.data();

        #pragma omp parallel for if(batch_size * local_seq_len * rope_dim > 1000)
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < local_seq_len; ++j) {
                size_t global_pos = rank * local_seq_len + j;
                for (size_t k = 0; k < rope_dim; ++k) {
                    cos_local_data[(i * local_seq_len + j) * rope_dim + k] = cos_full_data[global_pos * rope_dim + k];
                    sin_local_data[(i * local_seq_len + j) * rope_dim + k] = sin_full_data[global_pos * rope_dim + k];
                }
            }
        }

        cos = Tensor(std::move(cos_local_data), Shape({static_cast<long>(batch_size), static_cast<long>(local_seq_len), static_cast<long>(rope_dim)}));
        sin = Tensor(std::move(sin_local_data), Shape({static_cast<long>(batch_size), static_cast<long>(local_seq_len), static_cast<long>(rope_dim)}));
    } else {
        // Head-wise: use full cos/sin
        cos = rope_freqs.first;
        sin = rope_freqs.second;
    }

    // Process through layers using new API
    for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        const auto& layer = layers[layer_idx];

        hidden_states = qwen3_decoder_layer_mpi_avx(
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

    // For sequence parallel: Allgather to get full sequence
    if (strategy == ParallelStrategy::SEQUENCE) {
        size_t local_seq_len = hidden_normed.shape()[1];
        size_t hidden_size = hidden_normed.shape()[2];

        std::vector<float> hidden_full_data(batch_size * seq_len * hidden_size);

        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);

        for (int i = 0; i < size; ++i) {
            recvcounts[i] = batch_size * local_seq_len * hidden_size;
            displs[i] = i * batch_size * local_seq_len * hidden_size;
        }

        MPI_Allgatherv(
            hidden_normed.data(), recvcounts[rank], MPI_FLOAT,
            hidden_full_data.data(), recvcounts.data(), displs.data(), MPI_FLOAT, comm
        );

        hidden_normed = Tensor(std::move(hidden_full_data), Shape({
            static_cast<long>(batch_size),
            static_cast<long>(seq_len),
            static_cast<long>(hidden_size)
        }));
    }

    // LM head projection to get logits
    size_t hidden_size = hidden_normed.shape()[2];
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
// MPI+AVX2 Decoder Layer with KV Cache Support
// ============================================================================

Tensor qwen3_decoder_layer_mpi_avx_with_cache(
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
    // Split QKV projections to match AVX2 interface
    size_t q_size = num_attention_heads * head_dim;
    size_t k_size = num_key_value_heads * head_dim;
    size_t v_size = num_key_value_heads * head_dim;
    size_t hidden_size = qkv_projs.shape()[1];

    std::vector<float> q_data(q_size * hidden_size);
    std::vector<float> k_data(k_size * hidden_size);
    std::vector<float> v_data(v_size * hidden_size);

    for (size_t row = 0; row < q_size; ++row) {
        for (size_t col = 0; col < hidden_size; ++col) {
            q_data[row * hidden_size + col] = qkv_projs[row * hidden_size + col];
        }
    }

    for (size_t row = 0; row < k_size; ++row) {
        for (size_t col = 0; col < hidden_size; ++col) {
            k_data[row * hidden_size + col] = qkv_projs[(q_size + row) * hidden_size + col];
        }
    }

    for (size_t row = 0; row < v_size; ++row) {
        for (size_t col = 0; col < hidden_size; ++col) {
            v_data[row * hidden_size + col] = qkv_projs[(q_size + k_size + row) * hidden_size + col];
        }
    }

    Tensor q_proj(std::move(q_data), Shape({static_cast<long>(q_size), static_cast<long>(hidden_size)}));
    Tensor k_proj(std::move(k_data), Shape({static_cast<long>(k_size), static_cast<long>(hidden_size)}));
    Tensor v_proj(std::move(v_data), Shape({static_cast<long>(v_size), static_cast<long>(hidden_size)}));

    // Delegate to AVX2 implementation with KV cache
    // Convert MPIAttentionType to qwen3::AttentionType
    qwen3::AttentionType avx_attn_type = qwen3::AttentionType::STANDARD;
    if (attention_type == MPIAttentionType::STREAMING) {
        avx_attn_type = qwen3::AttentionType::STREAMING;
    }

    // TODO: Optimize with MPI data parallelism for MLP and attention
    return tensor_cpp::qwen3::avx2::qwen3_decoder_layer_avx_with_cache(
        hidden_states,
        kv_cache,
        layer_idx,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        rms_norm_eps,
        input_layernorm_weight,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_norm_weight,
        k_norm_weight,
        post_attention_layernorm_weight,
        gate_mlp,
        up_mlp,
        down_mlp,
        cos,
        sin,
        avx_attn_type
    );
}

// ============================================================================
// MPI+AVX2 Forward Pass with KV Cache Support
// ============================================================================

Tensor qwen3_forward_mpi_avx_with_cache(
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

    // Convert MPIAttentionType to qwen3::AttentionType
    qwen3::AttentionType avx_attn_type = qwen3::AttentionType::STANDARD;
    if (attention_type == MPIAttentionType::STREAMING) {
        avx_attn_type = qwen3::AttentionType::STREAMING;
    }

    // For now, delegate to AVX2 implementation with KV cache
    // TODO: Optimize with MPI data parallelism
    // All ranks compute the same result (data parallelism not yet implemented for cache)

    Tensor result = tensor_cpp::qwen3::avx2::qwen3_forward_avx_with_cache(
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
        avx_attn_type
    );

    return result;
}

// ============================================================================
// True Sequence Parallel with AVX2 Optimization
// ============================================================================

Tensor qwen3_attention_true_sequence_parallel_avx2(
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
    // hidden_states: [batch, local_seq_len, hidden_size] - DISTRIBUTED
    const Shape& hidden_shape = hidden_states.shape();
    size_t batch_size = hidden_shape[0];
    size_t local_seq_len = hidden_shape[1];
    size_t hidden_size = hidden_shape[2];

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Split combined QKV projections
    size_t q_size = num_attention_heads * head_dim;
    size_t k_size = num_key_value_heads * head_dim;
    size_t v_size = num_key_value_heads * head_dim;

    // Extract Q, K, V projections from combined qkv_projs
    std::vector<float> q_proj_data(q_size * hidden_size);
    std::vector<float> k_proj_data(k_size * hidden_size);
    std::vector<float> v_proj_data(v_size * hidden_size);

    #pragma omp parallel for if(q_size * hidden_size > 1000)
    for (size_t row = 0; row < q_size; ++row) {
        for (size_t col = 0; col < hidden_size; ++col) {
            q_proj_data[row * hidden_size + col] = qkv_projs[row * hidden_size + col];
        }
    }

    #pragma omp parallel for if(k_size * hidden_size > 1000)
    for (size_t row = 0; row < k_size; ++row) {
        for (size_t col = 0; col < hidden_size; ++col) {
            k_proj_data[row * hidden_size + col] = qkv_projs[(q_size + row) * hidden_size + col];
        }
    }

    #pragma omp parallel for if(v_size * hidden_size > 1000)
    for (size_t row = 0; row < v_size; ++row) {
        for (size_t col = 0; col < hidden_size; ++col) {
            v_proj_data[row * hidden_size + col] = qkv_projs[(q_size + k_size + row) * hidden_size + col];
        }
    }

    Tensor q_proj(std::move(q_proj_data), Shape({static_cast<long>(q_size), static_cast<long>(hidden_size)}));
    Tensor k_proj(std::move(k_proj_data), Shape({static_cast<long>(k_size), static_cast<long>(hidden_size)}));
    Tensor v_proj(std::move(v_proj_data), Shape({static_cast<long>(v_size), static_cast<long>(hidden_size)}));

    // Reshape hidden_states: [batch, local_seq_len, hidden_size] -> [batch * local_seq_len, hidden_size]
    Tensor hidden_reshaped = hidden_states.reshape({static_cast<long>(batch_size * local_seq_len), static_cast<long>(hidden_size)});

    // QKV projections using AVX2-optimized linear (NO MPI, NO Allgather)
    // Each rank computes full features for its local sequence positions
    Tensor q = linear_avx2(hidden_reshaped, q_proj, nullptr);  // [batch * local_seq_len, q_size]
    Tensor k = linear_avx2(hidden_reshaped, k_proj, nullptr);  // [batch * local_seq_len, k_size]
    Tensor v = linear_avx2(hidden_reshaped, v_proj, nullptr);  // [batch * local_seq_len, v_size]

    // Q normalization with AVX2
    {
        const float* q_data = q.data();
        const float* q_norm_data = q_norm_weight.data();
        std::vector<float> q_normalized_data(q.size());

        size_t num_q_elements = batch_size * local_seq_len * q_size;

        #pragma omp parallel for if(num_q_elements > 1000)
        for (size_t i = 0; i < num_q_elements; ++i) {
            float val = q_data[i];
            float norm_sq = val * val;
            float normalized = val / std::sqrt(norm_sq + 1e-6f);
            q_normalized_data[i] = normalized * q_norm_data[i % q_size];
        }

        q = Tensor(std::move(q_normalized_data), q.shape());
    }

    // K normalization with AVX2
    {
        const float* k_data = k.data();
        const float* k_norm_data = k_norm_weight.data();
        std::vector<float> k_normalized_data(k.size());

        size_t num_k_elements = batch_size * local_seq_len * k_size;

        #pragma omp parallel for if(num_k_elements > 1000)
        for (size_t i = 0; i < num_k_elements; ++i) {
            float val = k_data[i];
            float norm_sq = val * val;
            float normalized = val / std::sqrt(norm_sq + 1e-6f);
            k_normalized_data[i] = normalized * k_norm_data[i % k_size];
        }

        k = Tensor(std::move(k_normalized_data), k.shape());
    }

    // Reshape Q, K, V: [batch * local_seq_len, head_size] -> [batch, num_heads, local_seq_len, head_dim]
    q = q.reshape({static_cast<long>(batch_size), static_cast<long>(num_attention_heads), static_cast<long>(local_seq_len), static_cast<long>(head_dim)});
    k = k.reshape({static_cast<long>(batch_size), static_cast<long>(num_key_value_heads), static_cast<long>(local_seq_len), static_cast<long>(head_dim)});
    v = v.reshape({static_cast<long>(batch_size), static_cast<long>(num_key_value_heads), static_cast<long>(local_seq_len), static_cast<long>(head_dim)});

    // Apply RoPE (cos/sin are already distributed for local sequence)
    auto qk_rope = apply_rotary_pos_emb(q, k, cos, sin);
    q = qk_rope.first;
    k = qk_rope.second;

    // Compute global sequence length
    size_t global_seq_len = local_seq_len * size;

    // Sequence parallel attention with AVX2
    // This uses attention_sequence_online_softmax_avx2 which is already AVX2-optimized
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    Tensor attn_output = ops::mpi::attention_sequence_online_softmax_avx2(
        q, k, v, nullptr, scale,
        num_attention_heads, num_key_value_heads,
        global_seq_len, comm
    );  // [batch, num_heads, local_seq_len, head_dim]

    // Reshape: [batch, num_heads, local_seq_len, head_dim] -> [batch, local_seq_len, num_heads * head_dim]
    attn_output = attn_output.reshape({
        static_cast<long>(batch_size),
        static_cast<long>(local_seq_len),
        static_cast<long>(num_attention_heads * head_dim)
    });

    // Output projection using AVX2-optimized linear
    Tensor attn_output_reshaped = attn_output.reshape({static_cast<long>(batch_size * local_seq_len), static_cast<long>(num_attention_heads * head_dim)});
    Tensor output = linear_avx2(attn_output_reshaped, o_proj, nullptr);  // [batch * local_seq_len, hidden_size]

    // Reshape back: [batch * local_seq_len, hidden_size] -> [batch, local_seq_len, hidden_size]
    output = output.reshape({
        static_cast<long>(batch_size),
        static_cast<long>(local_seq_len),
        static_cast<long>(hidden_size)
    });

    // Return DISTRIBUTED output (NO Allgather here!)
    return output;  // [batch, local_seq_len, hidden_size] - DISTRIBUTED
}

Tensor qwen3_forward_true_sequence_parallel_avx2(
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

    // Get input dimensions
    const Shape& input_shape = input_ids.shape();
    size_t batch_size = input_shape[0];
    size_t seq_len = input_shape[1];

    // Calculate local sequence length for this rank
    size_t local_seq_len = seq_len / size;

    // Extract local input IDs for this rank (DISTRIBUTED sequence)
    size_t num_ids = batch_size * local_seq_len;
    std::vector<long> local_input_ids_data(num_ids);

    const long* input_ids_data = input_ids.data();
    #pragma omp parallel for if(num_ids > 1000)
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < local_seq_len; ++j) {
            size_t global_pos = rank * local_seq_len + j;
            local_input_ids_data[i * local_seq_len + j] = input_ids_data[i * seq_len + global_pos];
        }
    }

    TensorL local_input_ids(std::move(local_input_ids_data), Shape({static_cast<long>(batch_size), static_cast<long>(local_seq_len)}));

    // Embedding layer (NO Allgather - keeps sequence distributed)
    Tensor hidden_states = ops::mpi::embedding_mpi_omp_no_allgather(
        local_input_ids, token_embedding, -1, comm
    );  // [batch, local_seq_len, hidden_size] - DISTRIBUTED

    // Compute RoPE frequencies for full sequence
    auto rope_freqs = compute_rope_freqs(seq_len, head_dim);
    Tensor& cos_full = rope_freqs.first;   // [seq_len, head_dim/2]
    Tensor& sin_full = rope_freqs.second;  // [seq_len, head_dim/2]

    // Extract local cos/sin for this rank (DISTRIBUTED)
    size_t rope_dim = head_dim / 2;
    std::vector<float> cos_local_data(batch_size * local_seq_len * rope_dim);
    std::vector<float> sin_local_data(batch_size * local_seq_len * rope_dim);

    const float* cos_full_data = cos_full.data();
    const float* sin_full_data = sin_full.data();

    #pragma omp parallel for if(batch_size * local_seq_len * rope_dim > 1000)
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < local_seq_len; ++j) {
            size_t global_pos = rank * local_seq_len + j;
            for (size_t k = 0; k < rope_dim; ++k) {
                cos_local_data[(i * local_seq_len + j) * rope_dim + k] = cos_full_data[global_pos * rope_dim + k];
                sin_local_data[(i * local_seq_len + j) * rope_dim + k] = sin_full_data[global_pos * rope_dim + k];
            }
        }
    }

    Tensor cos_local(std::move(cos_local_data), Shape({static_cast<long>(batch_size), static_cast<long>(local_seq_len), static_cast<long>(rope_dim)}));
    Tensor sin_local(std::move(sin_local_data), Shape({static_cast<long>(batch_size), static_cast<long>(local_seq_len), static_cast<long>(rope_dim)}));

    // Process through layers (sequence stays DISTRIBUTED)
    for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        const auto& layer = layers[layer_idx];

        // Input layernorm with AVX2
        Tensor hidden_normed = rms_norm(hidden_states, &layer.input_layernorm_weight, rms_norm_eps);

        // Self-attention with TRUE sequence parallel and AVX2
        Tensor attn_output = qwen3_attention_true_sequence_parallel_avx2(
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
        );  // [batch, local_seq_len, hidden_size] - DISTRIBUTED

        // Residual connection
        hidden_states = hidden_states + attn_output;

        // Post-attention layernorm with AVX2
        Tensor residual = hidden_states;
        hidden_normed = rms_norm(hidden_states, &layer.post_attention_layernorm_weight, rms_norm_eps);

        // MLP with AVX2-optimized MPI (this does Allgather for intermediate features)
        // Note: This could be further optimized to avoid Allgather
        Tensor mlp_output = qwen3_mlp_mpi_avx(
            hidden_normed,
            layer.gate_proj,
            layer.up_proj,
            layer.down_proj,
            comm
        );  // [batch, seq_len, hidden_size] - GATHERED (due to MLP Allgather)

        // Residual connection
        hidden_states = residual + mlp_output;

        // After MLP, sequence is gathered. Need to redistribute for next layer's attention
        // For now, we keep it gathered and rely on attention to handle local sequence
        // This is a compromise - true sequence parallel would need MLP without Allgather too
    }

    // Final layernorm with AVX2
    Tensor hidden_normed = rms_norm(hidden_states, &norm_weight, rms_norm_eps);

    // Now we need to Allgather the sequence dimension
    // All ranks should have full [batch, seq_len, hidden_size]
    size_t hidden_size = hidden_states.shape()[2];
    std::vector<float> hidden_full_data(batch_size * seq_len * hidden_size);

    // Each rank contributes its local sequence portion
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);

    for (int i = 0; i < size; ++i) {
        recvcounts[i] = batch_size * local_seq_len * hidden_size;
        displs[i] = i * batch_size * local_seq_len * hidden_size;
    }

    const float* hidden_local_data = hidden_normed.data();
    MPI_Allgatherv(
        (void*)hidden_local_data, recvcounts[rank], MPI_FLOAT,
        hidden_full_data.data(), recvcounts.data(), displs.data(), MPI_FLOAT, comm
    );

    Tensor hidden_full(std::move(hidden_full_data), Shape({
        static_cast<long>(batch_size),
        static_cast<long>(seq_len),
        static_cast<long>(hidden_size)
    }));

    // LM head projection to get logits
    size_t vocab_size = lm_head.shape()[0];
    size_t num_samples = batch_size * seq_len;

    std::vector<float> logits_data(num_samples * vocab_size);

    // AVX2-optimized LM head projection
    #pragma omp parallel for if(num_samples * vocab_size > 1000)
    for (size_t s = 0; s < num_samples; ++s) {
        for (size_t v = 0; v < vocab_size; ++v) {
            float sum = 0.0f;
            size_t h = 0;

            // AVX2 dot product
            __m256 sum_vec = _mm256_setzero_ps();

            for (; h + 8 <= hidden_size; h += 8) {
                __m256 hidden_vec = _mm256_loadu_ps(&hidden_full_data[s * hidden_size + h]);
                __m256 weight_vec = _mm256_loadu_ps(&lm_head[v * hidden_size + h]);
                sum_vec = _mm256_fmadd_ps(hidden_vec, weight_vec, sum_vec);
            }

            sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
            sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
            float temp[8];
            _mm256_storeu_ps(temp, sum_vec);
            sum = temp[0] + temp[4];

            // Remaining elements
            for (; h < hidden_size; ++h) {
                sum += hidden_full_data[s * hidden_size + h] * lm_head[v * hidden_size + h];
            }

            logits_data[s * vocab_size + v] = sum;
        }
    }

    return Tensor(std::move(logits_data), Shape({
        static_cast<long>(batch_size),
        static_cast<long>(seq_len),
        static_cast<long>(vocab_size)
    }));
}

#endif // MPI_VERSION

} // namespace mpi_avx
} // namespace qwen3
} // namespace tensor_cpp
