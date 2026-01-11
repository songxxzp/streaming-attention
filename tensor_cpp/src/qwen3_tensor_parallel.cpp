/**
 * @file qwen3_tensor_parallel.cpp
 * @brief Implementation of Qwen3 tensor parallelism with MPI
 */

#include "tensor_cpp/qwen3_tensor_parallel.h"
#include "tensor_cpp/ops.h"
#include "tensor_cpp/ops_mpi.h"
#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/qwen3_ops_mpi.h"
#include <algorithm>

using namespace tensor_cpp::ops;

namespace tensor_cpp {
namespace qwen3 {
namespace tensor_parallel {

#ifdef MPI_VERSION

// ============================================================================
// Weight Distribution
// ============================================================================

Qwen3Weights distribute_weights(
    const Qwen3Weights& weights,
    int rank,
    int size
) {
    Qwen3Weights local_weights;

    // Distribute embedding: all ranks get full copy
    local_weights.embed_tokens = weights.embed_tokens;
    local_weights.lm_head = weights.lm_head;
    local_weights.norm_weight = weights.norm_weight;

    // Distribute each layer's weights
    local_weights.num_layers = weights.num_layers;
    local_weights.num_attention_heads = weights.num_attention_heads;
    local_weights.num_key_value_heads = weights.num_key_value_heads;
    local_weights.head_dim = weights.head_dim;
    local_weights.hidden_size = weights.hidden_size;
    local_weights.num_layers = weights.num_layers;

    size_t hidden_size = weights.hidden_size;
    size_t intermediate_size = 4 * hidden_size;  // Qwen3 uses 4x hidden_size
    size_t num_heads = weights.num_attention_heads;
    size_t head_dim = weights.head_dim;
    size_t kv_heads = weights.num_key_value_heads;

    size_t local_hidden = hidden_size / size;
    size_t local_intermediate = intermediate_size / size;
    size_t local_heads = num_heads / size;
    size_t local_kv_heads = kv_heads;

    // Distribute each layer
    for (const auto& global_layer : weights.layers) {
        Qwen3LayerWeights local_layer;

        // Attention weights - distribute QKV projection
        size_t q_out = num_heads * head_dim;
        size_t kv_out = kv_heads * head_dim;
        size_t total_qkv = q_out + 2 * kv_out;

        // Each rank gets a portion of QKV projection
        size_t local_qkv = total_qkv / size;
        size_t qkv_start = rank * local_qkv;

        // Extract local portion
        std::vector<float> local_qkv_data(local_qkv * hidden_size);
        for (size_t i = 0; i < local_qkv; ++i) {
            for (size_t j = 0; j < hidden_size; ++j) {
                local_qkv_data[i * hidden_size + j] =
                    global_layer.qkv_projs[(qkv_start + i) * hidden_size + j];
            }
        }
        local_layer.qkv_projs = Tensor(std::move(local_qkv_data),
                                       Shape({static_cast<long>(local_qkv), static_cast<long>(hidden_size)}));

        // Output projection - distribute columns
        std::vector<float> local_o_data(local_hidden * hidden_size);
        size_t o_start = rank * local_hidden;
        for (size_t i = 0; i < local_hidden; ++i) {
            for (size_t j = 0; j < hidden_size; ++j) {
                local_o_data[i * hidden_size + j] =
                    global_layer.o_proj[(o_start + i) * hidden_size + j];
            }
        }
        local_layer.o_proj = Tensor(std::move(local_o_data),
                                    Shape({static_cast<long>(local_hidden), static_cast<long>(hidden_size)}));

        // QKNorm weights - distribute per-head
        // Each rank gets norm weights for its local heads
        local_layer.q_norm_weight = global_layer.q_norm_weight;  // Simplified
        local_layer.k_norm_weight = global_layer.k_norm_weight;  // Simplified

        // Layer norms - all ranks get full copy
        local_layer.input_layernorm_weight = global_layer.input_layernorm_weight;
        local_layer.post_attention_layernorm_weight = global_layer.post_attention_layernorm_weight;

        // MLP weights - distribute intermediate dimension
        size_t gate_start = rank * local_intermediate;
        std::vector<float> local_gate_data(local_intermediate * hidden_size);
        for (size_t i = 0; i < local_intermediate; ++i) {
            for (size_t j = 0; j < hidden_size; ++j) {
                local_gate_data[i * hidden_size + j] =
                    global_layer.gate_proj[(gate_start + i) * hidden_size + j];
            }
        }
        local_layer.gate_proj = Tensor(std::move(local_gate_data),
                                       Shape({static_cast<long>(local_intermediate), static_cast<long>(hidden_size)}));

        std::vector<float> local_up_data(local_intermediate * hidden_size);
        for (size_t i = 0; i < local_intermediate; ++i) {
            for (size_t j = 0; j < hidden_size; ++j) {
                local_up_data[i * hidden_size + j] =
                    global_layer.up_proj[(gate_start + i) * hidden_size + j];
            }
        }
        local_layer.up_proj = Tensor(std::move(local_up_data),
                                    Shape({static_cast<long>(local_intermediate), static_cast<long>(hidden_size)}));

        // Down projection - distribute rows (input features)
        std::vector<float> local_down_data(hidden_size * local_intermediate);
        for (size_t i = 0; i < hidden_size; ++i) {
            for (size_t j = 0; j < local_intermediate; ++j) {
                local_down_data[i * local_intermediate + j] =
                    global_layer.down_proj[i * intermediate_size + (gate_start + j)];
            }
        }
        local_layer.down_proj = Tensor(std::move(local_down_data),
                                        Shape({static_cast<long>(hidden_size), static_cast<long>(local_intermediate)}));

        local_weights.layers.push_back(local_layer);
    }

    return local_weights;
}

// ============================================================================
// Tensor Parallel Operations
// ============================================================================

Tensor linear_tensor_parallel(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    MPI_Comm comm
) {
    // input: [seq_len, in_features]
    // weight: [out_features/size, in_features] (local portion)
    // output: [seq_len, out_features] (allreduced)

    size_t seq_len = input.shape()[0];
    size_t in_features = input.shape()[1];
    size_t local_out_features = weight.shape()[0];

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Local computation
    std::vector<float> local_output(seq_len * local_out_features);

    #pragma omp parallel for if(seq_len * local_out_features > 100)
    for (size_t s = 0; s < seq_len; ++s) {
        for (size_t o = 0; o < local_out_features; ++o) {
            float sum = 0.0f;
            #pragma omp simd reduction(+:sum)
            for (size_t i = 0; i < in_features; ++i) {
                sum += input[s * in_features + i] * weight[o * in_features + i];
            }
            local_output[s * local_out_features + o] = sum + (bias ? (*bias)[rank * local_out_features + o] : 0.0f);
        }
    }

    // Allgather to get full output
    size_t total_out_features = local_out_features * size;
    std::vector<float> full_output(seq_len * total_out_features);

    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);

    for (int i = 0; i < size; ++i) {
        recvcounts[i] = seq_len * local_out_features;
        displs[i] = i * seq_len * local_out_features;
    }

    MPI_Allgatherv(
        local_output.data(), local_output.size(), MPI_FLOAT,
        full_output.data(), recvcounts.data(), displs.data(), MPI_FLOAT, comm
    );

    Shape result_shape({static_cast<long>(seq_len), static_cast<long>(total_out_features)});
    return Tensor(std::move(full_output), result_shape);
}

Tensor attention_tensor_parallel(
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
    int rank,
    int size,
    MPI_Comm comm
) {
    // hidden_states: [batch, seq_len, hidden_size]

    // Use the MPI+OpenMP Qwen3 attention function which handles combined qkv_projs
    // Note: This is a simplified implementation that uses data parallelism
    // rather than true tensor parallelism for attention
    return qwen3::mpi::qwen3_attention_mpi_omp(
        hidden_states,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        qkv_projs,
        o_proj,
        q_norm_weight,
        k_norm_weight,
        cos,
        sin,
        comm
    );
}

// ============================================================================
// Full Forward Pass with Tensor Parallelism
// ============================================================================

Tensor forward_tensor_parallel(
    const TensorL& input_ids,
    const Qwen3Weights& local_weights,
    size_t num_layers,
    size_t num_attention_heads,
    size_t num_key_value_heads,
    size_t head_dim,
    float rms_norm_eps,
    int rank,
    int size,
    MPI_Comm comm
) {
    // Embed tokens (all ranks have full embedding)
    Tensor hidden_states = embedding(input_ids, local_weights.embed_tokens);

    // Compute RoPE frequencies
    size_t seq_len = input_ids.shape()[1];
    auto rope_freqs = compute_rope_freqs(seq_len, head_dim);
    Tensor cos = rope_freqs.first;
    Tensor sin = rope_freqs.second;

    // Process through layers with tensor parallelism
    for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        const auto& layer = local_weights.layers[layer_idx];

        // Input layernorm
        Tensor residual = hidden_states;
        Tensor hidden = rms_norm(hidden_states, &layer.input_layernorm_weight, rms_norm_eps);

        // Self-attention (simplified - use MPI attention)
        Tensor attn_output = attention_tensor_parallel(
            hidden, num_attention_heads, num_key_value_heads, head_dim,
            layer.qkv_projs, layer.o_proj,
            layer.q_norm_weight, layer.k_norm_weight,
            cos, sin, rank, size, comm
        );

        // Allreduce to combine tensor parallel results
        ops::mpi::all_reduce_sum(attn_output, comm);

        // Residual
        hidden = residual + attn_output;

        // Post-attention layernorm
        residual = hidden;
        hidden = rms_norm(hidden, &layer.post_attention_layernorm_weight, rms_norm_eps);

        // MLP
        // Simplified: use local MLP weights
        Tensor mlp_output = qwen3_mlp(
            hidden, layer.gate_proj, layer.up_proj, layer.down_proj
        );

        // Allreduce MLP output
        ops::mpi::all_reduce_sum(mlp_output, comm);

        // Residual
        hidden = residual + mlp_output;
    }

    // Final layernorm
    hidden_states = rms_norm(hidden_states, &local_weights.norm_weight, rms_norm_eps);

    return hidden_states;
}

#endif // MPI_VERSION

} // namespace tensor_parallel
} // namespace qwen3
} // namespace tensor_cpp
