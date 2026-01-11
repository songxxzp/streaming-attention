/**
 * @file qwen3_ops_mpi_avx.cpp
 * @brief MPI+AVX2 hybrid Qwen3 operator implementations
 */

#include "tensor_cpp/qwen3_ops_mpi_avx.h"
#include "tensor_cpp/ops.h"
#include "tensor_cpp/ops_mpi.h"
#include "tensor_cpp/ops_avx.h"
#include "tensor_cpp/qwen3_ops_mpi.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>

using namespace tensor_cpp::ops;

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
    // hidden_states: [batch, seq_len, hidden_size]
    const Shape& hidden_shape = hidden_states.shape();
    size_t batch = hidden_shape[0];
    size_t seq_len = hidden_shape[1];
    size_t hidden_size = hidden_shape[2];
    size_t intermediate_size = gate_proj.shape()[0];

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Distribute intermediate dimension across ranks
    int local_intermediate = intermediate_size / size;
    int start_intermediate = rank * local_intermediate;

    // Compute gate projection (local portion)
    std::vector<float> gate_local_data(batch * seq_len * local_intermediate);

    #pragma omp parallel for if(batch * seq_len * local_intermediate > 1000)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            size_t row_offset = (b * seq_len + s) * local_intermediate;
            size_t input_offset = (b * seq_len + s) * hidden_size;

            for (int i = 0; i < local_intermediate; ++i) {
                float sum = 0.0f;
                int global_i = start_intermediate + i;
                size_t weight_offset = global_i * hidden_size;

                // AVX2 dot product
                size_t j = 0;
                __m256 sum_vec = _mm256_setzero_ps();

                for (; j + 8 <= hidden_size; j += 8) {
                    __m256 hidden_vec = _mm256_loadu_ps(&hidden_states[input_offset + j]);
                    __m256 weight_vec = _mm256_loadu_ps(&gate_proj[weight_offset + j]);
                    sum_vec = _mm256_fmadd_ps(hidden_vec, weight_vec, sum_vec);
                }

                sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
                sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
                float temp[8];
                _mm256_storeu_ps(temp, sum_vec);
                sum = temp[0] + temp[4];

                for (; j < hidden_size; ++j) {
                    sum += hidden_states[input_offset + j] * gate_proj[weight_offset + j];
                }

                gate_local_data[row_offset + i] = sum;
            }
        }
    }

    // Allgather gate projections
    std::vector<float> gate_data(batch * seq_len * intermediate_size);
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);

    for (int i = 0; i < size; ++i) {
        recvcounts[i] = batch * seq_len * local_intermediate;
        displs[i] = i * batch * seq_len * local_intermediate;
    }

    MPI_Allgatherv(
        gate_local_data.data(), gate_local_data.size(), MPI_FLOAT,
        gate_data.data(), recvcounts.data(), displs.data(), MPI_FLOAT, comm
    );

    Tensor gate(std::move(gate_data), Shape({static_cast<long>(batch), static_cast<long>(seq_len), static_cast<long>(intermediate_size)}));

    // Compute up projection (local portion)
    std::vector<float> up_local_data(batch * seq_len * local_intermediate);

    #pragma omp parallel for if(batch * seq_len * local_intermediate > 1000)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            size_t row_offset = (b * seq_len + s) * local_intermediate;
            size_t input_offset = (b * seq_len + s) * hidden_size;

            for (int i = 0; i < local_intermediate; ++i) {
                float sum = 0.0f;
                int global_i = start_intermediate + i;
                size_t weight_offset = global_i * hidden_size;

                // AVX2 dot product
                size_t j = 0;
                __m256 sum_vec = _mm256_setzero_ps();

                for (; j + 8 <= hidden_size; j += 8) {
                    __m256 hidden_vec = _mm256_loadu_ps(&hidden_states[input_offset + j]);
                    __m256 weight_vec = _mm256_loadu_ps(&up_proj[weight_offset + j]);
                    sum_vec = _mm256_fmadd_ps(hidden_vec, weight_vec, sum_vec);
                }

                sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
                sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
                float temp[8];
                _mm256_storeu_ps(temp, sum_vec);
                sum = temp[0] + temp[4];

                for (; j < hidden_size; ++j) {
                    sum += hidden_states[input_offset + j] * up_proj[weight_offset + j];
                }

                up_local_data[row_offset + i] = sum;
            }
        }
    }

    // Allgather up projections
    std::vector<float> up_data(batch * seq_len * intermediate_size);

    MPI_Allgatherv(
        up_local_data.data(), up_local_data.size(), MPI_FLOAT,
        up_data.data(), recvcounts.data(), displs.data(), MPI_FLOAT, comm
    );

    Tensor up(std::move(up_data), Shape({static_cast<long>(batch), static_cast<long>(seq_len), static_cast<long>(intermediate_size)}));

    // SwiGLU activation with AVX2
    std::vector<float> swiglu_data(batch * seq_len * intermediate_size);

    #pragma omp parallel for if(batch * seq_len * intermediate_size > 1000)
    for (size_t i = 0; i < batch * seq_len * intermediate_size; i += 8) {
        if (i + 8 <= batch * seq_len * intermediate_size) {
            __m256 gate_vec = _mm256_loadu_ps(&gate[i]);
            __m256 up_vec = _mm256_loadu_ps(&up[i]);

            // Fast sigmoid approximation with manual abs
            __m256 sign_mask = _mm256_set1_ps(-0.0f);
            __m256 abs_up = _mm256_andnot_ps(sign_mask, up_vec);  // abs(x) = x & ~sign_bit
            __m256 ones = _mm256_set1_ps(1.0f);
            __m256 sigmoid = _mm256_div_ps(up_vec, _mm256_add_ps(abs_up, ones));

            __m256 result = _mm256_mul_ps(gate_vec, sigmoid);
            _mm256_storeu_ps(&swiglu_data[i], result);
        } else {
            for (size_t j = i; j < batch * seq_len * intermediate_size; ++j) {
                float up_val = up[j];
                float sigmoid_val = up_val / (1.0f + std::abs(up_val));
                swiglu_data[j] = gate[j] * sigmoid_val;
            }
        }
    }

    Tensor swiglu(std::move(swiglu_data), gate.shape());

    // Down projection (distribute rows)
    std::vector<float> down_local_data(batch * seq_len * hidden_size);

    #pragma omp parallel for if(batch * seq_len * hidden_size > 1000)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            size_t row_offset = (b * seq_len + s) * hidden_size;
            size_t input_offset = (b * seq_len + s) * intermediate_size;

            for (size_t i = 0; i < hidden_size; ++i) {
                float sum = 0.0f;
                size_t weight_offset = i * intermediate_size;

                // AVX2 dot product
                size_t j = 0;
                __m256 sum_vec = _mm256_setzero_ps();

                for (; j + 8 <= intermediate_size; j += 8) {
                    __m256 swiglu_vec = _mm256_loadu_ps(&swiglu[input_offset + j]);
                    __m256 weight_vec = _mm256_loadu_ps(&down_proj[weight_offset + j]);
                    sum_vec = _mm256_fmadd_ps(swiglu_vec, weight_vec, sum_vec);
                }

                sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
                sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
                float temp[8];
                _mm256_storeu_ps(temp, sum_vec);
                sum = temp[0] + temp[4];

                for (; j < intermediate_size; ++j) {
                    sum += swiglu[input_offset + j] * down_proj[weight_offset + j];
                }

                down_local_data[row_offset + i] = sum;
            }
        }
    }

    // Allreduce down projections
    Tensor down_result(std::move(down_local_data), hidden_shape);
    ops::mpi::all_reduce_sum(down_result, comm);

    return down_result;
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
    MPI_Comm comm
) {
    // Input layernorm
    Tensor residual = hidden_states;
    Tensor hidden = rms_norm(hidden_states, &input_layernorm_weight, rms_norm_eps);

    // Self-attention with MPI
    Tensor attn_output = qwen3::mpi::qwen3_attention_mpi_omp(
        hidden, num_attention_heads, num_key_value_heads, head_dim,
        qkv_projs, o_proj, q_norm_weight, k_norm_weight, cos, sin, comm
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
    size_t num_layers,
    size_t num_attention_heads,
    size_t num_key_value_heads,
    size_t head_dim,
    float rms_norm_eps,
    MPI_Comm comm
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
            comm
        );
    }

    // Final layernorm
    hidden_states = rms_norm(hidden_states, &norm_weight, rms_norm_eps);

    return hidden_states;
}

#endif // MPI_VERSION

} // namespace mpi_avx
} // namespace qwen3
} // namespace tensor_cpp
