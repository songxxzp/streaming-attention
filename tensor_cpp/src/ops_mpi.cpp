/**
 * @file ops_mpi.cpp
 * @brief Implementation of MPI+OpenMP parallelized operators
 */

#include "tensor_cpp/ops_mpi.h"
#include "tensor_cpp/ops.h"
#include <algorithm>
#include <limits>
#include <complex>
#include <cmath>
#include <cstring>

namespace tensor_cpp {
namespace ops {
namespace mpi {

#ifdef MPI_VERSION

// ============================================================================
// MPI Communication Helpers
// ============================================================================

void all_reduce_sum(Tensor& tensor, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size <= 1) return;  // No communication needed

    // Create a copy of the data
    std::vector<float> data(tensor.size());
    std::copy(tensor.data(), tensor.data() + tensor.size(), data.begin());

    MPI_Allreduce(MPI_IN_PLACE, data.data(), data.size(), MPI_FLOAT, MPI_SUM, comm);

    // Update tensor data
    tensor = Tensor(std::move(data), tensor.shape());
}

void broadcast(Tensor& tensor, int root, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size <= 1) return;

    // Create a copy of the data
    std::vector<float> data(tensor.size());
    std::copy(tensor.data(), tensor.data() + tensor.size(), data.begin());

    MPI_Bcast(data.data(), data.size(), MPI_FLOAT, root, comm);

    // Update tensor data
    tensor = Tensor(std::move(data), tensor.shape());
}

std::pair<int, int> get_mpi_info(MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    return {rank, size};
}

// ============================================================================
// MPI+OpenMP Matrix Multiplication
// ============================================================================

void matmul_mpi_omp(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    MPI_Comm comm
) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Each rank computes a subset of rows
    int rows_per_rank = (M + size - 1) / size;
    int start_row = rank * rows_per_rank;
    int end_row = std::min(start_row + rows_per_rank, M);
    int local_M = end_row - start_row;

    // Allocate local result buffer
    std::vector<float> C_local(local_M * N);

    // Local computation with OpenMP
    #pragma omp parallel for if(local_M * N * K > 1000)
    for (int i = 0; i < local_M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[(start_row + i) * K + k] * B[j * K + k];
            }
            C_local[i * N + j] = sum;
        }
    }

    // Gather results from all ranks
    if (size > 1) {
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);

        for (int i = 0; i < size; ++i) {
            int s_row = i * rows_per_rank;
            int e_row = std::min(s_row + rows_per_rank, M);
            recvcounts[i] = (e_row - s_row) * N;
            displs[i] = s_row * N;
        }

        MPI_Allgatherv(
            C_local.data(), local_M * N, MPI_FLOAT,
            C, recvcounts.data(), displs.data(), MPI_FLOAT, comm
        );
    } else {
        // Single rank, just copy
        std::copy(C_local.begin(), C_local.end(), C);
    }
}

// ============================================================================
// MPI+OpenMP Element-wise Operations
// ============================================================================

Tensor add_mpi_omp(
    const Tensor& input,
    const Tensor& other,
    float alpha,
    MPI_Comm comm
) {
    if (input.shape() != other.shape()) {
        throw std::invalid_argument("Shape mismatch for MPI add");
    }

    std::vector<float> result(input.size());

    #pragma omp parallel for if(input.size() > 1000)
    for (size_t i = 0; i < input.size(); ++i) {
        result[i] = input[i] + alpha * other[i];
    }

    // No MPI communication needed for element-wise ops
    // (assuming all ranks have the same input)
    return Tensor(std::move(result), input.shape());
}

// ============================================================================
// MPI+OpenMP RMSNorm
// ============================================================================

Tensor rms_norm_mpi_omp(
    const Tensor& input,
    const Tensor* weight,
    float eps,
    MPI_Comm comm
) {
    // input: [batch, seq_len, hidden_size]
    size_t batch_size = input.shape()[0];
    size_t seq_len = input.shape()[1];
    size_t hidden_size = input.shape()[2];

    std::vector<float> result(input.size());

    // Normalize along last dimension
    #pragma omp parallel for if(batch_size * seq_len > 10)
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            // Compute RMS
            float sum_sq = 0.0f;
            size_t offset = (b * seq_len + s) * hidden_size;

            #pragma omp simd reduction(+:sum_sq)
            for (size_t h = 0; h < hidden_size; ++h) {
                float val = input[offset + h];
                sum_sq += val * val;
            }

            float rms = std::sqrt(sum_sq / hidden_size + eps);

            // Normalize
            for (size_t h = 0; h < hidden_size; ++h) {
                float normalized = input[offset + h] / rms;
                if (weight) {
                    result[offset + h] = normalized * (*weight)[h];
                } else {
                    result[offset + h] = normalized;
                }
            }
        }
    }

    return Tensor(std::move(result), input.shape());
}

// ============================================================================
// MPI+OpenMP Rotary Position Embedding
// ============================================================================

Tensor rope_mpi_omp(
    const Tensor& input,
    const Tensor& cos,
    const Tensor& sin,
    MPI_Comm comm
) {
    // input: [batch, num_heads, seq_len, head_dim]
    size_t batch_size = input.shape()[0];
    size_t num_heads = input.shape()[1];
    size_t seq_len = input.shape()[2];
    size_t head_dim = input.shape()[3];

    std::vector<float> result(input.size());

    // Apply RoPE
    #pragma omp parallel for if(batch_size * num_heads * seq_len > 10)
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t s = 0; s < seq_len; ++s) {
                size_t base_idx = ((b * num_heads + h) * seq_len + s) * head_dim;
                size_t cos_idx = s * (head_dim / 2);

                for (size_t d = 0; d < head_dim / 2; ++d) {
                    float x0 = input[base_idx + 2 * d];
                    float x1 = input[base_idx + 2 * d + 1];
                    float c = cos[cos_idx + d];
                    float s_val = sin[cos_idx + d];

                    result[base_idx + 2 * d] = x0 * c - x1 * s_val;
                    result[base_idx + 2 * d + 1] = x0 * s_val + x1 * c;
                }
            }
        }
    }

    return Tensor(std::move(result), input.shape());
}

// ============================================================================
// MPI+OpenMP SwiGLU
// ============================================================================

Tensor swiglu_mpi_omp(
    const Tensor& x,
    const Tensor& gate,
    MPI_Comm comm
) {
    if (x.shape() != gate.shape()) {
        throw std::invalid_argument("Shape mismatch for MPI swiglu");
    }

    std::vector<float> result(x.size());

    // SwiGLU(x, gate) = SiLU(gate) * x = (gate / (1 + exp(-gate))) * x
    #pragma omp parallel for if(x.size() > 1000)
    for (size_t i = 0; i < x.size(); ++i) {
        float sigmoid_gate = gate[i] / (1.0f + std::exp(-gate[i]));
        result[i] = sigmoid_gate * x[i];
    }

    return Tensor(std::move(result), x.shape());
}

// ============================================================================
// MPI+OpenMP Self-Attention
// ============================================================================

Tensor self_attention_mpi_omp(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor* mask,
    float scale,
    int num_attention_heads,
    int num_key_value_heads,
    MPI_Comm comm
) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Distribute attention heads across MPI ranks
    int heads_per_rank = (num_attention_heads + size - 1) / size;
    int start_head = rank * heads_per_rank;
    int end_head = std::min(start_head + heads_per_rank, num_attention_heads);
    int local_num_heads = end_head - start_head;

    // Tensor shapes
    size_t batch_size = query.shape()[0];
    size_t q_seq_len = query.shape()[2];
    size_t k_seq_len = key.shape()[2];
    size_t head_dim = query.shape()[3];

    // Compute GQA repetition factor
    int n_rep = num_attention_heads / num_key_value_heads;

    // Repeat KV heads for GQA (only for local heads)
    Tensor k_repeated, v_repeated;

    if (local_num_heads > 0) {
        // Extract local KV heads
        int kv_start = start_head / n_rep;
        int kv_end = (end_head + n_rep - 1) / n_rep;
        int local_kv_heads = kv_end - kv_start;

        // Extract and repeat KV heads
        std::vector<float> k_local_data(local_kv_heads * batch_size * k_seq_len * head_dim);
        std::vector<float> v_local_data(local_kv_heads * batch_size * k_seq_len * head_dim);

        for (int h = 0; h < local_kv_heads; ++h) {
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t s = 0; s < k_seq_len; ++s) {
                    for (size_t d = 0; d < head_dim; ++d) {
                        size_t src_idx = ((b * num_key_value_heads + (kv_start + h)) * k_seq_len + s) * head_dim + d;
                        size_t dst_idx = ((b * local_kv_heads + h) * k_seq_len + s) * head_dim + d;
                        k_local_data[dst_idx] = key[src_idx];
                        v_local_data[dst_idx] = value[src_idx];
                    }
                }
            }
        }

        // Repeat for local attention heads
        std::vector<float> k_repeated_data(local_num_heads * batch_size * k_seq_len * head_dim);
        std::vector<float> v_repeated_data(local_num_heads * batch_size * k_seq_len * head_dim);

        for (int h_local = 0; h_local < local_num_heads; ++h_local) {
            int h_global = start_head + h_local;
            int h_kv = h_global / n_rep - kv_start;

            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t s = 0; s < k_seq_len; ++s) {
                    for (size_t d = 0; d < head_dim; ++d) {
                        size_t src_idx = ((b * local_kv_heads + h_kv) * k_seq_len + s) * head_dim + d;
                        size_t dst_idx = ((b * local_num_heads + h_local) * k_seq_len + s) * head_dim + d;
                        k_repeated_data[dst_idx] = k_local_data[src_idx];
                        v_repeated_data[dst_idx] = v_local_data[src_idx];
                    }
                }
            }
        }

        Shape local_shape({static_cast<long>(batch_size), static_cast<long>(local_num_heads),
                          static_cast<long>(k_seq_len), static_cast<long>(head_dim)});
        k_repeated = Tensor(std::move(k_repeated_data), local_shape);
        v_repeated = Tensor(std::move(v_repeated_data), local_shape);
    }

    // Compute attention scores for local heads
    std::vector<float> attn_output;
    if (local_num_heads > 0) {
        attn_output.resize(local_num_heads * batch_size * q_seq_len * head_dim, 0.0f);

        #pragma omp parallel for if(batch_size * local_num_heads * q_seq_len > 10)
        for (size_t b = 0; b < batch_size; ++b) {
            for (int h = 0; h < local_num_heads; ++h) {
                for (size_t i = 0; i < q_seq_len; ++i) {
                    // Compute attention scores
                    std::vector<float> scores(k_seq_len);
                    for (size_t j = 0; j < k_seq_len; ++j) {
                        float sum = 0.0f;
                        for (size_t d = 0; d < head_dim; ++d) {
                            size_t q_idx = ((b * local_num_heads + h) * q_seq_len + i) * head_dim + d;
                            size_t k_idx = ((b * local_num_heads + h) * k_seq_len + j) * head_dim + d;
                            // Extract from original query tensor
                            size_t q_src_idx = ((b * num_attention_heads + (start_head + h)) * q_seq_len + i) * head_dim + d;
                            sum += query[q_src_idx] * k_repeated[k_idx];
                        }
                        scores[j] = sum * scale;
                    }

                    // Apply mask if provided
                    if (mask) {
                        for (size_t j = 0; j < k_seq_len; ++j) {
                            size_t mask_idx = (b * q_seq_len + i) * k_seq_len + j;
                            if (std::isfinite((*mask)[mask_idx])) {
                                scores[j] += (*mask)[mask_idx];
                            } else {
                                scores[j] = -std::numeric_limits<float>::infinity();
                            }
                        }
                    }

                    // Softmax
                    float max_score = *std::max_element(scores.begin(), scores.end());
                    float sum_exp = 0.0f;
                    for (size_t j = 0; j < k_seq_len; ++j) {
                        scores[j] = std::exp(scores[j] - max_score);
                        sum_exp += scores[j];
                    }
                    for (size_t j = 0; j < k_seq_len; ++j) {
                        scores[j] /= sum_exp;
                    }

                    // Weighted sum of values
                    for (size_t d = 0; d < head_dim; ++d) {
                        float sum = 0.0f;
                        for (size_t j = 0; j < k_seq_len; ++j) {
                            size_t v_idx = ((b * local_num_heads + h) * k_seq_len + j) * head_dim + d;
                            sum += scores[j] * v_repeated[v_idx];
                        }
                        size_t out_idx = ((b * local_num_heads + h) * q_seq_len + i) * head_dim + d;
                        attn_output[out_idx] = sum;
                    }
                }
            }
        }
    }

    // Gather results from all ranks
    std::vector<float> result;
    if (size > 1) {
        result.resize(num_attention_heads * batch_size * q_seq_len * head_dim);

        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);

        for (int i = 0; i < size; ++i) {
            int s_h = i * heads_per_rank;
            int e_h = std::min(s_h + heads_per_rank, num_attention_heads);
            recvcounts[i] = (e_h - s_h) * batch_size * q_seq_len * head_dim;
            displs[i] = s_h * batch_size * q_seq_len * head_dim;
        }

        MPI_Allgatherv(
            attn_output.data(), attn_output.size(), MPI_FLOAT,
            result.data(), recvcounts.data(), displs.data(), MPI_FLOAT, comm
        );
    } else {
        result = std::move(attn_output);
    }

    Shape result_shape({static_cast<long>(batch_size), static_cast<long>(num_attention_heads),
                       static_cast<long>(q_seq_len), static_cast<long>(head_dim)});
    return Tensor(std::move(result), result_shape);
}

// ============================================================================
// MPI+OpenMP Streaming Attention
// ============================================================================

Tensor self_attention_mpi_streaming_omp(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor* mask,
    float scale,
    int num_attention_heads,
    int num_key_value_heads,
    MPI_Comm comm
) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Distribute attention heads across MPI ranks (same as standard version)
    int heads_per_rank = (num_attention_heads + size - 1) / size;
    int start_head = rank * heads_per_rank;
    int end_head = std::min(start_head + heads_per_rank, num_attention_heads);
    int local_num_heads = end_head - start_head;

    // Tensor shapes
    size_t batch_size = query.shape()[0];
    size_t q_seq_len = query.shape()[2];
    size_t k_seq_len = key.shape()[2];
    size_t head_dim = query.shape()[3];

    // Compute GQA repetition factor
    int n_rep = num_attention_heads / num_key_value_heads;

    // Extract local query heads
    std::vector<float> q_local_data;
    if (local_num_heads > 0) {
        q_local_data.resize(local_num_heads * batch_size * q_seq_len * head_dim);

        for (int h_local = 0; h_local < local_num_heads; ++h_local) {
            int h_global = start_head + h_local;
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t i = 0; i < q_seq_len; ++i) {
                    for (size_t d = 0; d < head_dim; ++d) {
                        size_t src_idx = ((b * num_attention_heads + h_global) * q_seq_len + i) * head_dim + d;
                        size_t dst_idx = ((b * local_num_heads + h_local) * q_seq_len + i) * head_dim + d;
                        q_local_data[dst_idx] = query[src_idx];
                    }
                }
            }
        }
    }

    // Repeat KV heads for GQA (only for local heads)
    Tensor k_repeated, v_repeated;

    if (local_num_heads > 0) {
        // Extract local KV heads
        int kv_start = start_head / n_rep;
        int kv_end = (end_head + n_rep - 1) / n_rep;
        int local_kv_heads = kv_end - kv_start;

        // Extract and repeat KV heads
        std::vector<float> k_local_data(local_kv_heads * batch_size * k_seq_len * head_dim);
        std::vector<float> v_local_data(local_kv_heads * batch_size * k_seq_len * head_dim);

        for (int h = 0; h < local_kv_heads; ++h) {
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t s = 0; s < k_seq_len; ++s) {
                    for (size_t d = 0; d < head_dim; ++d) {
                        size_t src_idx = ((b * num_key_value_heads + (kv_start + h)) * k_seq_len + s) * head_dim + d;
                        size_t dst_idx = ((b * local_kv_heads + h) * k_seq_len + s) * head_dim + d;
                        k_local_data[dst_idx] = key[src_idx];
                        v_local_data[dst_idx] = value[src_idx];
                    }
                }
            }
        }

        // Repeat for local attention heads
        std::vector<float> k_repeated_data(local_num_heads * batch_size * k_seq_len * head_dim);
        std::vector<float> v_repeated_data(local_num_heads * batch_size * k_seq_len * head_dim);

        for (int h_local = 0; h_local < local_num_heads; ++h_local) {
            int h_global = start_head + h_local;
            int h_kv = h_global / n_rep - kv_start;

            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t s = 0; s < k_seq_len; ++s) {
                    for (size_t d = 0; d < head_dim; ++d) {
                        size_t src_idx = ((b * local_kv_heads + h_kv) * k_seq_len + s) * head_dim + d;
                        size_t dst_idx = ((b * local_num_heads + h_local) * k_seq_len + s) * head_dim + d;
                        k_repeated_data[dst_idx] = k_local_data[src_idx];
                        v_repeated_data[dst_idx] = v_local_data[src_idx];
                    }
                }
            }
        }

        Shape local_shape({static_cast<long>(batch_size), static_cast<long>(local_num_heads),
                          static_cast<long>(k_seq_len), static_cast<long>(head_dim)});
        k_repeated = Tensor(std::move(k_repeated_data), local_shape);
        v_repeated = Tensor(std::move(v_repeated_data), local_shape);
    }

    // Compute streaming attention for local heads
    std::vector<float> attn_output;
    if (local_num_heads > 0) {
        // Create local query tensor
        Shape q_local_shape({static_cast<long>(batch_size), static_cast<long>(local_num_heads),
                            static_cast<long>(q_seq_len), static_cast<long>(head_dim)});
        Tensor q_local(std::move(q_local_data), q_local_shape);

        // Call streaming attention for local heads
        Tensor attn_local = ops::self_attention_streaming_blockwise(
            q_local, k_repeated, v_repeated, scale, 32, 64
        );

        // Copy result to output buffer
        attn_output.resize(local_num_heads * batch_size * q_seq_len * head_dim);
        std::memcpy(attn_output.data(), attn_local.data(),
                   attn_output.size() * sizeof(float));
    }

    // Gather results from all ranks (same as standard version)
    std::vector<float> result;
    if (size > 1) {
        result.resize(num_attention_heads * batch_size * q_seq_len * head_dim);

        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);

        for (int i = 0; i < size; ++i) {
            int s_h = i * heads_per_rank;
            int e_h = std::min(s_h + heads_per_rank, num_attention_heads);
            recvcounts[i] = (e_h - s_h) * batch_size * q_seq_len * head_dim;
            displs[i] = s_h * batch_size * q_seq_len * head_dim;
        }

        MPI_Allgatherv(
            attn_output.data(), attn_output.size(), MPI_FLOAT,
            result.data(), recvcounts.data(), displs.data(), MPI_FLOAT, comm
        );
    } else {
        result = std::move(attn_output);
    }

    Shape result_shape({static_cast<long>(batch_size), static_cast<long>(num_attention_heads),
                       static_cast<long>(q_seq_len), static_cast<long>(head_dim)});
    return Tensor(std::move(result), result_shape);
}

// ============================================================================
// MPI+OpenMP Linear Layer
// ============================================================================

Tensor linear_mpi_omp(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    MPI_Comm comm
) {
    // input: [seq_len, in_features]
    // weight: [out_features, in_features]
    // output: [seq_len, out_features]

    size_t seq_len = input.shape()[0];
    size_t in_features = input.shape()[1];
    size_t out_features = weight.shape()[0];

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Distribute output features across ranks
    int out_per_rank = (out_features + size - 1) / size;
    int start_out = rank * out_per_rank;
    int end_out = std::min(start_out + out_per_rank, static_cast<int>(out_features));
    int local_out = end_out - start_out;

    // FIX: Allocate only local result size, not full size!
    std::vector<float> result(seq_len * local_out);

    // Local computation
    #pragma omp parallel for if(seq_len * local_out > 100)
    for (size_t s = 0; s < seq_len; ++s) {
        for (int o = 0; o < local_out; ++o) {
            float sum = 0.0f;
            #pragma omp simd reduction(+:sum)
            for (size_t i = 0; i < in_features; ++i) {
                sum += input[s * in_features + i] * weight[(start_out + o) * in_features + i];
            }
            result[s * local_out + o] = sum + (bias ? (*bias)[start_out + o] : 0.0f);
        }
    }

    // Gather results
    if (size > 1) {
        std::vector<float> full_result(seq_len * out_features);

        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);

        for (int i = 0; i < size; ++i) {
            int s_o = i * out_per_rank;
            int e_o = std::min(s_o + out_per_rank, static_cast<int>(out_features));
            recvcounts[i] = seq_len * (e_o - s_o);
            displs[i] = seq_len * s_o;
        }

        MPI_Allgatherv(
            result.data(), result.size(), MPI_FLOAT,
            full_result.data(), recvcounts.data(), displs.data(), MPI_FLOAT, comm
        );

        result = std::move(full_result);
    }

    Shape result_shape({static_cast<long>(seq_len), static_cast<long>(out_features)});
    return Tensor(std::move(result), result_shape);
}

// ============================================================================
// MPI+OpenMP Embedding Lookup
// ============================================================================

Tensor embedding_mpi_omp(
    const TensorL& indices,
    const Tensor& weight,
    long padding_idx,
    MPI_Comm comm
) {
    // indices: [batch, seq_len]
    // weight: [vocab_size, hidden_size]
    // output: [batch, seq_len, hidden_size]

    size_t batch_size = indices.shape()[0];
    size_t seq_len = indices.shape()[1];
    size_t hidden_size = weight.shape()[1];

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Distribute sequence positions across ranks
    int seq_per_rank = (seq_len + size - 1) / size;
    int start_seq = rank * seq_per_rank;
    int end_seq = std::min(start_seq + seq_per_rank, static_cast<int>(seq_len));
    int local_seq = end_seq - start_seq;

    // FIX: Allocate only local result size, not full size!
    std::vector<float> result(batch_size * local_seq * hidden_size);

    // Local computation
    #pragma omp parallel for if(batch_size * local_seq > 10)
    for (size_t b = 0; b < batch_size; ++b) {
        for (int s = 0; s < local_seq; ++s) {
            long token_idx = indices[b * seq_len + (start_seq + s)];
            if (token_idx == padding_idx) {
                // Use zero embedding for padding
                for (size_t h = 0; h < hidden_size; ++h) {
                    result[(b * local_seq + s) * hidden_size + h] = 0.0f;
                }
            } else {
                for (size_t h = 0; h < hidden_size; ++h) {
                    result[(b * local_seq + s) * hidden_size + h] =
                        weight[token_idx * hidden_size + h];
                }
            }
        }
    }

    // Gather results
    if (size > 1) {
        std::vector<float> full_result(batch_size * seq_len * hidden_size);

        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);

        for (int i = 0; i < size; ++i) {
            int s_s = i * seq_per_rank;
            int e_s = std::min(s_s + seq_per_rank, static_cast<int>(seq_len));
            recvcounts[i] = batch_size * (e_s - s_s) * hidden_size;
            displs[i] = batch_size * s_s * hidden_size;
        }

        MPI_Allgatherv(
            result.data(), result.size(), MPI_FLOAT,
            full_result.data(), recvcounts.data(), displs.data(), MPI_FLOAT, comm
        );

        result = std::move(full_result);
    }

    Shape result_shape({static_cast<long>(batch_size), static_cast<long>(seq_len), static_cast<long>(hidden_size)});
    return Tensor(std::move(result), result_shape);
}

// ============================================================================
// Head-wise Parallelism (Renamed wrappers for backward compatibility)
// ============================================================================

Tensor attention_headwise_standard(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor* mask,
    float scale,
    int num_attention_heads,
    int num_key_value_heads,
    MPI_Comm comm
) {
    // Directly call the existing implementation
    return self_attention_mpi_omp(query, key, value, mask, scale,
                                  num_attention_heads, num_key_value_heads, comm);
}

Tensor attention_headwise_online_softmax(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor* mask,
    float scale,
    int num_attention_heads,
    int num_key_value_heads,
    MPI_Comm comm
) {
    // Directly call the existing implementation
    return self_attention_mpi_streaming_omp(query, key, value, mask, scale,
                                            num_attention_heads, num_key_value_heads, comm);
}

// ============================================================================
// Sequence Parallelism (New implementation)
// ============================================================================

Tensor attention_sequence_online_softmax(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor* mask,
    float scale,
    int num_attention_heads,
    int num_key_value_heads,
    size_t global_seq_len,
    MPI_Comm comm
) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Tensor shapes
    size_t batch_size = query.shape()[0];
    size_t local_q_seq_len = query.shape()[2];
    size_t local_kv_seq_len = key.shape()[2];
    size_t head_dim = query.shape()[3];

    // Compute GQA repetition factor
    int n_rep = num_attention_heads / num_key_value_heads;

    // Step 1: 本地 streaming attention 计算
    // 对本rank负责的K/V block，计算所有query positions的attention
    // 使用 online softmax 维护局部统计量

    // Initialize local statistics for each query position and each head
    // local_max[b][h][i] = maximum score for query i in head h
    // local_exp_sum[b][h][i] = sum of exp(score - max) for query i in head h
    // local_output[b][h][i][d] = accumulated weighted value for query i in head h, dimension d

    size_t num_stats = batch_size * num_attention_heads * local_q_seq_len;

    std::vector<float> local_max(num_stats, -std::numeric_limits<float>::infinity());
    std::vector<float> local_exp_sum(num_stats, 0.0f);

    // Initialize local weighted value accumulation
    std::vector<float> local_weighted_value(batch_size * num_attention_heads * local_q_seq_len * head_dim, 0.0f);

    // Process attention in blocks for memory efficiency
    const size_t KV_BLOCK_SIZE = 64;

    #pragma omp parallel for if(batch_size * num_attention_heads * local_q_seq_len > 100)
    for (size_t b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_attention_heads; ++h) {
            // Get corresponding KV head
            int h_kv = h / n_rep;

            for (size_t q_idx = 0; q_idx < local_q_seq_len; ++q_idx) {
                // Global query position (for causal mask)
                // Query positions are [rank * local_q_seq_len, (rank+1) * local_q_seq_len)
                // But we need to map this to global position
                size_t global_q_pos = rank * local_q_seq_len + q_idx;

                // Extract query vector for this position
                std::vector<float> q_vec(head_dim);
                for (size_t d = 0; d < head_dim; ++d) {
                    size_t idx = ((b * num_attention_heads + h) * local_q_seq_len + q_idx) * head_dim + d;
                    q_vec[d] = query[idx] * scale;
                }

                // Process KV blocks
                for (size_t kv_block_start = 0; kv_block_start < local_kv_seq_len; kv_block_start += KV_BLOCK_SIZE) {
                    size_t kv_block_end = std::min(kv_block_start + KV_BLOCK_SIZE, local_kv_seq_len);

                    // Causal constraint: query can only attend to positions <= global_q_pos
                    // Map local KV positions to global positions
                    // Rank 0 has KV positions [0, local_kv_seq_len)
                    // Rank 1 has KV positions [local_kv_seq_len, 2*local_kv_seq_len)
                    // etc.

                    // For current block, check if any positions are <= global_q_pos
                    bool block_has_valid_positions = false;
                    for (size_t kv_idx = kv_block_start; kv_idx < kv_block_end; ++kv_idx) {
                        size_t global_kv_pos = rank * local_kv_seq_len + kv_idx;
                        if (global_kv_pos <= global_q_pos) {
                            block_has_valid_positions = true;
                            break;
                        }
                    }

                    if (!block_has_valid_positions) {
                        continue; // Skip this block entirely
                    }

                    // Compute attention scores for this query against KV block
                    std::vector<float> block_scores(kv_block_end - kv_block_start);
                    std::vector<float> block_weighted_v((kv_block_end - kv_block_start) * head_dim);

                    for (size_t kv_idx = kv_block_start; kv_idx < kv_block_end; ++kv_idx) {
                        size_t global_kv_pos = rank * local_kv_seq_len + kv_idx;

                        // Causal mask: query can only attend to positions <= global_q_pos
                        if (global_kv_pos > global_q_pos) {
                            continue;
                        }

                        // Compute dot product Q @ K^T
                        float score = 0.0f;
                        #pragma omp simd reduction(+:score)
                        for (size_t d = 0; d < head_dim; ++d) {
                            size_t k_idx = ((b * num_key_value_heads + h_kv) * local_kv_seq_len + kv_idx) * head_dim + d;
                            score += q_vec[d] * key[k_idx];
                        }

                        size_t local_kv_idx = kv_idx - kv_block_start;
                        block_scores[local_kv_idx] = score;

                        // Compute weighted value: score * V
                        #pragma omp simd
                        for (size_t d = 0; d < head_dim; ++d) {
                            size_t v_idx = ((b * num_key_value_heads + h_kv) * local_kv_seq_len + kv_idx) * head_dim + d;
                            block_weighted_v[local_kv_idx * head_dim + d] = score * value[v_idx];
                        }
                    }

                    // Update online softmax statistics
                    size_t stat_idx = (b * num_attention_heads + h) * local_q_seq_len + q_idx;

                    // Find max in this block
                    float block_max = *std::max_element(block_scores.begin(), block_scores.end());

                    // Update global max
                    float old_max = local_max[stat_idx];
                    local_max[stat_idx] = std::max(local_max[stat_idx], block_max);

                    // Compute exp sum for this block
                    float block_exp_sum = 0.0f;
                    std::vector<float> block_exp_scores(kv_block_end - kv_block_start);

                    for (size_t kv_idx = kv_block_start; kv_idx < kv_block_end; ++kv_idx) {
                        size_t global_kv_pos = rank * local_kv_seq_len + kv_idx;
                        if (global_kv_pos > global_q_pos) {
                            continue;
                        }

                        size_t local_kv_idx = kv_idx - kv_block_start;
                        float exp_score = std::exp(block_scores[local_kv_idx] - local_max[stat_idx]);
                        block_exp_scores[local_kv_idx] = exp_score;
                        block_exp_sum += exp_score;
                    }

                    // Update global exp sum
                    float scale_factor = std::exp(old_max - local_max[stat_idx]);
                    local_exp_sum[stat_idx] = local_exp_sum[stat_idx] * scale_factor + block_exp_sum;

                    // Update weighted value
                    // The formula: O_new = O_old * scale_factor + sum(exp(score) * V)
                    size_t output_idx = ((b * num_attention_heads + h) * local_q_seq_len + q_idx) * head_dim;

                    #pragma omp simd
                    for (size_t d = 0; d < head_dim; ++d) {
                        local_weighted_value[output_idx + d] *= scale_factor;

                        // Accumulate weighted values from this block
                        for (size_t kv_idx = kv_block_start; kv_idx < kv_block_end; ++kv_idx) {
                            size_t global_kv_pos = rank * local_kv_seq_len + kv_idx;
                            if (global_kv_pos > global_q_pos) {
                                continue;
                            }

                            size_t local_kv_idx = kv_idx - kv_block_start;
                            local_weighted_value[output_idx + d] +=
                                block_exp_scores[local_kv_idx] * block_weighted_v[local_kv_idx * head_dim + d];
                        }
                    }
                }
            }
        }
    }

    // Step 2: 跨rank归约 (MPI_Allreduce)
    // 归约 global_max 和 global_exp_sum

    std::vector<float> global_max = local_max;
    std::vector<float> global_exp_sum = local_exp_sum;

    if (size > 1) {
        MPI_Allreduce(MPI_IN_PLACE, global_max.data(), num_stats, MPI_FLOAT, MPI_MAX, comm);
        MPI_Allreduce(MPI_IN_PLACE, global_exp_sum.data(), num_stats, MPI_FLOAT, MPI_SUM, comm);
    }

    // Step 3: 本地归一化修正
    // 使用 global_max 和 global_exp_sum 修正本地输出
    // O_final = local_weighted_value / global_exp_sum
    // 需要考虑 global_max 的差异

    std::vector<float> result_data(batch_size * num_attention_heads * local_q_seq_len * head_dim);

    #pragma omp parallel for if(batch_size * num_attention_heads * local_q_seq_len * head_dim > 1000)
    for (size_t idx = 0; idx < batch_size * num_attention_heads * local_q_seq_len * head_dim; ++idx) {
        size_t b = idx / (num_attention_heads * local_q_seq_len * head_dim);
        size_t remainder = idx % (num_attention_heads * local_q_seq_len * head_dim);
        size_t h = remainder / (local_q_seq_len * head_dim);
        remainder = remainder % (local_q_seq_len * head_dim);
        size_t q_idx = remainder / head_dim;

        size_t stat_idx = (b * num_attention_heads + h) * local_q_seq_len + q_idx;

        // Renormalize: account for difference between local_max and global_max
        float max_diff = local_max[stat_idx] - global_max[stat_idx];
        float correction = std::exp(max_diff);

        // Final normalization
        result_data[idx] = local_weighted_value[idx] * correction / global_exp_sum[stat_idx];
    }

    Shape result_shape({static_cast<long>(batch_size), static_cast<long>(num_attention_heads),
                       static_cast<long>(local_q_seq_len), static_cast<long>(head_dim)});
    return Tensor(std::move(result_data), result_shape);
}

#endif // MPI_VERSION

} // namespace mpi
} // namespace ops
} // namespace tensor_cpp
