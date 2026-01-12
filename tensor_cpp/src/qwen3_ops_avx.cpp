/**
 * @file qwen3_ops_avx_v2.cpp
 * @brief AVX2-optimized Qwen3 operators with pre-extracted QKV projections
 */

#include "tensor_cpp/qwen3_ops_avx.h"
#include "tensor_cpp/ops.h"
#include "tensor_cpp/attention_avx.h"
#include "tensor_cpp/qwen3_ops.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>

using namespace tensor_cpp::ops;
using namespace tensor_cpp::ops::avx2;

namespace tensor_cpp {
namespace qwen3 {
namespace avx2 {

// ============================================================================
// Qwen3 Attention with AVX2 (Optimized with pre-extracted QKV)
// ============================================================================

Tensor qwen3_attention_avx(
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
    const Tensor& sin
) {
    // hidden_states: [batch, seq_len, hidden_size]
    const Shape& hidden_shape = hidden_states.shape();
    size_t batch = hidden_shape[0];
    size_t seq_len = hidden_shape[1];
    size_t hidden_size = hidden_shape[2];

    // Compute Q, K, V projections using AVX2 linear
    Tensor hidden_reshaped = hidden_states.view({static_cast<long>(batch * seq_len), static_cast<long>(hidden_size)});

    Tensor q_proj_out = linear_avx2(hidden_reshaped, q_proj, nullptr);
    Tensor k_proj_out = linear_avx2(hidden_reshaped, k_proj, nullptr);
    Tensor v_proj_out = linear_avx2(hidden_reshaped, v_proj, nullptr);

    // Reshape to [batch, num_heads, seq_len, head_dim]
    size_t q_total_heads = num_attention_heads;
    size_t kv_total_heads = num_key_value_heads;

    Tensor q_reshaped = q_proj_out.view({batch, seq_len, q_total_heads, head_dim});
    Tensor k_reshaped = k_proj_out.view({batch, seq_len, kv_total_heads, head_dim});
    Tensor v_reshaped = v_proj_out.view({batch, seq_len, kv_total_heads, head_dim});

    // Transpose to [batch, num_heads, seq_len, head_dim]
    Tensor q = q_reshaped.transpose(1, 2);
    Tensor k = k_reshaped.transpose(1, 2);
    Tensor v = v_reshaped.transpose(1, 2);

    // ========== QKNorm: Apply RMS normalization to Q and K per-head ==========
    const float* q_norm_weight_data = q_norm_weight.data();
    const float* k_norm_weight_data = k_norm_weight.data();
    const float* q_data = q.data();
    const float* k_data = k.data();

    size_t q_total_elements = batch * q_total_heads * seq_len * head_dim;
    std::vector<float> q_normed_data(q_total_elements);

    #pragma omp parallel for if(q_total_elements > 1000)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < q_total_heads; ++h) {
            for (size_t s = 0; s < seq_len; ++s) {
                size_t base_idx = ((b * q_total_heads + h) * seq_len + s) * head_dim;

                // Compute RMS: sqrt(mean(x^2))
                float sum_sq = 0.0f;
                for (size_t d = 0; d < head_dim; ++d) {
                    float val = q_data[base_idx + d];
                    sum_sq += val * val;
                }
                float rms = std::sqrt(sum_sq / static_cast<float>(head_dim)) + 1e-6f;

                // Normalize: x / RMS * weight
                for (size_t d = 0; d < head_dim; ++d) {
                    q_normed_data[base_idx + d] = (q_data[base_idx + d] / rms) * q_norm_weight_data[d];
                }
            }
        }
    }
    Tensor q_normed(std::move(q_normed_data), q.shape());

    size_t kv_total_elements = batch * kv_total_heads * seq_len * head_dim;
    std::vector<float> k_normed_data(kv_total_elements);

    #pragma omp parallel for if(kv_total_elements > 1000)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < kv_total_heads; ++h) {
            for (size_t s = 0; s < seq_len; ++s) {
                size_t base_idx = ((b * kv_total_heads + h) * seq_len + s) * head_dim;

                float sum_sq = 0.0f;
                for (size_t d = 0; d < head_dim; ++d) {
                    float val = k_data[base_idx + d];
                    sum_sq += val * val;
                }
                float rms = std::sqrt(sum_sq / static_cast<float>(head_dim)) + 1e-6f;

                for (size_t d = 0; d < head_dim; ++d) {
                    k_normed_data[base_idx + d] = (k_data[base_idx + d] / rms) * k_norm_weight_data[d];
                }
            }
        }
    }
    Tensor k_normed(std::move(k_normed_data), k.shape());

    // Apply RoPE
    auto [q_rope, k_rope] = apply_rotary_pos_emb(q_normed, k_normed, cos, sin);

    // Repeat KV for GQA
    int n_rep = static_cast<int>(num_attention_heads / num_key_value_heads);
    Tensor k_repeated = repeat_kv(k_rope, n_rep);
    Tensor v_repeated = repeat_kv(v, n_rep);

    // Compute attention scores with AVX2
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    Tensor causal_mask = create_causal_mask(seq_len);
    Tensor mask = causal_mask.view({1, 1, seq_len, seq_len});

    Tensor attn_output = self_attention_avx2(q_rope, k_repeated, v_repeated, &mask, scale);

    // Transpose back: [batch, seq_len, num_heads, head_dim]
    Tensor attn_output_t = attn_output.transpose(1, 2);

    // Reshape to [batch, seq_len, hidden_size]
    Tensor attn_output_reshaped = attn_output_t.contiguous().view({batch, seq_len, q_total_heads * head_dim});

    // Apply output projection with AVX2
    Tensor output = linear_avx2(attn_output_reshaped, o_proj, nullptr);

    return output;
}

// ============================================================================
// Qwen3 MLP with AVX2 (Reused from avx2 namespace)
// ============================================================================

namespace {
// Internal MLP implementation (same as avx2::qwen3_mlp_avx)
Tensor qwen3_mlp_avx_internal(
    const Tensor& hidden_states,
    const Tensor& gate_proj,
    const Tensor& up_proj,
    const Tensor& down_proj
) {
    const Shape& hidden_shape = hidden_states.shape();
    size_t batch = hidden_shape[0];
    size_t seq_len = hidden_shape[1];
    size_t hidden_size = hidden_shape[2];
    size_t intermediate_size = gate_proj.shape()[0];

    const float* hidden_data = hidden_states.data();
    const float* gate_proj_data = gate_proj.data();
    const float* up_proj_data = up_proj.data();

    // Gate projection with AVX2
    size_t gate_data_size = batch * seq_len * intermediate_size;
    std::vector<float> gate_data(gate_data_size);

    #pragma omp parallel for if(batch * seq_len * intermediate_size > 1000)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            size_t row_offset = (b * seq_len + s) * intermediate_size;
            size_t input_offset = (b * seq_len + s) * hidden_size;

            for (size_t i = 0; i < intermediate_size; ++i) {
                float sum = 0.0f;
                size_t weight_offset = i * hidden_size;
                size_t j = 0;

                __m256 sum_vec = _mm256_setzero_ps();
                for (; j + 8 <= hidden_size; j += 8) {
                    __m256 hidden_vec = _mm256_loadu_ps(&hidden_data[input_offset + j]);
                    __m256 weight_vec = _mm256_loadu_ps(&gate_proj_data[weight_offset + j]);
                    sum_vec = _mm256_fmadd_ps(hidden_vec, weight_vec, sum_vec);
                }

                // Improved horizontal sum
                __m128 hi_quad = _mm256_extractf128_ps(sum_vec, 1);
                __m128 lo_quad = _mm256_castps256_ps128(sum_vec);
                __m128 sum_quad = _mm_add_ps(lo_quad, hi_quad);
                __m128 lo_dual = sum_quad;
                __m128 hi_dual = _mm_movehl_ps(sum_quad, sum_quad);
                __m128 sum_dual = _mm_add_ps(lo_dual, hi_dual);
                __m128 lo = sum_dual;
                __m128 hi = _mm_shuffle_ps(sum_dual, sum_dual, 1);
                __m128 sum_128 = _mm_add_ss(lo, hi);
                sum = _mm_cvtss_f32(sum_128);

                for (; j < hidden_size; ++j) {
                    sum += hidden_data[input_offset + j] * gate_proj_data[weight_offset + j];
                }

                gate_data[row_offset + i] = sum;
            }
        }
    }

    Shape gate_shape({static_cast<long>(batch), static_cast<long>(seq_len), static_cast<long>(intermediate_size)});
    Tensor gate(std::move(gate_data), gate_shape);

    // Up projection with AVX2
    std::vector<float> up_data(batch * seq_len * intermediate_size);

    #pragma omp parallel for if(batch * seq_len * intermediate_size > 1000)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            size_t row_offset = (b * seq_len + s) * intermediate_size;
            size_t input_offset = (b * seq_len + s) * hidden_size;

            for (size_t i = 0; i < intermediate_size; ++i) {
                float sum = 0.0f;
                size_t weight_offset = i * hidden_size;
                size_t j = 0;

                __m256 sum_vec = _mm256_setzero_ps();
                for (; j + 8 <= hidden_size; j += 8) {
                    __m256 hidden_vec = _mm256_loadu_ps(&hidden_data[input_offset + j]);
                    __m256 weight_vec = _mm256_loadu_ps(&up_proj_data[weight_offset + j]);
                    sum_vec = _mm256_fmadd_ps(hidden_vec, weight_vec, sum_vec);
                }

                __m128 hi_quad = _mm256_extractf128_ps(sum_vec, 1);
                __m128 lo_quad = _mm256_castps256_ps128(sum_vec);
                __m128 sum_quad = _mm_add_ps(lo_quad, hi_quad);
                __m128 lo_dual = sum_quad;
                __m128 hi_dual = _mm_movehl_ps(sum_quad, sum_quad);
                __m128 sum_dual = _mm_add_ps(lo_dual, hi_dual);
                __m128 lo = sum_dual;
                __m128 hi = _mm_shuffle_ps(sum_dual, sum_dual, 1);
                __m128 sum_128 = _mm_add_ss(lo, hi);
                sum = _mm_cvtss_f32(sum_128);

                for (; j < hidden_size; ++j) {
                    sum += hidden_data[input_offset + j] * up_proj_data[weight_offset + j];
                }

                up_data[row_offset + i] = sum;
            }
        }
    }

    Shape up_shape({static_cast<long>(batch), static_cast<long>(seq_len), static_cast<long>(intermediate_size)});
    Tensor up(std::move(up_data), up_shape);

    // SwiGLU activation with AVX2
    // Note: Not using OpenMP here as it's already vectorized with AVX2
    // and the loop structure is not thread-safe
    std::vector<float> swiglu_data(batch * seq_len * intermediate_size);

    const float* gate_tensor_data = gate.data();
    const float* up_tensor_data = up.data();

    for (size_t i = 0; i < batch * seq_len * intermediate_size; ++i) {
        float gate_val = gate_tensor_data[i];
        // SiLU(gate) = gate * sigmoid(gate) = gate / (1 + exp(-gate))
        float silu_val = gate_val * (1.0f / (1.0f + std::exp(-gate_val)));
        swiglu_data[i] = silu_val * up_tensor_data[i];
    }

    Tensor swiglu(std::move(swiglu_data), gate.shape());

    // Down projection with AVX2
    std::vector<float> output_data(batch * seq_len * hidden_size);
    const float* swiglu_data_ptr = swiglu.data();
    const float* down_proj_data = down_proj.data();

    #pragma omp parallel for if(batch * seq_len * hidden_size > 1000)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            size_t row_offset = (b * seq_len + s) * hidden_size;
            size_t input_offset = (b * seq_len + s) * intermediate_size;

            for (size_t i = 0; i < hidden_size; ++i) {
                float sum = 0.0f;
                size_t weight_offset = i * intermediate_size;
                size_t j = 0;

                __m256 sum_vec = _mm256_setzero_ps();
                for (; j + 8 <= intermediate_size; j += 8) {
                    __m256 swiglu_vec = _mm256_loadu_ps(&swiglu_data_ptr[input_offset + j]);
                    __m256 weight_vec = _mm256_loadu_ps(&down_proj_data[weight_offset + j]);
                    sum_vec = _mm256_fmadd_ps(swiglu_vec, weight_vec, sum_vec);
                }

                __m128 hi_quad = _mm256_extractf128_ps(sum_vec, 1);
                __m128 lo_quad = _mm256_castps256_ps128(sum_vec);
                __m128 sum_quad = _mm_add_ps(lo_quad, hi_quad);
                __m128 lo_dual = sum_quad;
                __m128 hi_dual = _mm_movehl_ps(sum_quad, sum_quad);
                __m128 sum_dual = _mm_add_ps(lo_dual, hi_dual);
                __m128 lo = sum_dual;
                __m128 hi = _mm_shuffle_ps(sum_dual, sum_dual, 1);
                __m128 sum_128 = _mm_add_ss(lo, hi);
                sum = _mm_cvtss_f32(sum_128);

                for (; j < intermediate_size; ++j) {
                    sum += swiglu_data_ptr[input_offset + j] * down_proj_data[weight_offset + j];
                }

                output_data[row_offset + i] = sum;
            }
        }
    }

    Shape output_shape(hidden_shape);
    return Tensor(std::move(output_data), output_shape);
}
} // anonymous namespace

// ============================================================================
// Qwen3 Decoder Layer with AVX2 (Optimized with pre-extracted QKV)
// ============================================================================

Tensor qwen3_decoder_layer_avx(
    const Tensor& hidden_states,
    size_t num_attention_heads,
    size_t num_key_value_heads,
    size_t head_dim,
    float rms_norm_eps,
    const Tensor& input_layernorm_weight,
    const Tensor& q_proj,
    const Tensor& k_proj,
    const Tensor& v_proj,
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
    static int layer_count = 0;
    if (layer_count == 0) {
    }

    // Input layernorm
    Tensor residual = hidden_states;
    Tensor hidden = rms_norm(hidden_states, &input_layernorm_weight, rms_norm_eps);

    // Self-attention with AVX2 (uses pre-extracted QKV)
    Tensor attn_output = qwen3_attention_avx(
        hidden, num_attention_heads, num_key_value_heads, head_dim,
        q_proj, k_proj, v_proj, o_proj, q_norm_weight, k_norm_weight, cos, sin
    );

    // Residual
    hidden = residual + attn_output;

    // Post-attention layernorm
    residual = hidden;
    hidden = rms_norm(hidden, &post_attention_layernorm_weight, rms_norm_eps);

    // MLP with AVX2
    Tensor mlp_output = qwen3_mlp_avx_internal(hidden, gate_mlp, up_mlp, down_mlp);

    // Residual
    hidden = residual + mlp_output;

    return hidden;
}

// ============================================================================
// Complete Qwen3 Forward Pass with AVX2 (Optimized with pre-extracted QKV)
// ============================================================================

Tensor qwen3_forward_avx(
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
    // Embed tokens
    Tensor hidden_states = embedding(input_ids, token_embedding);

    // Compute RoPE frequencies
    size_t seq_len = input_ids.shape()[1];
    auto rope_freqs = compute_rope_freqs(seq_len, head_dim, 1000000.0f);
    Tensor cos = rope_freqs.first;
    Tensor sin = rope_freqs.second;

    // Process through layers (using pre-extracted QKV projections)
    for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        const auto& layer = layers[layer_idx];

        hidden_states = qwen3_decoder_layer_avx(
            hidden_states,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            layer.input_layernorm_weight,
            layer.q_proj,  // Pre-extracted
            layer.k_proj,  // Pre-extracted
            layer.v_proj,  // Pre-extracted
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

    // Final layernorm
    hidden_states = rms_norm(hidden_states, &norm_weight, rms_norm_eps);

    return hidden_states;
}

// Export MLP function - wrapper for internal implementation
Tensor qwen3_mlp_avx(
    const Tensor& hidden_states,
    const Tensor& gate_proj,
    const Tensor& up_proj,
    const Tensor& down_proj
) {
    return qwen3_mlp_avx_internal(hidden_states, gate_proj, up_proj, down_proj);
}

// ============================================================================
// Qwen3 Decoder Layer with AVX2 and KV Cache Support
// ============================================================================

Tensor qwen3_decoder_layer_avx_with_cache(
    const Tensor& hidden_states,
    KVCache* kv_cache,
    size_t layer_idx,
    size_t num_attention_heads,
    size_t num_key_value_heads,
    size_t head_dim,
    float rms_norm_eps,
    const Tensor& input_layernorm_weight,
    const Tensor& q_proj,
    const Tensor& k_proj,
    const Tensor& v_proj,
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
    Tensor normed = rms_norm(hidden_states, &input_layernorm_weight, rms_norm_eps);

    const Shape& hidden_shape = hidden_states.shape();
    size_t batch = hidden_shape[0];
    size_t seq_len = hidden_shape[1];
    size_t hidden_size = hidden_shape[2];

    // Reshape for linear projection
    Tensor normed_reshaped = normed.view({static_cast<long>(batch * seq_len), static_cast<long>(hidden_size)});

    // Compute Q, K, V projections using AVX2 linear
    Tensor q_proj_out = linear_avx2(normed_reshaped, q_proj, nullptr);
    Tensor k_proj_out = linear_avx2(normed_reshaped, k_proj, nullptr);
    Tensor v_proj_out = linear_avx2(normed_reshaped, v_proj, nullptr);

    // Reshape to [batch, num_heads, seq_len, head_dim]
    Tensor q_reshaped = q_proj_out.view({static_cast<long>(batch), static_cast<long>(seq_len), static_cast<long>(num_attention_heads), static_cast<long>(head_dim)});
    Tensor k_reshaped = k_proj_out.view({static_cast<long>(batch), static_cast<long>(seq_len), static_cast<long>(num_key_value_heads), static_cast<long>(head_dim)});
    Tensor v_reshaped = v_proj_out.view({static_cast<long>(batch), static_cast<long>(seq_len), static_cast<long>(num_key_value_heads), static_cast<long>(head_dim)});

    // Transpose to [batch, num_heads, seq_len, head_dim]
    Tensor q = q_reshaped.transpose(1, 2);
    Tensor k = k_reshaped.transpose(1, 2);
    Tensor v = v_reshaped.transpose(1, 2);

    // Apply QKNorm to Q
    const float* q_norm_data = q_norm_weight.data();
    size_t q_total_elements = batch * num_attention_heads * seq_len * head_dim;
    std::vector<float> q_normed_data(q_total_elements);

    const float* q_data = q.data();
    #pragma omp parallel for if(batch * num_attention_heads * seq_len * head_dim > 1000)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < num_attention_heads; ++h) {
            for (size_t s = 0; s < seq_len; ++s) {
                size_t base_idx = ((b * num_attention_heads + h) * seq_len + s) * head_dim;

                // Compute RMS
                float sum_sq = 0.0f;
                for (size_t i = 0; i < head_dim; ++i) {
                    float val = q_data[base_idx + i];
                    sum_sq += val * val;
                }
                float rms = std::sqrt(sum_sq / static_cast<float>(head_dim)) + 1e-6f;

                // Normalize and scale
                for (size_t i = 0; i < head_dim; ++i) {
                    q_normed_data[base_idx + i] = (q_data[base_idx + i] / rms) * q_norm_data[i];
                }
            }
        }
    }
    Tensor q_normed(std::move(q_normed_data), q.shape());

    // Apply QKNorm to K
    const float* k_norm_data = k_norm_weight.data();
    size_t k_total_elements = batch * num_key_value_heads * seq_len * head_dim;
    std::vector<float> k_normed_data(k_total_elements);

    const float* k_data = k.data();
    #pragma omp parallel for if(batch * num_key_value_heads * seq_len * head_dim > 1000)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < num_key_value_heads; ++h) {
            for (size_t s = 0; s < seq_len; ++s) {
                size_t base_idx = ((b * num_key_value_heads + h) * seq_len + s) * head_dim;

                // Compute RMS
                float sum_sq = 0.0f;
                for (size_t i = 0; i < head_dim; ++i) {
                    float val = k_data[base_idx + i];
                    sum_sq += val * val;
                }
                float rms = std::sqrt(sum_sq / static_cast<float>(head_dim)) + 1e-6f;

                // Normalize and scale
                for (size_t i = 0; i < head_dim; ++i) {
                    k_normed_data[base_idx + i] = (k_data[base_idx + i] / rms) * k_norm_data[i];
                }
            }
        }
    }
    Tensor k_normed(std::move(k_normed_data), k.shape());

    // Apply RoPE with correct position offset
    Tensor cos_for_rope, sin_for_rope;
    size_t rope_offset = 0;

    if (kv_cache != nullptr && kv_cache->current_seq_len > 0) {
        // Decode phase: use cos/sin starting from current cached position
        rope_offset = kv_cache->current_seq_len;
        size_t half_dim = head_dim / 2;

        // Slice cos/sin to get only the rows we need [seq_len, half_dim]
        std::vector<float> cos_slice_data(seq_len * half_dim);
        std::vector<float> sin_slice_data(seq_len * half_dim);

        for (size_t s = 0; s < seq_len; ++s) {
            size_t src_pos = rope_offset + s;
            for (size_t i = 0; i < half_dim; ++i) {
                cos_slice_data[s * half_dim + i] = cos[src_pos * half_dim + i];
                sin_slice_data[s * half_dim + i] = sin[src_pos * half_dim + i];
            }
        }

        cos_for_rope = Tensor(std::move(cos_slice_data),
                             Shape({static_cast<long>(seq_len), static_cast<long>(half_dim)}));
        sin_for_rope = Tensor(std::move(sin_slice_data),
                             Shape({static_cast<long>(seq_len), static_cast<long>(half_dim)}));
    } else {
        // Prefill phase: use cos/sin starting from position 0
        cos_for_rope = cos;
        sin_for_rope = sin;
    }

    auto [q_rope, k_rope] = apply_rotary_pos_emb(q_normed, k_normed, cos_for_rope, sin_for_rope);

    // Handle KV cache
    Tensor k_final, v_final;

    if (kv_cache != nullptr && kv_cache->current_seq_len > 0) {
        // Has cache: concatenate cached K/V with new K/V
        size_t cached_seq_len = kv_cache->current_seq_len;

        // Get cached keys and values for this layer
        Tensor k_cached = kv_cache->get_cached_keys(layer_idx, cached_seq_len);
        Tensor v_cached = kv_cache->get_cached_values(layer_idx, cached_seq_len);

        // Reshape to [batch, num_kv_heads, cached_seq_len, head_dim]
        k_cached = k_cached.view({static_cast<long>(batch), static_cast<long>(num_key_value_heads), static_cast<long>(cached_seq_len), static_cast<long>(head_dim)});
        v_cached = v_cached.view({static_cast<long>(batch), static_cast<long>(num_key_value_heads), static_cast<long>(cached_seq_len), static_cast<long>(head_dim)});

        // Concatenate cached and new along seq_len dimension
        size_t total_seq_len = cached_seq_len + seq_len;
        std::vector<float> k_final_data(batch * num_key_value_heads * total_seq_len * head_dim);
        std::vector<float> v_final_data(batch * num_key_value_heads * total_seq_len * head_dim);

        const float* k_cached_data = k_cached.data();
        const float* v_cached_data = v_cached.data();
        const float* k_rope_data = k_rope.data();
        const float* v_data = v.data();

        // Copy cached part
        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < num_key_value_heads; ++h) {
                for (size_t s = 0; s < cached_seq_len; ++s) {
                    for (size_t d = 0; d < head_dim; ++d) {
                        size_t cached_idx = ((b * num_key_value_heads + h) * cached_seq_len + s) * head_dim + d;
                        size_t final_idx = ((b * num_key_value_heads + h) * total_seq_len + s) * head_dim + d;
                        k_final_data[final_idx] = k_cached_data[cached_idx];
                        v_final_data[final_idx] = v_cached_data[cached_idx];
                    }
                }

                // Copy new part
                for (size_t s_new = 0; s_new < seq_len; ++s_new) {
                    for (size_t d = 0; d < head_dim; ++d) {
                        size_t new_idx = ((b * num_key_value_heads + h) * seq_len + s_new) * head_dim + d;
                        size_t final_idx = ((b * num_key_value_heads + h) * total_seq_len + cached_seq_len + s_new) * head_dim + d;
                        k_final_data[final_idx] = k_rope_data[new_idx];
                        v_final_data[final_idx] = v_data[new_idx];
                    }
                }
            }
        }

        k_final = Tensor(std::move(k_final_data),
                         Shape({static_cast<long>(batch),
                               static_cast<long>(num_key_value_heads),
                               static_cast<long>(total_seq_len),
                               static_cast<long>(head_dim)}));

        v_final = Tensor(std::move(v_final_data),
                         Shape({static_cast<long>(batch),
                               static_cast<long>(num_key_value_heads),
                               static_cast<long>(total_seq_len),
                               static_cast<long>(head_dim)}));

        // Update cache with new K and V (after RoPE for K)
        kv_cache->update(layer_idx, k_rope, v);

    } else {
        // No cache: use new K/V directly
        k_final = k_rope;
        v_final = v;

        // Initialize cache if provided
        if (kv_cache != nullptr) {
            kv_cache->update(layer_idx, k_rope, v);
        }
    }

    // Repeat KV for GQA
    int n_rep = static_cast<int>(num_attention_heads / num_key_value_heads);
    Tensor k_repeated = repeat_kv(k_final, n_rep);
    Tensor v_repeated = repeat_kv(v_final, n_rep);

    // Compute attention with AVX2
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Create causal mask
    size_t total_seq_len = k_final.shape()[2];
    size_t q_seq_len = q_rope.shape()[2];

    Tensor mask;
    if (q_seq_len == 1 && total_seq_len > 1) {
        // Decode phase: single query token at the end
        std::vector<float> mask_data(total_seq_len);
        for (size_t j = 0; j < total_seq_len; ++j) {
            mask_data[j] = (j <= total_seq_len - 1) ? 0.0f : -std::numeric_limits<float>::infinity();
        }
        mask = Tensor(std::move(mask_data), Shape({1, static_cast<long>(total_seq_len)}));
        mask = mask.view({1, 1, 1, static_cast<long>(total_seq_len)});
    } else {
        // Prefill phase: standard causal mask
        Tensor causal_mask = create_causal_mask(total_seq_len);
        mask = causal_mask.view({1, 1, static_cast<long>(total_seq_len), static_cast<long>(total_seq_len)});
    }

    // Use AVX2-optimized attention
    Tensor attn_output = self_attention_avx2(q_rope, k_repeated, v_repeated, &mask, scale);

    // Transpose back: [batch, seq_len, num_heads, head_dim]
    Tensor attn_output_t = attn_output.transpose(1, 2);

    // Reshape to [batch, seq_len, hidden_size]
    Tensor attn_output_reshaped = attn_output_t.contiguous().view({static_cast<long>(batch), static_cast<long>(seq_len), static_cast<long>(num_attention_heads * head_dim)});

    // Apply output projection with AVX2
    Tensor attn_output_2d = attn_output_reshaped.view({static_cast<long>(batch * seq_len), static_cast<long>(num_attention_heads * head_dim)});
    Tensor output = linear_avx2(attn_output_2d, o_proj, nullptr);
    output = output.view({static_cast<long>(batch), static_cast<long>(seq_len), static_cast<long>(hidden_size)});

    // Residual 1
    Tensor hidden = residual + output;

    // Post attention layernorm
    residual = hidden;
    normed = rms_norm(hidden, &post_attention_layernorm_weight, rms_norm_eps);

    // MLP with AVX2
    Tensor mlp_output = qwen3_mlp_avx_internal(normed, gate_mlp, up_mlp, down_mlp);

    // Residual 2
    hidden = residual + mlp_output;

    return hidden;
}

// ============================================================================
// Complete Qwen3 Forward Pass with AVX2 and KV Cache Support
// ============================================================================

Tensor qwen3_forward_avx_with_cache(
    const TensorL& input_ids,
    KVCache* kv_cache,
    const Tensor& token_embedding,
    const std::vector<Qwen3LayerWeights>& layers,
    const Tensor& norm_weight,
    size_t num_layers,
    size_t num_attention_heads,
    size_t num_key_value_heads,
    size_t head_dim,
    float rms_norm_eps
) {
    // Embed tokens
    Tensor hidden_states = embedding(input_ids, token_embedding);

    // Compute RoPE frequencies
    size_t seq_len = input_ids.shape()[1];
    auto rope_freqs = compute_rope_freqs(seq_len, head_dim, 1000000.0f);
    Tensor cos = rope_freqs.first;
    Tensor sin = rope_freqs.second;

    // Process through layers
    for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        const auto& layer = layers[layer_idx];

        hidden_states = qwen3_decoder_layer_avx_with_cache(
            hidden_states,
            kv_cache,
            layer_idx,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            layer.input_layernorm_weight,
            layer.q_proj,
            layer.k_proj,
            layer.v_proj,
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

    // Increment cache sequence length once per forward pass
    if (kv_cache != nullptr) {
        kv_cache->increment_seq_len(seq_len);
    }

    // Final layernorm
    hidden_states = rms_norm(hidden_states, &norm_weight, rms_norm_eps);

    return hidden_states;
}

} // namespace avx2
} // namespace qwen3
} // namespace tensor_cpp
