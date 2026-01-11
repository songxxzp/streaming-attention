/**
 * @file kv_cache.h
 * @brief KV Cache for efficient autoregressive generation
 */

#ifndef TENSOR_CPP_KV_CACHE_H
#define TENSOR_CPP_KV_CACHE_H

#include "tensor_cpp/tensor.h"
#include <vector>
#include <memory>

namespace tensor_cpp {
namespace qwen3 {

/**
 * @struct KVCache
 * @brief Key-Value cache for transformer decoder layers
 *
 * Stores K and V tensors for each layer to avoid recomputing
 * attention keys and values for previous tokens.
 *
 * Layout:
 * - key_cache: [batch_size, num_kv_heads, seq_len, head_dim]
 * - value_cache: [batch_size, num_kv_heads, seq_len, head_dim]
 */
struct KVCache {
    // Per-layer cache
    std::vector<Tensor> key_cache;   ///< Cached keys for each layer
    std::vector<Tensor> value_cache; ///< Cached values for each layer

    size_t num_layers;         ///< Number of layers
    size_t batch_size;         ///< Batch size
    size_t num_kv_heads;       ///< Number of key-value heads
    size_t head_dim;           ///< Head dimension
    size_t max_seq_len;        ///< Maximum sequence length
    size_t current_seq_len;    ///< Current sequence length in cache

    /**
     * @brief Construct a new KV cache
     * @param num_layers Number of transformer layers
     * @param batch_size Batch size
     * @param num_kv_heads Number of KV heads
     * @param head_dim Head dimension
     * @param max_seq_len Maximum sequence length to cache
     */
    KVCache(
        size_t num_layers,
        size_t batch_size,
        size_t num_kv_heads,
        size_t head_dim,
        size_t max_seq_len = 4096
    ) : num_layers(num_layers),
        batch_size(batch_size),
        num_kv_heads(num_kv_heads),
        head_dim(head_dim),
        max_seq_len(max_seq_len),
        current_seq_len(0)
    {
        // Allocate cache for each layer
        key_cache.reserve(num_layers);
        value_cache.reserve(num_layers);

        // Each layer gets [batch, num_kv_heads, max_seq_len, head_dim]
        // Initialize with zeros
        std::vector<float> key_data(batch_size * num_kv_heads * max_seq_len * head_dim, 0.0f);
        std::vector<float> value_data(batch_size * num_kv_heads * max_seq_len * head_dim, 0.0f);

        for (size_t i = 0; i < num_layers; ++i) {
            key_cache.push_back(Tensor(
                key_data,
                Shape({static_cast<long>(batch_size),
                      static_cast<long>(num_kv_heads),
                      static_cast<long>(max_seq_len),
                      static_cast<long>(head_dim)})
            ));

            value_cache.push_back(Tensor(
                value_data,
                Shape({static_cast<long>(batch_size),
                      static_cast<long>(num_kv_heads),
                      static_cast<long>(max_seq_len),
                      static_cast<long>(head_dim)})
            ));
        }
    }

    /**
     * @brief Update cache with new keys and values
     * @param layer_idx Layer index
     * @param new_keys New keys to append [batch, num_kv_heads, seq_len, head_dim]
     * @param new_values New values to append [batch, num_kv_heads, seq_len, head_dim]
     */
    void update(size_t layer_idx, const Tensor& new_keys, const Tensor& new_values) {
        if (layer_idx >= num_layers) {
            throw std::out_of_range("Layer index out of range");
        }

        // Get cache views for this layer
        Tensor& layer_keys = key_cache[layer_idx];
        Tensor& layer_values = value_cache[layer_idx];

        // Copy new keys/values into cache starting at position current_seq_len
        size_t batch = new_keys.shape()[0];
        size_t heads = new_keys.shape()[1];
        size_t seq_len = new_keys.shape()[2];
        size_t head_dim = new_keys.shape()[3];

        if (current_seq_len + seq_len > max_seq_len) {
            throw std::runtime_error("KV cache full, cannot append more tokens");
        }

        // Copy each element
        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < heads; ++h) {
                for (size_t s = 0; s < seq_len; ++s) {
                    for (size_t d = 0; d < head_dim; ++d) {
                        // Source index in new_keys/new_values
                        size_t src_idx = ((b * heads + h) * seq_len + s) * head_dim + d;

                        // Destination index in cache
                        size_t dst_idx = ((b * heads + h) * max_seq_len + current_seq_len + s) * head_dim + d;

                        layer_keys[dst_idx] = new_keys[src_idx];
                        layer_values[dst_idx] = new_values[src_idx];
                    }
                }
            }
        }
    }

    /**
     * @brief Increment sequence length by amount (default 1)
     */
    void increment_seq_len(size_t amount = 1) {
        current_seq_len += amount;
    }

    /**
     * @brief Reset cache (clear all cached data)
     */
    void reset() {
        current_seq_len = 0;
        // Zero out all caches
        for (size_t i = 0; i < num_layers; ++i) {
            std::fill(key_cache[i].data(),
                     key_cache[i].data() + key_cache[i].size(),
                     0.0f);
            std::fill(value_cache[i].data(),
                     value_cache[i].data() + value_cache[i].size(),
                     0.0f);
        }
    }

    /**
     * @brief Get cached keys for a specific layer
     * @param layer_idx Layer index
     * @param seq_len Length of sequence to retrieve
     * @return View of cached keys [batch, num_kv_heads, seq_len, head_dim]
     */
    Tensor get_cached_keys(size_t layer_idx, size_t seq_len) const {
        if (layer_idx >= num_layers) {
            throw std::out_of_range("Layer index out of range");
        }

        if (seq_len > current_seq_len) {
            throw std::runtime_error("Requested seq_len exceeds cached length");
        }

        // Return view of cache (this would need a proper Tensor view implementation)
        // For now, return a copy
        const Tensor& full_cache = key_cache[layer_idx];
        size_t batch = batch_size;
        size_t heads = num_kv_heads;
        size_t dim = head_dim;

        std::vector<float> result_data(batch * heads * seq_len * dim);

        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < heads; ++h) {
                for (size_t s = 0; s < seq_len; ++s) {
                    for (size_t d = 0; d < dim; ++d) {
                        size_t src_idx = ((b * heads + h) * max_seq_len + s) * dim + d;
                        size_t dst_idx = ((b * heads + h) * seq_len + s) * dim + d;
                        result_data[dst_idx] = full_cache[src_idx];
                    }
                }
            }
        }

        return Tensor(std::move(result_data),
                     Shape({static_cast<long>(batch),
                           static_cast<long>(heads),
                           static_cast<long>(seq_len),
                           static_cast<long>(dim)}));
    }

    /**
     * @brief Get cached values for a specific layer
     * @param layer_idx Layer index
     * @param seq_len Length of sequence to retrieve
     * @return View of cached values [batch, num_kv_heads, seq_len, head_dim]
     */
    Tensor get_cached_values(size_t layer_idx, size_t seq_len) const {
        if (layer_idx >= num_layers) {
            throw std::out_of_range("Layer index out of range");
        }

        if (seq_len > current_seq_len) {
            throw std::runtime_error("Requested seq_len exceeds cached length");
        }

        const Tensor& full_cache = value_cache[layer_idx];
        size_t batch = batch_size;
        size_t heads = num_kv_heads;
        size_t dim = head_dim;

        std::vector<float> result_data(batch * heads * seq_len * dim);

        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < heads; ++h) {
                for (size_t s = 0; s < seq_len; ++s) {
                    for (size_t d = 0; d < dim; ++d) {
                        size_t src_idx = ((b * heads + h) * max_seq_len + s) * dim + d;
                        size_t dst_idx = ((b * heads + h) * seq_len + s) * dim + d;
                        result_data[dst_idx] = full_cache[src_idx];
                    }
                }
            }
        }

        return Tensor(std::move(result_data),
                     Shape({static_cast<long>(batch),
                           static_cast<long>(heads),
                           static_cast<long>(seq_len),
                           static_cast<long>(dim)}));
    }
};

} // namespace qwen3
} // namespace tensor_cpp

#endif // TENSOR_CPP_KV_CACHE_H
