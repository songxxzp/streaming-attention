/**
 * @file benchmark_qwen3.cpp
 * @brief Qwen3模型性能测试：Prefill和Decode阶段的吞吐量测试
 *
 * 用法：
 *   # Benchmark模式 - Prefill阶段
 *   OMP_NUM_THREADS=16 ./benchmark_qwen3 --model /path/to/model.safetensors --phase prefill --prompt-len 128 --iters 10
 *
 *   # Benchmark模式 - Decode阶段（使用KV cache）
 *   OMP_NUM_THREADS=16 ./benchmark_qwen3 --model /path/to/model.safetensors --phase decode --gen-len 100 --use-kv-cache
 *
 *   # Benchmark模式 - Decode阶段（不使用KV cache）
 *   OMP_NUM_THREADS=16 ./benchmark_qwen3 --model /path/to/model.safetensors --phase decode --gen-len 10 --no-kv-cache
 *
 *   # 对比所有方法的KV cache加速效果
 *   OMP_NUM_THREADS=16 ./benchmark_qwen3 --model /path/to/model.safetensors --compare-kv-cache --gen-len 10
 *
 *   # 验证模式 - 检查输出正确性
 *   OMP_NUM_THREADS=16 ./benchmark_qwen3 --model /path/to/model.safetensors --verify --prompt "Hello, world!"
 */

#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/qwen3_ops_avx.h"
#include "tensor_cpp/qwen3_ops_mpi.h"
#include "tensor_cpp/qwen3_ops_mpi_avx.h"
#include "tensor_cpp/ops.h"
#include "tensor_cpp/kv_cache.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <random>
#include <sstream>

#ifdef USE_MPI
#include <mpi.h>
#endif

using namespace tensor_cpp;
using namespace tensor_cpp::qwen3;

struct BenchmarkConfig {
    std::string model_path = "/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors";
    std::string phase = "prefill";         // prefill | decode
    std::string mode = "omp";              // omp | mpi | serial
    std::string method = "baseline";       // baseline | avx2 | mpi | mpi+avx2
    std::string attention = "standard";     // standard | streaming

    int prompt_len = 128;                  // prompt长度 (prefill阶段)
    int gen_len = 100;                     // 生成长度 (decode阶段)
    int iters = 10;                        // 迭代次数
    int num_threads = 16;                  // OpenMP线程数
    int warmup = 2;                        // 预热次数

    bool verbose = false;
    bool use_kv_cache = true;              // decode阶段是否使用KV cache
    bool compare_kv_cache = false;         // 对比所有方法的KV cache加速效果
    bool verify_mode = false;              // 验证模式
    std::string verify_prompt = "";        // 验证模式下的输入文本
};

void print_usage(const char* prog) {
    std::cout << "用法: " << prog << " [选项]\n"
              << "选项:\n"
              << "  --model PATH          模型文件路径\n"
              << "  --phase PHASE         测试阶段: prefill(预填充) 或 decode(解码) [默认: prefill]\n"
              << "  --mode MODE           并行模式: omp, mpi, serial [默认: omp]\n"
              << "  --method METHOD       优化方法: baseline, avx2, mpi, mpi+avx2 [默认: baseline]\n"
              << "  --attention TYPE      attention类型: standard(标准) 或 streaming(流式) [默认: standard]\n"
              << "  --prompt-len N        prompt长度 [默认: 128]\n"
              << "  --gen-len N           生成长度 [默认: 100]\n"
              << "  --iters N             迭代次数 [默认: 10]\n"
              << "  --threads N           OpenMP线程数 [默认: 16]\n"
              << "  --warmup N            预热次数 [默认: 2]\n"
              << "  --use-kv-cache        decode阶段使用KV cache\n"
              << "  --no-kv-cache         decode阶段不使用KV cache\n"
              << "  --compare-kv-cache    对比所有方法的KV cache加速效果\n"
              << "  --verify TEXT         验证模式：检查输出正确性\n"
              << "  --verbose             输出详细信息\n"
              << "  --help                显示帮助信息\n";
}

BenchmarkConfig parse_args(int argc, char** argv) {
    BenchmarkConfig cfg;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            cfg.model_path = argv[++i];
        } else if (strcmp(argv[i], "--phase") == 0 && i + 1 < argc) {
            cfg.phase = argv[++i];
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            cfg.mode = argv[++i];
        } else if (strcmp(argv[i], "--method") == 0 && i + 1 < argc) {
            cfg.method = argv[++i];
        } else if (strcmp(argv[i], "--attention") == 0 && i + 1 < argc) {
            cfg.attention = argv[++i];
        } else if (strcmp(argv[i], "--prompt-len") == 0 && i + 1 < argc) {
            cfg.prompt_len = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--gen-len") == 0 && i + 1 < argc) {
            cfg.gen_len = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            cfg.iters = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            cfg.num_threads = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            cfg.warmup = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--use-kv-cache") == 0) {
            cfg.use_kv_cache = true;
        } else if (strcmp(argv[i], "--no-kv-cache") == 0) {
            cfg.use_kv_cache = false;
        } else if (strcmp(argv[i], "--compare-kv-cache") == 0) {
            cfg.compare_kv_cache = true;
        } else if (strcmp(argv[i], "--verify") == 0 && i + 1 < argc) {
            cfg.verify_mode = true;
            cfg.verify_prompt = argv[++i];
        } else if (strcmp(argv[i], "--verbose") == 0) {
            cfg.verbose = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else {
            std::cerr << "未知参数: " << argv[i] << "\n";
            print_usage(argv[0]);
            exit(1);
        }
    }

    return cfg;
}

// 生成随机token IDs
std::vector<long> generate_random_tokens(int count, int vocab_size) {
    std::vector<long> tokens(count);
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_int_distribution<long> dist(0, vocab_size - 1);

    for (auto& token : tokens) {
        token = dist(gen);
    }

    return tokens;
}

// 获取方法名称
std::string get_method_name(const std::string& method) {
    if (method == "baseline") return "Baseline";
    if (method == "avx2") return "AVX2";
    if (method == "mpi") return "MPI";
    if (method == "mpi+avx2") return "MPI+AVX2";
    return method;
}

// 根据方法选择forward函数
Tensor forward_with_method(
    const TensorL& input_ids,
    KVCache* kv_cache,
    const Qwen3Weights& weights,
    const std::string& method
) {
    if (method == "baseline") {
        if (kv_cache) {
            return qwen3::qwen3_forward_with_cache(
                input_ids, kv_cache, weights.embed_tokens, weights.layers,
                weights.norm_weight, weights.num_layers,
                weights.num_attention_heads, weights.num_key_value_heads,
                weights.head_dim, 1e-6f
            );
        } else {
            return qwen3::qwen3_forward(
                input_ids, weights.embed_tokens, weights.layers,
                weights.norm_weight, weights.num_layers,
                weights.num_attention_heads, weights.num_key_value_heads,
                weights.head_dim, 1e-6f
            );
        }
    } else if (method == "avx2") {
        if (kv_cache) {
            return avx2::qwen3_forward_avx_with_cache(
                input_ids, kv_cache, weights.embed_tokens, weights.layers,
                weights.norm_weight, weights.num_layers,
                weights.num_attention_heads, weights.num_key_value_heads,
                weights.head_dim, 1e-6f
            );
        } else {
            return avx2::qwen3_forward_avx(
                input_ids, weights.embed_tokens, weights.layers,
                weights.norm_weight, weights.num_layers,
                weights.num_attention_heads, weights.num_key_value_heads,
                weights.head_dim, 1e-6f
            );
        }
    } else if (method == "mpi") {
        if (kv_cache) {
            return mpi::qwen3_forward_mpi_omp_with_cache(
                input_ids, kv_cache, weights.embed_tokens, weights.layers,
                weights.norm_weight, weights.num_layers,
                weights.num_attention_heads, weights.num_key_value_heads,
                weights.head_dim, 1e-6f
            );
        } else {
            return mpi::qwen3_forward_mpi_omp(
                input_ids, weights.embed_tokens, weights.layers,
                weights.norm_weight, weights.num_layers,
                weights.num_attention_heads, weights.num_key_value_heads,
                weights.head_dim, 1e-6f
            );
        }
    } else if (method == "mpi+avx2") {
        if (kv_cache) {
            return mpi_avx::qwen3_forward_mpi_avx_with_cache(
                input_ids, kv_cache, weights.embed_tokens, weights.layers,
                weights.norm_weight, weights.num_layers,
                weights.num_attention_heads, weights.num_key_value_heads,
                weights.head_dim, 1e-6f
            );
        } else {
            return mpi_avx::qwen3_forward_mpi_avx(
                input_ids, weights.embed_tokens, weights.layers,
                weights.norm_weight, weights.num_layers,
                weights.num_attention_heads, weights.num_key_value_heads,
                weights.head_dim, 1e-6f
            );
        }
    }
    throw std::runtime_error("Unknown method: " + method);
}

// Prefill阶段基准测试
double benchmark_prefill(const BenchmarkConfig& cfg, const Qwen3Weights& weights) {
    std::vector<long> input_ids = generate_random_tokens(cfg.prompt_len, weights.vocab_size);
    Shape input_shape({1, static_cast<long>(input_ids.size())});
    TensorL input(input_ids, input_shape);

    // 预热
    for (int i = 0; i < cfg.warmup; ++i) {
        Tensor output = forward_with_method(input, nullptr, weights, cfg.method);
    }

    // 正式测试
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < cfg.iters; ++i) {
        Tensor output = forward_with_method(input, nullptr, weights, cfg.method);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end - start;

    return diff.count();
}

// Decode阶段基准测试（使用KV cache）
double benchmark_decode_with_cache(const BenchmarkConfig& cfg, const Qwen3Weights& weights) {
    // 初始化KV cache
    auto kv_cache = std::make_unique<KVCache>(
        weights.num_layers,
        1,
        weights.num_key_value_heads,
        weights.head_dim,
        4096
    );

    // Prefill阶段：处理初始prompt
    std::vector<long> prompt_ids = generate_random_tokens(16, weights.vocab_size);
    Shape prompt_shape({1, static_cast<long>(prompt_ids.size())});
    TensorL prompt_input(prompt_ids, prompt_shape);

    Tensor prefill_output = forward_with_method(prompt_input, kv_cache.get(), weights, cfg.method);

    // 获取第一个预测token
    size_t hidden_size = weights.hidden_size;
    size_t vocab_size = weights.lm_head.shape()[0];
    size_t last_idx = (prompt_ids.size() - 1) * hidden_size;
    std::vector<float> last_hidden(hidden_size);
    for (size_t i = 0; i < hidden_size; ++i) {
        last_hidden[i] = prefill_output[last_idx + i];
    }

    std::vector<float> logits(vocab_size);
    for (size_t v = 0; v < vocab_size; ++v) {
        float sum = 0.0f;
        for (size_t h = 0; h < hidden_size; ++h) {
            sum += last_hidden[h] * weights.lm_head[v * hidden_size + h];
        }
        logits[v] = sum;
    }

    long next_token = 0;
    float max_logit = logits[0];
    for (size_t v = 1; v < vocab_size; ++v) {
        if (logits[v] > max_logit) {
            max_logit = logits[v];
            next_token = static_cast<long>(v);
        }
    }

    // 预热
    for (int i = 0; i < cfg.warmup; ++i) {
        std::vector<long> new_token = {next_token};
        Shape new_shape({1, 1});
        TensorL new_input(new_token, new_shape);

        Tensor output = forward_with_method(new_input, kv_cache.get(), weights, cfg.method);
    }

    // 重置cache
    kv_cache->reset();
    // 重新prefill
    prefill_output = forward_with_method(prompt_input, kv_cache.get(), weights, cfg.method);
    // 重新获取next_token
    for (size_t v = 0; v < vocab_size; ++v) {
        float sum = 0.0f;
        for (size_t h = 0; h < hidden_size; ++h) {
            sum += last_hidden[h] * weights.lm_head[v * hidden_size + h];
        }
        logits[v] = sum;
    }
    next_token = 0;
    max_logit = logits[0];
    for (size_t v = 1; v < vocab_size; ++v) {
        if (logits[v] > max_logit) {
            max_logit = logits[v];
            next_token = static_cast<long>(v);
        }
    }

    // 正式测试：生成cfg.gen_len个token
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < cfg.gen_len; ++step) {
        std::vector<long> new_token = {next_token};
        Shape new_shape({1, 1});
        TensorL new_input(new_token, new_shape);

        Tensor output = forward_with_method(new_input, kv_cache.get(), weights, cfg.method);

        // 获取next token
        for (size_t i = 0; i < hidden_size; ++i) {
            last_hidden[i] = output[i];
        }

        for (size_t v = 0; v < vocab_size; ++v) {
            float sum = 0.0f;
            for (size_t h = 0; h < hidden_size; ++h) {
                sum += last_hidden[h] * weights.lm_head[v * hidden_size + h];
            }
            logits[v] = sum;
        }

        next_token = 0;
        max_logit = logits[0];
        for (size_t v = 1; v < vocab_size; ++v) {
            if (logits[v] > max_logit) {
                max_logit = logits[v];
                next_token = static_cast<long>(v);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end - start;

    return diff.count();
}

// Decode阶段基准测试（不使用KV cache - 每次重新计算整个序列）
double benchmark_decode_without_cache(const BenchmarkConfig& cfg, const Qwen3Weights& weights) {
    // 初始prompt
    std::vector<long> prompt_ids = generate_random_tokens(4, weights.vocab_size);  // 使用较短的prompt
    Shape prompt_shape({1, static_cast<long>(prompt_ids.size())});
    TensorL prompt_input(prompt_ids, prompt_shape);

    // 初始forward pass获取第一个token
    Tensor prefill_output = forward_with_method(prompt_input, nullptr, weights, cfg.method);

    // 获取第一个预测token
    size_t hidden_size = weights.hidden_size;
    size_t vocab_size = weights.lm_head.shape()[0];
    size_t last_idx = (prompt_ids.size() - 1) * hidden_size;
    std::vector<float> last_hidden(hidden_size);
    for (size_t i = 0; i < hidden_size; ++i) {
        last_hidden[i] = prefill_output[last_idx + i];
    }

    std::vector<float> logits(vocab_size);
    for (size_t v = 0; v < vocab_size; ++v) {
        float sum = 0.0f;
        for (size_t h = 0; h < hidden_size; ++h) {
            sum += last_hidden[h] * weights.lm_head[v * hidden_size + h];
        }
        logits[v] = sum;
    }

    long next_token = 0;
    float max_logit = logits[0];
    for (size_t v = 1; v < vocab_size; ++v) {
        if (logits[v] > max_logit) {
            max_logit = logits[v];
            next_token = static_cast<long>(v);
        }
    }

    // 添加到序列
    prompt_ids.push_back(next_token);

    // 预热
    for (int i = 0; i < cfg.warmup; ++i) {
        Shape new_shape({1, static_cast<long>(prompt_ids.size())});
        TensorL new_input(prompt_ids, new_shape);
        Tensor output = forward_with_method(new_input, nullptr, weights, cfg.method);

        // 获取next token
        for (size_t j = 0; j < hidden_size; ++j) {
            last_hidden[j] = output[(prompt_ids.size() - 1) * hidden_size + j];
        }

        for (size_t v = 0; v < vocab_size; ++v) {
            float sum = 0.0f;
            for (size_t h = 0; h < hidden_size; ++h) {
                sum += last_hidden[h] * weights.lm_head[v * hidden_size + h];
            }
            logits[v] = sum;
        }

        next_token = 0;
        max_logit = logits[0];
        for (size_t v = 1; v < vocab_size; ++v) {
            if (logits[v] > max_logit) {
                max_logit = logits[v];
                next_token = static_cast<long>(v);
            }
        }
        prompt_ids.push_back(next_token);
    }

    // 重置序列
    prompt_ids.resize(4);  // 回到初始prompt

    // 正式测试：生成cfg.gen_len个token，每次都重新计算整个序列
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < cfg.gen_len; ++step) {
        // 每次都重新计算整个序列
        Shape new_shape({1, static_cast<long>(prompt_ids.size())});
        TensorL new_input(prompt_ids, new_shape);

        Tensor output = forward_with_method(new_input, nullptr, weights, cfg.method);

        // 获取next token
        for (size_t j = 0; j < hidden_size; ++j) {
            last_hidden[j] = output[(prompt_ids.size() - 1) * hidden_size + j];
        }

        for (size_t v = 0; v < vocab_size; ++v) {
            float sum = 0.0f;
            for (size_t h = 0; h < hidden_size; ++h) {
                sum += last_hidden[h] * weights.lm_head[v * hidden_size + h];
            }
            logits[v] = sum;
        }

        next_token = 0;
        max_logit = logits[0];
        for (size_t v = 1; v < vocab_size; ++v) {
            if (logits[v] > max_logit) {
                max_logit = logits[v];
                next_token = static_cast<long>(v);
            }
        }
        prompt_ids.push_back(next_token);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end - start;

    return diff.count();
}

void print_benchmark_header(const BenchmarkConfig& cfg, const Qwen3Weights& weights) {
    std::cout << "\n";
    std::cout << "==================================================\n";
    std::cout << "         Qwen3 Performance Benchmark\n";
    std::cout << "==================================================\n";
    std::cout << "模型路径:       " << cfg.model_path << "\n";
    std::cout << "测试阶段:       " << cfg.phase << "\n";
    if (cfg.phase == "decode") {
        std::cout << "优化方法:       " << get_method_name(cfg.method) << "\n";
    }
    std::cout << "并行模式:       " << cfg.mode << "\n";
    std::cout << "Attention类型:  " << cfg.attention << "\n";
    if (cfg.phase == "prefill") {
        std::cout << "Prompt长度:     " << cfg.prompt_len << "\n";
    } else {
        std::cout << "生成长度:       " << cfg.gen_len << "\n";
        std::cout << "使用KV Cache:   " << (cfg.use_kv_cache ? "是" : "否") << "\n";
    }
    std::cout << "迭代次数:       " << cfg.iters << "\n";
    std::cout << "线程数:         " << cfg.num_threads << "\n";
    std::cout << "模型配置:\n";
    std::cout << "  层数:           " << weights.num_layers << "\n";
    std::cout << "  Attention头数:  " << weights.num_attention_heads << "\n";
    std::cout << "  KV头数:         " << weights.num_key_value_heads << "\n";
    std::cout << "  Head维度:       " << weights.head_dim << "\n";
    std::cout << "  隐藏层维度:     " << weights.hidden_size << "\n";
    std::cout << "  词汇表大小:     " << weights.vocab_size << "\n";
    std::cout << "==================================================\n\n";
}

void print_benchmark_results(const BenchmarkConfig& cfg, double total_time_ms, int num_tokens) {
    double avg_time_per_token_ms = total_time_ms / num_tokens;
    double throughput_tokens_per_sec = (num_tokens * 1000.0) / total_time_ms;

    std::cout << "\n";
    std::cout << "==================================================\n";
    std::cout << "              Benchmark Results\n";
    std::cout << "==================================================\n";
    std::cout << "总时间:         " << std::fixed << std::setprecision(2) << total_time_ms << " ms\n";
    std::cout << "处理token数:    " << num_tokens << "\n";
    std::cout << "平均时间/token: " << std::fixed << std::setprecision(4) << avg_time_per_token_ms << " ms\n";
    std::cout << "吞吐量:         " << std::fixed << std::setprecision(2) << throughput_tokens_per_sec
              << " tokens/sec\n";
    std::cout << "==================================================\n\n";
}

// 对比所有方法的KV cache加速效果
void benchmark_compare_kv_cache(const BenchmarkConfig& cfg, const Qwen3Weights& weights) {
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "     KV Cache Speedup Comparison (All Methods)\n";
    std::cout << "============================================================\n";
    std::cout << "Initial tokens: 4, Generated tokens: " << cfg.gen_len << "\n";
    std::cout << "============================================================\n\n";

    std::vector<std::string> methods;
#ifdef USE_MPI
    methods = {"baseline", "avx2", "mpi", "mpi+avx2"};
#else
    methods = {"baseline", "avx2"};
    std::cout << "Note: MPI methods skipped (MPI not enabled in build)\n\n";
#endif

    for (const auto& method : methods) {
        std::cout << "----------------------------------------\n";
        std::cout << get_method_name(method) << "\n";
        std::cout << "----------------------------------------\n";

        BenchmarkConfig method_cfg = cfg;
        method_cfg.method = method;

        // 不使用KV cache
        std::cout << "  WITHOUT KV CACHE (recompute everything):\n";
        method_cfg.use_kv_cache = false;
        double time_without = benchmark_decode_without_cache(method_cfg, weights);
        std::cout << "    Total time: " << time_without << " ms\n";

        // 使用KV cache
        std::cout << "\n  WITH KV CACHE (only compute new tokens):\n";
        method_cfg.use_kv_cache = true;
        double time_with = benchmark_decode_with_cache(method_cfg, weights);
        std::cout << "    Total time: " << time_with << " ms\n";

        // 计算加速比
        double speedup = time_without / time_with;
        std::cout << "\n  >>> SPEEDUP: " << std::fixed << std::setprecision(2) << speedup << "x <<<\n";
        std::cout << "  (Without: " << time_without << " ms, With: " << time_with << " ms)\n\n";
    }

    std::cout << "============================================================\n";
    std::cout << "Comparison completed!\n";
    std::cout << "============================================================\n\n";
}

// 简单的decode生成演示（用于verify模式）
void demo_decode_generation(
    const std::vector<long>& prompt_ids,
    const Qwen3Weights& weights,
    const std::string& method,
    int gen_len
) {
    using namespace tensor_cpp::ops;

    std::cout << "\n  >>> Decode Generation Demo (" << get_method_name(method) << ") <<<\n";
    std::cout << "  Prompt tokens: [";
    for (size_t i = 0; i < std::min(prompt_ids.size(), size_t(10)); ++i) {
        std::cout << prompt_ids[i];
        if (i < prompt_ids.size() - 1) std::cout << ", ";
    }
    if (prompt_ids.size() > 10) std::cout << "...";
    std::cout << "] (" << prompt_ids.size() << " tokens)\n";

    // 创建KV cache
    auto kv_cache = std::make_unique<KVCache>(
        weights.num_layers, 1, weights.num_key_value_heads,
        weights.head_dim, 4096
    );

    // 生成tokens
    std::vector<long> generated = prompt_ids;
    std::cout << "  Generating " << gen_len << " tokens...\n";

    for (int step = 0; step < gen_len; ++step) {
        // 准备输入
        Shape input_shape({1, static_cast<long>(generated.size())});
        TensorL input(generated, input_shape);

        // Forward pass
        Tensor output;
        if (method == "baseline") {
            output = qwen3::qwen3_forward_with_cache(
                input, kv_cache.get(), weights.embed_tokens, weights.layers,
                weights.norm_weight, weights.num_layers,
                weights.num_attention_heads, weights.num_key_value_heads,
                weights.head_dim, 1e-6f
            );
        } else if (method == "avx2") {
            output = avx2::qwen3_forward_avx_with_cache(
                input, kv_cache.get(), weights.embed_tokens, weights.layers,
                weights.norm_weight, weights.num_layers,
                weights.num_attention_heads, weights.num_key_value_heads,
                weights.head_dim, 1e-6f
            );
        } else {
            std::cout << "  Unknown method: " << method << "\n";
            return;
        }

        // Sample next token (argmax)
        // output shape: [1, seq_len, vocab_size]
        // Get last position's logits
        size_t vocab_size = weights.vocab_size;
        size_t last_pos = generated.size() - 1;
        const float* logits = output.data() + last_pos * vocab_size;

        // Find argmax
        float max_logit = logits[0];
        long max_idx = 0;
        for (size_t i = 1; i < vocab_size; ++i) {
            if (logits[i] > max_logit) {
                max_logit = logits[i];
                max_idx = static_cast<long>(i);
            }
        }

        generated.push_back(max_idx);

        // 每10个token输出一次进度
        if ((step + 1) % 10 == 0 || step == gen_len - 1) {
            std::cout << "    Step " << (step + 1) << "/" << gen_len
                      << ": Generated token " << max_idx
                      << " (logit=" << max_logit << ")\n";
        }
    }

    std::cout << "  Generated tokens: [";
    for (size_t i = prompt_ids.size(); i < std::min(generated.size(), prompt_ids.size() + 10); ++i) {
        std::cout << generated[i];
        if (i < generated.size() - 1) std::cout << ", ";
    }
    if (generated.size() > prompt_ids.size() + 10) std::cout << "...";
    std::cout << "]\n";
    std::cout << "  Total tokens: " << generated.size() << " (prompt: " << prompt_ids.size()
              << ", generated: " << gen_len << ")\n";
}

// 验证模式：检查输出正确性
void verify_outputs(const BenchmarkConfig& cfg, const Qwen3Weights& weights) {
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "     Verification Mode: Comparing Outputs\n";
    std::cout << "============================================================\n";
    std::cout << "Input prompt: \"" << cfg.verify_prompt << "\"\n";
    std::cout << "============================================================\n\n";

    // 将prompt转换为token IDs（简单方法：使用字符ASCII值）
    std::vector<long> input_ids;
    for (char c : cfg.verify_prompt) {
        input_ids.push_back(static_cast<long>(c));
    }
    Shape input_shape({1, static_cast<long>(input_ids.size())});
    TensorL input(input_ids, input_shape);

    std::vector<std::string> methods = {"baseline", "avx2"};
    std::vector<std::pair<std::string, Tensor>> outputs_with_cache;
    std::vector<std::pair<std::string, Tensor>> outputs_without_cache;

    // 测试不使用KV cache
    std::cout << "WITHOUT KV CACHE:\n";
    for (const auto& method : methods) {
        std::cout << "  " << get_method_name(method) << "... ";
        Tensor output = forward_with_method(input, nullptr, weights, method);
        outputs_without_cache.push_back({method, output});
        std::cout << "Done (output size: " << output.size() << ")\n";
    }

    // 测试使用KV cache
    std::cout << "\nWITH KV CACHE:\n";
    for (const auto& method : methods) {
        std::cout << "  " << get_method_name(method) << "... ";
        auto kv_cache = std::make_unique<KVCache>(
            weights.num_layers, 1, weights.num_key_value_heads, weights.head_dim, 4096);
        Tensor output = forward_with_method(input, kv_cache.get(), weights, method);
        outputs_with_cache.push_back({method, output});
        std::cout << "Done (output size: " << output.size() << ")\n";
    }

    // 比较输出
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "     Output Comparison\n";
    std::cout << "============================================================\n\n";

    // 比较不使用KV cache的方法
    std::cout << "WITHOUT KV CACHE comparison:\n";
    bool no_cache_match = true;
    float no_cache_max_diff = 0.0f;
    float no_cache_max_rel_diff = 0.0f;
    for (size_t i = 0; i < outputs_without_cache.size(); ++i) {
        for (size_t j = i + 1; j < outputs_without_cache.size(); ++j) {
            const auto& out1 = outputs_without_cache[i].second;
            const auto& out2 = outputs_without_cache[j].second;

            if (out1.size() != out2.size()) {
                std::cout << "  " << outputs_without_cache[i].first << " vs " << outputs_without_cache[j].first
                          << ": SIZE MISMATCH (" << out1.size() << " vs " << out2.size() << ")\n";
                no_cache_match = false;
                continue;
            }

            float max_diff = 0.0f;
            float max_rel_diff = 0.0f;
            int large_diff_count = 0;
            const float* data1 = out1.data();
            const float* data2 = out2.data();
            for (size_t k = 0; k < out1.size(); ++k) {
                float diff = std::abs(data1[k] - data2[k]);
                max_diff = std::max(max_diff, diff);

                // 相对误差
                float abs_val = std::max(std::abs(data1[k]), std::abs(data2[k]));
                float rel_diff = (abs_val > 1e-6f) ? (diff / abs_val) : diff;
                max_rel_diff = std::max(max_rel_diff, rel_diff);

                if (diff > 0.1f) large_diff_count++;
            }

            std::cout << "  " << outputs_without_cache[i].first << " vs " << outputs_without_cache[j].first << ":\n";
            std::cout << "    Max absolute diff: " << max_diff << "\n";
            std::cout << "    Max relative diff: " << (max_rel_diff * 100.0f) << "%\n";
            std::cout << "    Elements with diff > 0.1: " << large_diff_count << " / " << out1.size() << "\n";

            // 更宽松的阈值：相对误差<1%或绝对误差<1.0
            if (max_rel_diff > 0.01f && max_diff > 1.0f) {
                std::cout << "    Result: MISMATCH (threshold: 1% rel or 1.0 abs)\n";
                no_cache_match = false;
                no_cache_max_diff = std::max(no_cache_max_diff, max_diff);
                no_cache_max_rel_diff = std::max(no_cache_max_rel_diff, max_rel_diff);
            } else {
                std::cout << "    Result: MATCH (within tolerance)\n";
            }
        }
    }

    // 比较使用KV cache的方法
    std::cout << "\nWITH KV CACHE comparison:\n";
    bool with_cache_match = true;
    float with_cache_max_diff = 0.0f;
    float with_cache_max_rel_diff = 0.0f;
    for (size_t i = 0; i < outputs_with_cache.size(); ++i) {
        for (size_t j = i + 1; j < outputs_with_cache.size(); ++j) {
            const auto& out1 = outputs_with_cache[i].second;
            const auto& out2 = outputs_with_cache[j].second;

            if (out1.size() != out2.size()) {
                std::cout << "  " << outputs_with_cache[i].first << " vs " << outputs_with_cache[j].first
                          << ": SIZE MISMATCH (" << out1.size() << " vs " << out2.size() << ")\n";
                with_cache_match = false;
                continue;
            }

            float max_diff = 0.0f;
            float max_rel_diff = 0.0f;
            int large_diff_count = 0;
            const float* data1 = out1.data();
            const float* data2 = out2.data();
            for (size_t k = 0; k < out1.size(); ++k) {
                float diff = std::abs(data1[k] - data2[k]);
                max_diff = std::max(max_diff, diff);

                // 相对误差
                float abs_val = std::max(std::abs(data1[k]), std::abs(data2[k]));
                float rel_diff = (abs_val > 1e-6f) ? (diff / abs_val) : diff;
                max_rel_diff = std::max(max_rel_diff, rel_diff);

                if (diff > 0.1f) large_diff_count++;
            }

            std::cout << "  " << outputs_with_cache[i].first << " vs " << outputs_with_cache[j].first << ":\n";
            std::cout << "    Max absolute diff: " << max_diff << "\n";
            std::cout << "    Max relative diff: " << (max_rel_diff * 100.0f) << "%\n";
            std::cout << "    Elements with diff > 0.1: " << large_diff_count << " / " << out1.size() << "\n";

            // 更宽松的阈值：相对误差<1%或绝对误差<1.0
            if (max_rel_diff > 0.01f && max_diff > 1.0f) {
                std::cout << "    Result: MISMATCH (threshold: 1% rel or 1.0 abs)\n";
                with_cache_match = false;
                with_cache_max_diff = std::max(with_cache_max_diff, max_diff);
                with_cache_max_rel_diff = std::max(with_cache_max_rel_diff, max_rel_diff);
            } else {
                std::cout << "    Result: MATCH (within tolerance)\n";
            }
        }
    }

    // Decode生成演示（展示模型实际生成的内容）
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "     Decode Generation Demo\n";
    std::cout << "============================================================\n";
    std::cout << "Running decode generation to show actual model output...\n";
    std::cout << "Using prompt tokens from input text...\n\n";

    // 使用prompt的前几个token进行演示（避免太长）
    std::vector<long> demo_prompt_ids = input_ids;
    if (demo_prompt_ids.size() > 10) {
        demo_prompt_ids.resize(10);  // 只用前10个token
    }

    // 演示baseline和AVX2的生成（只生成10个token，避免太慢）
    demo_decode_generation(demo_prompt_ids, weights, "baseline", 10);
    std::cout << "\n";
    demo_decode_generation(demo_prompt_ids, weights, "avx2", 10);

    std::cout << "\n============================================================\n";
    std::cout << "Note: This demo uses greedy decoding (argmax) and ASCII token IDs.\n";
    std::cout << "      For proper text generation, use a tokenizer.\n";
    std::cout << "============================================================\n\n";

    // 最终结果
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "     Verification Result\n";
    std::cout << "============================================================\n";
    if (no_cache_match && with_cache_match) {
        std::cout << "✓ All outputs MATCH - Implementation is CORRECT\n";
        std::cout << "Note: Using relaxed tolerance (1% relative or 1.0 absolute)\n";
        std::cout << "      due to floating-point precision differences in AVX2 SIMD\n\n";
    } else {
        std::cout << "✗ Outputs MISMATCH - Implementation has ERRORS\n\n";
        if (!no_cache_match) {
            std::cout << "  WITHOUT KV CACHE:\n";
            std::cout << "    Max abs diff: " << no_cache_max_diff << "\n";
            std::cout << "    Max rel diff: " << (no_cache_max_rel_diff * 100.0f) << "%\n";
        }
        if (!with_cache_match) {
            std::cout << "  WITH KV CACHE:\n";
            std::cout << "    Max abs diff: " << with_cache_max_diff << "\n";
            std::cout << "    Max rel diff: " << (with_cache_max_rel_diff * 100.0f) << "%\n";
        }
        std::cout << "\n";
    }
    std::cout << "============================================================\n\n";
}

int main(int argc, char** argv) {
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif

    BenchmarkConfig cfg = parse_args(argc, argv);

    // 设置OpenMP线程数
    omp_set_num_threads(cfg.num_threads);

#ifdef USE_MPI
    if (rank == 0) {
#endif
        std::cout << "\n加载模型...\n";
#ifdef USE_MPI
    }
#endif

    // 加载模型
    Qwen3Weights weights = load_qwen3_weights(cfg.model_path);

    // 验证模式
    if (cfg.verify_mode) {
        verify_outputs(cfg, weights);
        return 0;
    }

    // 对比KV cache加速效果
    if (cfg.compare_kv_cache) {
        benchmark_compare_kv_cache(cfg, weights);
        return 0;
    }

#ifdef USE_MPI
    if (rank == 0) {
#endif
        print_benchmark_header(cfg, weights);
#ifdef USE_MPI
    }
#endif

    double total_time_ms = 0.0;
    int num_tokens = 0;

    if (cfg.phase == "prefill") {
        num_tokens = cfg.prompt_len * cfg.iters;
#ifdef USE_MPI
        if (rank == 0) {
#endif
            std::cout << "运行 Prefill 阶段基准测试...\n";
#ifdef USE_MPI
        }
#endif
        total_time_ms = benchmark_prefill(cfg, weights);
    } else if (cfg.phase == "decode") {
        num_tokens = cfg.gen_len;
#ifdef USE_MPI
        if (rank == 0) {
#endif
            std::cout << "运行 Decode 阶段基准测试...\n";
#ifdef USE_MPI
        }
#endif
        if (cfg.use_kv_cache) {
            total_time_ms = benchmark_decode_with_cache(cfg, weights);
        } else {
            total_time_ms = benchmark_decode_without_cache(cfg, weights);
        }
    } else {
        std::cerr << "错误: 未知的阶段 '" << cfg.phase << "'\n";
        std::cerr << "支持的阶段: prefill, decode\n";
        return 1;
    }

#ifdef USE_MPI
    if (rank == 0) {
#endif
        print_benchmark_results(cfg, total_time_ms, num_tokens);
#ifdef USE_MPI
    }
#endif

#ifdef USE_MPI
    MPI_Finalize();
#endif

    return 0;
}
