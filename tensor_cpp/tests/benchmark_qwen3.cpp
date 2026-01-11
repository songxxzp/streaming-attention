/**
 * @file benchmark_qwen3.cpp
 * @brief Qwen3模型性能测试：Prefill和Decode阶段的吞吐量测试
 *
 * 用法：
 *   # OpenMP版本
 *   OMP_NUM_THREADS=16 ./benchmark_qwen3 --model /path/to/model.safetensors --phase prefill --prompt-len 128 --iters 10
 *
 *   # MPI版本
 *   mpirun -np 4 ./benchmark_qwen3 --mode mpi --model /path/to/model.safetensors --phase decode --gen-len 100
 */

#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/ops.h"
#include "tensor_cpp/kv_cache.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <random>

#ifdef USE_MPI
#include <mpi.h>
#endif

using namespace tensor_cpp;
using namespace tensor_cpp::qwen3;

struct BenchmarkConfig {
    std::string model_path = "/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors";
    std::string phase = "prefill";       // prefill | decode
    std::string mode = "omp";            // omp | mpi | serial
    std::string attention = "standard";   // standard | streaming

    int prompt_len = 128;                // prompt长度 (prefill阶段)
    int gen_len = 100;                   // 生成长度 (decode阶段)
    int iters = 10;                      // 迭代次数
    int num_threads = 16;                // OpenMP线程数
    int warmup = 2;                      // 预热次数

    bool verbose = false;
    bool use_kv_cache = true;            // decode阶段是否使用KV cache
};

void print_usage(const char* prog) {
    std::cout << "用法: " << prog << " [选项]\n"
              << "选项:\n"
              << "  --model PATH          模型文件路径\n"
              << "  --phase PHASE         测试阶段: prefill(预填充) 或 decode(解码) [默认: prefill]\n"
              << "  --mode MODE           并行模式: omp, mpi, serial [默认: omp]\n"
              << "  --attention TYPE      attention类型: standard(标准) 或 streaming(流式) [默认: standard]\n"
              << "  --prompt-len N        prompt长度 [默认: 128]\n"
              << "  --gen-len N           生成长度 [默认: 100]\n"
              << "  --iters N             迭代次数 [默认: 10]\n"
              << "  --threads N           OpenMP线程数 [默认: 16]\n"
              << "  --warmup N            预热次数 [默认: 2]\n"
              << "  --no-kv-cache         decode阶段不使用KV cache\n"
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
        } else if (strcmp(argv[i], "--no-kv-cache") == 0) {
            cfg.use_kv_cache = false;
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

// Prefill阶段基准测试
double benchmark_prefill(const BenchmarkConfig& cfg, const Qwen3Weights& weights) {
    std::vector<long> input_ids = generate_random_tokens(cfg.prompt_len, weights.vocab_size);
    Shape input_shape({1, static_cast<long>(input_ids.size())});
    TensorL input(input_ids, input_shape);

    // 预热
    for (int i = 0; i < cfg.warmup; ++i) {
        Tensor output = qwen3::qwen3_forward(
            input,
            weights.embed_tokens,
            weights.layers,
            weights.norm_weight,
            weights.num_layers,
            weights.num_attention_heads,
            weights.num_key_value_heads,
            weights.head_dim,
            1e-6f
        );
    }

    // 正式测试
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < cfg.iters; ++i) {
        Tensor output = qwen3::qwen3_forward(
            input,
            weights.embed_tokens,
            weights.layers,
            weights.norm_weight,
            weights.num_layers,
            weights.num_attention_heads,
            weights.num_key_value_heads,
            weights.head_dim,
            1e-6f
        );
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
    std::vector<long> prompt_ids = generate_random_tokens(16, weights.vocab_size);  // 短prompt初始化
    Shape prompt_shape({1, static_cast<long>(prompt_ids.size())});
    TensorL prompt_input(prompt_ids, prompt_shape);

    Tensor prefill_output = qwen3::qwen3_forward_with_cache(
        prompt_input,
        kv_cache.get(),
        weights.embed_tokens,
        weights.layers,
        weights.norm_weight,
        weights.num_layers,
        weights.num_attention_heads,
        weights.num_key_value_heads,
        weights.head_dim,
        1e-6f
    );

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

        Tensor output = qwen3::qwen3_forward_with_cache(
            new_input,
            kv_cache.get(),
            weights.embed_tokens,
            weights.layers,
            weights.norm_weight,
            weights.num_layers,
            weights.num_attention_heads,
            weights.num_key_value_heads,
            weights.head_dim,
            1e-6f
        );
    }

    // 重置cache
    kv_cache->reset();
    // 重新prefill
    prefill_output = qwen3::qwen3_forward_with_cache(
        prompt_input,
        kv_cache.get(),
        weights.embed_tokens,
        weights.layers,
        weights.norm_weight,
        weights.num_layers,
        weights.num_attention_heads,
        weights.num_key_value_heads,
        weights.head_dim,
        1e-6f
    );
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

        Tensor output = qwen3::qwen3_forward_with_cache(
            new_input,
            kv_cache.get(),
            weights.embed_tokens,
            weights.layers,
            weights.norm_weight,
            weights.num_layers,
            weights.num_attention_heads,
            weights.num_key_value_heads,
            weights.head_dim,
            1e-6f
        );

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

void print_benchmark_header(const BenchmarkConfig& cfg, const Qwen3Weights& weights) {
    std::cout << "\n";
    std::cout << "==================================================\n";
    std::cout << "         Qwen3 Performance Benchmark\n";
    std::cout << "==================================================\n";
    std::cout << "模型路径:       " << cfg.model_path << "\n";
    std::cout << "测试阶段:       " << cfg.phase << "\n";
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
            // TODO: 实现不使用KV cache的decode基准测试
            std::cerr << "错误: decode阶段必须使用KV cache\n";
            return 1;
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
