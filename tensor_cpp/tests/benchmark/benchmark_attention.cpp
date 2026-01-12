/**
 * @file benchmark_attention.cpp
 * @brief Attention算子性能对比测试：Standard vs Streaming
 *
 * 用法：
 *   ./benchmark_attention --mode standard|streaming --seq-len 1024 --hidden 128 --iters 100 --threads 16
 */

#include "tensor_cpp/ops.h"
#include "tensor_cpp/tensor.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <random>
#include <numeric>

using namespace tensor_cpp;

struct BenchmarkConfig {
    int seq_len = 1024;          // 序列长度
    int hidden_dim = 128;        // 隐藏维度 (head_dim)
    int num_heads = 16;          // attention heads
    int batch_size = 1;          // batch size
    int iters = 100;             // 迭代次数
    int num_threads = 16;        // OpenMP线程数
    std::string mode = "standard"; // standard | streaming
    int block_size = 64;         // streaming attention block size
    bool verbose = false;
};

void print_usage(const char* prog) {
    std::cout << "用法: " << prog << " [选项]\n"
              << "选项:\n"
              << "  --mode MODE           attention类型: standard(标准) 或 streaming(流式) [默认: standard]\n"
              << "  --seq-len N           序列长度 [默认: 1024]\n"
              << "  --hidden N            隐藏维度 [默认: 128]\n"
              << "  --heads N             attention头数 [默认: 16]\n"
              << "  --batch N             batch size [默认: 1]\n"
              << "  --iters N             迭代次数 [默认: 100]\n"
              << "  --threads N           OpenMP线程数 [默认: 16]\n"
              << "  --block-size N        streaming attention块大小 [默认: 64]\n"
              << "  --verbose             输出详细信息\n"
              << "  --help                显示帮助信息\n";
}

BenchmarkConfig parse_args(int argc, char** argv) {
    BenchmarkConfig cfg;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            cfg.mode = argv[++i];
        } else if (strcmp(argv[i], "--seq-len") == 0 && i + 1 < argc) {
            cfg.seq_len = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) {
            cfg.hidden_dim = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--heads") == 0 && i + 1 < argc) {
            cfg.num_heads = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            cfg.batch_size = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            cfg.iters = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            cfg.num_threads = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--block-size") == 0 && i + 1 < argc) {
            cfg.block_size = std::atoi(argv[++i]);
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

// 生成随机数据
Tensor generate_random_tensor(std::vector<size_t> shape) {
    size_t total_size = 1;
    for (size_t s : shape) total_size *= s;

    std::vector<float> data(total_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (auto& val : data) {
        val = dist(gen);
    }

    return Tensor(std::move(data), Shape(shape));
}

// 运行Standard Attention基准测试
double benchmark_standard_attention(const BenchmarkConfig& cfg) {
    // 创建输入张量
    // Q, K, V: [batch, num_heads, seq_len, hidden_dim]
    std::vector<float> q_data(cfg.batch_size * cfg.num_heads * cfg.seq_len * cfg.hidden_dim);
    std::vector<float> k_data(cfg.batch_size * cfg.num_heads * cfg.seq_len * cfg.hidden_dim);
    std::vector<float> v_data(cfg.batch_size * cfg.num_heads * cfg.seq_len * cfg.hidden_dim);

    // 随机初始化
    std::random_device rd;
    std::mt19937 gen(42); // 固定种子保证可重复性
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (auto& val : q_data) val = dist(gen);
    for (auto& val : k_data) val = dist(gen);
    for (auto& val : v_data) val = dist(gen);

    Tensor q(q_data, Shape({static_cast<size_t>(cfg.batch_size), static_cast<size_t>(cfg.num_heads),
                             static_cast<size_t>(cfg.seq_len), static_cast<size_t>(cfg.hidden_dim)}));
    Tensor k(k_data, Shape({static_cast<size_t>(cfg.batch_size), static_cast<size_t>(cfg.num_heads),
                             static_cast<size_t>(cfg.seq_len), static_cast<size_t>(cfg.hidden_dim)}));
    Tensor v(v_data, Shape({static_cast<size_t>(cfg.batch_size), static_cast<size_t>(cfg.num_heads),
                             static_cast<size_t>(cfg.seq_len), static_cast<size_t>(cfg.hidden_dim)}));

    float scale = 1.0f / std::sqrt(static_cast<float>(cfg.hidden_dim));

    // 预热
    for (int i = 0; i < 5; ++i) {
        Tensor output = ops::self_attention(q, k, v, nullptr, scale);
    }

    // 正式测试
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < cfg.iters; ++i) {
        Tensor output = ops::self_attention(q, k, v, nullptr, scale);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end - start;

    return diff.count();
}

// 运行Streaming Attention基准测试
double benchmark_streaming_attention(const BenchmarkConfig& cfg) {
    // Streaming attention是单query版本，输入格式不同
    // Q: [1, hidden_dim], K: [seq_len, hidden_dim], V: [seq_len, hidden_dim]

    size_t total_elements = cfg.seq_len * cfg.hidden_dim;

    std::vector<float> q_data(cfg.hidden_dim);
    std::vector<float> k_data(total_elements);
    std::vector<float> v_data(total_elements);

    // 随机初始化
    std::random_device rd;
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (auto& val : q_data) val = dist(gen);
    for (auto& val : k_data) val = dist(gen);
    for (auto& val : v_data) val = dist(gen);

    // 预热
    for (int i = 0; i < 5; ++i) {
        std::vector<float> output = ops::streaming_attention_omp(
            q_data.data(), k_data.data(), v_data.data(),
            cfg.seq_len, cfg.hidden_dim, cfg.block_size, cfg.num_threads
        );
    }

    // 正式测试
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < cfg.iters; ++i) {
        std::vector<float> output = ops::streaming_attention_omp(
            q_data.data(), k_data.data(), v_data.data(),
            cfg.seq_len, cfg.hidden_dim, cfg.block_size, cfg.num_threads
        );
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end - start;

    return diff.count();
}

void print_benchmark_results(const BenchmarkConfig& cfg, double total_time_ms) {
    double avg_time_ms = total_time_ms / cfg.iters;
    double throughput_tokens_per_sec = (cfg.seq_len * cfg.iters * 1000.0) / total_time_ms;
    double gflops = (2.0 * cfg.seq_len * cfg.hidden_dim * cfg.iters) / (total_time_ms / 1000.0) / 1e9;

    std::cout << "\n==================================================\n";
    std::cout << "              Benchmark Results\n";
    std::cout << "==================================================\n";
    std::cout << "模式:           " << cfg.mode << "\n";
    std::cout << "序列长度:       " << cfg.seq_len << "\n";
    std::cout << "隐藏维度:       " << cfg.hidden_dim << "\n";
    std::cout << "Attention头数:  " << cfg.num_heads << "\n";
    std::cout << "Batch大小:      " << cfg.batch_size << "\n";
    std::cout << "迭代次数:       " << cfg.iters << "\n";
    std::cout << "线程数:         " << cfg.num_threads << "\n";
    if (cfg.mode == "streaming") {
        std::cout << "块大小:         " << cfg.block_size << "\n";
    }
    std::cout << "--------------------------------------------------\n";
    std::cout << "总时间:         " << std::fixed << std::setprecision(2) << total_time_ms << " ms\n";
    std::cout << "平均时间:       " << std::fixed << std::setprecision(4) << avg_time_ms << " ms/iter\n";
    std::cout << "吞吐量:         " << std::fixed << std::setprecision(2) << throughput_tokens_per_sec
              << " tokens/sec\n";
    std::cout << "GFLOPS:         " << std::fixed << std::setprecision(2) << gflops << "\n";
    std::cout << "==================================================\n\n";
}

void print_detailed_results(const BenchmarkConfig& cfg, double total_time_ms) {
    print_benchmark_results(cfg, total_time_ms);

    if (cfg.verbose) {
        std::cout << "\n详细信息:\n";
        std::cout << "  每次迭代数据量: " << (cfg.batch_size * cfg.num_heads * cfg.seq_len * cfg.hidden_dim * sizeof(float) / 1024.0 / 1024.0)
                  << " MB\n";
        std::cout << "  总计算量: " << (cfg.batch_size * cfg.num_heads * cfg.seq_len * cfg.hidden_dim * 2.0 * cfg.iters / 1e9)
                  << " GFLOPs\n";
    }
}

int main(int argc, char** argv) {
    BenchmarkConfig cfg = parse_args(argc, argv);

    // 设置OpenMP线程数
    omp_set_num_threads(cfg.num_threads);

    std::cout << "\n";
    std::cout << "==================================================\n";
    std::cout << "       Attention Performance Benchmark\n";
    std::cout << "==================================================\n\n";

    double total_time_ms = 0.0;

    if (cfg.mode == "standard") {
        std::cout << "运行 Standard Attention 基准测试...\n";
        total_time_ms = benchmark_standard_attention(cfg);
    } else if (cfg.mode == "streaming") {
        std::cout << "运行 Streaming Attention 基准测试...\n";
        total_time_ms = benchmark_streaming_attention(cfg);
    } else {
        std::cerr << "错误: 未知的模式 '" << cfg.mode << "'\n";
        std::cerr << "支持的模式: standard, streaming\n";
        return 1;
    }

    print_detailed_results(cfg, total_time_ms);

    return 0;
}
