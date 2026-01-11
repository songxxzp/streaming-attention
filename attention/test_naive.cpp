#include "attention.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

// 外部函数（在naive_serial.cpp中定义）
extern std::vector<float> naive_attention_serial(
    const float* Q,
    const float* K,
    const float* V,
    int T,
    int d
);

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "用法: " << argv[0] << " <T> <d> <block_size>" << std::endl;
        std::cerr << "  T: 序列长度" << std::endl;
        std::cerr << "  d: 隐藏维度" << std::endl;
        std::cerr << "  block_size: (naive版本忽略此参数)" << std::endl;
        return 1;
    }

    int T = std::atoi(argv[1]);
    int d = std::atoi(argv[2]);
    int block_size = std::atoi(argv[3]);  // naive版本忽略此参数

    // 创建随机数据
    std::vector<float> Q(d);
    std::vector<float> K(T * d);
    std::vector<float> V(T * d);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < d; ++i) {
        Q[i] = dist(gen);
    }
    for (int i = 0; i < T * d; ++i) {
        K[i] = dist(gen);
        V[i] = dist(gen);
    }

    // 预热（由Python脚本控制）
    // 测试（由Python脚本控制，只运行1次）
    constexpr int ITERS = 1;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < ITERS; ++i) {
        auto result = naive_attention_serial(Q.data(), K.data(), V.data(), T, d);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end - start;

    double avg_time = diff.count() / ITERS;

    std::cout << "Time: " << avg_time << " ms" << std::endl;
    std::cout << "Throughput: " << (T * 1000 / avg_time) << " tokens/sec" << std::endl;

    return 0;
}
