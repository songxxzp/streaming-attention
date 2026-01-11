/**
 * @file test_mpi_simple.cpp
 * @brief 简单的MPI测试程序 - 验证MPI是否可用
 */

#include <mpi.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <iomanip>

// 简单的矩阵乘法（用于测试MPI性能）
void matrix_multiply_local(const float* A, const float* B, float* C,
                          int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << "==================================================\n";
        std::cout << "          MPI 测试程序\n";
        std::cout << "==================================================\n";
        std::cout << "MPI进程数: " << size << "\n";
        std::cout << "当前进程: " << rank << "\n";
        std::cout << "==================================================\n\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // 测试1: 基础通信
    if (rank == 0) {
        std::cout << "测试1: 基础点对点通信...\n";
    }

    int send_data = rank;
    int recv_data = -1;

    if (rank > 0) {
        MPI_Send(&send_data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    } else {
        for (int i = 1; i < size; ++i) {
            MPI_Recv(&recv_data, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::cout << "  进程 " << i << " 发送数据: " << recv_data << "\n";
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // 测试2: 广播
    if (rank == 0) {
        std::cout << "\n测试2: 广播操作...\n";
    }

    int broadcast_data = 42;
    MPI_Bcast(&broadcast_data, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        std::cout << "  进程 " << rank << " 接收广播数据: " << broadcast_data << "\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // 测试3: 矩阵乘法性能测试
    if (rank == 0) {
        std::cout << "\n测试3: 矩阵乘法性能测试...\n";
    }

    // 每个进程进行本地矩阵乘法
    int M = 64, N = 64, K = 64;
    int iters = 10;

    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N);

    // 初始化数据
    for (auto& val : A) val = 1.0f;
    for (auto& val : B) val = 2.0f;

    // 预热
    for (int i = 0; i < 3; ++i) {
        matrix_multiply_local(A.data(), B.data(), C.data(), M, N, K);
    }

    // 正式测试
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iters; ++i) {
        matrix_multiply_local(A.data(), B.data(), C.data(), M, N, K);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end - start;

    // 计算总吞吐量
    size_t total_flops = size * M * N * K * 2 * iters;  // 每个进程都计算
    double gflops = total_flops / (diff.count() / 1000.0) / 1e9;

    if (rank == 0) {
        std::cout << "  矩阵规模: " << M << "x" << N << "x" << K << "\n";
        std::cout << "  迭代次数: " << iters << "\n";
        std::cout << "  进程数: " << size << "\n";
        std::cout << "  总时间: " << diff.count() << " ms\n";
        std::cout << "  总性能: " << std::fixed << std::setprecision(2)
                  << gflops << " GFLOPS\n";
        std::cout << "  每进程性能: " << (gflops / size) << " GFLOPS\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // 测试4: 归约操作
    if (rank == 0) {
        std::cout << "\n测试4: 归约操作...\n";
    }

    int local_sum = rank * 10;
    int global_sum = 0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int expected_sum = 0;
        for (int i = 0; i < size; ++i) {
            expected_sum += i * 10;
        }
        std::cout << "  各进程求和: " << global_sum << " (期望: " << expected_sum << ")\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // 测试5: Allreduce
    if (rank == 0) {
        std::cout << "\n测试5: Allreduce操作...\n";
    }

    int local_max = rank * 100;
    int global_max = 0;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    std::cout << "  进程 " << rank << ": local_max=" << local_max
              << ", global_max=" << global_max << "\n";

    MPI_Finalize();

    if (rank == 0) {
        std::cout << "\n==================================================\n";
        std::cout << "MPI测试完成！MPI工作正常。\n";
        std::cout << "==================================================\n";
    }

    return 0;
}
