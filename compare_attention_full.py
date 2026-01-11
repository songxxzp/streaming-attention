#!/usr/bin/env python3
"""
Attention性能对比测试 - 完整版

测试相同规模下：
1. PyTorch F.scaled_dot_product_attention
2. C++ Naive Attention (串行)
3. C++ Naive Attention (OpenMP多线程)
4. C++ Streaming Attention (串行)
5. C++ Streaming Attention (OpenMP多线程)

测试配置:
- 序列长度: 512, 1024, 2048, 4096, 8192
- 隐藏维度: 128
- OpenMP线程数: 1, 2, 4, 8, 16
- 迭代次数: 预热2次，测试10次
"""

import subprocess
import time
import torch
import numpy as np
import os
from pathlib import Path
from typing import Tuple, List

# ============================================================================
# 配置参数
# ============================================================================

SEQ_LENS = [512, 1024, 2048, 4096, 8192]
HIDDEN_DIM = 128
WARMUP_RUNS = 2
TEST_RUNS = 10
BLOCK_SIZE = 64
THREADS_LIST = [1, 2, 4, 8, 16]

# C++可执行文件路径
CPP_NAIVE_SERIAL = "./attention/test_naive"
CPP_NAIVE_OMP = "./attention/test_naive_omp"
CPP_STREAMING_SERIAL = "./attention/test_streaming"
CPP_STREAMING_OMP = "./attention/test_streaming_omp"

# ============================================================================
# PyTorch SDPA测试
# ============================================================================

def test_pytorch_sdpa(seq_len: int, hidden_dim: int) -> float:
    """测试PyTorch SDPA性能 (预热2次，测试10次取平均)"""
    Q = torch.randn(1, 1, 1, hidden_dim, device='cpu', dtype=torch.float32)
    K = torch.randn(1, 1, seq_len, hidden_dim, device='cpu', dtype=torch.float32)
    V = torch.randn(1, 1, seq_len, hidden_dim, device='cpu', dtype=torch.float32)

    # 预热2次
    for _ in range(WARMUP_RUNS):
        _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    torch.cpu.synchronize()

    # 测试10次取平均
    times = []
    for _ in range(TEST_RUNS):
        start = time.time()
        _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        torch.cpu.synchronize()
        end = time.time()
        times.append((end - start) * 1000)

    avg_time_ms = np.mean(times)
    return avg_time_ms

# ============================================================================
# C++ Attention测试
# ============================================================================

def compile_cpp_executables():
    """编译C++可执行文件"""
    attention_dir = Path("./attention")

    if not attention_dir.exists():
        print("✗ 错误: attention目录不存在")
        return False

    print("检查C++可执行文件...")

    executables = [
        ("test_naive", "test_streaming.cpp", "naive_serial.cpp", ""),
        ("test_naive_omp", "test_naive_omp.cpp", "naive_omp.cpp", "-fopenmp"),
        ("test_streaming", "test_streaming.cpp", "streaming_serial.cpp", ""),
        ("test_streaming_omp", "test_streaming_omp.cpp", "streaming_omp.cpp streaming_serial.cpp", "-fopenmp"),
    ]

    for exe_name, test_file, impl_files, flags in executables:
        if not (attention_dir / exe_name).exists():
            print(f"  编译{exe_name}...")
            compile_cmd = f"""
            cd attention && \
            g++ -std=c++17 -O3 -march=native {flags} -I. \
                {test_file} {impl_files} \
                -o {exe_name}
            """
            result = subprocess.run(compile_cmd, shell=True, capture_output=True)
            if result.returncode != 0:
                print(f"  ✗ 编译失败: {result.stderr.decode()}")
                return False
            print(f"  ✓ {exe_name}编译成功")
        else:
            print(f"  ✓ {exe_name}已存在")

    print("✓ 所有可执行文件就绪\n")
    return True


def test_cpp_attention(seq_len: int, hidden_dim: int, block_size: int,
                        threads: int, executable: str) -> Tuple[float, bool]:
    """测试C++ Attention性能 (运行10次取平均)"""
    try:
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = str(threads)

        times = []
        for run in range(TEST_RUNS):
            result = subprocess.run(
                [executable, str(seq_len), str(hidden_dim), str(block_size)],
                capture_output=True,
                text=True,
                timeout=30,
                env=env
            )

            if result.returncode != 0:
                return 0.0, False

            # 解析输出
            for line in result.stdout.split('\n'):
                if 'Time' in line:
                    try:
                        time_val = float(line.split()[1])
                        times.append(time_val)
                        break
                    except (IndexError, ValueError):
                        continue

        if times:
            return np.mean(times), True
        return 0.0, False

    except subprocess.TimeoutExpired:
        return 0.0, False
    except FileNotFoundError:
        return 0.0, False
    except Exception as e:
        return 0.0, False

# ============================================================================
# 主测试函数
# ============================================================================

def run_comparison():
    """运行完整的对比测试"""

    print("=" * 100)
    print(" " * 25 + "Attention性能对比 - Naive vs Streaming vs PyTorch")
    print("=" * 100)
    print()

    print(f"测试配置:")
    print(f"  序列长度: {SEQ_LENS}")
    print(f"  隐藏维度: {HIDDEN_DIM}")
    print(f"  预热次数: {WARMUP_RUNS}")
    print(f"  测试次数: {TEST_RUNS}")
    print(f"  Block Size: {BLOCK_SIZE}")
    print(f"  OpenMP线程: {THREADS_LIST}")
    print()

    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print()

    # 编译C++程序
    if not compile_cpp_executables():
        print("\n请确保在项目根目录运行此脚本，并且attention目录存在")
        return

    # ============================================================================
    # 测试1: 串行性能对比 (PyTorch vs C++ Naive Serial vs C++ Streaming Serial)
    # ============================================================================
    print("=" * 100)
    print("  测试1: 串行性能对比")
    print("=" * 100)
    print()

    print("序列长度 | PyTorch(ms) | Naive(ms) | Streaming(ms) | PT-吞吐 | Naive-吞吐 | Streaming-吞吐")
    print("---------|------------|-----------|---------------|---------|-----------|--------------")

    for seq_len in SEQ_LENS:
        # PyTorch SDPA
        try:
            pytorch_time = test_pytorch_sdpa(seq_len, HIDDEN_DIM)
            pytorch_throughput = seq_len * 1000 / pytorch_time
        except Exception as e:
            print(f"  ✗ PyTorch测试失败: {e}")
            continue

        # C++ Naive Serial
        naive_time, naive_success = test_cpp_attention(seq_len, HIDDEN_DIM, BLOCK_SIZE, 1, CPP_NAIVE_SERIAL)

        # C++ Streaming Serial
        streaming_time, streaming_success = test_cpp_attention(seq_len, HIDDEN_DIM, BLOCK_SIZE, 1, CPP_STREAMING_SERIAL)

        if naive_success and streaming_success:
            naive_throughput = seq_len * 1000 / naive_time
            streaming_throughput = seq_len * 1000 / streaming_time
            print(f"{seq_len:8d} | {pytorch_time:11.4f} | {naive_time:10.4f} | {streaming_time:14.4f} | "
                  f"{pytorch_throughput:8.2f} | {naive_throughput:10.2f} | {streaming_throughput:13.2f}")
        elif naive_success:
            naive_throughput = seq_len * 1000 / naive_time
            print(f"{seq_len:8d} | {pytorch_time:11.4f} | {naive_time:10.4f} |      N/A       | "
                  f"{pytorch_throughput:8.2f} | {naive_throughput:10.2f} |      N/A      ")
        elif streaming_success:
            streaming_throughput = seq_len * 1000 / streaming_time
            print(f"{seq_len:8d} | {pytorch_time:11.4f} |     N/A    | {streaming_time:14.4f} | "
                  f"{pytorch_throughput:8.2f} |     N/A    | {streaming_throughput:13.2f}")
        else:
            print(f"{seq_len:8d} | {pytorch_time:11.4f} |     N/A    |      N/A       | "
                  f"{pytorch_throughput:8.2f} |     N/A    |      N/A      ")

    print()

    # ============================================================================
    # 测试2: Naive OpenMP扩展性 (seq_len=8192, 包含PyTorch对比)
    # ============================================================================
    print("=" * 100)
    print("  测试2: Naive Attention OpenMP扩展性 (seq_len=8192, 包含PyTorch对比)")
    print("=" * 100)
    print()

    SEQ_LEN_SCALABILITY = 8192

    print("实现方式       | 线程数 | 时间(ms) | 吞吐量(tokens/s) | 相对PyTorch加速 | 相对串行加速 | 效率")
    print("---------------|--------|---------|-----------------|----------------|-------------|------")

    # 首先获取PyTorch baseline
    try:
        pytorch_time_scalability = test_pytorch_sdpa(SEQ_LEN_SCALABILITY, HIDDEN_DIM)
        pytorch_throughput_scalability = SEQ_LEN_SCALABILITY * 1000 / pytorch_time_scalability
        print(f"{'PyTorch SDPA':15s} |   {'N/A':3s} | {pytorch_time_scalability:7.4f} | {pytorch_throughput_scalability:15.2f} | "
              f"{'  1.00':12s} | {'  N/A':9s} |  N/A")
    except Exception as e:
        print(f"  ✗ PyTorch测试失败: {e}")
        pytorch_time_scalability = None

    # 获取C++ Naive串行baseline
    naive_baseline_time, _ = test_cpp_attention(SEQ_LEN_SCALABILITY, HIDDEN_DIM, BLOCK_SIZE, 1, CPP_NAIVE_SERIAL)

    if naive_baseline_time > 0:
        naive_baseline_throughput = SEQ_LEN_SCALABILITY * 1000 / naive_baseline_time
        if pytorch_time_scalability:
            speedup_vs_pytorch = pytorch_time_scalability / naive_baseline_time
            print(f"{'Naive Serial':15s} |     {1:3d}   | {naive_baseline_time:7.4f} | {naive_baseline_throughput:15.2f} | "
                  f"{speedup_vs_pytorch:12.2f}x | {1.00:9.2f}x | 1.000")
        else:
            print(f"{'Naive Serial':15s} |     {1:3d}   | {naive_baseline_time:7.4f} | {naive_baseline_throughput:15.2f} | "
                  f"{'  N/A':12s} | {1.00:9.2f}x | 1.000")

        for threads in THREADS_LIST[1:]:
            time, success = test_cpp_attention(SEQ_LEN_SCALABILITY, HIDDEN_DIM, BLOCK_SIZE, threads, CPP_NAIVE_OMP)
            if success:
                throughput = SEQ_LEN_SCALABILITY * 1000 / time
                speedup_vs_serial = naive_baseline_time / time
                efficiency = speedup_vs_serial / threads

                if pytorch_time_scalability:
                    speedup_vs_pytorch = pytorch_time_scalability / time
                    print(f"{'Naive OpenMP':15s} |    {threads:3d}   | {time:7.4f} | {throughput:15.2f} | "
                          f"{speedup_vs_pytorch:12.2f}x | {speedup_vs_serial:9.2f}x | {efficiency:.3f}")
                else:
                    print(f"{'Naive OpenMP':15s} |    {threads:3d}   | {time:7.4f} | {throughput:15.2f} | "
                          f"{'  N/A':12s} | {speedup_vs_serial:9.2f}x | {efficiency:.3f}")

    print()

    # ============================================================================
    # 测试3: Streaming OpenMP扩展性 (seq_len=8192, 包含PyTorch对比)
    # ============================================================================
    print("=" * 100)
    print("  测试3: Streaming Attention OpenMP扩展性 (seq_len=8192, 包含PyTorch对比)")
    print("=" * 100)
    print()

    print("实现方式       | 线程数 | 时间(ms) | 吞吐量(tokens/s) | 相对PyTorch加速 | 相对串行加速 | 效率")
    print("---------------|--------|---------|-----------------|----------------|-------------|------")

    # PyTorch baseline (已在测试2中获取)

    if pytorch_time_scalability:
        print(f"{'PyTorch SDPA':15s} |   {'N/A':3s} | {pytorch_time_scalability:7.4f} | {pytorch_throughput_scalability:15.2f} | "
              f"{'  1.00':12s} | {'  N/A':9s} |  N/A")

    # 获取C++ Streaming串行baseline
    streaming_baseline_time, _ = test_cpp_attention(SEQ_LEN_SCALABILITY, HIDDEN_DIM, BLOCK_SIZE, 1, CPP_STREAMING_SERIAL)

    if streaming_baseline_time > 0:
        streaming_baseline_throughput = SEQ_LEN_SCALABILITY * 1000 / streaming_baseline_time
        if pytorch_time_scalability:
            speedup_vs_pytorch = pytorch_time_scalability / streaming_baseline_time
            print(f"{'Streaming Serial':15s} |     {1:3d}   | {streaming_baseline_time:7.4f} | {streaming_baseline_throughput:15.2f} | "
                  f"{speedup_vs_pytorch:12.2f}x | {1.00:9.2f}x | 1.000")
        else:
            print(f"{'Streaming Serial':15s} |     {1:3d}   | {streaming_baseline_time:7.4f} | {streaming_baseline_throughput:15.2f} | "
                  f"{'  N/A':12s} | {1.00:9.2f}x | 1.000")

        for threads in THREADS_LIST[1:]:
            time, success = test_cpp_attention(SEQ_LEN_SCALABILITY, HIDDEN_DIM, BLOCK_SIZE, threads, CPP_STREAMING_OMP)
            if success:
                throughput = SEQ_LEN_SCALABILITY * 1000 / time
                speedup_vs_serial = streaming_baseline_time / time
                efficiency = speedup_vs_serial / threads

                if pytorch_time_scalability:
                    speedup_vs_pytorch = pytorch_time_scalability / time
                    print(f"{'Streaming OpenMP':15s} |    {threads:3d}   | {time:7.4f} | {throughput:15.2f} | "
                          f"{speedup_vs_pytorch:12.2f}x | {speedup_vs_serial:9.2f}x | {efficiency:.3f}")
                else:
                    print(f"{'Streaming OpenMP':15s} |    {threads:3d}   | {time:7.4f} | {throughput:15.2f} | "
                          f"{'  N/A':12s} | {speedup_vs_serial:9.2f}x | {efficiency:.3f}")

    print()

    # ============================================================================
    # 总结分析
    # ============================================================================
    print("=" * 100)
    print("  总结分析")
    print("=" * 100)
    print()

    print("1. 算法对比:")
    print("   - Naive Attention: 计算完整Q@K^T矩阵，空间O(T²)，适合短序列")
    print("   - Streaming Attention: 分块计算+online softmax，空间O(Td)，适合长序列")
    print("   - PyTorch SDPA: 使用高度优化的BLAS库 (MKL/oneDNN)")
    print()

    print("2. 并行化策略:")
    print("   - Naive OpenMP: 并行化矩阵乘法 (每个线程处理一部分T)")
    print("   - Streaming OpenMP: 并行化block处理 (每个线程处理一部分blocks)")
    print()

    print("3. 性能特点:")
    print("   - PyTorch: 单线程最快（BLAS优化 + SIMD）")
    print("   - Naive OpenMP: 多线程可扩展性好，计算密集型")
    print("   - Streaming OpenMP: 内存友好，适合超长序列")
    print()

    print("=" * 100)


if __name__ == "__main__":
    run_comparison()
