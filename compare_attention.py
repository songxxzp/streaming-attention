#!/usr/bin/env python3
"""
Streaming Attention vs PyTorch SDPA 性能对比测试

测试相同规模下：
1. PyTorch F.scaled_dot_product_attention
2. C++ Streaming Attention (串行)
3. C++ Streaming Attention (OpenMP多线程)

测试配置:
- 序列长度: 512, 1024, 2048, 4096, 8192
- 隐藏维度: 128
- OpenMP线程数: 1, 2, 4, 8, 16
- 迭代次数: 100
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
    """
    测试PyTorch SDPA性能 (预热2次，测试10次取平均)

    Args:
        seq_len: 序列长度
        hidden_dim: 隐藏维度

    Returns:
        平均时间(ms)
    """
    # 创建测试数据 (Q=1x1xd, K,V=1xTxd)
    # 这里模拟单query的情况
    Q = torch.randn(1, 1, 1, hidden_dim, device='cpu', dtype=torch.float32)
    K = torch.randn(1, 1, seq_len, hidden_dim, device='cpu', dtype=torch.float32)
    V = torch.randn(1, 1, seq_len, hidden_dim, device='cpu', dtype=torch.float32)

    # 预热2次
    for _ in range(WARMUP_RUNS):
        _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V)

    # 同步确保预热完成
    torch.cpu.synchronize()

    # 测试10次取平均
    times = []
    for _ in range(TEST_RUNS):
        start = time.time()
        _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        torch.cpu.synchronize()
        end = time.time()
        times.append((end - start) * 1000)  # 转换为ms

    avg_time_ms = np.mean(times)
    return avg_time_ms


# ============================================================================
# C++ Streaming Attention测试
# ============================================================================

def compile_cpp_executables():
    """编译C++可执行文件"""
    attention_dir = Path("./attention")

    if not attention_dir.exists():
        print("✗ 错误: attention目录不存在")
        return False

    print("检查C++可执行文件...")

    # 编译naive serial版本
    if not (attention_dir / "test_naive").exists():
        print("  编译test_naive...")
        compile_cmd = """
        cd attention && \
        g++ -std=c++17 -O3 -march=native -I. \
            test_streaming.cpp naive_serial.cpp \
            -o test_naive
        """
        result = subprocess.run(compile_cmd, shell=True, capture_output=True)
        if result.returncode != 0:
            print(f"  ✗ 编译失败: {result.stderr.decode()}")
            return False
        print("  ✓ test_naive编译成功")
    else:
        print("  ✓ test_naive已存在")

    # 编译naive OpenMP版本
    if not (attention_dir / "test_naive_omp").exists():
        print("  编译test_naive_omp...")
        compile_cmd = """
        cd attention && \
        g++ -std=c++17 -O3 -march=native -fopenmp -I. \
            test_naive_omp.cpp naive_omp.cpp \
            -o test_naive_omp
        """
        result = subprocess.run(compile_cmd, shell=True, capture_output=True)
        if result.returncode != 0:
            print(f"  ✗ 编译失败: {result.stderr.decode()}")
            return False
        print("  ✓ test_naive_omp编译成功")
    else:
        print("  ✓ test_naive_omp已存在")

    # 编译streaming serial版本
    if not (attention_dir / "test_streaming").exists():
        print("  编译test_streaming...")
        compile_cmd = """
        cd attention && \
        g++ -std=c++17 -O3 -march=native -I. \
            test_streaming.cpp streaming_serial.cpp \
            -o test_streaming
        """
        result = subprocess.run(compile_cmd, shell=True, capture_output=True)
        if result.returncode != 0:
            print(f"  ✗ 编译失败: {result.stderr.decode()}")
            return False
        print("  ✓ test_streaming编译成功")
    else:
        print("  ✓ test_streaming已存在")

    # 编译streaming OpenMP版本
    if not (attention_dir / "test_streaming_omp").exists():
        print("  编译test_streaming_omp...")
        compile_cmd = """
        cd attention && \
        g++ -std=c++17 -O3 -march=native -fopenmp -I. \
            test_streaming_omp.cpp streaming_omp.cpp streaming_serial.cpp \
            -o test_streaming_omp
        """
        result = subprocess.run(compile_cmd, shell=True, capture_output=True)
        if result.returncode != 0:
            print(f"  ✗ 编译失败: {result.stderr.decode()}")
            return False
        print("  ✓ test_streaming_omp编译成功")
    else:
        print("  ✓ test_streaming_omp已存在")

    print("✓ 所有可执行文件就绪\n")
    return True


def test_cpp_attention(seq_len: int, hidden_dim: int, block_size: int,
                        threads: int, executable: str) -> Tuple[float, bool]:
    """
    测试C++ Attention性能 (运行10次取平均)

    Args:
        seq_len: 序列长度
        hidden_dim: 隐藏维度
        block_size: streaming block size (naive版本不需要)
        threads: OpenMP线程数
        executable: C++可执行文件路径

    Returns:
        (平均时间ms, 是否成功)
    """
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
    print(" " * 30 + "Streaming Attention vs PyTorch SDPA 性能对比")
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

    # 检查PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备: {torch.cuda.get_device_name(0)}")
    print()

    # 编译C++程序
    if not compile_cpp_executables():
        print("\n请确保在项目根目录运行此脚本，并且attention目录存在")
        return

    # ============================================================================
    # 测试1: 串行对比 (PyTorch vs C++ Serial)
    # ============================================================================
    print("=" * 100)
    print("  测试1: 串行性能对比 (PyTorch SDPA vs C++ Streaming Serial)")
    print("=" * 100)
    print()

    serial_results = []

    print("序列长度 | PyTorch(ms) | C++-Serial(ms) | PT-吞吐量 | C++-吞吐量 | 加速比(PT/Cpp)")
    print("---------|------------|---------------|----------|-----------|--------------")

    for seq_len in SEQ_LENS:
        # PyTorch SDPA
        try:
            pytorch_time = test_pytorch_sdpa(seq_len, HIDDEN_DIM)
            pytorch_throughput = seq_len * 1000 / pytorch_time
        except Exception as e:
            print(f"  ✗ PyTorch测试失败: {e}")
            continue

        # C++ Serial
        cpp_time, cpp_success = test_cpp_streaming(seq_len, HIDDEN_DIM, BLOCK_SIZE, 1, CPP_SERIAL)
        if cpp_success:
            cpp_throughput = seq_len * 1000 / cpp_time
            speedup = pytorch_time / cpp_time  # PyTorch是baseline
            print(f"{seq_len:8d} | {pytorch_time:11.4f} | {cpp_time:14.4f} | "
                  f"{pytorch_throughput:9.2f} | {cpp_throughput:10.2f} | {speedup:13.2f}x")

            serial_results.append({
                'seq_len': seq_len,
                'pt_time': pytorch_time,
                'cpp_time': cpp_time,
                'speedup': speedup
            })
        else:
            print(f"{seq_len:8d} | {pytorch_time:11.4f} |      N/A       | "
                  f"{pytorch_throughput:9.2f} |     N/A    |     N/A      ")

    print()

    # ============================================================================
    # 测试2: C++ OpenMP扩展性 (包含PyTorch对比)
    # ============================================================================
    print("=" * 100)
    print("  测试2: C++ Streaming OpenMP扩展性 (seq_len=8192, 包含PyTorch对比)")
    print("=" * 100)
    print()

    SEQ_LEN_SCALABILITY = 8192

    print("实现方式       | 线程数 | 时间(ms) | 吞吐量(tokens/s) | 相对PyTorch加速比 | 相对C++串行加速比 | 效率")
    print("---------------|--------|---------|-----------------|------------------|-----------------|------")

    scalability_results = []

    # 首先获取PyTorch baseline
    try:
        pytorch_time_scalability = test_pytorch_sdpa(SEQ_LEN_SCALABILITY, HIDDEN_DIM)
        pytorch_throughput_scalability = SEQ_LEN_SCALABILITY * 1000 / pytorch_time_scalability
        print(f"{'PyTorch SDPA':15s} |   {'N/A':3s} | {pytorch_time_scalability:7.4f} | {pytorch_throughput_scalability:15.2f} | "
              f"{'  1.00':14s} | {'  N/A':13s} |  N/A")
    except Exception as e:
        print(f"  ✗ PyTorch测试失败: {e}")
        pytorch_time_scalability = None

    # 获取C++串行baseline
    baseline_time, _ = test_cpp_streaming(SEQ_LEN_SCALABILITY, HIDDEN_DIM,
                                          BLOCK_SIZE, 1, CPP_SERIAL)

    if baseline_time > 0:
        baseline_throughput = SEQ_LEN_SCALABILITY * 1000 / baseline_time
        if pytorch_time_scalability:
            speedup_vs_pytorch = pytorch_time_scalability / baseline_time
            print(f"{'C++ Serial':15s} |   {1:3d}   | {baseline_time:7.4f} | {baseline_throughput:15.2f} | "
                  f"{speedup_vs_pytorch:14.2f}x | {1.0:13.2f}x | 1.000")
        else:
            print(f"{'C++ Serial':15s} |   {1:3d}   | {baseline_time:7.4f} | {baseline_throughput:15.2f} | "
                  f"{'  N/A':14s} | {1.0:13.2f}x | 1.000")
        scalability_results.append({'threads': 1, 'time': baseline_time, 'speedup': 1.0})

        for threads in THREADS_LIST[1:]:
            time, success = test_cpp_streaming(SEQ_LEN_SCALABILITY, HIDDEN_DIM,
                                              BLOCK_SIZE, threads, CPP_OMP)
            if success:
                throughput = SEQ_LEN_SCALABILITY * 1000 / time
                speedup_vs_cpp_serial = baseline_time / time
                efficiency = speedup_vs_cpp_serial / threads

                if pytorch_time_scalability:
                    speedup_vs_pytorch = pytorch_time_scalability / time
                    print(f"{'C++ OpenMP':15s} |  {threads:3d}   | {time:7.4f} | {throughput:15.2f} | "
                          f"{speedup_vs_pytorch:14.2f}x | {speedup_vs_cpp_serial:13.2f}x | {efficiency:.3f}")
                else:
                    print(f"{'C++ OpenMP':15s} |  {threads:3d}   | {time:7.4f} | {throughput:15.2f} | "
                          f"{'  N/A':14s} | {speedup_vs_cpp_serial:13.2f}x | {efficiency:.3f}")

                scalability_results.append({
                    'threads': threads,
                    'time': time,
                    'speedup': speedup_vs_cpp_serial,
                    'efficiency': efficiency
                })

    print()

    # ============================================================================
    # 生成总结报告
    # ============================================================================
    print("=" * 100)
    print("  总结分析")
    print("=" * 100)
    print()

    if serial_results:
        avg_speedup = np.mean([r['speedup'] for r in serial_results if r['speedup'] > 0])

        print("1. 串行性能对比 (C++ vs PyTorch, PyTorch为baseline):")
        print(f"   - C++ Streaming Attention平均相对加速比: {avg_speedup:.2f}x")
        if avg_speedup > 1:
            print(f"   - C++实现平均比PyTorch快 {avg_speedup:.2f}x")
        else:
            print(f"   - PyTorch平均比C++快 {1/avg_speedup:.2f}x")
        print()

        # 分析原因
        if avg_speedup < 1:
            print("   PyTorch更快的原因:")
            print("   - PyTorch使用了高度优化的内核 (如oneDNN, MKL)")
            print("   - SIMD指令优化 (AVX512等)")
            print("   - 更好的内存访问模式")
            print()

    if scalability_results:
        print("2. OpenMP扩展性:")
        max_speedup = max(r['speedup'] for r in scalability_results)
        optimal_threads = scalability_results[
            scalability_results.index(max(scalability_results, key=lambda x: x['speedup']))
        ]['threads']

        print(f"   - 最大加速比: {max_speedup:.2f}x ({optimal_threads}线程)")
        print(f"   - 最优线程数: {optimal_threads}")

        # 效率分析
        high_efficiency = [r for r in scalability_results if r.get('efficiency', 0) > 0.7]
        if high_efficiency:
            print(f"   - 高效配置(>70%): {len(high_efficiency)}/{len(scalability_results)}")
        print()

    print("=" * 100)
    print("  结论")
    print("=" * 100)
    print()

    print("1. 单线程性能:")
    if serial_results and avg_speedup < 1:
        print("   ✓ PyTorch的SDPA在单线程下性能优于朴素C++实现")
        print("   - 原因: PyTorch使用了深度优化的内核库")
    elif serial_results:
        print("   ✓ C++实现与PyTorch性能相当或更优")
    else:
        print("   - C++测试失败，无法对比")

    print()
    print("2. 多线程扩展:")
    if scalability_results:
        print(f"   ✓ C++ OpenMP版本可以利用多核CPU")
        print(f"   - 在{optimal_threads}线程下达到最佳性能 ({max_speedup:.2f}x加速)")
    else:
        print("   - OpenMP测试未运行")

    print()
    print("3. 实际应用建议:")
    print("   - PyTorch: 快速原型开发和小规模推理")
    print("   - C++ + OpenMP: 大规模并行计算和教学演示")
    print("   - 对于课程报告: 重点分析OpenMP的扩展性和效率")

    print()
    print("=" * 100)


if __name__ == "__main__":
    run_comparison()
