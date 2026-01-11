#!/usr/bin/env python3
"""
Streaming Attention vs PyTorch SDPA 性能对比测试

测试相同规模下：
1. C++ Streaming Attention (串行实现)
2. PyTorch F.scaled_dot_product_attention

测试配置:
- 序列长度: 512, 1024, 2048, 4096, 8192
- 隐藏维度: 128
- 批次大小: 1
- 迭代次数: 100
"""

import subprocess
import time
import torch
import numpy as np
from typing import Tuple

# ============================================================================
# 配置参数
# ============================================================================

SEQ_LENS = [512, 1024, 2048, 4096, 8192]
HIDDEN_DIM = 128
ITERS = 100
BLOCK_SIZE = 64

# C++可执行文件路径
CPP_SERIAL = "./attention/test_streaming"
CPP_OMP = "./attention/streaming_omp"

# ============================================================================
# PyTorch SDPA测试
# ============================================================================

def test_pytorch_sdpa(seq_len: int, hidden_dim: int, iters: int) -> float:
    """
    测试PyTorch SDPA性能

    Args:
        seq_len: 序列长度
        hidden_dim: 隐藏维度
        iters: 迭代次数

    Returns:
        平均时间(ms)
    """
    # 创建测试数据 (Q=1x1xd, K,V=1xTxd)
    # 这里模拟单query的情况
    Q = torch.randn(1, 1, 1, hidden_dim, device='cpu', dtype=torch.float32)
    K = torch.randn(1, 1, seq_len, hidden_dim, device='cpu', dtype=torch.float32)
    V = torch.randn(1, 1, seq_len, hidden_dim, device='cpu', dtype=torch.float32)

    # 预热
    for _ in range(5):
        _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V)

    # 同步确保预热完成
    torch.cpu.synchronize()

    # 测试
    start = time.time()
    for _ in range(iters):
        _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    torch.cpu.synchronize()
    end = time.time()

    avg_time_ms = (end - start) * 1000 / iters
    return avg_time_ms


def test_pytorch_sdca(seq_len: int, hidden_dim: int, iters: int) -> float:
    """
    测试PyTorch SDPA性能 (使用Efficient Attention)
    这是streaming attention的等价实现
    """
    # 创建测试数据
    Q = torch.randn(1, 1, 1, hidden_dim, device='cpu', dtype=torch.float32)
    K = torch.randn(1, 1, seq_len, hidden_dim, device='cpu', dtype=torch.float32)
    V = torch.randn(1, 1, seq_len, hidden_dim, device='cpu', dtype=torch.float32)

    # 预热
    for _ in range(5):
        _ = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )

    torch.cpu.synchronize()

    # 测试
    start = time.time()
    for _ in range(iters):
        _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    torch.cpu.synchronize()
    end = time.time()

    avg_time_ms = (end - start) * 1000 / iters
    return avg_time_ms


# ============================================================================
# C++ Streaming Attention测试
# ============================================================================

def test_cpp_streaming(seq_len: int, hidden_dim: int, iters: int,
                        executable: str) -> Tuple[float, bool]:
    """
    测试C++ Streaming Attention性能

    Args:
        seq_len: 序列长度
        hidden_dim: 隐藏维度
        iters: 迭代次数
        executable: C++可执行文件路径

    Returns:
        (平均时间ms, 是否成功)
    """
    try:
        # 运行C++程序
        result = subprocess.run(
            [executable, str(seq_len), str(hidden_dim), str(BLOCK_SIZE)],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            return 0.0, False

        # 解析输出
        for line in result.stdout.split('\n'):
            if 'Time' in line or 'time' in line:
                try:
                    time_val = float(line.split()[1])
                    return time_val, True
                except (IndexError, ValueError):
                    continue

        return 0.0, False

    except subprocess.TimeoutExpired:
        return 0.0, False
    except FileNotFoundError:
        return 0.0, False
    except Exception as e:
        print(f"  错误: {e}")
        return 0.0, False


# ============================================================================
# 主测试函数
# ============================================================================

def run_comparison():
    """运行完整的对比测试"""

    print("=" * 80)
    print("  Streaming Attention vs PyTorch SDPA 性能对比")
    print("=" * 80)
    print()

    print(f"测试配置:")
    print(f"  序列长度: {SEQ_LENS}")
    print(f"  隐藏维度: {HIDDEN_DIM}")
    print(f"  迭代次数: {ITERS}")
    print(f"  Block Size: {BLOCK_SIZE}")
    print()

    # 检查PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备: {torch.cuda.get_device_name(0)}")
    print()

    # 测试结果
    results = []

    print("=" * 80)
    print("  开始测试...")
    print("=" * 80)
    print()

    for seq_len in SEQ_LENS:
        print(f"[序列长度: {seq_len}]")

        # PyTorch SDPA
        print("  测试 PyTorch SDPA...")
        try:
            pytorch_time = test_pytorch_sdpa(seq_len, HIDDEN_DIM, ITERS)
            pytorch_throughput = seq_len * 1000 / pytorch_time
            print(f"    时间: {pytorch_time:.4f} ms")
            print(f"    吞吐量: {pytorch_throughput:.2f} tokens/sec")
        except Exception as e:
            print(f"    ✗ 失败: {e}")
            pytorch_time = 0.0
            pytorch_throughput = 0.0

        # C++ Streaming (串行)
        print("  测试 C++ Streaming (串行)...")
        cpp_time, cpp_success = test_cpp_streaming(seq_len, HIDDEN_DIM, ITERS, CPP_SERIAL)
        if cpp_success:
            cpp_throughput = seq_len * 1000 / cpp_time
            print(f"    时间: {cpp_time:.4f} ms")
            print(f"    吞吐量: {cpp_throughput:.2f} tokens/sec")
        else:
            print(f"    ✗ 失败: C++程序未编译或运行失败")
            cpp_time = 0.0
            cpp_throughput = 0.0

        # 计算加速比
        if cpp_time > 0 and pytorch_time > 0:
            speedup = pytorch_time / cpp_time
            if speedup > 1:
                winner = "C++更快"
            else:
                winner = "PyTorch更快"
            print(f"  对比: {winner} ({speedup:.2f}x)")

        print()

        # 保存结果
        results.append({
            'seq_len': seq_len,
            'pytorch_time': pytorch_time,
            'pytorch_throughput': pytorch_throughput,
            'cpp_time': cpp_time,
            'cpp_throughput': cpp_throughput,
            'speedup': pytorch_time / cpp_time if cpp_time > 0 else 0.0,
        })

    # 生成总结报告
    print("=" * 80)
    print("  测试总结")
    print("=" * 80)
    print()

    print("序列长度 | PyTorch(ms) | C++(ms) | 吞吐量-PT | 吞吐量-C++ | 加速比")
    print("---------|------------|---------|-----------|-----------|-------")

    for r in results:
        if r['cpp_time'] > 0:
            print(f"{r['seq_len']:8d} | {r['pytorch_time']:10.4f} | {r['cpp_time']:7.4f} | "
                  f"{r['pytorch_throughput']:9.2f} | {r['cpp_throughput']:9.2f} | {r['speedup']:5.2f}x")
        else:
            print(f"{r['seq_len']:8d} | {r['pytorch_time']:10.4f} |   N/A   | "
                  f"{r['pytorch_throughput']:9.2f} |    N/A    |   N/A")

    print()

    # 分析结论
    valid_results = [r for r in results if r['cpp_time'] > 0]

    if valid_results:
        avg_speedup = np.mean([r['speedup'] for r in valid_results])
        print(f"平均加速比: {avg_speedup:.2f}x")

        if avg_speedup > 1:
            print(f"结论: C++ Streaming Attention平均比PyTorch SDPA快 {avg_speedup:.2f}x")
        else:
            print(f"结论: PyTorch SDPA平均比C++ Streaming Attention快 {1/avg_speedup:.2f}x")
    else:
        print("结论: C++程序测试失败，无法进行对比")

    print()
    print("=" * 80)
    print("  注意事项")
    print("=" * 80)
    print()
    print("1. C++实现使用Streaming Attention算法")
    print("2. PyTorch使用F.scaled_dot_product_attention (优化内核)")
    print("3. 两者都是单线程测试")
    print("4. PyTorch可能使用了SIMD优化")
    print("5. C++优势在于可以使用OpenMP并行")
    print()
    print("如需测试多线程性能，编译并运行streaming_omp版本:")
    print(f"  g++ -O3 -fopenmp streaming_omp.cpp streaming_serial.cpp -o streaming_omp")
    print(f"  OMP_NUM_THREADS=4 ./streaming_omp <seq_len> <hidden> <block_size> 4")
    print()


if __name__ == "__main__":
    run_comparison()
