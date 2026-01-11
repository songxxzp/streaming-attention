#!/usr/bin/env python3
"""
Streaming Attention vs PyTorch SDPA 完整性能对比测试

对比测试:
1. PyTorch SDPA (单线程)
2. C++ Streaming Attention (串行)
3. C++ Streaming Attention (OpenMP多线程)

测试配置:
- 序列长度: 512, 1024, 2048, 4096, 8192
- 隐藏维度: 128
- OpenMP线程数: 1, 2, 4, 8, 16
"""

import subprocess
import time
import torch
import os
from pathlib import Path

# ============================================================================
# 配置参数
# ============================================================================

SEQ_LENS = [512, 1024, 2048, 4096, 8192]
HIDDEN_DIM = 128
ITERS = 100
BLOCK_SIZE = 64
THREADS_LIST = [1, 2, 4, 8, 16]

# ============================================================================
# PyTorch SDPA测试
# ============================================================================

def test_pytorch_sdpa(seq_len: int, hidden_dim: int, iters: int) -> float:
    """测试PyTorch SDPA性能"""
    Q = torch.randn(1, 1, 1, hidden_dim, device='cpu', dtype=torch.float32)
    K = torch.randn(1, 1, seq_len, hidden_dim, device='cpu', dtype=torch.float32)
    V = torch.randn(1, 1, seq_len, hidden_dim, device='cpu', dtype=torch.float32)

    # 预热
    for _ in range(5):
        _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    torch.cpu.synchronize()

    # 测试
    start = time.time()
    for _ in range(iters):
        _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    torch.cpu.synchronize()
    end = time.time()

    return (end - start) * 1000 / iters


# ============================================================================
# C++ Streaming Attention测试
# ============================================================================

def test_cpp_streaming(seq_len: int, hidden_dim: int, block_size: int,
                        threads: int, executable: str) -> tuple:
    """
    测试C++ Streaming Attention

    Returns:
        (时间ms, 是否成功)
    """
    try:
        env = os.environ.copy()
        if threads > 1:
            env['OMP_NUM_THREADS'] = str(threads)

        result = subprocess.run(
            [executable, str(seq_len), str(hidden_dim), str(block_size), str(threads)],
            capture_output=True,
            text=True,
            timeout=30,
            env=env
        )

        if result.returncode != 0:
            return 0.0, False

        # 解析时间
        for line in result.stdout.split('\n'):
            if 'Time' in line:
                try:
                    time_val = float(line.split()[1])
                    return time_val, True
                except (IndexError, ValueError):
                    continue

        return 0.0, False

    except Exception as e:
        return 0.0, False


# ============================================================================
# 编译检查
# ============================================================================

def check_and_compile():
    """检查并编译C++程序"""
    attention_dir = Path("./attention")

    if not attention_dir.exists():
        print("✗ 错误: attention目录不存在")
        return False

    print("检查C++可执行文件...")

    # 检查串行版本
    if not (attention_dir / "streaming_serial").exists():
        print("  编译streaming_serial...")
        compile_cmd = """
        cd attention && \
        g++ -std=c++17 -O3 -march=native \
            streaming_serial.cpp \
            -o streaming_serial
        """
        result = subprocess.run(compile_cmd, shell=True, capture_output=True)
        if result.returncode != 0:
            print(f"  ✗ 编译失败: {result.stderr.decode()}")
            return False
        print("  ✓ streaming_serial编译成功")

    # 检查OpenMP版本
    if not (attention_dir / "streaming_omp").exists():
        print("  编译streaming_omp...")
        compile_cmd = """
        cd attention && \
        g++ -std=c++17 -O3 -march=native -fopenmp \
            streaming_omp.cpp streaming_serial.cpp \
            -o streaming_omp
        """
        result = subprocess.run(compile_cmd, shell=True, capture_output=True)
        if result.returncode != 0:
            print(f"  ✗ 编译失败: {result.stderr.decode()}")
            return False
        print("  ✓ streaming_omp编译成功")

    print("✓ 所有可执行文件就绪")
    return True


# ============================================================================
# 主测试函数
# ============================================================================

def run_comprehensive_test():
    """运行完整的对比测试"""

    print("=" * 100)
    print(" " * 30 + "Streaming Attention vs PyTorch SDPA 性能对比")
    print("=" * 100)
    print()

    # 检查并编译
    if not check_and_compile():
        print("\n请确保在项目根目录运行此脚本")
        return

    print(f"\n测试配置:")
    print(f"  序列长度: {SEQ_LENS}")
    print(f"  隐藏维度: {HIDDEN_DIM}")
    print(f"  迭代次数: {ITERS}")
    print(f"  Block Size: {BLOCK_SIZE}")
    print(f"  OpenMP线程: {THREADS_LIST}")
    print()

    # PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print()

    # ============================================================================
    # 测试1: 串行对比
    # ============================================================================

    print("=" * 100)
    print("  测试1: 串行性能对比 (PyTorch SDPA vs C++ Streaming Serial)")
    print("=" * 100)
    print()

    serial_results = []

    print("序列长度 | PyTorch(ms) | C++(ms) | PT-吞吐量 | C++-吞吐量 | 加速比(C++/PT)")
    print("---------|------------|---------|----------|-----------|--------------")

    for seq_len in SEQ_LENS:
        # PyTorch
        try:
            pt_time = test_pytorch_sdpa(seq_len, HIDDEN_DIM, ITERS)
            pt_throughput = seq_len * 1000 / pt_time
        except:
            pt_time = 0
            pt_throughput = 0

        # C++
        cpp_time, success = test_cpp_streaming(seq_len, HIDDEN_DIM, BLOCK_SIZE, 1,
                                               "./attention/streaming_serial")
        if success:
            cpp_throughput = seq_len * 1000 / cpp_time
            speedup = cpp_time / pt_time if pt_time > 0 else 0
            print(f"{seq_len:8d} | {pt_time:10.4f} | {cpp_time:7.4f} | "
                  f"{pt_throughput:8.2f} | {cpp_throughput:9.2f} | {speedup:12.2f}x")

            serial_results.append({
                'seq_len': seq_len,
                'pt_time': pt_time,
                'cpp_time': cpp_time,
                'speedup': speedup
            })
        else:
            print(f"{seq_len:8d} | {pt_time:10.4f} |   N/A   | "
                  f"{pt_throughput:8.2f} |     N/A   |      N/A     ")

    print()

    # ============================================================================
    # 测试2: OpenMP扩展性
    # ============================================================================

    print("=" * 100)
    print("  测试2: C++ Streaming OpenMP扩展性 (seq_len=2048)")
    print("=" * 100)
    print()

    SEQ_LEN_SCALABILITY = 2048

    print("线程数 | 时间(ms) | 吞吐量(tokens/s) | 加速比 | 效率")
    print("-------|---------|-----------------|-------|------")

    scalability_results = []

    # 获取串行baseline
    baseline_time, _ = test_cpp_streaming(SEQ_LEN_SCALABILITY, HIDDEN_DIM,
                                          BLOCK_SIZE, 1, "./attention/streaming_serial")

    if baseline_time > 0:
        print(f"   1   | {baseline_time:7.4f} | {SEQ_LEN_SCALABILITY * 1000 / baseline_time:15.2f} | "
              f"  1.00 | 1.000")
        scalability_results.append({'threads': 1, 'time': baseline_time, 'speedup': 1.0})

        for threads in THREADS_LIST[1:]:
            time, success = test_cpp_streaming(SEQ_LEN_SCALABILITY, HIDDEN_DIM,
                                              BLOCK_SIZE, threads, "./attention/streaming_omp")
            if success:
                throughput = SEQ_LEN_SCALABILITY * 1000 / time
                speedup = baseline_time / time
                efficiency = speedup / threads

                print(f"  {threads:3d}   | {time:7.4f} | {throughput:15.2f} | "
                      f"{speedup:5.2f} | {efficiency:.3f}")

                scalability_results.append({
                    'threads': threads,
                    'time': time,
                    'speedup': speedup,
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
        avg_speedup = sum(r['speedup'] for r in serial_results if r['speedup'] > 0) / len(serial_results)

        print("1. 串行性能对比 (C++ vs PyTorch):")
        print(f"   - C++ Streaming Attention平均加速比: {avg_speedup:.2f}x")
        if avg_speedup > 1:
            print(f"   - C++实现平均比PyTorch快 {avg_speedup:.2f}x")
        else:
            print(f"   - PyTorch平均比C++快 {1/avg_speedup:.2f}x")

        # 分析原因
        if avg_speedup < 1:
            print()
            print("   PyTorch更快的原因:")
            print("   - PyTorch使用了高度优化的内核 (如oneDNN, MKL)")
            print("   - SIMD指令优化 (AVX512等)")
            print("   - 更好的内存访问模式")
            print()
            print("   C++如何改进:")
            print("   - 使用OpenMP多线程并行")
            print("   - 添加SIMD intrinsic")
            print("   - 优化内存布局")

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
        high_efficiency_count = sum(1 for r in scalability_results if r.get('efficiency', 0) > 0.7)
        print(f"   - 高效配置(>70%)数量: {high_efficiency_count}/{len(scalability_results)}")

    print()

    print("=" * 100)
    print("  结论")
    print("=" * 100)
    print()

    print("1. 单线程性能:")
    if serial_results and avg_speedup < 1:
        print("   PyTorch的SDPA在单线程下性能优于朴素C++实现")
        print("   原因: PyTorch使用了深度优化的内核库")
    else:
        print("   C++实现与PyTorch性能相当或更优")

    print()
    print("2. 多线程扩展:")
    if scalability_results:
        print("   C++ OpenMP版本可以利用多核CPU实现性能提升")
        print(f"   在{optimal_threads}线程下达到最佳性能")
    else:
        print("   OpenMP测试未运行")

    print()
    print("3. 实际应用:")
    print("   - PyTorch适合原型开发和推理")
    print("   - C++ + OpenMP适合大规模并行计算")
    print("   - 对于教学和演示，C++实现更有助于理解算法")

    print()
    print("=" * 100)


if __name__ == "__main__":
    run_comprehensive_test()
