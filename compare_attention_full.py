#!/usr/bin/env python3
"""
Attention性能对比测试 - 完整版

测试相同规模下：
1. PyTorch F.scaled_dot_product_attention
2. C++ Naive Attention (串行)
3. C++ Naive Attention (OpenMP多线程)
4. C++ Streaming Attention (串行)
5. C++ Streaming Attention (OpenMP多线程)
"""

import subprocess
import time
import os
import argparse
from pathlib import Path
from typing import Tuple, List, Optional

# 尝试导入torch和numpy
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("警告: PyTorch未安装，将跳过PyTorch相关测试")

# 不再需要numpy，使用Python内置方法

# ============================================================================
# 默认配置参数
# ============================================================================

DEFAULT_SEQ_LENS = [1024, 2048, 4096, 8192]
DEFAULT_HIDDEN_DIM = 128
DEFAULT_WARMUP_RUNS = 2
DEFAULT_TEST_RUNS = 10
DEFAULT_BLOCK_SIZES = [64, 128]
DEFAULT_THREADS_LIST = [1, 2, 4, 8, 16, 32, 64]
DEFAULT_SEQ_LEN_SCALES = [1024, 8192, 65536]
DEFAULT_BASELINE = "serial"
DEFAULT_MPI_RANKS_LIST = [1, 2, 4, 8]
DEFAULT_MPI_OMP_THREADS_LIST = [16, 32, 64]

# C++可执行文件默认路径
DEFAULT_CPP_NAIVE_SERIAL = "./attention/test_naive"
DEFAULT_CPP_NAIVE_OMP = "./attention/test_naive_omp"
DEFAULT_CPP_STREAMING_SERIAL = "./attention/test_streaming"
DEFAULT_CPP_STREAMING_OMP = "./attention/test_streaming_omp"
DEFAULT_CPP_NAIVE_MPI = "./attention/test_naive_mpi"
DEFAULT_CPP_STREAMING_MPI = "./attention/test_streaming_mpi"
DEFAULT_MPIRUN = "mpirun"

# ============================================================================
# PyTorch SDPA测试
# ============================================================================

def test_pytorch_sdpa(seq_len: int, hidden_dim: int, warmup: int, repeat: int) -> float:
    """测试PyTorch SDPA性能"""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch未安装，无法运行PyTorch测试")

    Q = torch.randn(1, 1, 1, hidden_dim, device='cpu', dtype=torch.float32)
    K = torch.randn(1, 1, seq_len, hidden_dim, device='cpu', dtype=torch.float32)
    V = torch.randn(1, 1, seq_len, hidden_dim, device='cpu', dtype=torch.float32)

    # 预热
    for _ in range(warmup):
        _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    torch.cpu.synchronize()

    # 测试
    times = []
    for _ in range(repeat):
        start = time.time()
        _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        torch.cpu.synchronize()
        end = time.time()
        times.append((end - start) * 1000)

    # 使用Python内置方法计算平均值
    avg_time_ms = sum(times) / len(times)
    return avg_time_ms

# ============================================================================
# C++ Attention测试
# ============================================================================

def compile_cpp_executables():
    """编译C++可执行文件（使用compile_attention.sh脚本）"""
    script_path = Path("./scripts/compile_attention.sh")

    if not script_path.exists():
        print("✗ 错误: compile_attention.sh脚本不存在")
        print("  请确保在项目根目录运行此脚本")
        return False

    print("检查并编译C++可执行文件...")

    result = subprocess.run(
        ["bash", str(script_path)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("✗ 编译失败:")
        print(result.stderr)
        return False

    # 显示编译输出（去掉颜色代码以保持整洁）
    for line in result.stdout.split('\n'):
        if line.strip():
            # 移除ANSI颜色代码
            clean_line = line
            clean_line = clean_line.replace('\033[0;31m', '').replace('\033[0;32m', '')
            clean_line = clean_line.replace('\033[0;33m', '').replace('\033[0;34m', '')
            clean_line = clean_line.replace('\033[0m', '')
            print(clean_line)

    return True


def test_cpp_attention(seq_len: int, hidden_dim: int, block_size: int,
                        threads: int, executable: str, repeat: int) -> Tuple[float, bool]:
    """测试C++ Attention性能"""
    try:
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = str(threads)

        times = []
        for run in range(repeat):
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
            # 使用Python内置方法计算平均值
            return sum(times) / len(times), True
        return 0.0, False

    except subprocess.TimeoutExpired:
        return 0.0, False
    except FileNotFoundError:
        return 0.0, False
    except Exception as e:
        return 0.0, False


def test_cpp_mpi_attention(seq_len: int, hidden_dim: int, block_size: int,
                           mpi_ranks: int, omp_threads: int,
                           executable: str, mpirun: str, repeat: int,
                           needs_block_size: bool = True) -> Tuple[float, bool]:
    """测试C++ MPI Attention性能

    Args:
        needs_block_size: True for streaming MPI, False for naive MPI
    """
    try:
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = str(omp_threads)

        times = []
        for run in range(repeat):
            # 根据类型构建命令行参数
            if needs_block_size:
                # Streaming MPI: <T_global> <d> <block_size> <num_omp_threads>
                cmd = [mpirun, "-np", str(mpi_ranks), executable,
                       str(seq_len), str(hidden_dim), str(block_size), str(omp_threads)]
            else:
                # Naive MPI: <T_global> <d> <num_omp_threads>
                cmd = [mpirun, "-np", str(mpi_ranks), executable,
                       str(seq_len), str(hidden_dim), str(omp_threads)]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                env=env
            )

            if result.returncode != 0:
                return 0.0, False

            # 解析输出
            for line in result.stdout.split('\n'):
                if 'Time' in line and 'ms' in line:
                    try:
                        time_val = float(line.split()[1])
                        times.append(time_val)
                        break
                    except (IndexError, ValueError):
                        continue

        if times:
            return sum(times) / len(times), True
        return 0.0, False

    except subprocess.TimeoutExpired:
        return 0.0, False
    except FileNotFoundError:
        return 0.0, False
    except Exception as e:
        return 0.0, False

# ============================================================================
# 参数解析
# ============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Attention性能对比测试 - Naive vs Streaming vs PyTorch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--baseline",
        type=str,
        choices=["torch", "serial"],
        default=DEFAULT_BASELINE,
        help="选择baseline: torch使用PyTorch SDPA, serial使用C++串行版本"
    )

    parser.add_argument(
        "--seqlen",
        type=int,
        nargs="+",
        default=DEFAULT_SEQ_LENS,
        help=f"序列长度列表 (默认: {DEFAULT_SEQ_LENS})"
    )

    parser.add_argument(
        "--dim",
        type=int,
        default=DEFAULT_HIDDEN_DIM,
        help=f"隐藏维度 (默认: {DEFAULT_HIDDEN_DIM})"
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP_RUNS,
        help=f"预热次数 (默认: {DEFAULT_WARMUP_RUNS})"
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=DEFAULT_TEST_RUNS,
        help=f"测试次数 (默认: {DEFAULT_TEST_RUNS})"
    )

    parser.add_argument(
        "--block-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_BLOCK_SIZES,
        help=f"Streaming block size列表 (默认: {DEFAULT_BLOCK_SIZES})"
    )

    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=DEFAULT_THREADS_LIST,
        help=f"OpenMP线程数列表 (默认: {DEFAULT_THREADS_LIST})"
    )

    parser.add_argument(
        "--seqlen-scales",
        type=int,
        nargs="+",
        default=DEFAULT_SEQ_LEN_SCALES,
        help=f"扩展性测试的序列长度列表 (默认: {DEFAULT_SEQ_LEN_SCALES})"
    )

    parser.add_argument(
        "--cpp-naive-serial",
        type=str,
        default=DEFAULT_CPP_NAIVE_SERIAL,
        help=f"C++ Naive串行版本可执行文件路径 (默认: {DEFAULT_CPP_NAIVE_SERIAL})"
    )

    parser.add_argument(
        "--cpp-naive-omp",
        type=str,
        default=DEFAULT_CPP_NAIVE_OMP,
        help=f"C++ Naive OpenMP版本可执行文件路径 (默认: {DEFAULT_CPP_NAIVE_OMP})"
    )

    parser.add_argument(
        "--cpp-streaming-serial",
        type=str,
        default=DEFAULT_CPP_STREAMING_SERIAL,
        help=f"C++ Streaming串行版本可执行文件路径 (默认: {DEFAULT_CPP_STREAMING_SERIAL})"
    )

    parser.add_argument(
        "--cpp-streaming-omp",
        type=str,
        default=DEFAULT_CPP_STREAMING_OMP,
        help=f"C++ Streaming OpenMP版本可执行文件路径 (默认: {DEFAULT_CPP_STREAMING_OMP})"
    )

    parser.add_argument(
        "--cpp-naive-mpi",
        type=str,
        default=DEFAULT_CPP_NAIVE_MPI,
        help=f"C++ Naive MPI版本可执行文件路径 (默认: {DEFAULT_CPP_NAIVE_MPI})"
    )

    parser.add_argument(
        "--cpp-streaming-mpi",
        type=str,
        default=DEFAULT_CPP_STREAMING_MPI,
        help=f"C++ Streaming MPI版本可执行文件路径 (默认: {DEFAULT_CPP_STREAMING_MPI})"
    )

    parser.add_argument(
        "--mpirun",
        type=str,
        default=DEFAULT_MPIRUN,
        help=f"MPI运行程序路径 (默认: {DEFAULT_MPIRUN})"
    )

    parser.add_argument(
        "--mpi-ranks",
        type=int,
        nargs="+",
        default=DEFAULT_MPI_RANKS_LIST,
        help=f"MPI进程数列表 (默认: {DEFAULT_MPI_RANKS_LIST})"
    )

    parser.add_argument(
        "--mpi-omp-threads",
        type=int,
        nargs="+",
        default=DEFAULT_MPI_OMP_THREADS_LIST,
        help=f"每个MPI进程的OpenMP线程数列表 (默认: {DEFAULT_MPI_OMP_THREADS_LIST})"
    )

    parser.add_argument(
        "--enable-mpi",
        action="store_true",
        help="启用MPI测试"
    )

    return parser.parse_args()

# ============================================================================
# 主测试函数
# ============================================================================

def run_comparison(args):
    """运行完整的对比测试"""

    print("=" * 100)
    print(" " * 25 + "Attention性能对比 - Naive vs Streaming vs PyTorch")
    print("=" * 100)
    print()

    print(f"测试配置:")
    print(f"  序列长度: {args.seqlen}")
    print(f"  隐藏维度: {args.dim}")
    print(f"  预热次数: {args.warmup}")
    print(f"  测试次数: {args.repeat}")
    print(f"  Block Sizes: {args.block_sizes}")
    print(f"  OpenMP线程: {args.threads}")
    print(f"  扩展性测试序列长度: {args.seqlen_scales}")
    print(f"  Baseline: {args.baseline}")
    print()

    if HAS_TORCH:
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
    else:
        print("PyTorch: 未安装（跳过PyTorch测试）")
    print()

    # 编译C++程序
    if not compile_cpp_executables():
        print("\n请确保在项目根目录运行此脚本，并且attention目录存在")
        return

    # ============================================================================
    # 测试1: 串行性能对比 (C++ Naive Serial vs C++ Streaming Serial)
    # ============================================================================
    print("=" * 100)
    print("  测试1: 串行性能对比")
    print("=" * 100)
    print()

    # 根据是否有PyTorch动态生成表头
    if HAS_TORCH:
        print("序列长度 | Block Size | PyTorch(ms) | Naive(ms) | Streaming(ms) | PT-吞吐 | Naive-吞吐 | Streaming-吞吐")
        print("---------|------------|------------|-----------|---------------|---------|-----------|--------------")
    else:
        print("序列长度 | Block Size | Naive(ms) | Streaming(ms) | Naive-吞吐 | Streaming-吞吐")
        print("---------|------------|-----------|---------------|-----------|--------------")

    for seq_len in args.seqlen:
        # PyTorch SDPA (如果可用)
        pytorch_time = None
        pytorch_throughput = None
        if HAS_TORCH:
            try:
                pytorch_time = test_pytorch_sdpa(seq_len, args.dim, args.warmup, args.repeat)
                pytorch_throughput = seq_len * 1000 / pytorch_time
            except Exception as e:
                print(f"  ✗ PyTorch测试失败: {e}")

        # 对每个block_size测试streaming
        for block_size in args.block_sizes:
            # C++ Naive Serial (naive不需要block_size，只测试一次)
            if block_size == args.block_sizes[0]:
                naive_time, naive_success = test_cpp_attention(seq_len, args.dim, block_size, 1, args.cpp_naive_serial, args.repeat)
            else:
                # 后续block_size复用naive结果
                pass

            # C++ Streaming Serial
            streaming_time, streaming_success = test_cpp_attention(seq_len, args.dim, block_size, 1, args.cpp_streaming_serial, args.repeat)

            # 根据是否有PyTorch选择输出格式
            if HAS_TORCH and pytorch_time is not None:
                if naive_success and streaming_success:
                    naive_throughput = seq_len * 1000 / naive_time
                    streaming_throughput = seq_len * 1000 / streaming_time
                    print(f"{seq_len:8d} | {block_size:10d} | {pytorch_time:11.4f} | {naive_time:10.4f} | {streaming_time:14.4f} | "
                          f"{pytorch_throughput:8.2f} | {naive_throughput:10.2f} | {streaming_throughput:13.2f}")
                elif naive_success:
                    naive_throughput = seq_len * 1000 / naive_time
                    print(f"{seq_len:8d} | {block_size:10d} | {pytorch_time:11.4f} | {naive_time:10.4f} |      N/A       | "
                          f"{pytorch_throughput:8.2f} | {naive_throughput:10.2f} |      N/A      ")
                elif streaming_success:
                    streaming_throughput = seq_len * 1000 / streaming_time
                    print(f"{seq_len:8d} | {block_size:10d} | {pytorch_time:11.4f} |     N/A    | {streaming_time:14.4f} | "
                          f"{pytorch_throughput:8.2f} |     N/A    | {streaming_throughput:13.2f}")
                else:
                    print(f"{seq_len:8d} | {block_size:10d} | {pytorch_time:11.4f} |     N/A    |      N/A       | "
                          f"{pytorch_throughput:8.2f} |     N/A    |      N/A      ")
            else:
                # 不包含PyTorch的输出
                if naive_success and streaming_success:
                    naive_throughput = seq_len * 1000 / naive_time
                    streaming_throughput = seq_len * 1000 / streaming_time
                    print(f"{seq_len:8d} | {block_size:10d} | {naive_time:10.4f} | {streaming_time:14.4f} | {naive_throughput:10.2f} | {streaming_throughput:13.2f}")
                elif naive_success:
                    naive_throughput = seq_len * 1000 / naive_time
                    print(f"{seq_len:8d} | {block_size:10d} | {naive_time:10.4f} |      N/A       | {naive_throughput:10.2f} |      N/A      ")
                elif streaming_success:
                    streaming_throughput = seq_len * 1000 / streaming_time
                    print(f"{seq_len:8d} | {block_size:10d} |     N/A    | {streaming_time:14.4f} |     N/A    | {streaming_throughput:13.2f}")
                else:
                    print(f"{seq_len:8d} | {block_size:10d} |     N/A    |      N/A       |     N/A    |      N/A      ")

    print()

    # ============================================================================
    # 测试2: Naive OpenMP扩展性
    # ============================================================================
    # 使用第一个block_size
    block_size = args.block_sizes[0]

    for seq_len_scale in args.seqlen_scales:
        print("=" * 100)
        if HAS_TORCH and args.baseline == "torch":
            print(f"  测试2: Naive Attention OpenMP扩展性 (seq_len={seq_len_scale}, 包含PyTorch对比)")
        else:
            print(f"  测试2: Naive Attention OpenMP扩展性 (seq_len={seq_len_scale})")
        print("=" * 100)
        print()

        print("实现方式       | 线程数 | 时间(ms) | 吞吐量(tokens/s) | 相对Baseline加速 | 相对串行加速 | 效率")
        print("---------------|--------|---------|-----------------|----------------|-------------|------")

        # 根据baseline选择获取哪个baseline时间
        baseline_time = None
        baseline_name = ""

        if args.baseline == "torch" and HAS_TORCH:
            try:
                baseline_time = test_pytorch_sdpa(seq_len_scale, args.dim, args.warmup, args.repeat)
                baseline_name = "PyTorch SDPA"
                baseline_throughput = seq_len_scale * 1000 / baseline_time
                print(f"{'PyTorch SDPA':15s} |   {'N/A':3s} | {baseline_time:7.4f} | {baseline_throughput:15.2f} | "
                      f"{'  1.00':14s} | {'  N/A':11s} |  N/A")
            except Exception as e:
                print(f"  ✗ PyTorch测试失败: {e}")

        # 获取C++ Naive串行baseline
        naive_baseline_time, _ = test_cpp_attention(seq_len_scale, args.dim, block_size, 1, args.cpp_naive_serial, args.repeat)

        if naive_baseline_time > 0:
            naive_baseline_throughput = seq_len_scale * 1000 / naive_baseline_time
            if baseline_time and args.baseline == "torch":
                speedup_vs_baseline = baseline_time / naive_baseline_time
                print(f"{'Naive Serial':15s} |     {1:3d}   | {naive_baseline_time:7.4f} | {naive_baseline_throughput:15.2f} | "
                      f"{speedup_vs_baseline:14.2f}x | {1.00:11.2f}x | 1.000")
            else:
                print(f"{'Naive Serial':15s} |     {1:3d}   | {naive_baseline_time:7.4f} | {naive_baseline_throughput:15.2f} | "
                      f"{'  N/A':14s} | {1.00:11.2f}x | 1.000")

            for threads in args.threads[1:]:
                time, success = test_cpp_attention(seq_len_scale, args.dim, block_size, threads, args.cpp_naive_omp, args.repeat)
                if success:
                    throughput = seq_len_scale * 1000 / time
                    speedup_vs_serial = naive_baseline_time / time
                    efficiency = speedup_vs_serial / threads

                    if baseline_time and args.baseline == "torch":
                        speedup_vs_baseline = baseline_time / time
                        print(f"{'Naive OpenMP':15s} |    {threads:3d}   | {time:7.4f} | {throughput:15.2f} | "
                              f"{speedup_vs_baseline:14.2f}x | {speedup_vs_serial:11.2f}x | {efficiency:.3f}")
                    else:
                        print(f"{'Naive OpenMP':15s} |    {threads:3d}   | {time:7.4f} | {throughput:15.2f} | "
                              f"{'  N/A':14s} | {speedup_vs_serial:11.2f}x | {efficiency:.3f}")

        print()

    # ============================================================================
    # 测试3: Streaming OpenMP扩展性 (测试不同block_size)
    # ============================================================================

    for seq_len_scale in args.seqlen_scales:
        print("=" * 100)
        if HAS_TORCH and args.baseline == "torch":
            print(f"  测试3: Streaming Attention OpenMP扩展性 (seq_len={seq_len_scale}, 包含PyTorch对比, 测试不同block_size)")
        else:
            print(f"  测试3: Streaming Attention OpenMP扩展性 (seq_len={seq_len_scale}, 测试不同block_size)")
        print("=" * 100)
        print()

        print("实现方式           | 线程数 | Block Size | 时间 | 吞吐量 | 相对Baseline加速 | 相对串行加速 | 效率")
        print("-------------------|--------|------------|---------|-----------------|----------------|-------------|------")

        # 获取这个序列长度的PyTorch baseline
        baseline_time = None
        if args.baseline == "torch" and HAS_TORCH:
            try:
                baseline_time = test_pytorch_sdpa(seq_len_scale, args.dim, args.warmup, args.repeat)
                baseline_throughput = seq_len_scale * 1000 / baseline_time
                print(f"{'PyTorch SDPA':19s} |   {'N/A':3s} |     {'N/A':8s} | {baseline_time:7.4f} | {baseline_throughput:15.2f} | "
                      f"{'  1.00':14s} | {'  N/A':11s} |  N/A")
            except Exception as e:
                print(f"  ✗ PyTorch测试失败: {e}")

        # 对每个block_size测试streaming attention
        for block_size in args.block_sizes:
            # 获取C++ Streaming串行baseline (只对第一个block_size测试)
            if block_size == args.block_sizes[0]:
                streaming_baseline_time, _ = test_cpp_attention(seq_len_scale, args.dim, block_size, 1, args.cpp_streaming_serial, args.repeat)
                if streaming_baseline_time > 0:
                    streaming_baseline_throughput = seq_len_scale * 1000 / streaming_baseline_time
                    if baseline_time and args.baseline == "torch":
                        speedup_vs_baseline = baseline_time / streaming_baseline_time
                        print(f"{'Streaming Serial':19s} |     {1:3d}   | {block_size:10d} | {streaming_baseline_time:7.4f} | {streaming_baseline_throughput:15.2f} | "
                              f"{speedup_vs_baseline:14.2f}x | {1.00:11.2f}x | 1.000")
                    else:
                        print(f"{'Streaming Serial':19s} |     {1:3d}   | {block_size:10d} | {streaming_baseline_time:7.4f} | {streaming_baseline_throughput:15.2f} | "
                              f"{'  N/A':14s} | {1.00:11.2f}x | 1.000")

            # 对每个线程数测试
            for threads in args.threads[1:]:
                time, success = test_cpp_attention(seq_len_scale, args.dim, block_size, threads, args.cpp_streaming_omp, args.repeat)
                if success:
                    throughput = seq_len_scale * 1000 / time
                    # 相对于串行baseline的加速比
                    speedup_vs_serial = streaming_baseline_time / time
                    efficiency = speedup_vs_serial / threads

                    if baseline_time and args.baseline == "torch":
                        speedup_vs_baseline = baseline_time / time
                        print(f"{'Streaming OpenMP':19s} |    {threads:3d}   | {block_size:10d} | {time:7.4f} | {throughput:15.2f} | "
                              f"{speedup_vs_baseline:14.2f}x | {speedup_vs_serial:11.2f}x | {efficiency:.3f}")
                    else:
                        print(f"{'Streaming OpenMP':19s} |    {threads:3d}   | {block_size:10d} | {time:7.4f} | {throughput:15.2f} | "
                              f"{'  N/A':14s} | {speedup_vs_serial:11.2f}x | {efficiency:.3f}")

        print()

    # ============================================================================
    # MPI测试 (如果启用)
    # ============================================================================
    if args.enable_mpi:
        # 使用第一个block_size
        block_size = args.block_sizes[0]

        # ========================================================================
        # 测试4: Naive MPI扩展性
        # ========================================================================
        for seq_len_scale in args.seqlen_scales:
            print("=" * 100)
            print(f"  测试4: Naive Attention MPI扩展性 (seq_len={seq_len_scale})")
            print("=" * 100)
            print()

            print("MPI配置         | MPI Ranks | OMP/Rank | 时间 | 吞吐量 | 相对串行加速 | 效率")
            print("----------------|-----------|----------|---------|-----------------|-------------|------")

            # 获取串行baseline
            naive_serial_time, _ = test_cpp_attention(seq_len_scale, args.dim, block_size, 1, args.cpp_naive_serial, 1)

            if naive_serial_time > 0:
                naive_serial_throughput = seq_len_scale * 1000 / naive_serial_time
                print(f"{'Naive Serial':16s} |     {1:3d}   |    {1:4d}  | {naive_serial_time:7.4f} | {naive_serial_throughput:15.2f} | {1.00:11.2f}x | 1.000")

                # 测试不同的MPI ranks × OMP threads组合
                for mpi_ranks in args.mpi_ranks:
                    for omp_threads in args.mpi_omp_threads:
                        time, success = test_cpp_mpi_attention(
                            seq_len_scale, args.dim, block_size,
                            mpi_ranks, omp_threads,
                            args.cpp_naive_mpi, args.mpirun, 1,
                            needs_block_size=False  # Naive MPI不需要block_size
                        )
                        if success:
                            throughput = seq_len_scale * 1000 / time
                            total_threads = mpi_ranks * omp_threads
                            speedup_vs_serial = naive_serial_time / time
                            efficiency = speedup_vs_serial / total_threads
                            print(f"{'Naive MPI+OMP':16s} |    {mpi_ranks:3d}   |   {omp_threads:4d}  | {time:7.4f} | {throughput:15.2f} | {speedup_vs_serial:11.2f}x | {efficiency:.3f}")

            print()

        # ========================================================================
        # 测试5: Streaming MPI扩展性 (测试不同block_size)
        # ========================================================================
        for seq_len_scale in args.seqlen_scales:
            print("=" * 100)
            print(f"  测试5: Streaming Attention MPI扩展性 (seq_len={seq_len_scale}, 测试不同block_size)")
            print("=" * 100)
            print()

            print("MPI配置            | MPI Ranks | OMP/Rank | Block Size | 时间 | 吞吐量 | 相对串行加速 | 效率")
            print("-------------------|-----------|----------|------------|---------|-----------------|-------------|------")

            # 对每个block_size测试streaming MPI
            for block_size in args.block_sizes:
                # 获取串行baseline (只对第一个block_size测试)
                if block_size == args.block_sizes[0]:
                    streaming_serial_time, _ = test_cpp_attention(seq_len_scale, args.dim, block_size, 1, args.cpp_streaming_serial, 1)

                if streaming_serial_time > 0:
                    # 只对第一个block_size显示串行baseline
                    if block_size == args.block_sizes[0]:
                        streaming_serial_throughput = seq_len_scale * 1000 / streaming_serial_time
                        print(f"{'Streaming Serial':19s} |     {1:3d}   |    {1:4d}  | {block_size:10d} | {streaming_serial_time:7.4f} | {streaming_serial_throughput:15.2f} | {1.00:11.2f}x | 1.000")

                    # 测试不同的MPI ranks × OMP threads组合
                    for mpi_ranks in args.mpi_ranks:
                        for omp_threads in args.mpi_omp_threads:
                            time, success = test_cpp_mpi_attention(
                                seq_len_scale, args.dim, block_size,
                                mpi_ranks, omp_threads,
                                args.cpp_streaming_mpi, args.mpirun, 1
                            )
                            if success:
                                throughput = seq_len_scale * 1000 / time
                                total_threads = mpi_ranks * omp_threads
                                speedup_vs_serial = streaming_serial_time / time
                                efficiency = speedup_vs_serial / total_threads
                                print(f"{'Streaming MPI+OMP':19s} |    {mpi_ranks:3d}   |   {omp_threads:4d}  | {block_size:10d} | {time:7.4f} | {throughput:15.2f} | {speedup_vs_serial:11.2f}x | {efficiency:.3f}")

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
    if HAS_TORCH:
        print("   - PyTorch SDPA: 使用高度优化的BLAS库 (MKL/oneDNN)")
    print()

    print("2. 并行化策略:")
    print("   - Naive OpenMP: 并行化矩阵乘法 (每个线程处理一部分T)")
    print("   - Streaming OpenMP: 并行化block处理 (每个线程处理一部分blocks)")
    if args.enable_mpi:
        print("   - MPI: 分布式计算，每个MPI rank处理一部分tokens")
        print("   - MPI+OpenMP混合: 跨节点(MPI) + 节点内(OpenMP)的两级并行")
    print()

    print("3. 性能特点:")
    if HAS_TORCH:
        print("   - PyTorch: 单线程最快（BLAS优化 + SIMD）")
    print("   - Naive OpenMP: 多线程可扩展性好，计算密集型")
    print("   - Streaming OpenMP: 内存友好，适合超长序列")
    if args.enable_mpi:
        print("   - MPI: 适合多节点并行，通信开销随ranks增加")
        print("   - MPI+OpenMP混合: 平衡计算和通信，最优配置需根据硬件调整")
    print()

    print("=" * 100)


if __name__ == "__main__":
    args = parse_args()
    run_comparison(args)
