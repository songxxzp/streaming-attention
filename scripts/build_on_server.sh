#!/bin/bash
# 服务器编译脚本 - tensor_cpp项目
# 用法: ./build_on_server.sh

set -e  # 出错时退出

echo "========================================"
echo "  Tensor C++ 服务器编译脚本"
echo "========================================"
echo ""

# 加载必要模块
echo "[1/5] 加载spack模块..."
if command -v spack &> /dev/null; then
    spack load cmake
    spack load openmpi
    echo "✓ 模块加载完成"
else
    echo "⚠ 警告: 未找到spack，尝试使用系统默认编译器"
fi

# 检查编译器
echo ""
echo "[2/5] 检查编译器..."
if command -v /usr/bin/mpicxx &> /dev/null; then
    echo "✓ MPI C++编译器: $(/usr/bin/mpicxx --version | head -1)"
else
    echo "✗ 错误: 未找到mpicxx"
    exit 1
fi

if command -v g++ &> /dev/null; then
    echo "✓ GCC版本: $(g++ --version | head -1)"
else
    echo "✗ 错误: 未找到g++"
    exit 1
fi

# 进入项目目录
echo ""
echo "[3/5] 进入项目目录..."
PROJECT_DIR="/media/song/LocalDisk/Weblearning/并行计算/final/tensor_cpp"
if [ -d "$PROJECT_DIR" ]; then
    cd "$PROJECT_DIR"
    echo "✓ 项目目录: $PROJECT_DIR"
else
    echo "✗ 错误: 项目目录不存在: $PROJECT_DIR"
    exit 1
fi

# 清理旧的构建
echo ""
echo "[4/5] 清理旧构建..."
make clean
echo "✓ 清理完成"

# 编译项目
echo ""
echo "[5/5] 编译项目..."
make all
echo "✓ 编译完成"

# 显示生成的可执行文件
echo ""
echo "========================================"
echo "  编译完成！生成的可执行文件:"
echo "========================================"
echo ""
echo "核心测试:"
ls -lh build/test_ops 2>/dev/null && echo "  ✓ test_ops"
ls -lh build/test_attention 2>/dev/null && echo "  ✓ test_attention"
ls -lh build/test_qwen3 2>/dev/null && echo "  ✓ test_qwen3"
echo ""
echo "性能测试:"
ls -lh build/benchmark_attention 2>/dev/null && echo "  ✓ benchmark_attention"
ls -lh build/benchmark_qwen3 2>/dev/null && echo "  ✓ benchmark_qwen3"
echo ""
echo "MPI测试:"
ls -lh build/test_mpi_simple 2>/dev/null && echo "  ✓ test_mpi_simple"
echo ""
echo "========================================"
echo "  使用示例:"
echo "========================================"
echo ""
echo "# 测试OpenMP版本"
echo "OMP_NUM_THREADS=16 ./build/benchmark_attention --mode streaming --seq-len 4096 --iters 10 --threads 16"
echo ""
echo "# 测试MPI版本 (使用原始attention项目)"
echo "mpirun -np 16 ../attention/streaming_mpi"
echo ""
echo "# 快速测试"
echo "./quick_test.sh"
echo ""
