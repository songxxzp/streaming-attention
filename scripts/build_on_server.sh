#!/bin/bash
# 服务器编译脚本 - tensor_cpp项目 (使用CMake)
# 参考: nbody项目的编译方式
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
    echo "⚠ 警告: 未找到spack，尝试使用系统默认cmake"
fi

# 检查cmake
echo ""
echo "[2/5] 检查cmake..."
if command -v cmake &> /dev/null; then
    echo "✓ CMake版本: $(cmake --version | head -1)"
else
    echo "✗ 错误: 未找到cmake"
    echo "  请运行: spack load cmake"
    exit 1
fi

# 进入项目目录
echo ""
echo "[3/5] 进入项目目录..."
# 自动检测项目目录（脚本在scripts/下，项目在上一级）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")/tensor_cpp"

if [ -d "$PROJECT_DIR" ]; then
    cd "$PROJECT_DIR"
    echo "✓ 项目目录: $PROJECT_DIR"
else
    echo "✗ 错误: 项目目录不存在: $PROJECT_DIR"
    echo "  脚本目录: $SCRIPT_DIR"
    exit 1
fi

# 清理旧的构建
echo ""
echo "[4/5] 清理旧构建..."
rm -rf build
mkdir -p build
echo "✓ 清理完成"

# 配置CMake
echo ""
echo "[5/5] 配置并编译项目..."
cd build
echo "运行CMake配置..."
cmake .. \
    -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="/usr/lib/x86_64-linux-gnu"

echo ""
echo "编译项目..."
make -j$(nproc)
echo "✓ 编译完成"

# 设置库路径配置文件
echo ""
echo "配置运行环境..."
cat > "$PROJECT_DIR/set_env.sh" << 'EOF'
#!/bin/bash
# Tensor C++ 运行环境设置
# 使用方式: source tensor_cpp/set_env.sh

# 设置MPI库路径
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# 设置默认OpenMP线程数
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}

echo "✓ 运行环境已设置:"
echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
EOF
chmod +x "$PROJECT_DIR/set_env.sh"
echo "✓ 运行环境配置完成"
echo "  使用方式: source tensor_cpp/set_env.sh"

# 显示生成的可执行文件
echo ""
echo "========================================"
echo "  编译完成！生成的可执行文件:"
echo "========================================"
echo ""
echo "核心测试:"
ls -lh test_ops 2>/dev/null && echo "  ✓ test_ops"
ls -lh test_attention 2>/dev/null && echo "  ✓ test_attention"
ls -lh test_qwen3 2>/dev/null && echo "  ✓ test_qwen3"
echo ""
echo "性能测试:"
ls -lh benchmark_attention 2>/dev/null && echo "  ✓ benchmark_attention"
ls -lh benchmark_qwen3 2>/dev/null && echo "  ✓ benchmark_qwen3"
echo ""
echo "MPI测试:"
ls -lh test_mpi_simple 2>/dev/null && echo "  ✓ test_mpi_simple"
echo ""
echo "========================================"
echo "  使用示例:"
echo "========================================"
echo ""
echo "# 设置运行环境（首次运行需要）"
echo "source tensor_cpp/set_env.sh"
echo ""
echo "# 测试OpenMP版本"
echo "LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH \\"
echo "  OMP_NUM_THREADS=16 \\"
echo "  ./build/benchmark_attention --mode streaming --seq-len 4096 --iters 10"
echo ""
echo "# 测试MPI版本 (使用原始attention项目)"
echo "LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH \\"
echo "  mpirun -np 16 ../attention/streaming_mpi"
echo ""
echo "# 快速测试"
echo "cd tensor_cpp && ./quick_test.sh"
echo ""
