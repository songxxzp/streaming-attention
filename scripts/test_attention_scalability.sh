#!/bin/bash
# Attention性能测试脚本 - 串行、OpenMP、MPI+OpenMP
# 测试配置: 8节点，每节点2CPU×26核 = 共416核
#
# 用法: ./test_attention_scalability.sh [seq_len]

set -e

# 配置参数
SEQ_LEN=${1:-4096}  # 默认序列长度4096
HIDDEN_DIM=128       # 隐藏维度
BLOCK_SIZE=64        # Streaming attention块大小
ITERS=10             # 每个配置的迭代次数

# 路径配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ATTENTION_DIR="$PROJECT_DIR/attention"
TENSOR_CPP_DIR="$PROJECT_DIR/tensor_cpp"
RESULTS_DIR="$PROJECT_DIR/results/attention_scalability_$(date +%Y%m%d_%H%M%S)"

# 创建结果目录
mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "  Attention性能扩展性测试"
echo "========================================"
echo "序列长度: $SEQ_LEN"
echo "隐藏维度: $HIDDEN_DIM"
echo "块大小: $BLOCK_SIZE"
echo "迭代次数: $ITERS"
echo "结果目录: $RESULTS_DIR"
echo "========================================"
echo ""

# ============================================================================
# 1. 串行版本测试
# ============================================================================
echo "[1/3] 测试串行版本..."

if [ -f "$ATTENTION_DIR/naive_serial" ]; then
    echo "找到串行版本可执行文件"
else
    echo "编译串行版本..."
    cd "$ATTENTION_DIR"
    g++ -std=c++17 -O3 -march=native \
        naive_serial.cpp streaming_serial.cpp \
        -o naive_serial
fi

cd "$ATTENTION_DIR"
echo "运行串行版本 (seq_len=$SEQ_LEN, iters=$ITERS)..."
ITER=0
while [ $ITER -lt $ITERS ]; do
    ./naive_serial $SEQ_LEN $HIDDEN_DIM 2>&1 | tee -a "$RESULTS_DIR/serial_log.txt"
    ITER=$((ITER+1))
done

echo "✓ 串行版本测试完成"
echo ""

# ============================================================================
# 2. OpenMP版本测试
# ============================================================================
echo "[2/3] 测试OpenMP版本..."

# 编译OpenMP版本
if [ ! -f "$ATTENTION_DIR/streaming_omp" ]; then
    echo "编译OpenMP版本..."
    cd "$ATTENTION_DIR"
    g++ -std=c++17 -O3 -march=native -fopenmp \
        streaming_omp.cpp streaming_serial.cpp \
        ../utils/softmax_online.cpp \
        -o streaming_omp -lm
fi

# 测试不同线程数
# 对于8节点×2CPU×26核=416核，我们测试一些关键点
THREADS_LIST=(1 2 4 8 16 26 52 104 208 416)

cd "$ATTENTION_DIR"
echo "测试OpenMP版本 (seq_len=$SEQ_LEN, iters=$ITERS)..."
echo ""

for NUM_THREADS in "${THREADS_LIST[@]}"; do
    echo "  线程数: $NUM_THREADS"
    export OMP_NUM_THREADS=$NUM_THREADS

    ITER=0
    while [ $ITER -lt $ITERS ]; do
        ./streaming_omp $SEQ_LEN $HIDDEN_DIM $BLOCK_SIZE $NUM_THREADS 2>&1 | \
            tee -a "$RESULTS_DIR/omp_threads_${NUM_THREADS}_log.txt"
        ITER=$((ITER+1))
    done
    echo ""
done

echo "✓ OpenMP版本测试完成"
echo ""

# ============================================================================
# 3. MPI+OpenMP混合版本测试
# ============================================================================
echo "[3/3] 测试MPI+OpenMP混合版本..."

# 编译MPI版本
if [ ! -f "$ATTENTION_DIR/streaming_mpi" ]; then
    echo "编译MPI版本..."
    cd "$ATTENTION_DIR"
    /usr/bin/mpicxx -std=c++17 -O3 -march=native -fopenmp \
        streaming_mpi.cpp streaming_omp.cpp streaming_serial.cpp \
        ../utils/softmax_online.cpp \
        -o streaming_mpi -lm
fi

# MPI配置测试
# 每个进程的OpenMP线程数
OMP_PER_RANK=26

# MPI进程数配置 (总核数416 / 每进程26核 = 16进程)
MPI_RANKS_LIST=(1 2 4 8 16)

cd "$ATTENTION_DIR"
echo "测试MPI+OpenMP混合版本 (seq_len=$SEQ_LEN, iters=$ITERS)..."
echo "每进程OpenMP线程数: $OMP_PER_RANK"
echo ""

for NUM_RANKS in "${MPI_RANKS_LIST[@]}"; do
    TOTAL_CORES=$((NUM_RANKS * OMP_PER_RANK))
    echo "  MPI进程数: $NUM_RANKS, 总核数: $TOTAL_CORES"

    ITER=0
    while [ $ITER -lt $ITERS ]; do
        export OMP_NUM_THREADS=$OMP_PER_RANK
        /usr/bin/mpirun -np $NUM_RANKS \
            ./streaming_mpi $((SEQ_LEN / NUM_RANKS)) $SEQ_LEN $HIDDEN_DIM $BLOCK_SIZE \
            2>&1 | tee -a "$RESULTS_DIR/mpi_ranks_${NUM_RANKS}_log.txt"
        ITER=$((ITER+1))
    done
    echo ""
done

echo "✓ MPI+OpenMP混合版本测试完成"
echo ""

# ============================================================================
# 生成性能报告
# ============================================================================
echo "========================================"
echo "  生成性能报告..."
echo "========================================"

cd "$RESULTS_DIR"

# 提取性能数据
echo "" > "summary.txt"
echo "Attention性能扩展性测试总结" >> "summary.txt"
echo "=================================" >> "summary.txt"
echo "测试配置:" >> "summary.txt"
echo "  序列长度: $SEQ_LEN" >> "summary.txt"
echo "  隐藏维度: $HIDDEN_DIM" >> "summary.txt"
echo "  块大小: $BLOCK_SIZE" >> "summary.txt"
echo "  迭代次数: $ITERS" >> "summary.txt"
echo "" >> "summary.txt"

# 串行版本
echo "[串行版本]" >> "summary.txt"
if [ -f "serial_log.txt" ]; then
    # 提取平均时间
    AVG_TIME=$(grep "Time" serial_log.txt | awk '{sum+=$2; n++} END {print sum/n}')
    echo "  平均时间: ${AVG_TIME} ms" >> "summary.txt"
fi
echo "" >> "summary.txt"

# OpenMP版本
echo "[OpenMP版本]" >> "summary.txt"
for NUM_THREADS in "${THREADS_LIST[@]}"; do
    if [ -f "omp_threads_${NUM_THREADS}_log.txt" ]; then
        AVG_TIME=$(grep "Time" omp_threads_${NUM_THREADS}_log.txt | awk '{sum+=$2; n++} END {print sum/n}')
        SPEEDUP=$(echo "scale=2; ${AVG_TIME_SERIAL:-1.0} / $AVG_TIME" | bc 2>/dev/null || echo "N/A")
        printf "  线程数: %3d, 平均时间: %8.2f ms, 加速比: %4.2fx\n" \
            $NUM_THREADS $AVG_TIME $SPEEDUP >> "summary.txt"
    fi
done
echo "" >> "summary.txt"

# MPI+OpenMP版本
echo "[MPI+OpenMP混合版本]" >> "summary.txt"
echo "每进程OpenMP线程数: $OMP_PER_RANK" >> "summary.txt"
for NUM_RANKS in "${MPI_RANKS_LIST[@]}"; do
    if [ -f "mpi_ranks_${NUM_RANKS}_log.txt" ]; then
        AVG_TIME=$(grep "Time" mpi_ranks_${NUM_RANKS}_log.txt | awk '{sum+=$2; n++} END {print sum/n}')
        printf "  MPI进程数: %2d, 总核数: %3d, 平均时间: %8.2f ms\n" \
            $NUM_RANKS $((NUM_RANKS * OMP_PER_RANK)) $AVG_TIME >> "summary.txt"
    fi
done
echo "" >> "summary.txt"

echo "✓ 性能报告已生成: $RESULTS_DIR/summary.txt"
echo ""
cat "$RESULTS_DIR/summary.txt"

echo ""
echo "========================================"
echo "  测试完成！"
echo "========================================"
echo "结果保存在: $RESULTS_DIR"
echo ""
