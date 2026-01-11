#!/bin/bash
# 运行MPI版本的Attention性能测试

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认参数
MPIRUN="${MPIRUN:-/usr/bin/mpirun}"
SEQ_LEN="${SEQ_LEN:-2048}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --mpirun)
            MPIRUN="$2"
            shift 2
            ;;
        --seq-len)
            SEQ_LEN="$2"
            shift 2
            ;;
        --dim)
            HIDDEN_DIM="$2"
            shift 2
            ;;
        --block-size)
            BLOCK_SIZE="$2"
            shift 2
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --mpirun PATH       MPI运行程序路径 (默认: /usr/bin/mpirun)"
            echo "  --seq-len N         序列长度 (默认: 2048)"
            echo "  --dim N             隐藏维度 (默认: 128)"
            echo "  --block-size N      Block size (默认: 64)"
            echo "  --help              显示此帮助信息"
            exit 0
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  MPI Attention性能测试${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "测试配置:"
echo "  序列长度: $SEQ_LEN"
echo "  隐藏维度: $HIDDEN_DIM"
echo "  Block Size: $BLOCK_SIZE"
echo "  mpirun: $MPIRUN"
echo ""

# 检查可执行文件是否存在
if [ ! -f "./attention/test_naive_mpi" ]; then
    echo -e "${RED}错误: ./attention/test_naive_mpi 不存在${NC}"
    echo "请先运行: bash scripts/compile_mpi.sh"
    exit 1
fi

if [ ! -f "./attention/test_streaming_mpi" ]; then
    echo -e "${RED}错误: ./attention/test_streaming_mpi 不存在${NC}"
    echo "请先运行: bash scripts/compile_mpi.sh"
    exit 1
fi

# ==============================================================================
# Naive MPI测试
# ==============================================================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Naive Attention MPI扩展性测试${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "MPI配置         | MPI Ranks | OMP/Rank | 时间 | 吞吐量 | 相对串行加速 | 效率"
echo "----------------|-----------|----------|---------|-----------------|-------------|------"

# 获取串行baseline (使用test_naive)
SERIAL_OUTPUT=$(./attention/test_naive $SEQ_LEN $HIDDEN_DIM $BLOCK_SIZE 2>/dev/null)
SERIAL_TIME=$(echo "$SERIAL_OUTPUT" | grep "Time:" | awk '{print $2}')
if [ -z "$SERIAL_TIME" ]; then
    echo -e "${RED}无法获取串行baseline时间${NC}"
    exit 1
fi

SERIAL_THROUGHPUT=$(echo "scale=2; $SEQ_LEN * 1000 / $SERIAL_TIME" | bc)
printf "Naive Serial     |      %3d   |    %4d  | %7.4f | %15.2f | %11.2fx | 1.000\n" 1 1 "$SERIAL_TIME" "$SERIAL_THROUGHPUT" 1.00

# 测试不同的MPI ranks × OMP threads组合
for mpi_ranks in 1 2 4; do
    for omp_threads in 2 4; do
        # 运行MPI测试 (naive_mpi不需要block_size参数)
        OUTPUT=$($MPIRUN -np $mpi_ranks ./attention/test_naive_mpi $SEQ_LEN $HIDDEN_DIM $omp_threads 2>/dev/null)
        TIME=$(echo "$OUTPUT" | grep "Time:" | awk '{print $2}')

        if [ -n "$TIME" ]; then
            THROUGHPUT=$(echo "scale=2; $SEQ_LEN * 1000 / $TIME" | bc)
            TOTAL_THREADS=$((mpi_ranks * omp_threads))
            SPEEDUP=$(echo "scale=2; $SERIAL_TIME / $TIME" | bc)
            EFFICIENCY=$(echo "scale=3; $SPEEDUP / $TOTAL_THREADS" | bc)

            printf "Naive MPI+OMP     |    %3d   |   %4d  | %7.4f | %15.2f | %11.2fx | %.3f\n" \
                   $mpi_ranks $omp_threads "$TIME" "$THROUGHPUT" "$SPEEDUP" "$EFFICIENCY"
        fi
    done
done

echo ""

# ==============================================================================
# Streaming MPI测试
# ==============================================================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Streaming Attention MPI扩展性测试${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "MPI配置            | MPI Ranks | OMP/Rank | 时间 | 吞吐量 | 相对串行加速 | 效率"
echo "-------------------|-----------|----------|---------|-----------------|-------------|------"

# 获取串行baseline (使用test_streaming)
SERIAL_OUTPUT=$(./attention/test_streaming $SEQ_LEN $HIDDEN_DIM $BLOCK_SIZE 2>/dev/null)
SERIAL_TIME=$(echo "$SERIAL_OUTPUT" | grep "Time:" | awk '{print $2}')
if [ -z "$SERIAL_TIME" ]; then
    echo -e "${RED}无法获取串行baseline时间${NC}"
    exit 1
fi

SERIAL_THROUGHPUT=$(echo "scale=2; $SEQ_LEN * 1000 / $SERIAL_TIME" | bc)
printf "Streaming Serial  |      %3d   |    %4d  | %7.4f | %15.2f | %11.2fx | 1.000\n" 1 1 "$SERIAL_TIME" "$SERIAL_THROUGHPUT" 1.00

# 测试不同的MPI ranks × OMP threads组合
for mpi_ranks in 1 2 4; do
    for omp_threads in 2 4; do
        # 运行MPI测试
        OUTPUT=$($MPIRUN -np $mpi_ranks ./attention/test_streaming_mpi $SEQ_LEN $HIDDEN_DIM $BLOCK_SIZE $omp_threads 2>/dev/null)
        TIME=$(echo "$OUTPUT" | grep "Time:" | awk '{print $2}')

        if [ -n "$TIME" ]; then
            THROUGHPUT=$(echo "scale=2; $SEQ_LEN * 1000 / $TIME" | bc)
            TOTAL_THREADS=$((mpi_ranks * omp_threads))
            SPEEDUP=$(echo "scale=2; $SERIAL_TIME / $TIME" | bc)
            EFFICIENCY=$(echo "scale=3; $SPEEDUP / $TOTAL_THREADS" | bc)

            printf "Streaming MPI+OMP  |    %3d   |   %4d  | %7.4f | %15.2f | %11.2fx | %.3f\n" \
                   $mpi_ranks $omp_threads "$TIME" "$THROUGHPUT" "$SPEEDUP" "$EFFICIENCY"
        fi
    done
done

echo ""
echo -e "${GREEN}测试完成！${NC}"
