#!/bin/bash
# 编译Attention测试程序
# 支持编译串行、OpenMP、MPI版本的所有attention算子

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认参数
MPICXX="${MPICXX:-/usr/bin/mpicxx}"
ATTENTION_DIR="./attention"
CLEAN=false

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --mpicxx)
            MPICXX="$2"
            shift 2
            ;;
        --attention-dir)
            ATTENTION_DIR="$2"
            shift 2
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --mpicxx PATH       mpicxx编译器路径 (默认: /usr/bin/mpicxx)"
            echo "  --attention-dir DIR  attention目录路径 (默认: ./attention)"
            echo "  --clean             清理所有编译的可执行文件"
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
echo -e "${BLUE}  编译Attention测试程序${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "mpicxx: $MPICXX"
echo "attention目录: $ATTENTION_DIR"
echo ""

# 检查目录是否存在
if [ ! -d "$ATTENTION_DIR" ]; then
    echo -e "${RED}错误: attention目录不存在: $ATTENTION_DIR${NC}"
    exit 1
fi

cd "$ATTENTION_DIR"

# 清理选项
if [ "$CLEAN" = true ]; then
    echo -e "${YELLOW}清理可执行文件...${NC}"
    rm -f test_naive test_naive_omp test_streaming test_streaming_omp
    rm -f test_naive_mpi test_streaming_mpi
    echo -e "${GREEN}✓ 清理完成${NC}"
    echo ""
    exit 0
fi

# 定义要编译的文件列表
# 格式: (可执行文件名, 测试文件, 实现文件, 编译标志, 是否需要MPI)
declare -a EXECUTABLES=(
    # 串行版本
    "test_naive|test_naive.cpp|naive_serial.cpp||false"
    "test_streaming|test_streaming.cpp|streaming_serial.cpp||false"

    # OpenMP版本
    "test_naive_omp|test_naive_omp.cpp|naive_omp.cpp|-fopenmp|false"
    "test_streaming_omp|test_streaming_omp.cpp|streaming_omp.cpp streaming_serial.cpp|-fopenmp|false"

    # MPI版本
    "test_naive_mpi|test_naive_mpi.cpp|naive_mpi.cpp|-fopenmp|true"
    "test_streaming_mpi|test_streaming_mpi.cpp|streaming_mpi.cpp streaming_serial.cpp|-fopenmp|true"
)

echo -e "${GREEN}检查并编译Attention测试程序...${NC}"
echo ""

# 检查并编译MPI版本的可执行文件
MPI_AVAILABLE=false
if [ -f "$MPICXX" ]; then
    # 检查mpicxx是否可用
    if $MPICXX --version &>/dev/null; then
        MPI_AVAILABLE=true
        echo -e "${GREEN}✓ MPI编译器可用: $MPICXX${NC}"
    else
        echo -e "${YELLOW}⚠ MPI编译器不可用: $MPICXX${NC}"
    fi
else
    echo -e "${YELLOW}⚠ MPI编译器不存在: $MPICXX${NC}"
fi
echo ""

# 编译非MPI版本
echo -e "${BLUE}编译串行和OpenMP版本...${NC}"
for exec_info in "${EXECUTABLES[@]}"; do
    IFS='|' read -r exe_name test_file impl_files flags needs_mpi <<< "$exec_info"

    if [ "$needs_mpi" = "true" ]; then
        continue
    fi

    if [ ! -f "$exe_name" ]; then
        echo -ne "${YELLOW}  编译${exe_name}...${NC}"
        compile_cmd="g++ -std=c++17 -O3 -march=native ${flags} -I. ${test_file} ${impl_files} -o ${exe_name}"
        if eval $compile_cmd >/dev/null 2>&1; then
            echo -e "\r${GREEN}  ✓ ${exe_name}编译成功${NC}"
        else
            echo -e "\r${RED}  ✗ ${exe_name}编译失败${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}  ✓ ${exe_name}已存在${NC}"
    fi
done

echo ""

# 编译MPI版本
if [ "$MPI_AVAILABLE" = true ]; then
    echo -e "${BLUE}编译MPI版本...${NC}"
    for exec_info in "${EXECUTABLES[@]}"; do
        IFS='|' read -r exe_name test_file impl_files flags needs_mpi <<< "$exec_info"

        if [ "$needs_mpi" = "false" ]; then
            continue
        fi

        if [ ! -f "$exe_name" ]; then
            echo -ne "${YELLOW}  编译${exe_name}...${NC}"
            compile_cmd="$MPICXX -std=c++17 -O3 -march=native ${flags} -I. ${test_file} ${impl_files} -o ${exe_name}"
            if eval $compile_cmd >/dev/null 2>&1; then
                echo -e "\r${GREEN}  ✓ ${exe_name}编译成功${NC}"
            else
                echo -e "\r${RED}  ✗ ${exe_name}编译失败${NC}"
                exit 1
            fi
        else
            echo -e "${GREEN}  ✓ ${exe_name}已存在${NC}"
        fi
    done
else
    echo -e "${YELLOW}跳过MPI版本编译（MPI编译器不可用）${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  编译完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "可执行文件列表:"

# 列出所有可执行文件
for exec_info in "${EXECUTABLES[@]}"; do
    IFS='|' read -r exe_name test_file impl_files flags needs_mpi <<< "$exec_info"
    if [ -f "$exe_name" ]; then
        if [ "$needs_mpi" = "true" ]; then
            echo -e "  ${GREEN}✓${NC} $exe_name (MPI)"
        else
            echo -e "  ${GREEN}✓${NC} $exe_name"
        fi
    else
        if [ "$needs_mpi" = "true" ]; then
            if [ "$MPI_AVAILABLE" = "true" ]; then
                echo -e "  ${RED}✗${NC} $exe_name (MPI) - 未编译"
            else
                echo -e "  ${YELLOW}○${NC} $exe_name (MPI) - MPI不可用"
            fi
        else
            echo -e "  ${RED}✗${NC} $exe_name - 未编译"
        fi
    fi
done

echo ""
echo "使用方法:"
echo "  # 串行测试"
echo "  ./attention/test_naive <T> <d> <block_size>"
echo "  ./attention/test_streaming <T> <d> <block_size>"
echo ""
echo "  # OpenMP测试"
echo "  OMP_NUM_THREADS=4 ./attention/test_naive_omp <T> <d> <block_size>"
echo "  OMP_NUM_THREADS=4 ./attention/test_streaming_omp <T> <d> <block_size>"
echo ""
if [ "$MPI_AVAILABLE" = "true" ]; then
    echo "  # MPI测试"
    echo "  $mpirun -np 2 ./attention/test_naive_mpi <T> <d> <omp_threads>"
    echo "  $mpirun -np 2 ./attention/test_streaming_mpi <T> <d> <block_size> <omp_threads>"
else
    echo "  # MPI测试 (需要可用的MPI编译器)"
    echo "  $mpirun -np 2 ./attention/test_naive_mpi <T> <d> <omp_threads>"
    echo "  $mpirun -np 2 ./attention/test_streaming_mpi <T> <d> <block_size> <omp_threads>"
fi
echo ""
echo "清理命令:"
echo "  bash scripts/compile_attention.sh --clean"
