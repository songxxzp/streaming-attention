#!/bin/bash
# 编译MPI版本的Attention测试程序

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 默认参数
MPICXX="${MPICXX:-/usr/bin/mpicxx}"
ATTENTION_DIR="./attention"

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
        --help)
            echo "用法: $0 [--mpicxx MPICXX] [--attention-dir DIR]"
            echo ""
            echo "选项:"
            echo "  --mpicxx PATH       mpicxx编译器路径 (默认: /usr/bin/mpicxx)"
            echo "  --attention-dir DIR  attention目录路径 (默认: ./attention)"
            echo "  --help              显示此帮助信息"
            exit 0
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}编译MPI版本的Attention测试程序${NC}"
echo "mpicxx: $MPICXX"
echo "attention目录: $ATTENTION_DIR"
echo ""

# 检查目录是否存在
if [ ! -d "$ATTENTION_DIR" ]; then
    echo -e "${RED}错误: attention目录不存在: $ATTENTION_DIR${NC}"
    exit 1
fi

# 检查mpicxx是否存在
if [ ! -f "$MPICXX" ]; then
    echo -e "${RED}错误: mpicxx不存在: $MPICXX${NC}"
    echo "提示: 可以使用 --mpicxx 参数指定mpicxx路径"
    echo "或者设置环境变量: export MPICXX=/path/to/mpicxx"
    exit 1
fi

cd "$ATTENTION_DIR"

# 编译test_naive_mpi
echo -e "${YELLOW}编译 test_naive_mpi...${NC}"
if [ ! -f "test_naive_mpi" ]; then
    $MPICXX -std=c++17 -O3 -march=native -fopenmp -I. \
        test_naive_mpi.cpp naive_mpi.cpp \
        -o test_naive_mpi
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ test_naive_mpi 编译成功${NC}"
    else
        echo -e "${RED}✗ test_naive_mpi 编译失败${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ test_naive_mpi 已存在${NC}"
fi

# 编译test_streaming_mpi
echo -e "${YELLOW}编译 test_streaming_mpi...${NC}"
if [ ! -f "test_streaming_mpi" ]; then
    $MPICXX -std=c++17 -O3 -march=native -fopenmp -I. \
        test_streaming_mpi.cpp streaming_mpi.cpp streaming_serial.cpp \
        -o test_streaming_mpi
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ test_streaming_mpi 编译成功${NC}"
    else
        echo -e "${RED}✗ test_streaming_mpi 编译失败${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ test_streaming_mpi 已存在${NC}"
fi

cd ..

echo ""
echo -e "${GREEN}所有MPI版本的程序编译完成！${NC}"
echo "可执行文件位于:"
echo "  - $ATTENTION_DIR/test_naive_mpi"
echo "  - $ATTENTION_DIR/test_streaming_mpi"
