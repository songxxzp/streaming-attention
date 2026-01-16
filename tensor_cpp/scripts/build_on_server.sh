#!/bin/bash
###############################################################################
# 服务器编译脚本 - tensor_cpp
#
# 用途: 在服务器集群上编译tensor_cpp库和benchmark程序
# 环境: 支持MPI、OpenMP、AVX2的服务器环境
#
# 使用方法:
#   ./build_on_server.sh           # 标准编译
#   ./build_on_server.sh clean      # 清理后重新编译
#   ./build_on_server.sh verbose    # 详细输出
###############################################################################

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 配置
BUILD_DIR="${PROJECT_DIR}/build"
BUILD_TYPE="Release"
PARALLEL_JOBS=$(nproc)  # 使用所有可用的CPU核心

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}         Tensor C++ 库 - 服务器编译脚本${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "项目目录: ${PROJECT_DIR}"
echo -e "构建目录: ${BUILD_DIR}"
echo -e "构建类型: ${BUILD_TYPE}"
echo -e "并行编译: ${PARALLEL_JOBS} jobs"
echo -e "${BLUE}============================================================${NC}"
echo ""

# 处理命令行参数
if [[ "$1" == "clean" ]]; then
    echo -e "${YELLOW}清理旧的构建文件...${NC}"
    rm -rf "${BUILD_DIR}"
    echo -e "${GREEN}✓ 清理完成${NC}"
    echo ""
fi

# 创建构建目录
echo -e "${BLUE}创建构建目录...${NC}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
echo -e "${GREEN}✓ 构建目录: ${BUILD_DIR}${NC}"
echo ""

# 检查必要的依赖
echo -e "${BLUE}检查编译环境...${NC}"

# 检查C++编译器
if command -v g++ &> /dev/null; then
    GCC_VERSION=$(g++ --version | head -n1)
    echo -e "${GREEN}✓ GCC: ${GCC_VERSION}${NC}"
else
    echo -e "${RED}✗ 未找到g++编译器${NC}"
    exit 1
fi

# 检查CMake
if command -v cmake &> /dev/null; then
    CMAKE_VERSION=$(cmake --version | head -n1)
    echo -e "${GREEN}✓ CMake: ${CMAKE_VERSION}${NC}"
else
    echo -e "${RED}✗ 未找到cmake${NC}"
    exit 1
fi

# 检查MPI
if command -v mpicxx &> /dev/null; then
    MPI_INFO=$(mpicxx --version | head -n1)
    echo -e "${GREEN}✓ MPI C++: ${MPI_INFO}${NC}"
else
    echo -e "${RED}✗ 未找到mpicxx${NC}"
    echo -e "${YELLOW}提示: 请先加载MPI模块 (module load openmpi)${NC}"
    exit 1
fi

# 检查AVX2支持
if grep -q avx2 /proc/cpuinfo 2>/dev/null; then
    echo -e "${GREEN}✓ CPU支持AVX2指令集${NC}"
else
    echo -e "${YELLOW}⚠ CPU可能不支持AVX2，编译会失败或运行时会出错${NC}"
fi

echo ""

# 运行CMake配置
echo -e "${BLUE}配置CMake...${NC}"
CMAKE_OUTPUT="cmake .. -DCMAKE_BUILD_TYPE=${BUILD_TYPE}"

if [[ "$2" == "verbose" ]]; then
    eval $CMAKE_OUTPUT
else
    eval $CMAKE_OUTPUT > /dev/null
fi

echo -e "${GREEN}✓ CMake配置完成${NC}"
echo ""

# 编译
echo -e "${BLUE}开始编译 (使用${PARALLEL_JOBS}个并行任务)...${NC}"
echo -e "${YELLOW}这可能需要几分钟...${NC}"
echo ""

MAKE_CMD="make -j${PARALLEL_JOBS}"

if [[ "$1" == "verbose" || "$2" == "verbose" ]]; then
    eval $MAKE_CMD
else
    eval $MAKE_CMD
fi

echo ""
echo -e "${GREEN}✓ 编译完成！${NC}"
echo ""

# 显示编译结果
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}         编译结果${NC}"
echo -e "${BLUE}============================================================${NC}"

# 检查关键的benchmark程序
CRITICAL_BINARIES=(
    "benchmark_qwen3"
)

BINARY_COUNT=0
for binary in "${CRITICAL_BINARIES[@]}"; do
    if [[ -f "${BUILD_DIR}/${binary}" ]]; then
        BINARY_COUNT=$((BINARY_COUNT + 1))
        echo -e "${GREEN}✓${NC} ${binary}"
    else
        echo -e "${RED}✗${NC} ${binary} (编译失败)"
    fi
done

echo ""

# 显示所有可执行文件
echo -e "${BLUE}所有可执行文件 (${BUILD_DIR}/):${NC}"
ls -lh "${BUILD_DIR}" | grep -E '^-rwx' | awk '{print "  " $9, "("$5")"}'

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}         编译完成！${NC}"
echo -e "${BLUE}============================================================${NC}"

if [[ $BINARY_COUNT -eq ${#CRITICAL_BINARIES[@]} ]]; then
    echo -e "${GREEN}所有关键程序编译成功！${NC}"
    echo ""
    echo -e "运行benchmark:"
    echo -e "  ${BUILD_DIR}/benchmark_qwen3 --model /media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors --help"
    echo ""
    echo -e "运行实验:"
    echo -e "  bash ${SCRIPT_DIR}/exp1_serial_baseline.sh"
    echo -e "  bash ${SCRIPT_DIR}/exp2_single_node_16threads.sh"
    echo ""
    echo -e "在集群上运行MPI实验:"
    echo -e "  srun --mpi=pmix -p student -N 8 --ntasks=8 --ntasks-per-node=1 --cpus-per-task=16 \\"
    echo -e "    bash ${SCRIPT_DIR}/exp3_mpi_parallel.sh"
else
    echo -e "${RED}某些程序编译失败，请检查错误信息${NC}"
    exit 1
fi

echo ""
