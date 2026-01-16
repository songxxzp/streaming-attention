#!/bin/bash
###############################################################################
# 服务器快速开始脚本
#
# 用途: 在服务器上从零开始部署和运行实验
# 步骤: 1. 检查环境 -> 2. 编译 -> 3. 运行测试
###############################################################################

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}         Qwen3 性能测试 - 服务器快速开始${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# 配置
MODEL_PATH="/student/2025310707/Qwen3-0.6B/model.safetensors"
BENCHMARK="${PROJECT_DIR}/build/benchmark_qwen3"

# ============================================================================
# 步骤1: 检查环境
# ============================================================================
echo -e "${BLUE}[步骤 1/5] 检查服务器环境${NC}"
echo "------------------------------------------------------------"

# 检查模型文件
if [[ -f "$MODEL_PATH" ]]; then
    MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    echo -e "${GREEN}✓${NC} 模型文件存在: $MODEL_PATH ($MODEL_SIZE)"
else
    echo -e "${RED}✗${NC} 模型文件不存在: $MODEL_PATH"
    exit 1
fi

# 检查MPI
if command -v mpicxx &> /dev/null; then
    echo -e "${GREEN}✓${NC} MPI已安装: $(mpicxx --version | head -n1)"
else
    echo -e "${RED}✗${NC} MPI未安装"
    echo -e "${YELLOW}提示: 运行 'module load openmpi' 或 'module load mpich'${NC}"
    exit 1
fi

# 检查CPU核心数
CORES=$(nproc)
echo -e "${GREEN}✓${NC} CPU核心数: $CORES"

# 检查AVX2支持
if grep -q avx2 /proc/cpuinfo 2>/dev/null; then
    echo -e "${GREEN}✓${NC} CPU支持AVX2"
else
    echo -e "${YELLOW}⚠${NC} CPU可能不支持AVX2"
fi

echo ""

# ============================================================================
# 步骤2: 编译
# ============================================================================
echo -e "${BLUE}[步骤 2/5] 编译项目${NC}"
echo "------------------------------------------------------------"

if [[ -f "$BENCHMARK" ]]; then
    echo -e "${YELLOW}发现已编译的benchmark${NC}"
    read -p "是否重新编译? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        bash "${SCRIPT_DIR}/build_on_server.sh" clean
    else
        echo -e "${GREEN}跳过编译，使用现有二进制文件${NC}"
    fi
else
    bash "${SCRIPT_DIR}/build_on_server.sh"
fi

echo ""

# ============================================================================
# 步骤3: 快速测试
# ============================================================================
echo -e "${BLUE}[步骤 3/5] 运行快速测试${NC}"
echo "------------------------------------------------------------"

echo "运行单次prefill测试 (seq_len=128, 1 thread)..."
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
OMP_NUM_THREADS=1 \
timeout 60 \
"$BENCHMARK" \
    --model "$MODEL_PATH" \
    --method avx2 \
    --parallel-strategy headwise \
    --attention-algo online_softmax \
    --batch-size 1 \
    --prompt-len 128 \
    --phase prefill \
    --iters 1 \
    --warmup 0 \
    --threads 1

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}✓ 快速测试通过${NC}"
else
    echo -e "${RED}✗ 快速测试失败${NC}"
    exit 1
fi

echo ""

# ============================================================================
# 步骤4: 选择实验
# ============================================================================
echo -e "${BLUE}[步骤 4/5] 选择实验类型${NC}"
echo "------------------------------------------------------------"
echo "可用的实验:"
echo "  1) 实验1: 串行Baseline (本地测试, ~5分钟)"
echo "  2) 实验2: 单机16线程 (本地测试, ~10分钟)"
echo "  3) 实验3: MPI并行 (需要集群环境, ~30分钟)"
echo "  4) 运行全部实验 (本地测试 + 集群测试)"
echo "  5) 仅编译，不运行实验"
echo ""
read -p "请选择 (1-5): " choice

echo ""

case $choice in
    1)
        echo -e "${BLUE}运行实验1: 串行Baseline${NC}"
        bash "${SCRIPT_DIR}/exp1_serial_baseline.sh"
        ;;
    2)
        echo -e "${BLUE}运行实验2: 单机16线程${NC}"
        bash "${SCRIPT_DIR}/exp2_single_node_16threads.sh"
        ;;
    3)
        echo -e "${YELLOW}注意: 实验3需要在SLURM集群环境运行${NC}"
        read -p "节点数 (1/2/4/8): " nodes
        echo -e "${BLUE}在${nodes}个节点上运行实验3${NC}"
        echo "SLURM命令:"
        echo "srun --mpi=pmix -p student -N ${nodes} --ntasks=${nodes} --ntasks-per-node=1 --cpus-per-task=16 \\"
        echo "  bash ${SCRIPT_DIR}/exp3_mpi_parallel.sh"
        echo ""
        read -p "是否现在运行? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            srun --mpi=pmix -p student -N $nodes --ntasks=$nodes --ntasks-per-node=1 --cpus-per-task=16 \
                bash "${SCRIPT_DIR}/exp3_mpi_parallel.sh"
        fi
        ;;
    4)
        echo -e "${BLUE}运行全部本地实验 (实验1 + 实验2)${NC}"
        bash "${SCRIPT_DIR}/exp1_serial_baseline.sh"
        bash "${SCRIPT_DIR}/exp2_single_node_16threads.sh"
        echo -e "${GREEN}本地实验完成！${NC}"
        echo -e "${YELLOW}如需运行实验3 (MPI)，请使用选项3${NC}"
        ;;
    5)
        echo -e "${GREEN}编译完成，跳过实验${NC}"
        ;;
    *)
        echo -e "${RED}无效选择${NC}"
        exit 1
        ;;
esac

echo ""

# ============================================================================
# 步骤5: 结果汇总
# ============================================================================
echo -e "${BLUE}[步骤 5/5] 实验完成${NC}"
echo "------------------------------------------------------------"

RESULTS_DIR="${PROJECT_DIR}/results"
if [[ -d "$RESULTS_DIR" ]]; then
    echo -e "${GREEN}实验结果保存在:${NC}"
    find "$RESULTS_DIR" -name "*.csv" -type f | while read file; do
        echo "  - $file"
    done
else
    echo -e "${YELLOW}未找到结果目录${NC}"
fi

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}         快速开始脚本执行完毕！${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo -e "查看实验结果:"
echo -e "  cat ${RESULTS_DIR}/exp1_serial_baseline/serial_baseline_results.csv"
echo -e "  cat ${RESULTS_DIR}/exp2_single_node_16threads/single_node_16threads_results.csv"
echo ""
echo -e "运行更多实验:"
echo -e "  bash ${SCRIPT_DIR}/exp1_serial_baseline.sh"
echo -e "  bash ${SCRIPT_DIR}/exp2_single_node_16threads.sh"
echo ""
echo -e "在集群上运行MPI实验:"
echo -e "  srun --mpi=pmix -p student -N 8 --ntasks=8 --ntasks-per-node=1 --cpus-per-task=16 \\"
echo -e "    bash ${SCRIPT_DIR}/exp3_mpi_parallel.sh"
echo ""
