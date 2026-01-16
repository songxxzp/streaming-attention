#!/bin/bash
###############################################################################
# 运行全部实验脚本
#
# 依次执行：
#   - 实验1: 串行Baseline性能测试
#   - 实验2: 单机16线程性能测试
#   - 实验3: MPI并行性能测试
###############################################################################

set -e  # 遇到错误立即退出

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "         Qwen3 Prefill 性能测试套件"
echo "============================================================"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 实验1: 串行Baseline
echo "=========================================="
echo "开始 实验1: 串行Baseline性能测试"
echo "=========================================="
bash "${SCRIPT_DIR}/exp1_serial_baseline.sh"
echo "✅ 实验1 完成"
echo ""

# 实验2: 单机16线程
echo "=========================================="
echo "开始 实验2: 单机16线程性能测试"
echo "=========================================="
bash "${SCRIPT_DIR}/exp2_single_node_16threads.sh"
echo "✅ 实验2 完成"
echo ""

# 实验3: MPI并行
echo "=========================================="
echo "开始 实验3: MPI并行性能测试"
echo "=========================================="
bash "${SCRIPT_DIR}/exp3_mpi_parallel.sh"
echo "✅ 实验3 完成"
echo ""

echo "============================================================"
echo "         全部实验完成！"
echo "============================================================"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "结果汇总:"
echo "  - 实验1: results/exp1_serial_baseline/serial_baseline_results.csv"
echo "  - 实验2: results/exp2_single_node_16threads/single_node_16threads_results.csv"
echo "  - 实验3: results/exp3_mpi_parallel/mpi_parallel_results.csv"
echo ""
