#!/bin/bash
# ============================================================================
# 并行计算课程完整实验脚本
# ============================================================================
# 测试内容：
# 1. Attention算子性能扩展性测试（串行、OpenMP、MPI+OpenMP）
# 2. Qwen3模型吞吐量测试（Prefill和Decode阶段）
#
# 用法: ./run_all_experiments.sh
#
# 结果保存在: results/experiment_<timestamp>/
# ============================================================================

set -e

# ============================================================================
# 配置参数
# ============================================================================

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 模型路径（自动查找）
find_model_path() {
    POSSIBLE_PATHS=(
        "/student/2025310707/Qwen3-0.6B/model.safetensors"
        "/media/song/LocalDisk/Weblearning/并行计算/final/models/Qwen3-0.6B/model.safetensors"
        "/home/$(whoami)/checkpoints/Qwen3-0.6B/model.safetensors"
        "~/checkpoints/Qwen3-0.6B/model.safetensors"
        "./models/Qwen3-0.6B/model.safetensors"
    )

    for path in "${POSSIBLE_PATHS[@]}"; do
        expanded_path="${path/#\~/$HOME}"
        if [ -f "$expanded_path" ]; then
            echo "$expanded_path"
            return 0
        fi
    done
    return 1
}

MODEL_PATH=$(find_model_path)
if [ $? -ne 0 ]; then
    echo "✗ 错误: 未找到Qwen3模型文件"
    echo "  请将模型放在以下路径之一："
    echo "    - /student/2025310707/Qwen3-0.6B/model.safetensors"
    echo "    - ~/checkpoints/Qwen3-0.6B/model.safetensors"
    exit 1
fi

# 实验配置
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$PROJECT_DIR/results/experiment_$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

# 运行环境
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# ============================================================================
# 实验设计
# ============================================================================

cat > "$RESULTS_DIR/experiment_design.txt" << 'EOF'
=============================================================================
并行计算课程实验设计
=============================================================================

实验1: Attention算子性能扩展性测试
----------------------------------------
目的: 对比串行、OpenMP、MPI+OpenMP三种并行模式的性能

测试配置:
  - 隐藏维度: 128
  - 块大小: 64
  - 迭代次数: 10

变量:
  - 序列长度: 512, 1024, 2048, 4096, 8192
  - 并行模式:
    * 串行 (baseline)
    * OpenMP: 1, 2, 4, 8, 16, 26, 52, 104, 208, 416线程
    * MPI+OpenMP: 1, 2, 4, 8, 16进程 × 26线程

测试指标:
  - 执行时间 (ms)
  - 吞吐量 (tokens/sec)
  - 加速比 = 串行时间 / 并行时间
  - 并行效率 = 加速比 / 处理器数

实验2: Qwen3模型吞吐量测试
----------------------------------------
目的: 测试Prefill和Decode阶段的性能扩展性

2.1 Prefill阶段测试:
  - Prompt长度: 512, 1024, 2048, 4096
  - 线程数: 1, 2, 4, 8, 16, 26, 52, 104, 208, 416
  - 迭代次数: 5

2.2 Decode阶段测试:
  - 生成长度: 512, 1024
  - 线程数: 1, 2, 4, 8, 16, 26, 52, 104, 208, 416
  - 迭代次数: 1
  - 使用KV Cache

测试指标:
  - 总时间 (ms)
  - 吞吐量 (tokens/sec)
  - 加速比和并行效率

硬件配置:
  - 8节点 × 2CPU × 26核 = 416核
  - 每个CPU的OpenMP线程数: 26

EOF

echo "✓ 实验设计已保存: $RESULTS_DIR/experiment_design.txt"

# ============================================================================
# 打印实验信息
# ============================================================================

echo ""
echo "============================================================================="
echo "  并行计算课程完整实验"
echo "============================================================================="
echo ""
echo "开始时间: $(date)"
echo "模型路径: $MODEL_PATH"
echo "结果目录: $RESULTS_DIR"
echo ""
echo "实验内容:"
echo "  1. Attention算子性能测试（串行、OpenMP、MPI+OpenMP）"
echo "  2. Qwen3 Prefill阶段性能测试"
echo "  3. Qwen3 Decode阶段性能测试"
echo ""
echo "============================================================================="
echo ""

# ============================================================================
# 实验1: Attention算子测试
# ============================================================================

echo "[实验1] Attention算子性能扩展性测试"
echo "============================================================================="
echo ""

# 检查attention目录是否存在
if [ ! -d "$PROJECT_DIR/attention" ]; then
    echo "⚠ 警告: Attention项目未找到"
    echo "  路径: $PROJECT_DIR/attention"
    echo "  跳过Attention测试..."
    echo ""
else
# 1.1 串行baseline
echo "[1.1] 串行baseline测试..."
cd "$PROJECT_DIR/attention"

# 编译串行版本
if [ ! -f "naive_serial" ]; then
    echo "  编译串行版本..."
    if g++ -std=c++17 -O3 -march=native \
        naive_serial.cpp streaming_serial.cpp \
        -o naive_serial 2>&1 | tee "$RESULTS_DIR/attention_serial_compile.log"; then
        echo "  ✓ 编译成功"
    else
        echo "  ✗ 编译失败，跳过串行测试"
    fi
fi

SERIAL_LOG="$RESULTS_DIR/attention_serial.txt"
echo "  序列长度 | 时间(ms) | 吞吐量(tokens/s)" | tee "$SERIAL_LOG"
echo "  ---------|---------|------------------" | tee -a "$SERIAL_LOG"

for SEQ_LEN in 512 1024 2048 4096 8192; do
    echo "    测试 seq_len=$SEQ_LEN..."
    OUTPUT=$(./naive_serial $SEQ_LEN 128 2>&1)
    TIME=$(echo "$OUTPUT" | grep "Time" | awk '{print $2}')
    if [ -z "$TIME" ] || [ "$TIME" = "0" ]; then
        echo "      跳过（程序未正确输出时间）" | tee -a "$SERIAL_LOG"
        continue
    fi
    THROUGHPUT=$(awk "BEGIN {printf \"%.2f\", $SEQ_LEN * 1000 / $TIME}")
    printf "    %6d   | %7.2f | %15.2f\n" $SEQ_LEN $TIME $THROUGHPUT | tee -a "$SERIAL_LOG"
done
echo "  ✓ 串行baseline完成"
echo ""

# 1.2 OpenMP版本
echo "[1.2] OpenMP并行测试..."

# 编译OpenMP版本
if [ ! -f "streaming_omp" ]; then
    echo "  编译OpenMP版本..."
    g++ -std=c++17 -O3 -march=native -fopenmp \
        streaming_omp.cpp streaming_serial.cpp \
        ../utils/softmax_online.cpp \
        -o streaming_omp -lm 2>&1 | tee "$RESULTS_DIR/attention_omp_compile.log"
fi

THREADS_LIST=(1 2 4 8 16 26 52 104 208 416)
SEQ_LEN=2048  # 中等规模用于线程扩展性测试

OMP_LOG="$RESULTS_DIR/attention_omp.txt"
echo "  线程数 | 时间(ms) | 吞吐量(tokens/s) | 加速比 | 效率" | tee "$OMP_LOG"
echo "  ------|---------|------------------|--------|------" | tee -a "$OMP_LOG"

# 获取串行baseline时间
SERIAL_TIME=$(grep "2048" "$SERIAL_LOG" | awk '{print $2}')

for NUM_THREADS in "${THREADS_LIST[@]}"; do
    echo "    测试 threads=$NUM_THREADS..."
    TIME=$(OMP_NUM_THREADS=$NUM_THREADS ./streaming_omp $SEQ_LEN 128 64 $NUM_THREADS 2>&1 | grep "Time" | awk '{print $2}')
    THROUGHPUT=$(echo "scale=2; $SEQ_LEN * 1000 / $TIME" | bc)
    SPEEDUP=$(echo "scale=2; $SERIAL_TIME / $TIME" | bc)
    EFFICIENCY=$(echo "scale=3; $SPEEDUP / $NUM_THREADS" | bc)
    printf "    %4d   | %7.2f | %15.2f | %6.2f | %.3f\n" \
        $NUM_THREADS $TIME $THROUGHPUT $SPEEDUP $EFFICIENCY | tee -a "$OMP_LOG"
done
echo "  ✓ OpenMP测试完成"
echo ""

# 1.3 MPI+OpenMP混合版本
echo "[1.3] MPI+OpenMP混合并行测试..."

# 编译MPI版本
if [ ! -f "streaming_mpi" ]; then
    echo "  编译MPI版本..."
    /usr/bin/mpicxx -std=c++17 -O3 -march=native -fopenmp \
        streaming_mpi.cpp streaming_omp.cpp streaming_serial.cpp \
        ../utils/softmax_online.cpp \
        -o streaming_mpi -lm 2>&1 | tee "$RESULTS_DIR/attention_mpi_compile.log"
fi

MPI_LOG="$RESULTS_DIR/attention_mpi.txt"
echo "  进程数 | 线程/进程 | 总核数 | 时间(ms) | 加速比 | 效率" | tee "$MPI_LOG"
echo "  ------|----------|--------|---------|-------|------" | tee -a "$MPI_LOG"

OMP_PER_RANK=26
MPI_RANKS_LIST=(1 2 4 8 16)

for NUM_RANKS in "${MPI_RANKS_LIST[@]}"; do
    TOTAL_CORES=$((NUM_RANKS * OMP_PER_RANK))
    echo "    测试 ranks=$NUM_RANKS, cores=$TOTAL_CORES..."

    TIME=$(LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
           OMP_NUM_THREADS=$OMP_PER_RANK \
           /usr/bin/mpirun -np $NUM_RANKS \
           ./streaming_mpi $((SEQ_LEN / NUM_RANKS)) $SEQ_LEN 128 64 \
           2>&1 | grep "Time" | awk '{print $2}')

    THROUGHPUT=$(echo "scale=2; $SEQ_LEN * 1000 / $TIME" | bc)
    SPEEDUP=$(echo "scale=2; $SERIAL_TIME / $TIME" | bc)
    EFFICIENCY=$(echo "scale=3; $SPEEDUP / $TOTAL_CORES" | bc)
    printf "    %4d   |    %4d  |  %4d  | %7.2f | %6.2f | %.3f\n" \
        $NUM_RANKS $OMP_PER_RANK $TOTAL_CORES $TIME $SPEEDUP $EFFICIENCY | tee -a "$MPI_LOG"
done
echo "  ✓ MPI+OpenMP测试完成"
echo ""

# ============================================================================
# 实验2: Qwen3模型测试
# ============================================================================

echo "[实验2] Qwen3模型吞吐量测试"
echo "============================================================================="
echo ""

cd "$PROJECT_DIR/tensor_cpp"

# 检查可执行文件
if [ ! -f "build/benchmark_qwen3" ]; then
    echo "✗ 错误: benchmark_qwen3未编译"
    echo "  请先运行: ../scripts/build_on_server.sh"
    exit 1
fi

BINARY="./build/benchmark_qwen3"

# 2.1 Prefill阶段测试
echo "[2.1] Prefill阶段性能测试..."
echo ""

PREFILL_LOG="$RESULTS_DIR/qwen3_prefill.txt"
echo "  Prompt长度 | 线程数 | 时间(ms) | 吞吐量(tokens/s) | 加速比 | 效率" | tee "$PREFILL_LOG"
echo "  ----------|-------|---------|-----------------|--------|------" | tee -a "$PREFILL_LOG"

for PROMPT_LEN in 512 1024 2048 4096; do
    echo "  测试 prompt_len=$PROMPT_LEN..."

    # 串行baseline
    SERIAL_TIME=$(LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
        OMP_NUM_THREADS=1 $BINARY \
        --model "$MODEL_PATH" \
        --phase prefill \
        --prompt-len $PROMPT_LEN \
        --iters 3 \
        --threads 1 2>&1 | grep "平均时间" | awk '{print $2}')

    printf "  %8d   | %5d | %7.2f | %15.2f | %6.2f | %.3f\n" \
        $PROMPT_LEN 1 $SERIAL_TIME \
        $(echo "scale=2; $PROMPT_LEN * 1000 / $SERIAL_TIME" | bc) \
        1.0 1.0 | tee -a "$PREFILL_LOG"

    # 多线程测试（选择几个关键点）
    for NUM_THREADS in 4 16 52 104 208 416; do
        TIME=$(LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
            OMP_NUM_THREADS=$NUM_THREADS $BINARY \
            --model "$MODEL_PATH" \
            --phase prefill \
            --prompt-len $PROMPT_LEN \
            --iters 3 \
            --threads $NUM_THREADS 2>&1 | grep "平均时间" | awk '{print $2}')

        THROUGHPUT=$(echo "scale=2; $PROMPT_LEN * 1000 / $TIME" | bc)
        SPEEDUP=$(echo "scale=2; $SERIAL_TIME / $TIME" | bc)
        EFFICIENCY=$(echo "scale=3; $SPEEDUP / $NUM_THREADS" | bc)

        printf "  %8d   | %5d | %7.2f | %15.2f | %6.2f | %.3f\n" \
            $PROMPT_LEN $NUM_THREADS $TIME $THROUGHPUT $SPEEDUP $EFFICIENCY | tee -a "$PREFILL_LOG"
    done

    echo "" | tee -a "$PREFILL_LOG"
done
echo "  ✓ Prefill阶段测试完成"
echo ""

# 2.2 Decode阶段测试
echo "[2.2] Decode阶段性能测试..."
echo ""

DECODE_LOG="$RESULTS_DIR/qwen3_decode.txt"
echo "  生成长度 | 线程数 | 时间(ms) | 吞吐量(tokens/s) | 加速比 | 效率" | tee "$DECODE_LOG"
echo "  --------|-------|---------|-----------------|--------|------" | tee -a "$DECODE_LOG"

for GEN_LEN in 512 1024; do
    echo "  测试 gen_len=$GEN_LEN..."

    # 串行baseline
    SERIAL_TIME=$(LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
        OMP_NUM_THREADS=1 $BINARY \
        --model "$MODEL_PATH" \
        --phase decode \
        --gen-len $GEN_LEN \
        --iters 1 \
        --threads 1 2>&1 | grep "平均时间" | awk '{print $2}')

    printf "  %6d   | %5d | %7.2f | %15.2f | %6.2f | %.3f\n" \
        $GEN_LEN 1 $SERIAL_TIME \
        $(echo "scale=2; $GEN_LEN * 1000 / $SERIAL_TIME" | bc) \
        1.0 1.0 | tee -a "$DECODE_LOG"

    # 多线程测试
    for NUM_THREADS in 4 16 52 104 208 416; do
        TIME=$(LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
            OMP_NUM_THREADS=$NUM_THREADS $BINARY \
            --model "$MODEL_PATH" \
            --phase decode \
            --gen-len $GEN_LEN \
            --iters 1 \
            --threads $NUM_THREADS 2>&1 | grep "平均时间" | awk '{print $2}')

        THROUGHPUT=$(echo "scale=2; $GEN_LEN * 1000 / $TIME" | bc)
        SPEEDUP=$(echo "scale=2; $SERIAL_TIME / $TIME" | bc)
        EFFICIENCY=$(echo "scale=3; $SPEEDUP / $NUM_THREADS" | bc)

        printf "  %6d   | %5d | %7.2f | %15.2f | %6.2f | %.3f\n" \
            $GEN_LEN $NUM_THREADS $TIME $THROUGHPUT $SPEEDUP $EFFICIENCY | tee -a "$DECODE_LOG"
    done

    echo "" | tee -a "$DECODE_LOG"
done
echo "  ✓ Decode阶段测试完成"
echo ""

# ============================================================================
# 生成总结报告
# ============================================================================

echo "============================================================================="
echo "  生成总结报告"
echo "============================================================================="
echo ""

cat > "$RESULTS_DIR/SUMMARY.md" << EOF
# 并行计算课程实验总结报告

**实验时间**: $(date)
**实验者**: $(whoami)@$(hostname)
**硬件配置**: 8节点 × 2CPU × 26核 = 416核

---

## 实验设计

详见 \`experiment_design.txt\`

---

## 实验1: Attention算子性能扩展性

### 1.1 串行Baseline

\`\`\`
$(cat $SERIAL_LOG)
\`\`\`

### 1.2 OpenMP并行扩展性 (seq_len=2048)

\`\`\`
$(cat $OMP_LOG)
\`\`\`

**观察**:
- 最优线程数: $(grep -v "^$" $OMP_LOG | tail -n +2 | awk '{print $5}' | sort -n | tail -1) 线程
- 最大加速比: $(grep -v "^$" $OMP_LOG | tail -n +2 | awk '{print $5}' | sort -n | tail -1)x

### 1.3 MPI+OpenMP混合并行 (seq_len=2048)

\`\`\`
$(cat $MPI_LOG)
\`\`\`

**观察**:
- 最优配置: $(grep -v "^$" $MPI_LOG | tail -n +2 | awk '{print $5}' | sort -n | tail -1)x (进程×线程配置)
- 最大加速比: $(grep -v "^$" $MPI_LOG | tail -n +2 | awk '{print $6}' | sort -n | tail -1)x

---

## 实验2: Qwen3模型吞吐量

### 2.1 Prefill阶段

\`\`\`
$(cat $PREFILL_LOG)
\`\`\`

### 2.2 Decode阶段

\`\`\`
$(cat $DECODE_LOG)
\`\`\`

---

## 课程报告要点

### 1. 加速比曲线
- 使用实验1的OpenMP数据绘制加速比vs线程数曲线
- 使用实验1的MPI数据绘制加速比vs总核数曲线

### 2. 并行效率
- OpenMP效率 = 加速比 / 线程数
- MPI+OpenMP效率 = 加速比 / 总核数
- 分析效率下降原因

### 3. 可扩展性分析
- **强扩展性**: 固定问题规模(seq_len=2048)，增加处理器数
- **弱扩展性**: 问题规模随处理器数成比例增长
- **Prefill vs Decode**: 计算密集 vs 内存带宽密集

### 4. 最优处理器数
- Attention: $(grep -v "^$" $OMP_LOG | tail -n +2 | awk '{print $1, $5}' | sort -k2 -n | head -1)
- Prefill: 待分析
- Decode: 待分析

---

## 数据文件

- \`attention_serial.txt\`: 串行baseline数据
- \`attention_omp.txt\`: OpenMP并行数据
- \`attention_mpi.txt\`: MPI+OpenMP混合并行数据
- \`qwen3_prefill.txt\`: Prefill阶段数据
- \`qwen3_decode.txt\`: Decode阶段数据

---

## 原始日志

所有测试的原始输出保存在当前目录。
EOF

echo "✓ 总结报告已生成: $RESULTS_DIR/SUMMARY.md"

# ============================================================================
# 完成
# ============================================================================

echo ""
echo "============================================================================="
echo "  实验完成！"
echo "============================================================================="
echo ""
echo "结束时间: $(date)"
echo "结果目录: $RESULTS_DIR"
echo ""
echo "文件列表:"
ls -lh "$RESULTS_DIR"
echo ""
echo "快速查看总结:"
echo "  cat $RESULTS_DIR/SUMMARY.md"
echo ""
echo "复制到本地（如果有网络）:"
echo "  scp -r $RESULTS_DIR user@local:/path/to/destination/"
echo ""
