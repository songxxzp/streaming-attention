#!/bin/bash
###############################################################################
# 实验6: Block Size调参实验
#
# 目标：评估不同block size对streaming attention性能的影响
# 配置：batch=1, seq_len=[128, 1024]
#       测试多组(q_block_size, kv_block_size)组合
###############################################################################

set -e  # 遇到错误立即退出

# 配置
MODEL_PATH="/student/2025310707/Qwen3-0.6B/model.safetensors"
BENCHMARK="./build/benchmark_qwen3"
OUTPUT_DIR="results/exp6_block_size_tuning"
mkdir -p "$OUTPUT_DIR"

# 实验配置
METHOD="mpi+avx2"
NUM_THREADS=26
BATCH_SIZE=1  # 固定

# 序列长度
declare -a SEQ_LENS=(128 1024)

# Block size配置：格式为 "q_block_size:kv_block_size"
# 测试不同的Q和KV block size组合
declare -a BLOCK_CONFIGS=(
    "32:64"      # 基准配置：小Q块，中等KV块
    "64:128"     # 中等配置：中等Q块，较大KV块
    "64:64"      # 对称配置：中等Q和KV块
    "128:128"    # 对称配置：较大Q和KV块
    "128:256"    # 扩展配置：较大Q块，大KV块
)

# 并行策略 (仅使用sequence + online_softmax)
PARALLEL_STRATEGY="sequence"
ATTENTION_ALGO="online_softmax"

# 迭代配置
ITERS=2
WARMUP=1

echo "============================================================"
echo "       实验6: Block Size调参实验 (扩展版)"
echo "============================================================"
echo "模型: $MODEL_PATH"
echo "方法: ${METHOD}"
echo "线程数: ${NUM_THREADS}"
echo "Batch大小: ${BATCH_SIZE} (固定)"
echo "序列长度: ${SEQ_LENS[@]}"
echo "Block配置数量: ${#BLOCK_CONFIGS[@]}"
echo "并行策略: ${PARALLEL_STRATEGY}"
echo "Attention算法: ${ATTENTION_ALGO}"
echo ""
echo "测试配置:"
for config in "${BLOCK_CONFIGS[@]}"; do
    echo "  - q_block=${config%%:*}, kv_block=${config##*:}"
done
echo "============================================================"
echo ""

# 结果文件头
RESULTS_FILE="${OUTPUT_DIR}/block_size_tuning_results.csv"
echo "seq_len,q_block_size,kv_block_size,total_time_ms,time_per_token_ms,throughput_tok_s,timestamp" > "$RESULTS_FILE"

# 遍历所有序列长度和block配置组合
for SEQ_LEN in "${SEQ_LENS[@]}"; do
    for CONFIG in "${BLOCK_CONFIGS[@]}"; do
        # 解析配置
        Q_BLOCK_SIZE=${CONFIG%%:*}
        KV_BLOCK_SIZE=${CONFIG##*:}

        echo "------------------------------------------------------------"
        echo "测试: seq_len=${SEQ_LEN} | q_block=${Q_BLOCK_SIZE}, kv_block=${KV_BLOCK_SIZE}"
        echo "------------------------------------------------------------"

        # 运行benchmark
        OUTPUT="${OUTPUT_DIR}/seq${SEQ_LEN}_q${Q_BLOCK_SIZE}_kv${KV_BLOCK_SIZE}.log"

        srun --mpi=pmix \
                -p student \
                -N 8 \
                --ntasks=16 \
                --ntasks-per-node=2 \
                --cpus-per-task=$NUM_THREADS \
                env OMP_NUM_THREADS=$NUM_THREADS \
                LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
                $BENCHMARK \
                    --model "$MODEL_PATH" \
                    --method "$METHOD" \
                    --parallel-strategy "$PARALLEL_STRATEGY" \
                    --attention-algo "$ATTENTION_ALGO" \
                    --batch-size $BATCH_SIZE \
                    --prompt-len $SEQ_LEN \
                    --q-block-size $Q_BLOCK_SIZE \
                    --kv-block-size $KV_BLOCK_SIZE \
                    --phase prefill \
                    --iters $ITERS \
                    --warmup $WARMUP \
                    --threads $NUM_THREADS \
                    2>&1 | tee "$OUTPUT"

        # 提取结果
        TOTAL_TIME=$(grep "总时间:" "$OUTPUT" | awk '{print $2}' | tr -d ' ')
        TIME_PER_TOKEN=$(grep "平均时间/token:" "$OUTPUT" | awk '{print $2}' | tr -d ' ')
        THROUGHPUT=$(grep "吞吐量:" "$OUTPUT" | awk '{print $2}' | tr -d ' ')

        # 写入CSV
        TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
        echo "${SEQ_LEN},${Q_BLOCK_SIZE},${KV_BLOCK_SIZE},${TOTAL_TIME},${TIME_PER_TOKEN},${THROUGHPUT},${TIMESTAMP}" >> "$RESULTS_FILE"

        echo "结果: 总时间=${TOTAL_TIME}ms, 吞吐量=${THROUGHPUT} tok/s"
        echo ""
    done
done

echo "============================================================"
echo "       实验6完成！"
echo "============================================================"
echo "结果保存在: $RESULTS_FILE"
echo "详细日志: $OUTPUT_DIR/"
echo ""
echo "分析建议:"
echo "  - 对比不同q_block_size和kv_block_size组合的吞吐量"
echo "  - 分析Q块和KV块大小对性能的独立影响"
echo "  - 查找最优的(q_block, kv_block)配置组合"
echo "  - 评估配置对不同序列长度的敏感性"
echo ""
