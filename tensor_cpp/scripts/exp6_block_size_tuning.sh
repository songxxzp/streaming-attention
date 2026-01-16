#!/bin/bash
###############################################################################
# 实验6: Block Size调参实验
#
# 目标：评估不同block size对streaming attention性能的影响
# 配置：batch=1, seq_len=[128, 1024]
#       block_size=[64, 128, 256] (q_block_size = kv_block_size)
###############################################################################

set -e  # 遇到错误立即退出

# 配置
MODEL_PATH="/student/2025310707/Qwen3-0.6B/model.safetensors"
BENCHMARK="./build/benchmark_qwen3"
OUTPUT_DIR="results/exp6_block_size_tuning"
mkdir -p "$OUTPUT_DIR"

# 实验配置
METHOD="avx2"
NUM_THREADS=26
BATCH_SIZE=1  # 固定

# 序列长度
declare -a SEQ_LENS=(128 1024)

# Block size范围 (q_block_size = kv_block_size)
declare -a BLOCK_SIZES=(64 128 256)

# 并行策略 (仅使用sequence + online_softmax)
PARALLEL_STRATEGY="sequence"
ATTENTION_ALGO="online_softmax"

# 迭代配置
ITERS=3
WARMUP=1

echo "============================================================"
echo "       实验6: Block Size调参实验"
echo "============================================================"
echo "模型: $MODEL_PATH"
echo "方法: ${METHOD}"
echo "线程数: ${NUM_THREADS}"
echo "Batch大小: ${BATCH_SIZE} (固定)"
echo "序列长度: ${SEQ_LENS[@]}"
echo "Block Size: ${BLOCK_SIZES[@]} (q_block_size = kv_block_size)"
echo "并行策略: ${PARALLEL_STRATEGY}"
echo "Attention算法: ${ATTENTION_ALGO}"
echo "============================================================"
echo ""

# 结果文件头
RESULTS_FILE="${OUTPUT_DIR}/block_size_tuning_results.csv"
echo "seq_len,block_size,q_block_size,kv_block_size,total_time_ms,time_per_token_ms,throughput_tok_s,timestamp" > "$RESULTS_FILE"

# 遍历所有序列长度和block size组合
for SEQ_LEN in "${SEQ_LENS[@]}"; do
    for BLOCK_SIZE in "${BLOCK_SIZES[@]}"; do

        echo "------------------------------------------------------------"
        echo "测试: seq_len=${SEQ_LEN} | block_size=${BLOCK_SIZE}"
        echo "------------------------------------------------------------"

        # 运行benchmark
        OUTPUT="${OUTPUT_DIR}/seq${SEQ_LEN}_block${BLOCK_SIZE}.log"

        LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
        OMP_NUM_THREADS=$NUM_THREADS \
        $BENCHMARK \
            --model "$MODEL_PATH" \
            --method "$METHOD" \
            --parallel-strategy "$PARALLEL_STRATEGY" \
            --attention-algo "$ATTENTION_ALGO" \
            --batch-size $BATCH_SIZE \
            --prompt-len $SEQ_LEN \
            --q-block-size $BLOCK_SIZE \
            --kv-block-size $BLOCK_SIZE \
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
        echo "${SEQ_LEN},${BLOCK_SIZE},${BLOCK_SIZE},${BLOCK_SIZE},${TOTAL_TIME},${TIME_PER_TOKEN},${THROUGHPUT},${TIMESTAMP}" >> "$RESULTS_FILE"

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
echo "  - 对比不同block size的吞吐量"
echo "  - 查找最优block size配置"
echo "  - 分析block size对不同序列长度的影响"
echo ""
