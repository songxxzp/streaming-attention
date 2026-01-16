#!/bin/bash
###############################################################################
# 实验4: OpenMP线程扩展性测试
#
# 目标：评估不同线程数下的性能扩展性
# 配置：batch*len = 1*128 (固定)
#       threads = 1/2/4/8/16/26/32
#       method = avx2 (固定)
###############################################################################

set -e  # 遇到错误立即退出

# 配置
MODEL_PATH="/student/2025310707/Qwen3-0.6B/model.safetensors"
BENCHMARK="./build/benchmark_qwen3"
OUTPUT_DIR="results/exp4_thread_scaling"
mkdir -p "$OUTPUT_DIR"

# 序列长度（固定）
SEQ_LEN=128
BATCH_SIZE=1

# 线程数范围
declare -a THREADS=(32 26 16 8 4 2 1)

# 实验配置（仅AVX2）
METHOD="avx2"
declare -a PARALLEL_STRATEGIES=("sequence") # "headwise" 
declare -a ATTENTION_ALGOS=("standard" "online_softmax")

# 迭代配置
ITERS=2
WARMUP=1

echo "============================================================"
echo "       实验4: OpenMP线程扩展性测试"
echo "============================================================"
echo "模型: $MODEL_PATH"
echo "配置: batch=${BATCH_SIZE}, seq_len=${SEQ_LEN} (固定)"
echo "方法: ${METHOD} (固定)"
echo "线程数: ${THREADS[@]}"
echo "并行策略: ${PARALLEL_STRATEGIES[@]}"
echo "Attention算法: ${ATTENTION_ALGOS[@]}"
echo "============================================================"
echo ""

# 结果文件头
RESULTS_FILE="${OUTPUT_DIR}/thread_scaling_results.csv"
echo "threads,parallel_strategy,attention_algo,total_time_ms,time_per_token_ms,throughput_tok_s,speedup,efficiency,timestamp" > "$RESULTS_FILE"

# 首先运行单线程baseline（用于计算加速比）
BASELINE_THROUGHPUT=0
for NUM_THREADS in 1; do
    for STRATEGY in "sequence"; do
        for ALGO in "standard"; do
            echo "------------------------------------------------------------"
            echo "运行baseline: threads=${NUM_THREADS} | ${STRATEGY} | ${ALGO}"
            echo "------------------------------------------------------------"

            OUTPUT="${OUTPUT_DIR}/baseline_threads${NUM_THREADS}_${STRATEGY}_${ALGO}.log"

            LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
            OMP_NUM_THREADS=$NUM_THREADS \
            $BENCHMARK \
                --model "$MODEL_PATH" \
                --method "$METHOD" \
                --parallel-strategy "$STRATEGY" \
                --attention-algo "$ALGO" \
                --batch-size $BATCH_SIZE \
                --prompt-len $SEQ_LEN \
                --phase prefill \
                --iters $ITERS \
                --warmup $WARMUP \
                --threads $NUM_THREADS \
                2>&1 | tee "$OUTPUT"

            BASELINE_THROUGHPUT=$(grep "吞吐量:" "$OUTPUT" | awk '{print $2}' | tr -d ' ')
            echo "Baseline吞吐量: ${BASELINE_THROUGHPUT} tok/s"
            echo ""
        done
    done
done

# 遍历所有线程数和策略组合
for NUM_THREADS in "${THREADS[@]}"; do
    for STRATEGY in "${PARALLEL_STRATEGIES[@]}"; do
        for ALGO in "${ATTENTION_ALGOS[@]}"; do

            echo "------------------------------------------------------------"
            echo "测试: threads=${NUM_THREADS} | ${STRATEGY} | ${ALGO}"
            echo "------------------------------------------------------------"

            # 运行benchmark
            OUTPUT="${OUTPUT_DIR}/threads${NUM_THREADS}_${STRATEGY}_${ALGO}.log"

            srun --mpi=pmix \
                    -p student \
                    -N 1 \
                    --ntasks=1 \
                    --ntasks-per-node=1 \
                    --cpus-per-task=$NUM_THREADS \
                    env OMP_NUM_THREADS=$NUM_THREADS \
                    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
                    $BENCHMARK \
                        --model "$MODEL_PATH" \
                        --method "$METHOD" \
                        --parallel-strategy "$STRATEGY" \
                        --attention-algo "$ALGO" \
                        --batch-size $BATCH_SIZE \
                        --prompt-len $SEQ_LEN \
                        --phase prefill \
                        --iters $ITERS \
                        --warmup $WARMUP \
                        --threads $NUM_THREADS \
                        2>&1 | tee "$OUTPUT"

            # 提取结果
            TOTAL_TIME=$(grep "总时间:" "$OUTPUT" | awk '{print $2}' | tr -d ' ')
            TIME_PER_TOKEN=$(grep "平均时间/token:" "$OUTPUT" | awk '{print $2}' | tr -d ' ')
            THROUGHPUT=$(grep "吞吐量:" "$OUTPUT" | awk '{print $2}' | tr -d ' ')

            # 计算加速比和效率
            SPEEDUP=$(awk "BEGIN {printf \"%.2f\", ${THROUGHPUT} / ${BASELINE_THROUGHPUT}}")
            EFFICIENCY=$(awk "BEGIN {printf \"%.2f\", ${SPEEDUP} / ${NUM_THREADS} * 100}")

            # 写入CSV
            TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
            echo "${NUM_THREADS},${STRATEGY},${ALGO},${TOTAL_TIME},${TIME_PER_TOKEN},${THROUGHPUT},${SPEEDUP},${EFFICIENCY},${TIMESTAMP}" >> "$RESULTS_FILE"

            echo "结果: 总时间=${TOTAL_TIME}ms, 吞吐量=${THROUGHPUT} tok/s"
            echo "      加速比=${SPEEDUP}x, 效率=${EFFICIENCY}%"
            echo ""
        done
    done
done

echo "============================================================"
echo "       实验4完成！"
echo "============================================================"
echo "结果保存在: $RESULTS_FILE"
echo "详细日志: $OUTPUT_DIR/"
echo ""
echo "Baseline吞吐量: ${BASELINE_THROUGHPUT} tok/s (1 thread)"
echo ""
