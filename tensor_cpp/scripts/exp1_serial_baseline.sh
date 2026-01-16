#!/bin/bash
###############################################################################
# 实验1: 串行Baseline性能测试
#
# 目标：评估单线程环境下不同优化策略和序列长度的性能
# 配置：batch*len = 1*(16/32/64/128)
###############################################################################

set -e  # 遇到错误立即退出

# 配置
MODEL_PATH="/student/2025310707/Qwen3-0.6B/model.safetensors"
BENCHMARK="./build/benchmark_qwen3"
OUTPUT_DIR="results/exp1_serial_baseline"
mkdir -p "$OUTPUT_DIR"

# 序列长度
declare -a SEQ_LENS=(16 32 64 128)
BATCH_SIZE=1

# 实验配置组合
declare -a METHODS=("baseline" "avx2")
declare -a PARALLEL_STRATEGIES=("headwise")  #  "sequence"
declare -a ATTENTION_ALGOS=("standard")  #  "online_softmax"

# 迭代配置
ITERS=2
WARMUP=1

echo "============================================================"
echo "          实验1: 串行Baseline性能测试"
echo "============================================================"
echo "模型: $MODEL_PATH"
echo "配置: batch=${BATCH_SIZE}, seq_lens=${SEQ_LENS[@]}, iters=${ITERS}, warmup=${WARMUP}"
echo "OMP线程: 1 (串行)"
echo "============================================================"
echo ""

# 结果文件头
RESULTS_FILE="${OUTPUT_DIR}/serial_baseline_results.csv"
echo "seq_len,method,parallel_strategy,attention_algo,total_time_ms,time_per_token_ms,throughput_tok_s,timestamp" > "$RESULTS_FILE"

# 遍历所有序列长度和方法组合
for SEQ_LEN in "${SEQ_LENS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        for STRATEGY in "${PARALLEL_STRATEGIES[@]}"; do
            for ALGO in "${ATTENTION_ALGOS[@]}"; do

                # 跳过 sequence + standard 组合
                if [[ "$STRATEGY" == "sequence" && "$ALGO" == "standard" ]]; then
                    echo "跳过: seq_len=${SEQ_LEN}, ${METHOD} + ${STRATEGY} + ${ALGO} (不推荐)"
                    continue
                fi

                echo "------------------------------------------------------------"
                echo "测试: seq_len=${SEQ_LEN} | ${METHOD} | ${STRATEGY} | ${ALGO}"
                echo "------------------------------------------------------------"

                # 运行benchmark
                OUTPUT="${OUTPUT_DIR}/seq${SEQ_LEN}_${METHOD}_${STRATEGY}_${ALGO}.log"

                LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
                OMP_NUM_THREADS=1 \
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
                    --threads 1 \
                    2>&1 | tee "$OUTPUT"

                # 提取结果
                TOTAL_TIME=$(grep "总时间:" "$OUTPUT" | awk '{print $2}' | tr -d ' ')
                TIME_PER_TOKEN=$(grep "平均时间/token:" "$OUTPUT" | awk '{print $2}' | tr -d ' ')
                THROUGHPUT=$(grep "吞吐量:" "$OUTPUT" | awk '{print $2}' | tr -d ' ')

                # 写入CSV
                TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
                echo "${SEQ_LEN},${METHOD},${STRATEGY},${ALGO},${TOTAL_TIME},${TIME_PER_TOKEN},${THROUGHPUT},${TIMESTAMP}" >> "$RESULTS_FILE"

                echo "结果: 总时间=${TOTAL_TIME}ms, 吞吐量=${THROUGHPUT} tok/s"
                echo ""
            done
        done
    done
done

echo "============================================================"
echo "          实验1完成！"
echo "============================================================"
echo "结果保存在: $RESULTS_FILE"
echo "详细日志: $OUTPUT_DIR/"
echo ""
