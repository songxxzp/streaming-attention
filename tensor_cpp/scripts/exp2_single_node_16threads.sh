#!/bin/bash
###############################################################################
# 实验2: 单机16线程性能测试
#
# 目标：评估单机多线程环境下不同序列长度、batch大小和并行策略的性能
# 配置：batch*len = 1*128 / 8*128 / 1*1024
###############################################################################

set -e  # 遇到错误立即退出

# 配置
MODEL_PATH="/student/2025310707/Qwen3-0.6B/model.safetensors"
BENCHMARK="./build/benchmark_qwen3"
OUTPUT_DIR="results/exp2_single_node_16threads"
mkdir -p "$OUTPUT_DIR"

# 实验配置
METHOD="mpi+avx2"
NUM_THREADS=26
declare -a SEQ_LENS=(128 1024)  # 序列长度
declare -a BATCH_SIZES=(1 8)    # batch大小

# 并行策略组合
declare -a PARALLEL_STRATEGIES=("sequence")
declare -a ATTENTION_ALGOS=("standard" "online_softmax")

# 迭代配置
ITERS=2
WARMUP=1

echo "============================================================"
echo "       实验2: 单机26线程性能测试"
echo "============================================================"
echo "模型: $MODEL_PATH"
echo "方法: ${METHOD}"
echo "线程数: ${NUM_THREADS}"
echo "序列长度: ${SEQ_LENS[@]}"
echo "Batch大小: ${BATCH_SIZES[@]}"
echo "并行策略: ${PARALLEL_STRATEGIES[@]}"
echo "Attention算法: ${ATTENTION_ALGOS[@]}"
echo "============================================================"
echo ""

# 结果文件头
RESULTS_FILE="${OUTPUT_DIR}/single_node_16threads_results.csv"
echo "batch_size,seq_len,parallel_strategy,attention_algo,total_time_ms,time_per_token_ms,throughput_tok_s,timestamp" > "$RESULTS_FILE"

# 遍历所有batch大小、序列长度和策略组合
for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    for SEQ_LEN in "${SEQ_LENS[@]}"; do
        for STRATEGY in "${PARALLEL_STRATEGIES[@]}"; do
            for ALGO in "${ATTENTION_ALGOS[@]}"; do

                # 跳过 batch=8 且 seq_len=1024 的组合
                if [[ $BATCH_SIZE -eq 8 && $SEQ_LEN -eq 1024 ]]; then
                    echo "跳过: batch=${BATCH_SIZE}, seq_len=${SEQ_LEN} (计算量太大)"
                    continue
                fi

                echo "------------------------------------------------------------"
                echo "测试: batch=${BATCH_SIZE} | seq_len=${SEQ_LEN} | ${STRATEGY} | ${ALGO}"
                echo "------------------------------------------------------------"

                # 运行benchmark
                OUTPUT="${OUTPUT_DIR}/batch${BATCH_SIZE}_seq${SEQ_LEN}_${STRATEGY}_${ALGO}.log"

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

                # 写入CSV
                TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
                echo "${BATCH_SIZE},${SEQ_LEN},${STRATEGY},${ALGO},${TOTAL_TIME},${TIME_PER_TOKEN},${THROUGHPUT},${TIMESTAMP}" >> "$RESULTS_FILE"

                echo "结果: 总时间=${TOTAL_TIME}ms, 吞吐量=${THROUGHPUT} tok/s"
                echo ""
            done
        done
    done
done

echo "============================================================"
echo "       实验2完成！"
echo "============================================================"
echo "结果保存在: $RESULTS_FILE"
echo "详细日志: $OUTPUT_DIR/"
echo ""
