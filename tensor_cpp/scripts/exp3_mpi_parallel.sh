#!/bin/bash
###############################################################################
# 实验3: MPI并行性能测试
#
# 目标：评估多节点MPI环境下不同节点数和序列长度的性能
# 配置：1/2/4/8 nodes, 16 threads/node
#       batch*len = 1*128 / 1*1024
###############################################################################

set -e  # 遇到错误立即退出

# 配置
MODEL_PATH="/student/2025310707/Qwen3-0.6B/model.safetensors"
BENCHMARK="./build/benchmark_qwen3"
OUTPUT_DIR="results/exp3_mpi_parallel"
mkdir -p "$OUTPUT_DIR"

# 实验配置
METHOD="mpi+avx2"
NUM_THREADS_PER_NODE=26
declare -a NODES=(1 2 4 8)  # 节点数
declare -a SEQ_LENS=(128 1024)  # 序列长度
declare -a BATCH_SIZES=(1)  # batch size

# 并行策略组合
declare -a PARALLEL_STRATEGIES=("sequence")
declare -a ATTENTION_ALGOS=("standard" "online_softmax")

# 迭代配置
ITERS=2
WARMUP=1

echo "============================================================"
echo "       实验3: MPI并行性能测试"
echo "============================================================"
echo "模型: $MODEL_PATH"
echo "方法: ${METHOD}"
echo "节点配置: ${NODES[@]} nodes × ${NUM_THREADS_PER_NODE} threads"
echo "序列长度: ${SEQ_LENS[@]}"
echo "Batch size: ${BATCH_SIZES[@]}"
echo "并行策略: ${PARALLEL_STRATEGIES[@]}"
echo "Attention算法: ${ATTENTION_ALGOS[@]}"
echo "============================================================"
echo ""

# 结果文件头
RESULTS_FILE="${OUTPUT_DIR}/mpi_parallel_results.csv"
echo "nodes,processes_per_node,total_processes,threads,seq_len,batch_size,parallel_strategy,attention_algo,total_time_ms,time_per_token_ms,throughput_tok_s,timestamp" > "$RESULTS_FILE"

# 遍历所有配置组合
for NUM_NODES in "${NODES[@]}"; do
    # 根据节点数确定每节点的进程数
    # - 单节点时，测试每节点1或2个进程
    # - 多节点时，仅测试每节点2个进程
    if [[ $NUM_NODES -eq 1 ]]; then
        declare -a PROCESSES_PER_NODE_LIST=(1 2)
    else
        declare -a PROCESSES_PER_NODE_LIST=(2)
    fi

    for PROCESSES_PER_NODE in "${PROCESSES_PER_NODE_LIST[@]}"; do
        NUM_PROCESSES=$((NUM_NODES * PROCESSES_PER_NODE))

        for SEQ_LEN in "${SEQ_LENS[@]}"; do

        for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
            # 序列长度 > 128时，只使用batch_size=1
            if [[ $SEQ_LEN -gt 128 && $BATCH_SIZE -ne 1 ]]; then
                echo "跳过: nodes=${NUM_NODES}, batch=${BATCH_SIZE}, seq_len=${SEQ_LEN} (seq_len>128仅测试batch=1)"
                continue
            fi

            # 序列长度 > 128时，只使用num_nodes >= 4
            if [[ $SEQ_LEN -gt 128 && $NUM_NODES -lt 4 ]]; then
                echo "跳过: nodes=${NUM_NODES}, batch=${BATCH_SIZE}, seq_len=${SEQ_LEN} (seq_len>128仅测试num_nodes>=4)"
                continue
            fi

            for STRATEGY in "${PARALLEL_STRATEGIES[@]}"; do
                for ALGO in "${ATTENTION_ALGOS[@]}"; do

                    echo "------------------------------------------------------------"
                    echo "测试: nodes=${NUM_NODES} | ppn=${PROCESSES_PER_NODE} | batch=${BATCH_SIZE} | seq_len=${SEQ_LEN} | ${STRATEGY} | ${ALGO}"
                    echo "------------------------------------------------------------"

                    # 运行benchmark
                    OUTPUT="${OUTPUT_DIR}/nodes${NUM_NODES}_ppn${PROCESSES_PER_NODE}_batch${BATCH_SIZE}_seq${SEQ_LEN}_${STRATEGY}_${ALGO}.log"

                    # SLURM srun命令格式
                    srun --mpi=pmix \
                         -p student \
                         -N $NUM_NODES \
                         --ntasks=$NUM_PROCESSES \
                         --ntasks-per-node=$PROCESSES_PER_NODE \
                         --cpus-per-task=$NUM_THREADS_PER_NODE \
                         env OMP_NUM_THREADS=$NUM_THREADS_PER_NODE \
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
                            --threads $NUM_THREADS_PER_NODE \
                            --q-block-size 64 \
                            --kv-block-size 64 \
                            2>&1 | tee "$OUTPUT"

                    # 提取结果
                    TOTAL_TIME=$(grep "总时间:" "$OUTPUT" | awk '{print $2}' | tr -d ' ')
                    TIME_PER_TOKEN=$(grep "平均时间/token:" "$OUTPUT" | awk '{print $2}' | tr -d ' ')
                    THROUGHPUT=$(grep "吞吐量:" "$OUTPUT" | awk '{print $2}' | tr -d ' ')

                    # 写入CSV
                    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
                    TOTAL_THREADS=$((NUM_NODES * NUM_THREADS_PER_NODE))
                    echo "${NUM_NODES},${PROCESSES_PER_NODE},${NUM_PROCESSES},${TOTAL_THREADS},${SEQ_LEN},${BATCH_SIZE},${STRATEGY},${ALGO},${TOTAL_TIME},${TIME_PER_TOKEN},${THROUGHPUT},${TIMESTAMP}" >> "$RESULTS_FILE"

                    echo "结果: 总时间=${TOTAL_TIME}ms, 吞吐量=${THROUGHPUT} tok/s"
                    echo ""
                done
            done
        done
        done  # SEQ_LEN loop
        done  # PROCESSES_PER_NODE loop
done

echo "============================================================"
echo "       实验3完成！"
echo "============================================================"
echo "结果保存在: $RESULTS_FILE"
echo "详细日志: $OUTPUT_DIR/"
echo ""
