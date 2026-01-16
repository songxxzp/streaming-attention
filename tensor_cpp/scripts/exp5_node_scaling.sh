#!/bin/bash
###############################################################################
# 实验5: 节点扩展性测试 - 相同核心数下单机vs多机
#
# 目标：评估在相同总核心数下，集中在一台机器还是分散在多台机器更好
# 配置：节点数 × 每节点进程数 × 线程数 = 16 × k
#       k = 1/2/4/8/16
#       batch*len = 1*1024 (固定)
###############################################################################

set -e  # 遇到错误立即退出

# 配置
MODEL_PATH="/student/2025310707/Qwen3-0.6B/model.safetensors"
BENCHMARK="./build/benchmark_qwen3"
OUTPUT_DIR="results/exp5_node_scaling"
mkdir -p "$OUTPUT_DIR"

# 实验配置
METHOD="mpi+avx2"
declare -a K_VALUES=(1 2 4 8 16)  # 扩展因子
declare -a SEQ_LENS=(128 1024)     # 序列长度
declare -a PARALLEL_STRATEGIES=("sequence")
declare -a ATTENTION_ALGOS=("standard" "online_softmax")

# 迭代配置
ITERS=2
WARMUP=1
MAX_THREADS=26  # 单机最大线程数
MAX_NODES=8

echo "============================================================"
echo "       实验5: 节点扩展性测试 (相同核心数单机vs多机)"
echo "============================================================"
echo "模型: $MODEL_PATH"
echo "方法: ${METHOD}"
echo "k值: ${K_VALUES[@]} (总核心数 = 16 × k)"
echo "序列长度: ${SEQ_LENS[@]}"
echo "并行策略: ${PARALLEL_STRATEGIES[@]}"
echo "Attention算法: ${ATTENTION_ALGOS[@]}"
echo "最大线程数/进程: ${MAX_THREADS}"
echo "============================================================"
echo ""

# 结果文件头
RESULTS_FILE="${OUTPUT_DIR}/node_scaling_results.csv"
echo "seq_len,k,total_cores,nodes,processes_per_node,total_processes,threads_per_process,parallel_strategy,attention_algo,total_time_ms,time_per_token_ms,throughput_tok_s,timestamp" > "$RESULTS_FILE"

# 遍历所有k值
for K in "${K_VALUES[@]}"; do
    TOTAL_CORES=$((16 * K))

    echo "=========================================="
    echo "测试配置: k=${K}, 总核心数=${TOTAL_CORES}"
    echo "=========================================="

    # 为每个k值生成不同的节点配置
    # 目标：比较相同核心数下单节点vs多节点的性能

    case $K in
        1)
            # k=1, 总核心=16
            declare -a CONFIGS=(
                "1:1:16"  # 1节点×1进程×16线程=16核心
                "1:2:8"   # 1节点×2进程×8线程=16核心
                "2:1:8"
                "2:2:4"
                "4:1:4"
                "4:2:2"
                "8:1:2"
                "8:2:1"
            )
            ;;
        2)
            # k=2, 总核心=32
            declare -a CONFIGS=(
                "1:2:16"
                "2:1:16"
                "2:2:8"
                "4:1:8"
                "4:2:4"
                "8:1:4"
                "8:2:2"
            )
            ;;
        4)
            # k=4, 总核心=64
            declare -a CONFIGS=(
                "2:2:16"
                "4:1:16"
                "4:2:8"
                "8:1:8"
                "8:2:4"
            )
            ;;
        8)
            # k=8, 总核心=128
            declare -a CONFIGS=(
                "4:2:16"
                "8:1:16"
                "8:2:8"
            )
            ;;
        16)
            # k=16, 总核心=256
            declare -a CONFIGS=(
                "8:2:16"
            )
            ;;
    esac

    # 遍历所有配置
    for CONFIG in "${CONFIGS[@]}"; do
        IFS=':' read -r NUM_NODES PROCESSES_PER_NODE THREADS_PER_PROCESS <<< "$CONFIG"

        # 检查线程数是否超过限制
        if [[ $THREADS_PER_PROCESS -gt $MAX_THREADS ]]; then
            echo "跳过: k=${K}, ${NUM_NODES}节点×${PROCESSES_PER_NODE}进程×${THREADS_PER_PROCESS}线程 (线程数>${MAX_THREADS})"
            continue
        fi

        if [[ $NUM_NODES -gt $MAX_NODES ]]; then
            echo "跳过: k=${K}, ${NUM_NODES}节点×${PROCESSES_PER_NODE}进程×${THREADS_PER_PROCESS}线程 (节点数>${MAX_NODES})"
            continue
        fi

        NUM_PROCESSES=$((NUM_NODES * PROCESSES_PER_NODE))

        echo "配置: ${NUM_NODES}节点 × ${PROCESSES_PER_NODE}进程/节点 × ${THREADS_PER_PROCESS}线程/进程 = ${TOTAL_CORES}核心"

        # 遍历所有序列长度
        for SEQ_LEN in "${SEQ_LENS[@]}"; do
            echo "  序列长度: ${SEQ_LEN}"

            for STRATEGY in "${PARALLEL_STRATEGIES[@]}"; do
                for ALGO in "${ATTENTION_ALGOS[@]}"; do

                    echo "    测试: ${STRATEGY} | ${ALGO}"

                    # 运行benchmark
                    OUTPUT="${OUTPUT_DIR}/k${K}_nodes${NUM_NODES}_ppn${PROCESSES_PER_NODE}_tpp${THREADS_PER_PROCESS}_seq${SEQ_LEN}_${STRATEGY}_${ALGO}.log"

                    # SLURM srun命令格式
                    srun --mpi=pmix \
                         -p student \
                         -N $NUM_NODES \
                         --ntasks=$NUM_PROCESSES \
                         --ntasks-per-node=$PROCESSES_PER_NODE \
                         --cpus-per-task=$THREADS_PER_PROCESS \
                         env OMP_NUM_THREADS=$THREADS_PER_PROCESS \
                         LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
                         timeout 1200 \
                         $BENCHMARK \
                            --model "$MODEL_PATH" \
                            --method "$METHOD" \
                            --parallel-strategy "$STRATEGY" \
                            --attention-algo "$ALGO" \
                            --batch-size 1 \
                            --prompt-len $SEQ_LEN \
                            --phase prefill \
                            --iters $ITERS \
                            --warmup $WARMUP \
                            --threads $THREADS_PER_PROCESS \
                            2>&1 | tee "$OUTPUT"

                    # 提取结果
                    TOTAL_TIME=$(grep "总时间:" "$OUTPUT" | awk '{print $2}' | tr -d ' ')
                    TIME_PER_TOKEN=$(grep "平均时间/token:" "$OUTPUT" | awk '{print $2}' | tr -d ' ')
                    THROUGHPUT=$(grep "吞吐量:" "$OUTPUT" | awk '{print $2}' | tr -d ' ')

                    # 写入CSV
                    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
                    echo "${SEQ_LEN},${K},${TOTAL_CORES},${NUM_NODES},${PROCESSES_PER_NODE},${NUM_PROCESSES},${THREADS_PER_PROCESS},${STRATEGY},${ALGO},${TOTAL_TIME},${TIME_PER_TOKEN},${THROUGHPUT},${TIMESTAMP}" >> "$RESULTS_FILE"

                    echo "      结果: 总时间=${TOTAL_TIME}ms, 吞吐量=${THROUGHPUT} tok/s"
                done
            done
            echo ""
        done
    done
done

echo "============================================================"
echo "       实验5完成！"
echo "============================================================"
echo "结果保存在: $RESULTS_FILE"
echo "详细日志: $OUTPUT_DIR/"
echo ""
