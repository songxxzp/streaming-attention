#!/bin/bash
# MPI Prefill Benchmark Script
# 对比Standard和Streaming attention在Qwen3 prefill阶段的性能

set -e

# 配置
MODEL_PATH=${MODEL_PATH:-"/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors"}
NUM_PROCS=${NUM_PROCS:-2}
NUM_THREADS=${NUM_THREADS:-8}
PHASE=${PHASE:-"prefill"}
PROMPT_LEN=${PROMPT_LEN:-128}
ITERS=${ITERS:-5}
METHOD=${METHOD:-"mpi"}

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================"
echo "  MPI Prefill Benchmark - Qwen3"
echo "============================================"
echo "Model: $MODEL_PATH"
echo "MPI Processes: $NUM_PROCS"
echo "OpenMP Threads: $NUM_THREADS"
echo "Phase: $PHASE"
echo "Prompt Length: $PROMPT_LEN"
echo "Iterations: $ITERS"
echo "Method: $METHOD"
echo "============================================"
echo ""

# 设置OpenMP线程数
export OMP_NUM_THREADS=$NUM_THREADS
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# 1. Benchmark Standard Attention
echo -e "${YELLOW}Running Standard Attention...${NC}"
mpirun -np $NUM_PROCS --bind-to none ./benchmark_qwen3 \
    --model "$MODEL_PATH" \
    --phase $PHASE \
    --method $METHOD \
    --attention standard \
    --prompt-len $PROMPT_LEN \
    --iters $ITERS \
    --threads $NUM_THREADS

echo ""
echo "============================================"
echo ""

# 2. Benchmark Streaming Attention
echo -e "${YELLOW}Running Streaming Attention...${NC}"
mpirun -np $NUM_PROCS --bind-to none ./benchmark_qwen3 \
    --model "$MODEL_PATH" \
    --phase $PHASE \
    --method $METHOD \
    --attention streaming \
    --prompt-len $PROMPT_LEN \
    --iters $ITERS \
    --threads $NUM_THREADS

echo ""
echo "============================================"
echo -e "${GREEN}Benchmark Complete!${NC}"
echo "============================================"
echo ""
echo "Tips:"
echo "  - Try different NUM_PROCS: 1, 2, 4, 8"
echo "  - Try different PROMPT_LEN: 32, 64, 128, 256, 512"
echo "  - For longer sequences, streaming should be faster"
echo ""
