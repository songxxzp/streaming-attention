#!/bin/bash
# @file quick_test.sh
# @brief 快速验证脚本：小数据量测试

set -e

echo "=================================================="
echo "       Quick Benchmark Verification"
echo "=================================================="
echo ""

# 创建结果目录
RESULTS_DIR="./benchmark_results_quick"
mkdir -p "$RESULTS_DIR"

# ====================================================================
# Part 1: Attention算子快速测试
# ====================================================================
echo "Part 1: Attention算子快速测试"
echo "=================================================="

echo ""
echo "1.1 Standard Attention (seq_len=128, threads=4)..."
OMP_NUM_THREADS=4 ./build/benchmark_attention \
    --mode standard \
    --seq-len 128 \
    --hidden 128 \
    --iters 10 \
    --threads 4 \
    > "$RESULTS_DIR/attention_standard_quick.txt" 2>&1
echo "  ✓ 完成"

echo ""
echo "1.2 Streaming Attention (seq_len=128, block=64, threads=4)..."
OMP_NUM_THREADS=4 ./build/benchmark_attention \
    --mode streaming \
    --seq-len 128 \
    --hidden 128 \
    --iters 10 \
    --threads 4 \
    --block-size 64 \
    > "$RESULTS_DIR/attention_streaming_quick.txt" 2>&1
echo "  ✓ 完成"

echo ""
echo "1.3 线程扩展性测试 (seq_len=256, threads=1,2,4,8)..."
for threads in 1 2 4 8; do
    echo "  - 线程数: $threads"
    OMP_NUM_THREADS=$threads ./build/benchmark_attention \
        --mode standard \
        --seq-len 256 \
        --hidden 128 \
        --iters 5 \
        --threads $threads \
        > "$RESULTS_DIR/attention_scalability_t${threads}.txt" 2>&1
done
echo "  ✓ 完成"

# ====================================================================
# Part 2: Qwen3模型快速测试
# ====================================================================
echo ""
echo "Part 2: Qwen3模型快速测试"
echo "=================================================="

# 查找模型文件
MODEL_PATH=""

# 常见模型路径
POSSIBLE_PATHS=(
    "/student/2025310707/Qwen3-0.6B/model.safetensors"
    "/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors"
    "/home/$(whoami)/checkpoints/Qwen3-0.6B/model.safetensors"
    "/home/$(whoami)/models/Qwen3-0.6B/model.safetensors"
    "~/checkpoints/Qwen3-0.6B/model.safetensors"
    "./models/Qwen3-0.6B/model.safetensors"
)

for path in "${POSSIBLE_PATHS[@]}"; do
    expanded_path="${path/#\~/$HOME}"
    if [ -f "$expanded_path" ]; then
        MODEL_PATH="$expanded_path"
        break
    fi
done

if [ -z "$MODEL_PATH" ]; then
    echo "⚠ 警告: 未找到Qwen3模型文件"
    echo "  请指定模型路径:"
    echo "  export MODEL_PATH=/path/to/Qwen3-0.6B/model.safetensors"
    echo "  或修改脚本中的MODEL_PATH变量"
    echo ""
    echo "  跳过Qwen3测试..."
else
    echo "✓ 找到模型: $MODEL_PATH"
fi

echo ""
echo "2.1 Prefill阶段 (prompt_len=16, threads=4)..."
if [ -n "$MODEL_PATH" ]; then
    OMP_NUM_THREADS=4 ./build/benchmark_qwen3 \
        --model "$MODEL_PATH" \
        --phase prefill \
        --prompt-len 16 \
        --iters 3 \
        --threads 4 \
        > "$RESULTS_DIR/qwen3_prefill_quick.txt" 2>&1
    echo "  ✓ 完成"
else
    echo "  ⊗ 跳过（未找到模型文件）"
fi

echo ""
echo "2.2 Decode阶段 (gen_len=10, threads=4)..."
if [ -n "$MODEL_PATH" ]; then
    OMP_NUM_THREADS=4 ./build/benchmark_qwen3 \
        --model "$MODEL_PATH" \
        --phase decode \
        --gen-len 10 \
        --iters 1 \
        --threads 4 \
        > "$RESULTS_DIR/qwen3_decode_quick.txt" 2>&1
    echo "  ✓ 完成"
else
    echo "  ⊗ 跳过（未找到模型文件）"
fi

# ====================================================================
# Part 3: 生成快速对比报告
# ====================================================================
echo ""
echo "=================================================="
echo "生成快速对比报告..."
echo "=================================================="

REPORT="$RESULTS_DIR/quick_summary.txt"
echo "Qwen3 快速性能测试报告" > "$REPORT"
echo "生成时间: $(date)" >> "$REPORT"
echo "" >> "$REPORT"
echo "==================================================" >> "$REPORT"
echo "" >> "$REPORT"

# Attention对比
echo "## Attention算子对比 (seq_len=128, threads=4)" >> "$REPORT"
echo "" >> "$REPORT"

std_file="$RESULTS_DIR/attention_standard_quick.txt"
stream_file="$RESULTS_DIR/attention_streaming_quick.txt"

if [ -f "$std_file" ]; then
    std_time=$(grep "平均时间:" "$std_file" | awk '{print $2}')
    std_throughput=$(grep "吞吐量:" "$std_file" | awk '{print $2}')
    echo "Standard Attention:" >> "$REPORT"
    echo "  平均时间: ${std_time} ms/iter" >> "$REPORT"
    echo "  吞吐量: ${std_throughput} tokens/sec" >> "$REPORT"
fi

if [ -f "$stream_file" ]; then
    stream_time=$(grep "平均时间:" "$stream_file" | awk '{print $2}')
    stream_throughput=$(grep "吞吐量:" "$stream_file" | awk '{print $2}')
    echo "" >> "$REPORT"
    echo "Streaming Attention:" >> "$REPORT"
    echo "  平均时间: ${stream_time} ms/iter" >> "$REPORT"
    echo "  吞吐量: ${stream_throughput} tokens/sec" >> "$REPORT"
fi

if [ -f "$std_file" ] && [ -f "$stream_file" ]; then
    speedup=$(echo "scale=2; $std_time / $stream_time" | bc)
    echo "" >> "$REPORT"
    echo "加速比: ${speedup}x" >> "$REPORT"
fi

echo "" >> "$REPORT"
echo "## 线程扩展性 (seq_len=256)" >> "$REPORT"
echo "" >> "$REPORT"
echo "线程数 | 吞吐量(tokens/sec) | 加速比" >> "$REPORT"
echo "-------|-------------------|-------" >> "$REPORT"

base_throughput=0
for threads in 1 2 4 8; do
    file="$RESULTS_DIR/attention_scalability_t${threads}.txt"
    if [ -f "$file" ]; then
        throughput=$(grep "吞吐量:" "$file" | awk '{print $2}')
        if [ $threads -eq 1 ]; then
            base_throughput=$throughput
            speedup="1.00"
        else
            speedup=$(echo "scale=2; $throughput / $base_throughput" | bc)
        fi
        printf "%-6d | %-17s | %s\n" $threads "$throughput" "$speedup" >> "$REPORT"
    fi
done

echo "" >> "$REPORT"
echo "## Qwen3模型性能 (threads=4)" >> "$REPORT"
echo "" >> "$REPORT"

# Prefill
prefill_file="$RESULTS_DIR/qwen3_prefill_quick.txt"
if [ -f "$prefill_file" ]; then
    prefill_throughput=$(grep "吞吐量:" "$prefill_file" | awk '{print $2}')
    echo "Prefill阶段: ${prefill_throughput} tokens/sec" >> "$REPORT"
fi

# Decode
decode_file="$RESULTS_DIR/qwen3_decode_quick.txt"
if [ -f "$decode_file" ]; then
    decode_throughput=$(grep "吞吐量:" "$decode_file" | awk '{print $2}')
    echo "Decode阶段: ${decode_throughput} tokens/sec" >> "$REPORT"
fi

echo "" >> "$REPORT"
echo "==================================================" >> "$REPORT"
echo "详细结果保存在: $RESULTS_DIR" >> "$REPORT"
echo "==================================================" >> "$REPORT"

# 显示报告摘要
echo ""
cat "$REPORT"

echo ""
echo "=================================================="
echo "快速测试完成！"
echo "详细结果: $RESULTS_DIR"
echo "汇总报告: $REPORT"
echo "=================================================="
