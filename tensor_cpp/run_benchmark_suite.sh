#!/bin/bash
# @file run_benchmark_suite.sh
# @brief 自动化性能测试脚本：测试不同线程数配置下的性能

set -e

# 配置
MODEL_PATH="${MODEL_PATH:-/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors}"
RESULTS_DIR="${RESULTS_DIR:-./benchmark_results}"
TIMINGS=()

# 创建结果目录
mkdir -p "$RESULTS_DIR"

echo "=================================================="
echo "     Qwen3 Performance Benchmark Suite"
echo "=================================================="
echo "模型路径: $MODEL_PATH"
echo "结果目录: $RESULTS_DIR"
echo "=================================================="
echo ""

# ====================================================================
# Part 1: Attention算子级别性能测试
# ====================================================================
echo "Part 1: Attention算子级别性能测试"
echo "=================================================="

# 测试Standard Attention
echo ""
echo "测试 Standard Attention (不同序列长度)..."
for seq_len in 64 128 256 512 1024; do
    echo "  序列长度: $seq_len"
    OUTPUT="$RESULTS_DIR/attention_standard_seq${seq_len}.txt"
    OMP_NUM_THREADS=16 ./build/benchmark_attention \
        --mode standard \
        --seq-len $seq_len \
        --hidden 128 \
        --iters 50 \
        --threads 16 \
        > "$OUTPUT" 2>&1
    echo "    完成: $OUTPUT"
done

# 测试Streaming Attention
echo ""
echo "测试 Streaming Attention (不同序列长度和块大小)..."
for seq_len in 64 128 256 512 1024; do
    for block_size in 32 64 128; do
        echo "  序列长度: $seq_len, 块大小: $block_size"
        OUTPUT="$RESULTS_DIR/attention_streaming_seq${seq_len}_block${block_size}.txt"
        OMP_NUM_THREADS=16 ./build/benchmark_attention \
            --mode streaming \
            --seq-len $seq_len \
            --hidden 128 \
            --iters 50 \
            --threads 16 \
            --block-size $block_size \
            > "$OUTPUT" 2>&1
        echo "    完成: $OUTPUT"
    done
done

# 测试不同线程数的扩展性
echo ""
echo "测试 Standard Attention 线程扩展性..."
for threads in 1 2 4 8 12 16 20 24 28 32; do
    echo "  线程数: $threads"
    OUTPUT="$RESULTS_DIR/attention_standard_threads${threads}.txt"
    OMP_NUM_THREADS=$threads ./build/benchmark_attention \
        --mode standard \
        --seq-len 512 \
        --hidden 128 \
        --iters 20 \
        --threads $threads \
        > "$OUTPUT" 2>&1
    echo "    完成: $OUTPUT"
done

# ====================================================================
# Part 2: Qwen3模型级别性能测试
# ====================================================================
echo ""
echo "=================================================="
echo "Part 2: Qwen3模型级别性能测试"
echo "=================================================="

# 测试Prefill阶段
echo ""
echo "测试 Prefill 阶段 (不同prompt长度)..."
for prompt_len in 16 32 64 128 256; do
    echo "  Prompt长度: $prompt_len"
    OUTPUT="$RESULTS_DIR/qwen3_prefill_len${prompt_len}.txt"
    OMP_NUM_THREADS=16 ./build/benchmark_qwen3 \
        --model "$MODEL_PATH" \
        --phase prefill \
        --prompt-len $prompt_len \
        --iters 5 \
        --threads 16 \
        > "$OUTPUT" 2>&1
    echo "    完成: $OUTPUT"
done

# 测试Decode阶段
echo ""
echo "测试 Decode 阶段 (不同生成长度)..."
for gen_len in 10 20 50 100; do
    echo "  生成长度: $gen_len"
    OUTPUT="$RESULTS_DIR/qwen3_decode_len${gen_len}.txt"
    OMP_NUM_THREADS=16 ./build/benchmark_qwen3 \
        --model "$MODEL_PATH" \
        --phase decode \
        --gen-len $gen_len \
        --iters 1 \
        --threads 16 \
        > "$OUTPUT" 2>&1
    echo "    完成: $OUTPUT"
done

# 测试线程扩展性 (Prefill)
echo ""
echo "测试 Prefill 阶段线程扩展性..."
for threads in 1 2 4 8 12 16; do
    echo "  线程数: $threads"
    OUTPUT="$RESULTS_DIR/qwen3_prefill_threads${threads}.txt"
    OMP_NUM_THREADS=$threads ./build/benchmark_qwen3 \
        --model "$MODEL_PATH" \
        --phase prefill \
        --prompt-len 64 \
        --iters 5 \
        --threads $threads \
        > "$OUTPUT" 2>&1
    echo "    完成: $OUTPUT"
done

# 测试线程扩展性 (Decode)
echo ""
echo "测试 Decode 阶段线程扩展性..."
for threads in 1 2 4 8 12 16; do
    echo "  线程数: $threads"
    OUTPUT="$RESULTS_DIR/qwen3_decode_threads${threads}.txt"
    OMP_NUM_THREADS=$threads ./build/benchmark_qwen3 \
        --model "$MODEL_PATH" \
        --phase decode \
        --gen-len 20 \
        --iters 1 \
        --threads $threads \
        > "$OUTPUT" 2>&1
    echo "    完成: $OUTPUT"
done

# ====================================================================
# Part 3: 生成对比报告
# ====================================================================
echo ""
echo "=================================================="
echo "生成性能对比报告..."
echo "=================================================="

# 生成汇总报告
REPORT="$RESULTS_DIR/summary_report.txt"
echo "Qwen3 Performance Benchmark Summary Report" > "$REPORT"
echo "Generated: $(date)" >> "$REPORT"
echo "" >> "$REPORT"
echo "==================================================" >> "$REPORT"
echo "" >> "$REPORT"

# Attention算子对比
echo "## Attention算子性能对比" >> "$REPORT"
echo "" >> "$REPORT"
for seq_len in 64 128 256 512 1024; do
    echo "### 序列长度: $seq_len" >> "$REPORT"
    echo "" >> "$REPORT"

    # Standard
    std_file="$RESULTS_DIR/attention_standard_seq${seq_len}.txt"
    if [ -f "$std_file" ]; then
        std_throughput=$(grep "吞吐量:" "$std_file" | awk '{print $2}')
        echo "Standard: $std_throughput tokens/sec" >> "$REPORT"
    fi

    # Streaming (最佳块大小)
    best_throughput=0
    best_block=0
    for block_size in 32 64 128; do
        stream_file="$RESULTS_DIR/attention_streaming_seq${seq_len}_block${block_size}.txt"
        if [ -f "$stream_file" ]; then
            throughput=$(grep "吞吐量:" "$stream_file" | awk '{print $2}')
            if (( $(echo "$throughput > $best_throughput" | bc -l) )); then
                best_throughput=$throughput
                best_block=$block_size
            fi
        fi
    done
    echo "Streaming (block=$best_block): $best_throughput tokens/sec" >> "$REPORT"

    if [ ! -z "$std_throughput" ] && [ ! -z "$best_throughput" ]; then
        speedup=$(echo "scale=2; $best_throughput / $std_throughput" | bc)
        echo "加速比: ${speedup}x" >> "$REPORT"
    fi
    echo "" >> "$REPORT"
done

# Qwen3模型性能对比
echo "" >> "$REPORT"
echo "## Qwen3模型性能对比" >> "$REPORT"
echo "" >> "$REPORT"

# Prefill
echo "### Prefill阶段 (prompt长度=64, 线程数变化)" >> "$REPORT"
echo "" >> "$REPORT"
echo "线程数 | 吞吐量(tokens/sec)" >> "$REPORT"
echo "-------|---------------------" >> "$REPORT"
for threads in 1 2 4 8 12 16; do
    file="$RESULTS_DIR/qwen3_prefill_threads${threads}.txt"
    if [ -f "$file" ]; then
        throughput=$(grep "吞吐量:" "$file" | awk '{print $2}')
        echo "$threads | $throughput" >> "$REPORT"
    fi
done
echo "" >> "$REPORT"

# Decode
echo "### Decode阶段 (生成长度=20, 线程数变化)" >> "$REPORT"
echo "" >> "$REPORT"
echo "线程数 | 吞吐量(tokens/sec)" >> "$REPORT"
echo "-------|---------------------" >> "$REPORT"
for threads in 1 2 4 8 12 16; do
    file="$RESULTS_DIR/qwen3_decode_threads${threads}.txt"
    if [ -f "$file" ]; then
        throughput=$(grep "吞吐量:" "$file" | awk '{print $2}')
        echo "$threads | $throughput" >> "$REPORT"
    fi
done

echo ""
echo "=================================================="
echo "测试完成！"
echo "结果保存在: $RESULTS_DIR"
echo "汇总报告: $REPORT"
echo "=================================================="
