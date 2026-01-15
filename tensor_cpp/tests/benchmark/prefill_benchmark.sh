#!/bin/bash
# Performance comparison: Standard vs Streaming attention for prefill

set -e

BENCHMARK="./build/benchmark_qwen3"
MODEL_PATH="--model /media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors"

echo "========================================="
echo "  Prefill Performance Benchmark"
echo "  Standard vs Streaming Attention"
echo "========================================="
echo ""

# Test different prefill lengths
LENGTHS=(4 8 16 32 64 128)
THREADS=4

for len in "${LENGTHS[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Prefill Length: $len tokens"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Generate prompt tokens (simple pattern)
    PROMPT="151644,872"
    for ((i=2; i<len; i++)); do
        PROMPT="$PROMPT,198"
    done

    # Test Standard Attention
    echo ""
    echo "[Standard Attention]"
    echo "Running..."
    start_std=$(date +%s%N)
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
        OMP_NUM_THREADS=$THREADS $BENCHMARK \
        $MODEL_PATH \
        --verify $PROMPT \
        --gen-len 0 \
        --attention standard 2>&1 | grep -E "(Step 1|Forward pass)" | head -1
    end_std=$(date +%s%N)
    time_std_ms=$(( (end_std - start_std) / 1000000 ))

    # Test Streaming Attention
    echo ""
    echo "[Streaming Attention]"
    echo "Running..."
    start_stream=$(date +%s%N)
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
        OMP_NUM_THREADS=$THREADS $BENCHMARK \
        $MODEL_PATH \
        --verify $PROMPT \
        --gen-len 0 \
        --attention streaming 2>&1 | grep -E "(Step 1|Forward pass)" | head -1
    end_stream=$(date +%s%N)
    time_stream_ms=$(( (end_stream - start_stream) / 1000000 ))

    # Calculate speedup
    if [ $time_stream_ms -gt 0 ]; then
        speedup=$(echo "scale=2; $time_std_ms / $time_stream_ms" | bc)
        faster=$(echo "$speedup > 1" | bc)
        if [ $faster -eq 1 ]; then
            result="Streaming is ${speedup}x faster ✓"
        else
            speedup_inv=$(echo "scale=2; $time_stream_ms / $time_std_ms" | bc)
            result="Standard is ${speedup_inv}x faster"
        fi
    else
        result="Too fast to measure accurately"
    fi

    echo ""
    echo "Results:"
    echo "  Standard:  ${time_std_ms} ms"
    echo "  Streaming: ${time_stream_ms} ms"
    echo "  └─ $result"
    echo ""
done

echo "========================================="
echo "  Test Complete!"
echo "========================================="
