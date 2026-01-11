#!/bin/bash
# Qwen3吞吐量测试脚本 - Prefill和Decode阶段
# 测试配置: Prefill长度4096, Decode长度1024
#
# 用法: ./test_qwen3_throughput.sh [model_path]

set -e

# 配置参数
# 如果没有提供模型路径，自动查找
if [ -z "$1" ]; then
    POSSIBLE_PATHS=(
        "/student/2025310707/Qwen3-0.6B/model.safetensors"
        "/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors"
        "/home/$(whoami)/checkpoints/Qwen3-0.6B/model.safetensors"
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
        echo "✗ 错误: 未找到Qwen3模型文件"
        echo ""
        echo "用法: $0 <model_path>"
        echo "示例: $0 /path/to/Qwen3-0.6B/model.safetensors"
        exit 1
    fi
else
    MODEL_PATH="$1"
fi

PREFILL_LEN=4096      # Prefill阶段长度
DECODE_LEN=1024       # Decode阶段生成长度
ITERS=5               # 每个配置的迭代次数

# 线程数配置
# 对于8节点×2CPU×26核=416核，测试关键配置点
THREADS_LIST=(1 2 4 8 16 26 52 104 208 416)

# 路径配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")/tensor_cpp"
RESULTS_DIR="$PROJECT_DIR/results/qwen3_throughput_$(date +%Y%m%d_%H%M%S)"

# 创建结果目录
mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "  Qwen3吞吐量测试"
echo "========================================"
echo "模型路径: $MODEL_PATH"
echo "Prefill长度: $PREFILL_LEN"
echo "Decode长度: $DECODE_LEN"
echo "迭代次数: $ITERS"
echo "结果目录: $RESULTS_DIR"
echo "========================================"
echo ""

# 检查模型文件
if [ ! -f "$MODEL_PATH" ]; then
    echo "✗ 错误: 模型文件不存在: $MODEL_PATH"
    echo ""
    echo "用法: $0 <model_path>"
    echo "示例: $0 /path/to/Qwen3-0.6B/model.safetensors"
    exit 1
fi

# 进入项目目录
cd "$PROJECT_DIR"

# 检查benchmark_qwen3是否存在
if [ ! -f "build/benchmark_qwen3" ]; then
    echo "✗ 错误: benchmark_qwen3未编译"
    echo "请先运行: $SCRIPT_DIR/build_on_server.sh"
    exit 1
fi

BINARY="$PROJECT_DIR/build/benchmark_qwen3"

# 设置运行环境
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# ============================================================================
# 1. Prefill阶段测试 (串行)
# ============================================================================
echo "[1/4] 测试Prefill阶段 (串行)..."
echo ""

LOG_FILE="$RESULTS_DIR/prefill_serial_log.txt"
ITER=0
while [ $ITER -lt $ITERS ]; do
    echo "迭代 $((ITER+1))/$ITERS"
    $BINARY \
        --model "$MODEL_PATH" \
        --phase prefill \
        --prompt-len $PREFILL_LEN \
        --iters 1 \
        --threads 1 \
        2>&1 | tee -a "$LOG_FILE"
    ITER=$((ITER+1))
done

echo "✓ Prefill串行测试完成"
echo ""

# ============================================================================
# 2. Prefill阶段测试 (OpenMP多线程)
# ============================================================================
echo "[2/2] 测试Prefill阶段 (OpenMP)..."
echo ""

for NUM_THREADS in "${THREADS_LIST[@]}"; do
    LOG_FILE="$RESULTS_DIR/prefill_threads_${NUM_THREADS}_log.txt"
    echo "线程数: $NUM_THREADS"

    ITER=0
    while [ $ITER -lt $ITERS ]; do
        echo "  迭代 $((ITER+1))/$ITERS"
        OMP_NUM_THREADS=$NUM_THREADS $BINARY \
            --model "$MODEL_PATH" \
            --phase prefill \
            --prompt-len $PREFILL_LEN \
            --iters 1 \
            --threads $NUM_THREADS \
            2>&1 | tee -a "$LOG_FILE"
        ITER=$((ITER+1))
    done
    echo ""
done

echo "✓ Prefill多线程测试完成"
echo ""

# ============================================================================
# 3. Decode阶段测试 (串行)
# ============================================================================
echo "[3/3] 测试Decode阶段 (串行)..."
echo ""

LOG_FILE="$RESULTS_DIR/decode_serial_log.txt"
ITER=0
while [ $ITER -lt $ITERS ]; do
    echo "迭代 $((ITER+1))/$ITERS"
    $BINARY \
        --model "$MODEL_PATH" \
        --phase decode \
        --gen-len $DECODE_LEN \
        --iters 1 \
        --threads 1 \
        2>&1 | tee -a "$LOG_FILE"
    ITER=$((ITER+1))
done

echo "✓ Decode串行测试完成"
echo ""

# ============================================================================
# 4. Decode阶段测试 (OpenMP多线程)
# ============================================================================
echo "[4/4] 测试Decode阶段 (OpenMP)..."
echo ""

for NUM_THREADS in "${THREADS_LIST[@]}"; do
    LOG_FILE="$RESULTS_DIR/decode_threads_${NUM_THREADS}_log.txt"
    echo "线程数: $NUM_THREADS"

    ITER=0
    while [ $ITER -lt $ITERS ]; do
        echo "  迭代 $((ITER+1))/$ITERS"
        OMP_NUM_THREADS=$NUM_THREADS $BINARY \
            --model "$MODEL_PATH" \
            --phase decode \
            --gen-len $DECODE_LEN \
            --iters 1 \
            --threads $NUM_THREADS \
            2>&1 | tee -a "$LOG_FILE"
        ITER=$((ITER+1))
    done
    echo ""
done

echo "✓ Decode多线程测试完成"
echo ""

# ============================================================================
# 生成性能报告
# ============================================================================
echo "========================================"
echo "  生成性能报告..."
echo "========================================"

cd "$RESULTS_DIR"

# 生成Python分析脚本
cat > analyze_results.py << 'PYTHON_EOF'
#!/usr/bin/env python3
import re
import os

def extract_time_from_log(filename):
    """从日志文件中提取平均时间"""
    if not os.path.exists(filename):
        return None

    times = []
    with open(filename, 'r') as f:
        content = f.read()
        # 查找 "总时间" 或 "平均时间"
        matches = re.findall(r'(?:总时间|平均时间):\s*([\d.]+)\s*ms', content)
        if matches:
            return float(matches[0])

    return None

def extract_throughput_from_log(filename):
    """从日志文件中提取吞吐量"""
    if not os.path.exists(filename):
        return None

    with open(filename, 'r') as f:
        content = f.read()
        # 查找 "吞吐量"
        match = re.search(r'吞吐量:\s*([\d.]+)\s*tokens/sec', content)
        if match:
            return float(match.group(1))

    return None

print("=" * 60)
print("Qwen3吞吐量测试总结")
print("=" * 60)
print()

# Prefill阶段
print("[Prefill阶段]")
print(f"序列长度: {os.environ.get('PREFILL_LEN', 4096)}")
print()

threads = [1, 2, 4, 8, 16, 26, 52, 104, 208, 416]
serial_time = extract_time_from_log('prefill_serial_log.txt')
serial_throughput = extract_throughput_from_log('prefill_serial_log.txt')

if serial_time:
    print(f"串行: 时间={serial_time:.2f}ms, 吞吐量={serial_throughput:.2f} tokens/sec")
    print()

print("线程数 | 时间(ms) | 吞吐量(tokens/s) | 加速比")
print("-" * 60)

for t in threads:
    log_file = f'prefill_threads_{t}_log.txt'
    time_val = extract_time_from_log(log_file)
    throughput = extract_throughput_from_log(log_file)

    if time_val:
        speedup = serial_time / time_val if serial_time else 0
        print(f"{t:6d} | {time_val:8.2f} | {throughput:15.2f} | {speedup:6.2f}x")

print()
print()

# Decode阶段
print("[Decode阶段]")
print(f"生成长度: {os.environ.get('DECODE_LEN', 1024)}")
print()

serial_time = extract_time_from_log('decode_serial_log.txt')
serial_throughput = extract_throughput_from_log('decode_serial_log.txt')

if serial_time:
    print(f"串行: 时间={serial_time:.2f}ms, 吞吐量={serial_throughput:.2f} tokens/sec")
    print()

print("线程数 | 时间(ms) | 吞吐量(tokens/s) | 加速比")
print("-" * 60)

for t in threads:
    log_file = f'decode_threads_{t}_log.txt'
    time_val = extract_time_from_log(log_file)
    throughput = extract_throughput_from_log(log_file)

    if time_val:
        speedup = serial_time / time_val if serial_time else 0
        print(f"{t:6d} | {time_val:8.2f} | {throughput:15.2f} | {speedup:6.2f}x")

print()
print("=" * 60)
PYTHON_EOF

chmod +x analyze_results.py
export PREFILL_LEN=$PREFILL_LEN
export DECODE_LEN=$DECODE_LEN

if command -v python3 &> /dev/null; then
    python3 analyze_results.py 2>&1 | tee summary.txt
    echo "✓ 性能报告已生成: $RESULTS_DIR/summary.txt"
else
    echo "⚠ 警告: 未找到Python3，跳过详细分析"
    echo "原始日志文件保存在: $RESULTS_DIR"
fi

echo ""
echo "========================================"
echo "  测试完成！"
echo "========================================"
echo "结果保存在: $RESULTS_DIR"
echo ""
echo "快速查看结果:"
echo "  cat $RESULTS_DIR/summary.txt"
echo ""
