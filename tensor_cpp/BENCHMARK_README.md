# Qwen3性能测试脚本使用说明

本目录包含完整的Qwen3模型和Attention算子性能测试脚本，用于并行计算课程报告。

## 测试脚本说明

### 1. Attention算子级别测试 (`benchmark_attention`)

测试Standard Attention和Streaming Attention的性能对比。

**用法：**
```bash
# Standard Attention
OMP_NUM_THREADS=16 ./build/benchmark_attention \
    --mode standard \
    --seq-len 1024 \
    --hidden 128 \
    --iters 100 \
    --threads 16

# Streaming Attention
OMP_NUM_THREADS=16 ./build/benchmark_attention \
    --mode streaming \
    --seq-len 1024 \
    --hidden 128 \
    --iters 100 \
    --threads 16 \
    --block-size 64
```

**参数说明：**
- `--mode`: attention类型 (`standard` 或 `streaming`)
- `--seq-len`: 序列长度
- `--hidden`: 隐藏维度 (head_dim)
- `--heads`: attention头数 (仅standard模式)
- `--batch`: batch size
- `--iters`: 迭代次数
- `--threads`: OpenMP线程数
- `--block-size`: streaming attention块大小 (仅streaming模式)

### 2. Qwen3模型级别测试 (`benchmark_qwen3`)

测试Qwen3模型的Prefill和Decode阶段性能。

**用法：**
```bash
# Prefill阶段
OMP_NUM_THREADS=16 ./build/benchmark_qwen3 \
    --model /path/to/model.safetensors \
    --phase prefill \
    --prompt-len 128 \
    --iters 10 \
    --threads 16

# Decode阶段 (使用KV cache)
OMP_NUM_THREADS=16 ./build/benchmark_qwen3 \
    --model /path/to/model.safetensors \
    --phase decode \
    --gen-len 100 \
    --iters 1 \
    --threads 16
```

**参数说明：**
- `--model`: 模型文件路径
- `--phase`: 测试阶段 (`prefill` 或 `decode`)
- `--prompt-len`: prompt长度 (仅prefill阶段)
- `--gen-len`: 生成长度 (仅decode阶段)
- `--iters`: 迭代次数
- `--threads`: OpenMP线程数
- `--warmup`: 预热次数
- `--no-kv-cache`: decode阶段不使用KV cache

## 自动化测试脚本

### 快速验证测试 (`quick_test.sh`)

小数据量快速验证，用于本地测试和调试。

```bash
./quick_test.sh
```

**测试内容：**
- Attention算子对比 (seq_len=128)
- 线程扩展性测试 (threads=1,2,4,8)
- Qwen3 Prefill/Decode快速测试

**结果输出：** `./benchmark_results_quick/`

### 完整测试套件 (`run_benchmark_suite.sh`)

完整性能测试，用于生成课程报告所需的所有数据。

```bash
./run_benchmark_suite.sh
```

**测试内容：**
1. **Attention算子测试**
   - 不同序列长度: 64, 128, 256, 512, 1024
   - 不同块大小: 32, 64, 128
   - 线程扩展性: 1-32线程

2. **Qwen3模型测试**
   - Prefill阶段: 不同prompt长度
   - Decode阶段: 不同生成长度
   - 线程扩展性测试

**结果输出：** `./benchmark_results/`

## 服务器上运行

### 服务器环境要求

- OpenMPI
- OpenMP
- C++17编译器
- 无需Python环境

### 加载模块

```bash
spack load cmake
spack load openmpi
```

### 编译项目

```bash
mkdir -p build
cd build
cmake ..
make -j
```

### 运行测试

**单节点测试 (OpenMP):**
```bash
# 设置线程数
export OMP_NUM_THREADS=16

# 运行完整测试套件
./quick_test.sh

# 或运行单个测试
OMP_NUM_THREADS=16 ./build/benchmark_attention \
    --mode standard \
    --seq-len 1024 \
    --iters 50 \
    --threads 16
```

**多节点测试 (MPI + OpenMP):**
```bash
# 节点配置
NODES=1              # 节点数
NTASKS=4             # 总进程数
NTASKS_PER_NODE=4    # 每节点进程数
CPUS_PER_TASK=4      # 每进程线程数

# 运行MPI任务
srun --mpi=pmix \
    -p student \
    -N $NODES \
    --ntasks=$NTASKS \
    --ntasks-per-node=$NTASKS_PER_NODE \
    --cpus-per-task=$CPUS_PER_TASK \
    env OMP_NUM_THREADS=$CPUS_PER_TASK \
    ./build/benchmark_qwen3_mpi \
    --model /path/to/model.safetensors \
    --phase prefill \
    --prompt-len 128 \
    --iters 10 \
    --mode mpi
```

**注意：** MPI版本需要在代码中实现MPI并行。当前实现是OpenMP并行。

## 性能分析指标说明

### 1. 吞吐量 (Throughput)
- 单位: tokens/sec
- 定义: 每秒处理的token数量
- 计算公式: `吞吐量 = (token数 × 迭代次数) / 总时间`

### 2. 加速比 (Speedup)
- 定义: 相对于串行版本的性能提升倍数
- 计算公式: `加速比 = 串行执行时间 / 并行执行时间`

### 3. 并行效率 (Parallel Efficiency)
- 定义: 加速比与处理器数目的比值
- 计算公式: `并行效率 = 加速比 / 处理器数目`

### 4. 可扩展性 (Scalability)
- 描述性能随处理器数量增长的变化趋势
- 强扩展性: 问题规模固定，增加处理器数目
- 弱扩展性: 处理器数目增加，问题规模成比例增长

## 课程报告撰写指南

### 1. 问题与建模
- **问题背景**: 大语言模型推理中的计算瓶颈
- **物理建模**: Transformer Attention机制的并行化
- **数学建模**:
  - Standard Attention复杂度: O(n²)
  - Streaming Attention复杂度: O(n²) 但cache-friendly

### 2. 算法设计
- **串行算法**: 标准Attention计算
- **并行算法**:
  - OpenMP并行: 共享内存并行
  - Streaming Attention: 分块计算策略
  - KV Cache: 避免重复计算
- **任务划分**:
  - 按attention头划分
  - 按序列位置分块
- **通信策略**: OpenMP共享内存，无需显式通信

### 3. 程序实现
- **硬件环境**:
  - CPU: Intel/AMD多核处理器
  - 内存: DDR4/DDR5
  - 编译器: GCC with OpenMP
- **关键代码**:
  - `src/ops.cpp`: Attention实现
  - `src/qwen3_ops.cpp`: Qwen3模型实现
  - `include/tensor_cpp/kv_cache.h`: KV cache实现

### 4. 性能分析
运行测试套件并收集数据：
```bash
./run_benchmark_suite.sh
```

**分析维度：**
1. **吞吐量对比**: Standard vs Streaming
2. **加速比曲线**: 不同线程数下的性能提升
3. **并行效率**: 资源利用率分析
4. **最优处理器数**: 性能拐点分析
5. **Prefill vs Decode**: 不同阶段的性能特征

**图表建议：**
- 吞吐量vs线程数
- 加速比vs线程数
- 并行效率vs线程数
- 序列长度vs吞吐量

### 5. 总结展望
- **结论**: Streaming Attention的优势和适用场景
- **不足**: 当前实现的局限性
- **未来改进**:
  - MPI分布式扩展
  - 混合精度计算
  - Flash Attention优化
  - GPU加速

## 测试结果示例

### Attention算子对比 (seq_len=128)
```
Standard Attention:
  平均时间: 38.82 ms/iter
  吞吐量: 3297.32 tokens/sec

Streaming Attention:
  平均时间: 0.04 ms/iter
  吞吐量: 3307895.27 tokens/sec

加速比: 1003.08x
```

**注意**: Streaming Attention的测试配置不同（单query vs 多query），直接对比需要注意公平性。更合理的对比是测试相同配置下的性能。

### 线程扩展性 (seq_len=256)
```
线程数 | 吞吐量(tokens/sec) | 加速比
-------|-------------------|-------
1      | 1626.53           | 1.00
2      | 1631.79           | 1.00
4      | 1580.16           | 0.97
8      | 1446.53           | 0.88
```

观察：当前测试规模下，多线程扩展性不佳，可能是因为：
1. 问题规模太小
2. 内存带宽限制
3. OpenMP开销

建议增加序列长度和迭代次数以获得更明显的并行效果。

## 故障排除

### 编译错误
```bash
# 检查OpenMP支持
g++ --version

# 手动编译测试
g++ -std=c++17 -O3 -march=native -fopenmp \
    -I./include -o build/benchmark_attention \
    src/tensor.o src/ops.o tests/benchmark_attention.cpp
```

### 运行时错误
```bash
# 检查模型文件是否存在
ls -lh /path/to/model.safetensors

# 检查OpenMP线程数设置
echo $OMP_NUM_THREADS

# 使用valgrind检查内存错误
valgrind --leak-check=full ./build/benchmark_attention --mode standard
```

### 性能异常
```bash
# 检查CPU频率
cat /proc/cpuinfo | grep MHz

# 检查系统负载
top

# 使用perf分析性能瓶颈
perf stat -e cycles,instructions,cache-misses ./build/benchmark_attention --mode standard
```

## 文件结构

```
tensor_cpp/
├── include/tensor_cpp/
│   ├── tensor.h              # 张量定义
│   ├── ops.h                # 算子定义
│   ├── qwen3_ops.h          # Qwen3模型算子
│   ├── qwen3_loader.h       # 模型加载
│   └── kv_cache.h           # KV cache实现
├── src/
│   ├── tensor.cpp            # 张量实现
│   ├── ops.cpp              # 算子实现
│   ├── qwen3_ops.cpp        # Qwen3模型实现
│   └── qwen3_loader.cpp     # 模型加载实现
├── tests/
│   ├── benchmark_attention.cpp    # Attention性能测试
│   ├── benchmark_qwen3.cpp        # Qwen3模型性能测试
│   └── ...其他测试文件
├── build/                     # 编译输出
│   ├── benchmark_attention
│   └── benchmark_qwen3
├── run_benchmark_suite.sh    # 完整测试套件
├── quick_test.sh              # 快速验证测试
└── BENCHMARK_README.md        # 本文档
```

## 联系方式

如有问题，请查看项目README或联系助教。
