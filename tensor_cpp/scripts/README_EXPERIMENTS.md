# Qwen3 Prefill 性能测试实验套件

本实验套件用于评估Qwen3-0.6B模型在不同并行配置下的prefill性能。

## 实验环境

- **模型**: Qwen3-0.6B
- **集群配置**: 8 nodes × 2 sockets × 26 CPUs (每节点16线程用于测试)
- **模型路径**: `/student/2025310707/Qwen3-0.6B/model.safetensors`
- **Benchmark程序**: `./build/benchmark_qwen3`

## 实验设计

### 实验1: 串行Baseline性能测试 (`exp1_serial_baseline.sh`)

**目标**: 评估单线程环境下不同优化策略的性能

**配置**:
- OMP线程数: 1 (串行)
- 方法: baseline, avx2
- 并行策略: headwise, sequence
- Attention算法: standard, online_softmax
- 序列长度: batch*len = 1*128
- 迭代: 5次 (warmup: 2次)

**输出**: `results/exp1_serial_baseline/serial_baseline_results.csv`

**数据列**: method, parallel_strategy, attention_algo, total_time_ms, time_per_token_ms, throughput_tok_s, timestamp

---

### 实验2: 单机16线程性能测试 (`exp2_single_node_16threads.sh`)

**目标**: 评估单机多线程环境下不同序列长度和并行策略的性能

**配置**:
- OMP线程数: 16
- 方法: avx2
- 并行策略: headwise, sequence
- Attention算法: standard, online_softmax
- 序列长度: batch*len = 1*128 / 1*512 / 1*1024
- 迭代: 5次 (warmup: 2次)

**输出**: `results/exp2_single_node_16threads/single_node_16threads_results.csv`

**数据列**: seq_len, parallel_strategy, attention_algo, total_time_ms, time_per_token_ms, throughput_tok_s, timestamp

---

### 实验3: MPI并行性能测试 (`exp3_mpi_parallel.sh`)

**目标**: 评估多节点MPI环境下不同节点数和序列长度的性能

**配置**:
- 节点数: 1, 2, 4, 8
- 每节点MPI进程数: 1
- 每进程OMP线程数: 16
- 方法: avx2
- 并行策略: headwise, sequence
- Attention算法: standard, online_softmax
- 序列长度: batch*len = 1*128 / 8*128 / 1*512 / 1*1024 / 1*4096
- 迭代: 5次 (warmup: 2次)

**SLURM配置**:
```bash
srun --mpi=pmix \
     -p student \
     -N <nodes> \
     --ntasks=<processes> \
     --ntasks-per-node=1 \
     --cpus-per-task=16 \
     env OMP_NUM_THREADS=16 \
     ./build/benchmark_qwen3 --args
```

**输出**: `results/exp3_mpi_parallel/mpi_parallel_results.csv`

**数据列**: nodes, processes, threads, seq_len, batch_size, parallel_strategy, attention_algo, total_time_ms, time_per_token_ms, throughput_tok_s, timestamp

---

## 并行策略说明

### Head-wise Parallelism
- **原理**: 将attention heads分布到不同进程/线程
- **通信量**: O(batch × seq_len × d_model)
- **适用算法**: standard, online_softmax

### Sequence Parallelism
- **原理**: 将序列tokens分布到不同进程/线程
- **通信量**:
  - + online_softmax: O(batch × d_head × P) ← 推荐
  - + standard: O(batch × seq_len × d) ← 不推荐 (500x通信量)

### 组合策略
✅ **推荐组合**:
- headwise + standard
- headwise + online_softmax
- sequence + online_softmax

❌ **不推荐组合**:
- sequence + standard (通信量巨大，完全抵消sequence并行优势)

---

## 运行方式

### 运行单个实验

```bash
# 实验1: 串行Baseline
cd /media/song/LocalDisk/Weblearning/并行计算/final/tensor_cpp
./scripts/exp1_serial_baseline.sh

# 实验2: 单机16线程
./scripts/exp2_single_node_16threads.sh

# 实验3: MPI并行 (需要在集群环境提交)
./scripts/exp3_mpi_parallel.sh
```

### 运行全部实验

```bash
# 依次运行所有实验
./scripts/run_all_experiments.sh
```

### 在SLURM集群运行实验3

```bash
# 交互式运行
srun --mpi=pmix -p student -N 8 --ntasks=8 --ntasks-per-node=1 --cpus-per-task=16 \
  bash /media/song/LocalDisk/Weblearning/并行计算/final/tensor_cpp/scripts/exp3_mpi_parallel.sh

# 或提交作业
sbatch --mpi=pmix -p student -N 8 --ntasks=8 --ntasks-per-node=1 --cpus-per-task=16 \
  --wrap="bash /media/song/LocalDisk/Weblearning/并行计算/final/tensor_cpp/scripts/exp3_mpi_parallel.sh"
```

---

## 结果分析

### CSV结果文件格式

**实验1**:
```csv
method,parallel_strategy,attention_algo,total_time_ms,time_per_token_ms,throughput_tok_s,timestamp
baseline,headwise,standard,1234.56,9.64,103.73,2025-01-16 13:00:00
```

**实验2**:
```csv
seq_len,parallel_strategy,attention_algo,total_time_ms,time_per_token_ms,throughput_tok_s,timestamp
128,headwise,standard,1234.56,9.64,103.73,2025-01-16 13:00:00
```

**实验3**:
```csv
nodes,processes,threads,seq_len,batch_size,parallel_strategy,attention_algo,total_time_ms,time_per_token_ms,throughput_tok_s,timestamp
8,8,128,1024,1,headwise,standard,1234.56,1.21,826.45,2025-01-16 13:00:00
```

### 分析维度

1. **串行 vs 并行加速比** (实验1 vs 实验2)
   - 对比单线程 vs 16线程性能
   - 评估AVX2优化效果

2. **扩展性分析** (实验3)
   - 强扩展: 固定问题规模，增加节点数
   - 弱扩展: 按比例增加问题规模和节点数

3. **算法对比**
   - Standard vs Online Softmax
   - Head-wise vs Sequence并行

4. **序列长度影响**
   - 短序列 (128): 测试延迟敏感场景
   - 中序列 (512-1024): 测试平衡场景
   - 长序列 (4096+): 测试吞吐量场景

---

## 预期实验结果

### 性能指标

- **总时间 (ms)**: 完成整个forward pass的时间
- **平均时间/token (ms)**: 每个token的平均延迟
- **吞吐量 (tok/s)**: 每秒处理的token数

### 预期趋势

1. **线程扩展**: 16线程应该接近8-12x加速比 (考虑内存带宽限制)
2. **MPI扩展**: 2-4节点应该有良好的线性扩展，8节点可能受通信限制
3. **算法优势**:
   - Online softmax在长序列下优势明显
   - Head-wise在短序列下表现更好
   - Sequence在长序列下表现更好
4. **AVX2加速**: 应该有2-3x性能提升

---

## 故障排除

### 常见问题

1. **Core dump/Segmentation fault**
   - 检查KV cache配置是否正确
   - 检查batch_size与输入数据是否匹配

2. **MPI初始化失败**
   - 确保在集群环境运行，不在本地运行MPI实验
   - 检查SLURM配置是否正确

3. **性能异常**
   - 检查OMP_NUM_THREADS设置
   - 检查CPU亲和性 (numa绑定)
   - 检查turbo boost和电源管理

---

## 实验检查清单

### 实验前准备

- [ ] 确认benchmark已编译: `./build/benchmark_qwen3`
- [ ] 确认模型文件存在: `/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors`
- [ ] 确认results目录可写
- [ ] 集群环境已加载MPI模块
- [ ] 检查SLURM分区可用性

### 实验中监控

- [ ] 监控CPU使用率: `htop`
- [ ] 监控内存使用: `free -h`
- [ ] 监控MPI通信: `mpirun --mca btl_tcp_if_include ib0 ...`

### 实验后验证

- [ ] 检查所有CSV文件完整
- [ ] 检查日志文件无错误
- [ ] 验证结果合理性（无异常值）
- [ ] 备份结果数据

---

## 文件结构

```
tensor_cpp/
├── scripts/
│   ├── README_EXPERIMENTS.md      # 本文档
│   ├── exp1_serial_baseline.sh    # 实验1脚本
│   ├── exp2_single_node_16threads.sh  # 实验2脚本
│   ├── exp3_mpi_parallel.sh       # 实验3脚本
│   └── run_all_experiments.sh     # 总控脚本
└── results/
    ├── exp1_serial_baseline/
    │   ├── serial_baseline_results.csv
    │   ├── baseline_headwise_standard.log
    │   ├── avx2_headwise_online_softmax.log
    │   └── ...
    ├── exp2_single_node_16threads/
    │   ├── single_node_16threads_results.csv
    │   └── ...
    └── exp3_mpi_parallel/
        ├── mpi_parallel_results.csv
        └── ...
```

---

## 修改历史

- 2025-01-16: 创建初始实验脚本
- 基于tensor_cpp/tests/benchmark/benchmark_qwen3.cpp实现

---

## 联系信息

如有问题，请检查:
1. benchmark源码: `tests/benchmark/benchmark_qwen3.cpp`
2. MPI实现: `src/ops_mpi.cpp`
3. AVX2实现: `src/qwen3_ops_avx.cpp`
