# Streaming Block Attention & Qwen3 Tensor Library

面向多 NUMA、多节点 CPU 集群的 Streaming Block Attention 并行化实现与 Qwen3 LLM 推理库。

## 📄 完整论文

**本项目的研究报告已整理为学术论文，请查看：[Paper.pdf](Paper.pdf)**

该论文包含完整的：
- 研究背景与动机
- 方法设计与实现
- 实验结果与分析（串行、OpenMP、MPI 并行性能对比）
- 序列并行 vs 头维并行的系统性研究
- Streaming Attention 算法优化效果
- 在 Qwen3-0.6B 模型上的端到端性能评估

**主要结论**：
- Streaming Attention 相比传统方法提升 2.61× - 6.41×
- 8 节点达到 14.29 tok/s 吞吐量
- 序列并行 + online softmax 是最优组合
- 算法优化优于算子优化（5.8× 收益比）

## 📁 项目结构

```
final/
├── README.md                      # 项目主文档
│
├── docs/                          # 📚 项目文档
│   ├── ATTENTION_REPORT.md        # Attention 算子实验报告
│   ├── MPI_IMPLEMENTATION_COMPARISON.md  # MPI 实现对比分析
│   └── QWEN3_MPI_GUIDE.md         # Qwen3 MPI 使用指南
│
├── attention/                     # 🎯 Streaming Block Attention 实现
│   ├── src/                       # 源代码
│   │   ├── attention.h            # 公共头文件
│   │   ├── naive_serial.cpp       # Naive 串行实现
│   │   ├── streaming_serial.cpp   # Streaming 串行实现
│   │   ├── streaming_omp.cpp      # OpenMP 并行实现
│   │   └── streaming_mpi.cpp      # MPI+OpenMP 混合并行实现
│   ├── tests/                     # 测试代码
│   └── scripts/                   # compare_attention_full.py 性能对比脚本
│
├── tensor_cpp/                    # 🧠 Qwen3 C++ Tensor 库
│   ├── README.md                  # 详细文档
│   ├── include/tensor_cpp/        # 头文件
│   │   ├── tensor.h               # Tensor 类定义
│   │   ├── ops.h                  # 基础算子
│   │   ├── ops_avx.h              # AVX SIMD 算子
│   │   ├── ops_mpi.h              # MPI 并行算子
│   │   ├── qwen3_ops.h            # Qwen3 前向传播
│   │   ├── qwen3_ops_avx.h        # AVX2 优化版本
│   │   ├── qwen3_ops_mpi.h        # MPI 版本
│   │   └── kv_cache.h             # KV Cache 实现
│   ├── src/                       # 源文件实现
│   ├── tests/benchmark/           # 性能测试
│   ├── scripts/                   # 实验脚本
│   │   ├── exp1_serial_baseline.sh
│   │   ├── exp2_single_node_n_threads.sh
│   │   ├── exp3_mpi_parallel.sh
│   │   ├── exp4_thread_scaling.sh
│   │   ├── exp5_node_scaling.sh
│   │   ├── exp6_block_size_tuning.sh
│   │   └── README_EXPERIMENTS.md  # 实验脚本完整文档
│   └── results/                   # 实验结果
│
├── experiments/                   # 📊 实验数据和可视化
│   ├── data/                      # 原始 CSV 数据
│   └── figures/                   # 绘图脚本和图表
│
├── scripts/                       # 🔧 通用工具脚本
├── utils/                         # 工具库
└── build/                         # 编译输出
```

## 📖 文档导航

- **[Attention 算子实验报告](docs/ATTENTION_REPORT.md)** - Streaming Attention 性能分析
- **[MPI 实现对比](docs/MPI_IMPLEMENTATION_COMPARISON.md)** - MPI vs MPI+AVX2 详细对比
- **[Qwen3 MPI 使用指南](docs/QWEN3_MPI_GUIDE.md)** - MPI 并行配置和运行
- **[Tensor_cpp README](tensor_cpp/README.md)** - C++ 库详细文档
- **[实验脚本指南](tensor_cpp/scripts/README_EXPERIMENTS.md)** - Qwen3 性能实验脚本完整说明

## 🚀 快速开始

### Attention 算子测试

```bash
# 编译并测试
cd attention
make test_mpi
mpirun -np 4 ./test_mpi --T 8192 --d 128 --block 64

# 性能对比分析
python scripts/compare_attention_full.py
```

### Qwen3 推理

```bash
# 编译
cd tensor_cpp/build
cmake ..
make -j

# Prefill 阶段基准测试 (处理长提示词)
./benchmark_qwen3 --model /path/to/qwen3-0.6B/model.safetensors \
  --method mpi+avx2 \
  --parallel-strategy sequence \
  --attention-algo online_softmax \
  --prompt-len 128 \
  --iters 3

# Decode 阶段性能验证 (自回归生成)
mpirun -np 2 ./benchmark_qwen3 \
  --model /path/to/qwen3-0.6B/model.safetensors \
  --method mpi+avx2 \
  --parallel-strategy sequence \
  --attention-algo online_softmax \
  --prompt-len 128 \
  --generate 100 \
  --threads 8

# 正确性验证 (与 PyTorch 输出对比)
mpirun -np 2 ./benchmark_qwen3 \
  --model /path/to/qwen3-0.6B/model.safetensors \
  --method mpi+avx2 \
  --parallel-strategy sequence \
  --attention-algo online_softmax \
  --prompt-len 32 \
  --verify
```

### Attention 算子性能对比

```bash
# 方式1: 在 attention/ 目录编译和运行
cd attention
make                    # 编译串行和 OpenMP 版本
make mpi                # 编译 MPI 版本 (可选)

# 运行完整性能对比测试
python scripts/compare_attention_full.py \
  --seq-lens 1024 8192 \
  --hidden-dim 128 \
  --threads 1 2 4 8 \
  --block-sizes 64 128

# 方式2: 从项目根目录运行
cd /path/to/final
python attention/scripts/compare_attention_full.py --help

# 快速测试
python attention/scripts/compare_attention_full.py \
  --seqlen 512 --dim 64 --threads 1 --repeat 2
```

**Makefile 目标**:
- `make` 或 `make all` - 编译串行和 OpenMP 版本
- `make serial` - 仅编译串行版本
- `make openmp` - 仅编译 OpenMP 版本
- `make mpi` - 编译 MPI 版本 (需要 mpicxx)
- `make clean` - 清理编译产物
- `make help` - 显示帮助信息

**测试项目**:
- PyTorch `F.scaled_dot_product_attention` (baseline)
- C++ Naive Attention (串行 / OpenMP / MPI)
- C++ Streaming Attention (串行 / OpenMP / MPI)

**输出**:
- 各实现的延迟和吞吐量对比
- 加速比分析
- 通信开销统计 (MPI版本)

**路径自动检测**: `compare_attention_full.py` 支持从任意位置运行：
- 项目根目录 → 自动使用 `./attention/` 路径
- `attention/` 目录 → 自动使用当前目录
- `attention/scripts/` 目录 → 自动使用 `..` 路径

**运行单个测试**:
```bash
cd attention
./test_naive 1024 128 64              # Naive 串行
OMP_NUM_THREADS=4 ./test_naive_omp 1024 128 64  # Naive OpenMP
mpirun -np 2 ./test_naive_mpi 1024 128 4       # Naive MPI
```

### Qwen3 性能实验脚本

```bash
cd tensor_cpp

# 运行单个实验
./scripts/exp1_serial_baseline.sh        # 串行baseline
./scripts/exp2_single_node_n_threads.sh  # 单机多线程
./scripts/exp3_mpi_parallel.sh           # MPI并行 (集群)

# 运行所有实验
./scripts/run_all_experiments.sh
```

**实验系列**:
1. **exp1_serial_baseline**: 串行 baseline (baseline vs avx2)
2. **exp2_single_node_n_threads**: 单机多线程扩展性
3. **exp3_mpi_parallel**: 多节点 MPI 并行 (1/2/4/8 nodes)
4. **exp4_thread_scaling**: 线程扩展性分析
5. **exp5_node_scaling**: 节点扩展性分析
6. **exp6_block_size_tuning**: Block size 调优

详细说明见 [实验脚本完整文档](tensor_cpp/scripts/README_EXPERIMENTS.md)

## 📊 性能亮点

### Attention 算子

- ✅ OpenMP 并行: 4.5x 加速比 (16 线程)
- ✅ MPI 强扩展: 支持多节点分布式计算
- ✅ 内存优化: Online Softmax 降低内存占用

### Qwen3 推理

- ✅ **MPI+AVX2 (8节点)**: 70+ tok/s (序列并行，长度128)
- ✅ **真序列并行**: 消除冗余计算
- ✅ **AVX2 优化**: 27% 性能提升
- ✅ **正确性验证**: 与 PyTorch 输出一致

详细性能数据见 [实验结果](experiments/data/)。

## 🔧 开发环境

- **编译器**: GCC 9+ (支持 C++17)
- **MPI**: OpenMPI 4+
- **SIMD**: AVX2 支持
- **系统**: Linux x86_64

## 📜 许可证

本项目用于学术研究和教学目的。
