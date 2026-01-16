# Streaming Block Attention & Qwen3 Tensor Library

面向多 NUMA、多节点 CPU 集群的 Streaming Block Attention 并行化实现与 Qwen3 LLM 推理库。

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
│   └── scripts/                   # compare_attention_full.py
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
- **[实验脚本指南](scripts/EXPERIMENT_GUIDE.md)** - 实验脚本使用说明

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

# 单线程基准测试
./benchmark_qwen3 --model /path/to/qwen3-0.6b

# MPI 并行推理 (2节点, 序列并行)
mpirun -np 2 ./benchmark_qwen3 \
  --model /path/to/qwen3-0.6b \
  --method mpi+avx2 \
  --parallel-strategy sequence \
  --attention-algo online_softmax \
  --prompt-len 128 \
  --iters 3
```

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
