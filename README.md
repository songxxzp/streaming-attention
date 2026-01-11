# Streaming Block Attention & Tensor Library - 并行计算课程项目

面向多 NUMA、多节点 CPU 集群的 Streaming Block Attention 并行化实现与性能分析，以及完整的 Qwen3 LLM 推理实现。

## 项目结构

```
project/
├── attention/                 # Streaming Block Attention 并行实现
│   ├── naive_serial.cpp       # Phase 1: Naive串行实现
│   ├── streaming_serial.cpp   # Phase 1: Streaming串行实现
│   ├── streaming_omp.cpp      # Phase 2: OpenMP并行实现
│   ├── streaming_mpi.cpp      # Phase 3: MPI+OpenMP混合并行实现
│   └── attention.h            # 公共头文件
│
├── tensor_cpp/                # C++ Tensor 库 & Qwen3 模型实现
│   ├── include/tensor_cpp/    # 头文件
│   │   ├── tensor.h           # Tensor类定义
│   │   ├── ops.h              # 基础算子（linear, rms_norm, rope等）
│   │   ├── qwen3_ops.h       # Qwen3前向传播
│   │   ├── qwen3_loader.h    # Qwen3模型加载器
│   │   └── kv_cache.h        # KV Cache实现
│   ├── src/                   # 源文件
│   ├── tests/                 # 测试程序（30+个）
│   │   ├── test_qwen3_logits.cpp           # Forward pass示例
│   │   ├── test_qwen3_generate.cpp         # 自回归生成示例
│   │   └── test_qwen3_generate_with_cache.cpp # KV Cache优化 ⭐
│   └── README.md             # tensor_cpp 详细文档
│
├── utils/
│   ├── timer.h                # 性能计时工具
│   └── softmax_online.h       # Online Softmax核心算法
│
├── experiments/
│   ├── run_single_node.sh     # 单节点实验脚本
│   └── run_multi_node.sh      # 多节点实验脚本
│
├── test_correctness.cpp       # Phase 1: 正确性测试
├── test_omp.cpp               # Phase 2: OpenMP性能测试
├── test_mpi.cpp               # Phase 3: MPI性能测试
│
├── Makefile                   # attention 编译配置
└── README.md                  # 本文件
```

---

## 第一部分：Streaming Block Attention 并行化

基于 Online Softmax 算法的内存高效 Attention 实现，支持串行、OpenMP 并行和 MPI 多节点并行。

### 快速开始

```bash
# 进入 attention 目录
cd attention

# Phase 1: 串行实现
make test_correctness
./test_correctness

# Phase 2: OpenMP 并行
make test_omp
export OMP_NUM_THREADS=8
./test_omp --T 8192 --d 128 --block 64

# Phase 3: MPI 多节点
make test_mpi
mpirun -np 4 ./test_mpi --T 8192 --d 128 --block 64
```

### 性能结果

**OpenMP 加速比** (T=8192, d=256):

| 线程数 | 时间 (ms) | 加速比 | 效率 |
|--------|----------|--------|------|
| 1      | 1.416    | 1.00x  | 100% |
| 8      | 0.447    | 3.48x  | 44%  |
| 16     | 0.345    | 4.50x  | 28%  |

**MPI 强扩展性**: 支持多节点分布式计算，详细数据见 `REPORT.md`

---

## 第二部分：Tensor C++ Library & Qwen3 实现

高性能的 C++ Tensor 库，包含 Qwen3-0.6B 大语言模型的完整推理实现。支持 KV Cache 优化，实现 **1.74x 性能提升**。

### 核心特性

- ✅ **完整的 LLM 推理**: Qwen3-0.6B 模型（28层 Transformer，151K 词汇表）
- ✅ **KV Cache 优化**: Decode 阶段性能提升 1.74x
- ✅ **OpenMP 并行**: 多线程矩阵运算加速
- ✅ **正确性保证**: 修复 self_attention 索引和 causal mask bug
- ✅ **易于使用**: 清晰的 API 和详细的测试示例

### 快速开始

```bash
# 进入 tensor_cpp 目录
cd tensor_cpp

# 编译项目
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 设置环境变量（如果使用 anaconda）
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# 运行测试
./test_qwen3_logits              # Forward pass 示例
./test_qwen3_generate            # 自回归生成（无 cache）
./test_qwen3_generate_with_cache # 自回归生成（有 KV Cache）⭐
```

### 性能数据

**测试环境**: Intel Xeon, GCC 13.3.0, `-O3 -march=native`

| 方法 | 平均时间/token | 吞吐量 | 加速比 |
|------|----------------|--------|--------|
| 不用 KV Cache | 1041 ms | 0.96 tokens/s | 1.0x |
| **用 KV Cache** | **600 ms** | **1.66 tokens/s** | **1.74x** |

**生成结果一致性**:
- 不用 cache: `"Okay, the user said "Hello" and I"`
- 用 cache: `"Okay, the user said "Hello" and I"` ✅ 完全一致

### 技术亮点

1. **KV Cache 实现细节**:
   - Prefill 阶段：处理初始 prompt，初始化 cache
   - Decode 阶段：只处理新 token，复用缓存的 K/V
   - 内存布局：`[batch, num_kv_heads, max_seq_len, head_dim]`

2. **Bug 修复**:
   - **self_attention 索引错误**: query 和 key seq_len 不同时的索引计算
   - **causal mask 错误**: decode 阶段单个 query 的 mask 处理

3. **支持的操作**:
   - Linear, RMSNorm, RoPE (Rotary Position Embedding)
   - Self-attention with GQA (Grouped Query Attention)
   - SwiGLU activation (MLP)
   - Embedding lookup

详细文档请查看 [tensor_cpp/README.md](tensor_cpp/README.md)

---

## 依赖项

### 必需
- C++17 编译器 (g++ 7.0+ 或 clang++ 5.0+)
- OpenMP 4.5+ (通常编译器自带)
- CMake 3.16+ (用于 tensor_cpp)
- Make (用于 attention)

### 可选
- MPI 实现 (OpenMPI 或 MPICH) - 用于多节点 attention 测试
  ```bash
  sudo apt-get install openmpi-bin openmpi-dev libopenmpi-dev
  ```

### 系统要求
- **操作系统**: Linux (测试环境：Ubuntu 22.04)
- **内存**: 至少 4GB（Qwen3-0.6B 模型需要约 2.4GB）
- **磁盘**: 约 2.4GB (model.safetensors)

---

## 详细文档

### Streaming Block Attention

**算法说明**:

Naive Attention (串行):
```
O = softmax(Q @ K^T) @ V
```
完整计算，需要构造完整 QK^T 矩阵。

Streaming Block Attention (Online Softmax):
```
初始化: m = -∞, l = 1, O = 0

对每个 block b:
    S_b = Q @ K_b^T
    m_new = max(m, max(S_b))
    l_new = l * exp(m - m_new) + Σ exp(S_b - m_new)
    O_new = O * (l * exp(m - m_new) / l_new) + Σ exp(S_b - m_new) * V_b / l_new
```
分块计算，使用 online softmax 避免构造完整矩阵。

**并行化策略**:
- **OpenMP**: Chunk-level 并行，Tree Reduction 合并结果
- **MPI**: Data Parallelism，KV cache 分布在多个 ranks
- **NUMA-aware**: First-touch 数据分配策略

**运行测试**:

```bash
# Phase 1: 正确性验证
cd attention
./test_correctness
# 预期: L2 Error < 1e-6, Max Error < 1e-7

# Phase 2: OpenMP 性能测试
export OMP_NUM_THREADS=8
./test_omp --T 4096 --d 128 --block 64
# 测试: 线程扩展性、Block size 影响、NUMA 优化

# Phase 3: MPI 多节点测试
mpirun -np 4 ./test_mpi --T 8192 --d 128 --block 64
# 测试: 正确性、强扩展性、通信开销
```

### Tensor Library & Qwen3

**模型规格** (Qwen3-0.6B):
```
层数: 28
隐藏层维度: 1024
Attention heads: 16
KV heads: 8 (GQA)
Head维度: 128
词汇表大小: 151,936
```

**代码示例**:

```cpp
#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/kv_cache.h"

// 加载模型
auto weights = load_qwen3_weights("model.safetensors");

// 创建 KV Cache
auto kv_cache = std::make_unique<KVCache>(
    28, 8, 128, 4096  // layers, kv_heads, head_dim, max_seq_len
);

// Prefill 阶段
Tensor output = qwen3_forward_with_cache(
    input_prompt, kv_cache.get(), weights, ...
);

// Decode 阶段（逐个生成）
for (int i = 0; i < max_tokens; ++i) {
    Tensor new_token = {last_token};  // 只处理新 token
    Tensor output = qwen3_forward_with_cache(
        new_token, kv_cache.get(), weights, ...
    );
}
```

**支持的操作**:
- Linear, RMSNorm, RoPE (Rotary Position Embedding)
- Self-attention with GQA (Grouped Query Attention)
- SwiGLU activation (MLP)
- Embedding lookup
- KV Cache 管理

---

## 性能对比

### Attention 并行化性能

**OpenMP 加速比** (T=8192, d=256):

| 线程数 | 时间 (ms) | 加速比 | 效率 |
|--------|----------|--------|------|
| 1      | 1.416    | 1.00x  | 100% |
| 8      | 0.447    | 3.48x  | 44%  |
| 16     | 0.345    | 4.50x  | 28%  |

### Qwen3 推理性能

**测试环境**: Intel Xeon, GCC 13.3.0, `-O3 -march=native`

| 方法 | 平均时间/token | 吞吐量 | 加速比 |
|------|----------------|--------|--------|
| 不用 KV Cache | 1041 ms | 0.96 tokens/s | 1.0x |
| **用 KV Cache** | **600 ms** | **1.66 tokens/s** | **1.74x** |

---

## 实验分析

### 1. 正确性分析
- **Attention**: Naive vs Streaming, OMP vs Serial, MPI vs Serial 误差均 < 1e-6
- **Qwen3**: KV Cache 版本与无 Cache 版本输出完全一致

### 2. 复杂度分析

**Attention**:
- 计算复杂度: O(Td)
- 空间复杂度:
  - Naive: O(T) for scores
  - Streaming: O(block_size) for scores
  - OMP: O(block_size × n_threads) for partial results
  - MPI: O(block_size × n_ranks) for communication

**Qwen3**:
- Prefill: O(seq_len × hidden_size² × num_layers)
- Decode (无 cache): O(seq_len × hidden_size² × num_layers)
- Decode (有 cache): O(hidden_size² × num_layers) - **常数时间！**

### 3. 扩展性分析
- **OpenMP**: 最优线程数 8-16，过饱和导致效率下降
- **MPI**: 理论上线性加速，实际受通信开销限制
- **KV Cache**: Decode 阶段 1.74x 加速，不受序列长度影响

### 4. 优化技巧
- **NUMA-aware**: First-touch 数据分配，单节点提升有限，多节点显著
- **KV Cache**: 避免重复计算，Decode 阶段性能关键
- **Bug 修复**: 正确的索引和 mask 对结果准确性至关重要

---

## 故障排除

### MPI 编译错误
```bash
sudo apt-get update
sudo apt-get install openmpi-bin openmpi-dev
```

### OpenMP 线程数设置
```bash
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```

### Anaconda 环境变量冲突
```bash
# tensor_cpp 需要系统库
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

### 多节点 SSH 配置
```bash
ssh-keygen -t rsa
ssh-copy-id user@remote_host

# 创建 hosts 文件
cat > hosts << EOF
node1
node2
node3
node4
EOF
```

---

## 参考文献

1. "Online Normalizer Calculation for Softmax" - Parallel softmax technique
2. "Flash Attention" - Fast attention with IO awareness
3. "Efficient Attention: Attention with Linear Complexities" - Linear attention variants
4. "Qwen3 Technical Report" - Model architecture and design

---

## 许可证

课程项目，仅供学习使用。

---

## 项目链接

- **代码仓库**: [https://github.com/songxxzp/streaming-attention](https://github.com/songxxzp/streaming-attention)
- **详细报告**: 见 [REPORT.md](REPORT.md)
- **Tensor 库文档**: 见 [tensor_cpp/README.md](tensor_cpp/README.md)
