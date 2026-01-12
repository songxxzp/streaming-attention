# Tensor C++ Library - Qwen3 Implementation with Parallel Optimizations

高性能的 C++ Tensor 库，包含 Qwen3-0.6B 模型的完整实现，支持 OpenMP、MPI、AVX2 优化和 KV Cache。

## 🎯 特性

### 核心功能
- ✅ **Qwen3-0.6B 模型完整实现**: 28层 Transformer 架构
- ✅ **KV Cache 支持**: 大幅提升 decode 阶段性能（1.74x 加速）
- ✅ **分组查询注意力 (GQA)**: 优化注意力机制
- ✅ **RoPE (旋转位置编码)**: 正确实现
- ✅ **Safetensors 格式**: 支持 HuggingFace 模型权重

### 性能优化
- ⚡ **OpenMP 并行**: 多线程加速
- ⚡ **AVX2 SIMD**: 向量化计算（1.6-3.3x 加速）
- ⚡ **MPI 数据并行**: 多节点分布式训练/推理
- ⚡ **张量并行**: 模型切分优化

### 正确性保证
- ✅ **数值验证**: 与 PyTorch 实现对比验证
- ✅ **完整测试**: 单元测试、集成测试、性能测试

---

## 📁 目录结构

```
tensor_cpp/
├── include/tensor_cpp/       # 公共头文件
│   ├── tensor.h              # Tensor 类定义
│   ├── tensor_impl.tpp       # Tensor 模板实现
│   ├── ops.h                 # 基础算子（matmul, add, rms_norm, rope）
│   ├── ops_avx.h             # AVX SIMD 算子
│   ├── ops_mpi.h             # MPI 并行算子
│   ├── attention.h           # 注意力机制
│   ├── kv_cache.h            # KV Cache 实现
│   ├── qwen3_loader.h       # 模型权重加载
│   ├── qwen3_ops.h          # Qwen3 前向传播
│   ├── qwen3_ops_mpi.h       # MPI 版本
│   ├── qwen3_ops_avx.h      # AVX2 优化版本
│   ├── qwen3_tensor_parallel.h # 张量并行
│   └── avx2_helpers.h        # AVX2 辅助函数库 ⭐
│
├── src/                     # 实现文件
│   ├── tensor.cpp
│   ├── ops.cpp
│   ├── ops_avx.cpp
│   ├── ops_mpi.cpp
│   ├── attention_avx.cpp
│   ├── qwen3_loader.cpp
│   ├── qwen3_ops.cpp        # 基础实现
│   ├── qwen3_ops_avx.cpp    # AVX2 优化（旧版）
│   ├── qwen3_ops_mpi_avx.cpp # MPI + AVX2
│   └── qwen3_tensor_parallel.cpp
│
├── tests/                   # 测试套件（已重新组织）
│   ├── unit/                # 单元测试（9个）
│   ├── integration/         # 集成测试（6个）
│   ├── benchmark/           # 性能测试（5个）
│   ├── validation/          # 验证测试（3个）
│   └── README.md            # 测试文档
│
├── examples/                # 示例代码
│   └── basic_usage.cpp
│
├── CMakeLists.txt
└── README.md
```

---

## 🚀 快速开始

### 前置要求

- GCC 9+ 或 Clang 10+（支持 C++17）
- OpenMP 4.5+
- MPI 3.0+（可选，用于分布式功能）
- CPU 支持 AVX2（推荐）

### 1. 编译项目

```bash
cd tensor_cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 2. 运行环境配置

**如果使用 anaconda，需要设置系统库路径：**

```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

---

## 📊 性能基准

### Qwen3-0.6B 模型性能 (OMP_NUM_THREADS=16)

| 版本 | seq_len=4 | seq_len=16 | seq_len=32 | vs Baseline |
|------|-----------|------------|------------|--------------|
| **Baseline** | 4.04s | 6.81s | 15.59s | 1.0x |
| **AVX2** | 1.23s | 4.16s | 7.67s | **3.3x / 1.6x / 2.0x** |
| **MPI (2进程)** | 2.88s | 5.12s | 11.20s | 1.4x / 1.3x / 1.4x |
| **MPI+AVX2** | 1.01s | 3.45s | 6.98s | **4.0x / 2.0x / 2.2x** |

**硬件**: Intel CPU, AVX2 支持

### 组件级优化

| 组件 | Baseline | AVX2 | 加速比 |
|------|----------|------|--------|
| MLP (SwiGLU) | 172ms | 28ms | **6.1x** |
| Linear Layer | - | - | **2.9x** |
| Horizontal Sum | - | - | **~20% faster** |

---

## 🧪 测试

### 运行测试

```bash
cd build

# 单元测试
./test_simple
./test_ops
./test_attention

# 集成测试
./test_qwen3                    # 完整前向传播
./test_qwen3_generate          # 自回归生成
./test_qwen3_generate_with_cache # 带 KV cache

# 性能测试
OMP_NUM_THREADS=16 ./benchmark_qwen3
OMP_NUM_THREADS=16 ./benchmark_avx2_versions

# MPI 测试
mpirun -np 2 ./test_qwen3_mpi_simple
```

详细测试说明请参考 [tests/README.md](tests/README.md)

---

## 🎚️ 使用示例

### 基础前向传播

```cpp
#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/qwen3_ops.h"

using namespace tensor_cpp;
using namespace tensor_cpp::qwen3;

// 加载模型
Qwen3Weights weights = load_qwen3_weights(
    "/path/to/Qwen3-0.6B/model.safetensors"
);

// 准备输入
std::vector<long> ids = {1, 2, 3, 4};
TensorL input_ids(ids, Shape({1, 4}));

// 前向传播
Tensor output = qwen3::qwen3_forward(
    input_ids,
    weights.embed_tokens,
    weights.layers,
    weights.norm_weight,
    weights.num_layers,
    weights.num_attention_heads,
    weights.num_key_value_heads,
    weights.head_dim,
    1e-6f  // rms_norm_eps
);
```

### 使用 AVX2 优化版本

```cpp
#include "tensor_cpp/qwen3_ops_avx.h"

using namespace tensor_cpp::qwen3::avx2;

Tensor output = avx2::qwen3_forward_avx(
    input_ids,
    weights.embed_tokens,
    weights.layers,
    weights.norm_weight,
    weights.num_layers,
    weights.num_attention_heads,
    weights.num_key_value_heads,
    weights.head_dim,
    1e-6f
);
```

### 使用 KV Cache 加速生成

```cpp
#include "tensor_cpp/qwen3_ops.h"

TensorKVCache kv_cache(
    weights.num_layers,
    weights.num_key_value_heads,
    128,  // max_seq_len
    1024  // hidden_size
);

// Prefill 阶段
Tensor output = qwen3_forward_with_cache(
    input_ids,
    weights,
    kv_cache
);

// Decode 阶段（迭代生成）
for (int i = 0; i < 10; ++i) {
    Tensor next_token = qwen3_forward_with_cache(
        last_token,
        weights,
        kv_cache
    );
}
```

---

## 🔧 实现版本对比

| 实现版本 | 命名空间 | 特性 | 性能 | 推荐场景 |
|---------|---------|------|------|---------|
| **基础版** | `qwen3::` | 标准 OpenMP | 基准 | 功能验证、调试 |
| **AVX2** | `qwen3::avx2::` | MLP 优化 | 1.6-3.3x | 单机推理 |
| **AVX2 V2** | `qwen3::avx2_v2::` | 全面优化 | 最高 | 单机推理（推荐） |
| **MPI** | `qwen3::mpi::` | 数据并行 | 1.3-1.4x | 多节点 |
| **MPI+AVX2** | `qwen3::mpi_avx::` | 混合并行 | 最高 | 多节点（推荐） |
| **张量并行** | `qwen3::tensor_parallel::` | 模型切分 | - | 大模型 |

### 推荐使用

**单机推理：**
```cpp
using namespace tensor_cpp::qwen3::avx2;  // 或 avx2_v2（最优）
```

**分布式推理：**
```cpp
using namespace tensor_cpp::qwen3::mpi_avx;
```

---

## 📈 优化技术

### 1. AVX2 SIMD 优化

**水平求和优化**（`avx2_helpers.h`）:
```cpp
// 旧方法（使用 hadd）
__m256 sum = _mm256_hadd_ps(v, v);
sum = _mm256_hadd_ps(sum, sum);

// 新方法（使用 shuffle，快20%）
float result = avx2_helpers::hsum_avx2(v);
```

**MLP 优化**:
- Gate/Up 投影：AVX2 向量化
- SwiGLU 激活：快速 sigmoid 近似
- Down 投影：AVX2 向量化
- **总体加速**: 6.1x

### 2. KV Cache 优化

- **Prefill 阶段**: 一次性处理所有 token
- **Decode 阶段**: 复用缓存的 K/V，只计算新 token
- **性能提升**: 1.74x（decode 阶段）

### 3. 预提取 QKV 投影

**优化前**（每次前向传播）:
```cpp
// 每层都需要提取 Q, K, V
for (int layer = 0; layer < 28; ++layer) {
    // 从 qkv_projs 提取 Q, K, V
    // 28 层 × 3 次 = 84 次矩阵复制
}
```

**优化后**（模型加载时）:
```cpp
// 预提取并保存
layer.q_proj = extract_q_proj(qkv_projs);
layer.k_proj = extract_k_proj(qkv_projs);
layer.v_proj = extract_v_proj(qkv_projs);
// 节省：~336MB 内存复制 + 84 次矩阵创建
```

### 4. MPI 数据并行

- 每个进程处理部分数据
- AllReduce 聚合梯度
- 支持 2-16 进程

---

## 🛠️ 开发指南

### 添加新的优化实现

1. **创建新文件**: `src/qwen3_ops_<optimization>.cpp`
2. **命名空间**: `namespace tensor_cpp::qwen3::<optimization>`
3. **导出函数**:
   ```cpp
   Tensor qwen3_forward_<optimization>(...);
   ```
4. **更新 CMakeLists.txt**: 添加编译目标和标志
5. **添加测试**: 在 `tests/integration/` 或 `tests/benchmark/`

### 使用 AVX2 辅助函数

```cpp
#include "tensor_cpp/avx2_helpers.h"

// 使用优化的水平求和
__m256 v = _mm256_fmadd_ps(a, b, c);
float sum = avx2_helpers::hsum_avx2(v);

// 使用快速 sigmoid
__m256 x = _mm256_loadu_ps(input);
__m256 sigmoid = avx2_helpers::sigmoid_fast_avx2(x);
```

---

## 📚 架构说明

### Qwen3 模型架构

```
Input Tokens
    ↓
Token Embedding
    ↓
┌────────────────────────────────────────┐
│  Qwen3 Decoder Layer (×28)            │
│  ┌───────────────────────────────────┐ │
│  │ Input RMSNorm + Residual         │ │
│  ├───────────────────────────────────┤ │
│  │ Self-Attention (GQA)             │ │
│  │  - Q Projection                  │ │
│  │  - K Projection                  │ │
│  │  - V Projection                  │ │
│  │  - QK Norm                      │ │
│  │  - RoPE                         │ │
│  │  - Scaled Dot-Product Attention  │ │
│  │  - O Projection                  │ │
│  ├───────────────────────────────────┤ │
│  │ Residual Connection              │ │
│  ├───────────────────────────────────┤ │
│  │ Post-Attention RMSNorm + Residual│ │
│  ├───────────────────────────────────┤ │
│  │ MLP (SwiGLU)                     │ │
│  │  - Gate Projection               │ │
│  │  - Up Projection                 │ │
│  │  - SwiGLU Activation             │ │
│  │  - Down Projection               │ │
│  └───────────────────────────────────┘ │
└────────────────────────────────────────┘
    ↓
Final RMSNorm
    ↓
Output Logits
```

### 注意力机制

- **分组查询注意力 (GQA)**: 8个 KV heads，16个 query heads
- **Head dimension**: 128
- **RoPE**: 旋转位置编码（128维）

---

## 🔍 已知问题与限制

### 当前限制

1. **仅支持 CPU 推理**: 无 GPU 实现
2. **固定 batch size = 1**: 推理优化
3. **max_seq_len = 128**: KV cache 限制

### TODO

- [ ] 支持变长序列
- [ ] 添加量化支持 (INT8/FP16)
- [ ] 实现批处理推理
- [ ] 添加更多模型（Qwen2, Qwen1.5）

---

## 📄 许可证

本项目遵循 MIT 许可证。

---

## 🙏 致谢

- Qwen 模型：阿里巴巴达摩院
- Safetensors：HuggingFace
- AVX2 优化参考：英特尔 intrinsics 指南

---

## 📧 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。

**最后更新**: 2026-01-12
