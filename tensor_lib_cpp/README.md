# Tensor Library C++ with OpenMP and MPI

高性能的 C++ Tensor 库，支持深度学习常用算子，并集成 OpenMP 和 MPI 并行能力。

## 特性

- ✅ **PyTorch 风格 API**: 简洁易用的接口设计
- ✅ **完整算子支持**: Add, Argmax, Embedding, Linear, RMS Norm, RoPE, SwiGLU, Self/Cross-Attention
- ✅ **OpenMP 并行**: 自动利用多核 CPU
- ✅ **MPI 分布式**: 支持多节点训练
- ✅ **高性能**: 基于 C++17，优化编译
- ✅ **Header-Only**: 模板库，无需额外编译

## 目录结构

```
tensor_lib_cpp/
├── include/tensor_lib/      # 公共头文件（Header-Only模板库）
│   ├── tensor.h             # Tensor 类声明
│   ├── tensor_impl.tpp      # Tensor 模板实现
│   └── ops.h                # 所有算子实现
│
├── tests/                   # 测试套件
│   ├── test_ops.cpp         # 基础算子测试
│   └── test_attention.cpp   # Attention专项测试
│
├── examples/                # 使用示例
│   └── basic_usage.cpp      # 完整示例代码
│
├── docs/                    # 文档
│   └── PROJECT_STRUCTURE.md # 项目结构说明
│
├── build/                   # 编译输出（自动生成）
├── results/                 # 测试结果（自动生成）
├── Makefile                 # 构建配置
└── README.md                # 本文件
```

> **注意**: 这是一个 **Header-Only 模板库**。所有实现都在头文件中，`src/` 目录被删除是因为模板库不需要单独的 `.cpp` 实现文件。

## 编译

### 依赖

- C++17 编译器 (g++ 7.0+)
- OpenMP (通常编译器自带)
- OpenMPI (可选，用于 MPI 支持)

### 快速开始

```bash
# 编译所有测试和示例
make

# 运行基础算子测试
make run

# 运行 Attention 测试
make test-attention

# 运行示例
make run-examples

# 查看所有命令
make help
```

### 编译选项

```bash
# 仅编译 OpenMP 版本
make build/test_ops

# 编译 MPI 版本
make build/test_ops_mpi

# 编译示例
make examples
```

## 运行测试

### 单机测试 (OpenMP)

```bash
# 基础算子测试
make run

# Attention 测试（Self + Cross）
make test-attention
```

### 多节点测试 (MPI + OpenMP)

```bash
# 使用 4 个进程运行基础测试
make run-mpi

# 使用 4 个进程运行 Attention 测试
make test-attention-mpi

# 自定义进程数
N=8 make run-mpi-n
```

### 测试结果

测试结果保存在 `results/` 目录：
- `test_results.txt` - 基础算子测试
- `attention_test_results.txt` - Attention 测试

## 支持的算子

| 算子 | 功能 | OpenMP | MPI |
|------|------|--------|-----|
| `add` | 逐元素加法 | ✅ | - |
| `argmax` | 找最大值索引 | ✅ | - |
| `embedding` | 词嵌入查找 | ✅ | - |
| `linear` | 线性层 (y=xA^T+b) | ✅ | - |
| `rms_norm` | 均方根归一化 | ✅ | - |
| `rope` | 旋转位置编码 | ✅ | - |
| `swiglu` | SwiGLU 激活函数 | ✅ | - |
| `self_attention` | 自注意力机制 | ✅ | ✅ |
| `cross_attention` | 交叉注意力机制 | ✅ | ✅ |
| `all_reduce_sum` | MPI 全局求和 | - | ✅ |
| `broadcast` | MPI 广播 | - | ✅ |

## 性能基准

### Self-Attention OpenMP 扩展性

输入形状: (4, 8, 64, 64)

| 线程数 | 时间 | 加速比 |
|-------|------|--------|
| 1 | 10.064 ms | 1.00x |
| 2 | 5.010 ms | 2.01x |
| 4 | 4.184 ms | 2.41x |
| 8 | 4.157 ms | 2.42x |

### Cross-Attention OpenMP 扩展性

Query: (4, 8, 32, 64), Key/Value: (4, 8, 128, 64)

| 线程数 | 时间 | 加速比 |
|-------|------|--------|
| 1 | 14.951 ms | 1.00x |
| 2 | 7.233 ms | 2.07x |
| 4 | 3.812 ms | 3.92x |
| 8 | 3.535 ms | 4.23x |

## 使用示例

### 创建 Tensor

```cpp
#include "tensor_lib/tensor.h"
#include "tensor_lib/ops.h"

using namespace tensor_lib;
using namespace ops;

// 创建零张量
TensorF zeros = TensorF::zeros(Shape({2, 3}));

// 创建随机张量
TensorF random = TensorF::randn(Shape({2, 2}));

// 从数据创建
std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
TensorF x(data, Shape({2, 2}));
```

### 线性层

```cpp
// 输入: (batch=2, in_features=4)
TensorF input = TensorF::randn(Shape({2, 4}));

// 权重: (out_features=3, in_features=4)
TensorF weight = TensorF::randn(Shape({3, 4}));

// 偏置: (out_features=3)
TensorF bias = TensorF::randn(Shape({3}));

// 前向传播
TensorF output = linear(input, weight, &bias);
// 输出形状: (2, 3)
```

### Self-Attention

```cpp
// Q, K, V: (batch=1, heads=2, seq_len=4, head_dim=8)
size_t batch = 1, heads = 2, seq_len = 4, head_dim = 8;

TensorF query = TensorF::randn(Shape({batch, heads, seq_len, head_dim}));
TensorF key = TensorF::randn(Shape({batch, heads, seq_len, head_dim}));
TensorF value = TensorF::randn(Shape({batch, heads, seq_len, head_dim}));

float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

TensorF output = self_attention(query, key, value,
                                 static_cast<const TensorF*>(nullptr),
                                 scale);
// 输出形状: (1, 2, 4, 8)
```

### Cross-Attention

```cpp
// Query: (batch=1, heads=2, query_len=4, head_dim=8)
// Key/Value: (batch=1, heads=2, kv_len=6, head_dim=8)
size_t query_len = 4, kv_len = 6;

TensorF query = TensorF::randn(Shape({1, 2, query_len, 8}));
TensorF key = TensorF::randn(Shape({1, 2, kv_len, 8}));
TensorF value = TensorF::randn(Shape({1, 2, kv_len, 8}));

TensorF output = cross_attention(query, key, value,
                                  static_cast<const TensorF*>(nullptr),
                                  0.353f);
// 输出形状: (1, 2, 4, 8) - 与 query 形状相同
```

### Element-wise 操作

```cpp
TensorF a = TensorF::ones(Shape({2, 2}));
TensorF b = a * 2.0f;      // 标量乘法
TensorF c = a + b;         // 张量加法
TensorF d = a.sqrt();      // 平方根
TensorF e = a.exp();       // 指数
```

## 集成到你的项目

### 方式 1: 复制头文件

```bash
cp -r include/tensor_lib /path/to/your/project/include/
```

### 方式 2: 作为 Git Submodule

```bash
git submodule add <repo-url> external/tensor_lib_cpp
```

在 Makefile 中：
```makefile
CXXFLAGS += -Iexternal/tensor_lib_cpp/include -fopenmp -std=c++17
```

### 方式 3: 直接使用

```bash
# 在你的项目中
g++ -std=c++17 -O3 -I/path/to/tensor_lib_cpp/include \
    -fopenmp your_code.cpp -o your_program
```

## Makefile 目标

| 命令 | 说明 |
|------|------|
| `make` | 编译所有测试和示例 |
| `make run` | 运行基础算子测试 |
| `make run-mpi` | 运行 MPI 测试 (4进程) |
| `make test-attention` | 运行 Attention 测试 |
| `make test-attention-mpi` | 运行 Attention MPI 测试 |
| `make examples` | 编译示例 |
| `make run-examples` | 编译并运行示例 |
| `make clean` | 清理编译产物 |
| `make help` | 显示所有命令 |

## 代码规范

1. **命名约定**:
   - 类名: `PascalCase` (如 `Tensor`, `Shape`)
   - 函数名: `snake_case` (如 `self_attention`, `add`)
   - 模板参数: `<typename T>`

2. **内存管理**:
   - 使用 `std::move` 返回大张量
   - 移动语义优化

3. **并行化**:
   - OpenMP: `#pragma omp parallel for if(condition)`
   - MPI: 使用 `#ifdef MPI_VERSION` 包裹

## 文档

- [项目结构说明](docs/PROJECT_STRUCTURE.md) - 详细的目录组织和设计说明
- `examples/basic_usage.cpp` - 完整的使用示例

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
