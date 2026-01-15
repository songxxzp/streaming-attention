# Parallel Strategies and Attention Algorithms

## 概述

本文档说明了Qwen3 MPI实现中的并行策略和注意力算法的新的清晰命名约定。

## 新的命名约定

### 并行策略 (ParallelStrategy)

- **HEAD_WISE** (headwise): 按注意力头并行
  - 每个MPI进程处理一部分注意力头
  - 所有序列位置都在每个进程上计算
  - 通信复杂度: O(batch × seq_len × d_model)

- **SEQUENCE** (sequence): 按序列维度并行
  - 每个MPI进程处理一部分序列tokens
  - 所有注意力头都在每个进程上计算
  - 通信复杂度: O(batch × d_head × P)
  - **对于长序列更高效！**

### 注意力算法 (AttentionAlgorithm)

- **STANDARD** (standard): 标准注意力
  - 显式构造QK^T矩阵
  - 内存消耗大: O(seq_len^2)
  - 数值精度高

- **ONLINE_SOFTMAX** (online_softmax): 在线softmax / 流式注意力
  - 逐块计算，避免构造完整矩阵
  - 内存效率高: O(seq_len × block_size)
  - 数值特性不同但正确

## 使用方式

### 1. 命令行 (benchmark_qwen3)

```bash
# Head-wise + Standard
mpirun -np 2 ./benchmark_qwen3 \
    --method mpi \
    --parallel-strategy headwise \
    --attention-algo standard

# Head-wise + Online Softmax
mpirun -np 2 ./benchmark_qwen3 \
    --method mpi \
    --parallel-strategy headwise \
    --attention-algo online_softmax

# Sequence + Online Softmax (新实现!)
mpirun -np 2 ./benchmark_qwen3 \
    --method mpi \
    --parallel-strategy sequence \
    --attention-algo online_softmax
```

### 2. C++ API

```cpp
#include "tensor_cpp/qwen3_ops_mpi.h"

using namespace tensor_cpp::qwen3::mpi;

// 方式1: 使用新的分离参数
Tensor output = qwen3_forward_mpi_omp(
    input_ids, token_embedding, layers, norm, lm_head,
    num_layers, num_heads, num_kv_heads, head_dim, rms_eps,
    MPI_COMM_WORLD,
    ParallelStrategy::SEQUENCE,           // 并行策略
    AttentionAlgorithm::ONLINE_SOFTMAX    // 注意力算法
);

// 方式2: 使用旧API (已弃用但向后兼容)
Tensor output = qwen3_forward_mpi_omp(
    input_ids, token_embedding, layers, norm, lm_head,
    num_layers, num_heads, num_kv_heads, head_dim, rms_eps,
    MPI_COMM_WORLD,
    MPIAttentionType::STREAMING  // 映射到 HEAD_WISE + ONLINE_SOFTMAX
);
```

### 3. 直接使用底层attention函数

```cpp
#include "tensor_cpp/ops_mpi.h"

using namespace tensor_cpp::ops::mpi;

// Head-wise + Standard
Tensor output = attention_headwise_standard(
    q, k, v, mask, scale,
    num_heads, num_kv_heads,
    MPI_COMM_WORLD
);

// Head-wise + Online Softmax
Tensor output = attention_headwise_online_softmax(
    q, k, v, mask, scale,
    num_heads, num_kv_heads,
    MPI_COMM_WORLD
);

// Sequence + Online Softmax (新!)
Tensor output = attention_sequence_online_softmax(
    q, k, v, mask, scale,
    num_heads, num_kv_heads,
    global_seq_len,  // 全局序列长度
    MPI_COMM_WORLD
);
```

## 性能对比

### Head-wise vs Sequence 并行

| 维度 | Head-wise | Sequence |
|------|-----------|----------|
| 数据分布 | 部分头，全序列 | 全头，部分序列 |
| 通信量 | O(seq_len × d_model) | O(d_head × P) |
| 适用场景 | 短序列 | **长序列** |
| MPI进程 | 通常≤16 | 可扩展到更多进程 |

### Standard vs Online Softmax

| 维度 | Standard | Online Softmax |
|------|----------|----------------|
| 内存 | O(seq_len^2) | O(seq_len × block_size) |
| 精度 | 高 | 略不同但正确 |
| 速度 | 慢 | **快** |
| 适用场景 | 短序列 | **长序列** |

## 推荐组合

1. **短序列 (< 512 tokens)**:
   - `HEAD_WISE + STANDARD`
   - 简单可靠

2. **中等序列 (512-2048 tokens)**:
   - `HEAD_WISE + ONLINE_SOFTMAX`
   - 内存效率高

3. **长序列 (> 2048 tokens)**:
   - `SEQUENCE + ONLINE_SOFTMAX`
   - **最优性能！**
   - 通信最少，内存效率高

## 向后兼容

旧的`--attention streaming/standard`选项仍然可用：

```bash
# 旧方式 (已弃用)
mpirun -np 2 ./benchmark_qwen3 --method mpi --attention streaming

# 自动映射到新方式
mpirun -np 2 ./benchmark_qwen3 --method mpi \
    --parallel-strategy headwise \
    --attention-algo online_softmax
```

## 测试

运行测试程序验证所有策略：

```bash
# 测试所有并行策略
mpirun -np 2 ./test_parallel_strategies

# 测试新的benchmark选项
bash /tmp/test_new_api.sh
```

## 实现细节

### Sequence Parallelism算法

Sequence parallelism使用三步在线softmax：

1. **本地计算**: 每个rank计算本地部分的统计量
   - local_max[i], local_exp_sum[i], local_weighted_value[i]

2. **跨rank归约**: 使用MPI_Allreduce聚合统计量
   - global_max[i] = Max(local_max[i]) across all ranks
   - global_exp_sum[i] = Sum(local_exp_sum[i]) across all ranks

3. **本地重归一化**: 使用全局统计量修正本地输出
   - output[i] = local_weighted_value[i] / global_exp_sum[i]

**通信优势**: 只传输统计量（d_head × P个float），而不是完整注意力矩阵（seq_len^2个float）。

## 迁移指南

如果您有使用旧API的代码：

```cpp
// 旧代码
Tensor output = qwen3::mpi::qwen3_forward_mpi_omp(
    ..., MPIAttentionType::STREAMING
);

// 新代码 (推荐)
Tensor output = qwen3::mpi::qwen3_forward_mpi_omp(
    ...,
    MPI_COMM_WORLD,
    ParallelStrategy::HEAD_WISE,
    AttentionAlgorithm::ONLINE_SOFTMAX
);
```

## 相关文档

- [RENAMING_IMPLEMENTATION_PLAN.md](../RENAMING_IMPLEMENTATION_PLAN.md) - 实现计划
- [README.md](../README.md) - 项目总览
