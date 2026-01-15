# MPI Attention 重命名和Sequence Parallelism实现计划

## 已完成 ✅

### 1. ops_mpi.h/cpp 重命名
- ✅ 添加 `attention_headwise_standard()` - Head-wise + Standard
- ✅ 添加 `attention_headwise_online_softmax()` - Head-wise + Online Softmax
- ✅ 实现 `attention_sequence_online_softmax()` - Sequence + Online Softmax
- ✅ 保留旧函数作为deprecated wrapper

### 2. qwen3_ops_mpi.h 枚举更新
- ✅ 添加 `ParallelStrategy` 枚举 (HEAD_WISE, SEQUENCE)
- ✅ 添加 `AttentionAlgorithm` 枚举 (STANDARD, ONLINE_SOFTMAX)
- ✅ 保留 `MPIAttentionType` 作为deprecated
- ✅ 添加新的 `qwen3_attention_mpi_omp()` 重载

## 待完成 🚧

### 3. qwen3_ops_mpi.cpp 实现新重载

需要在 `qwen3_attention_mpi_omp()` 旧版本 (line 62-175) 后添加：

```cpp
Tensor qwen3_attention_mpi_omp(
    const Tensor& hidden_states,
    size_t num_attention_heads,
    size_t num_key_value_heads,
    size_t head_dim,
    const Tensor& qkv_projs,
    const Tensor& o_proj,
    const Tensor& q_norm_weight,
    const Tensor& k_norm_weight,
    const Tensor& cos,
    const Tensor& sin,
    MPI_Comm comm,
    ParallelStrategy strategy,
    AttentionAlgorithm algorithm
) {
    // 准备 Q, K, V (与旧版本相同)
    // ... (lines 76-146 from old version)

    // 根据strategy和algorithm选择底层函数
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    Tensor attn_output;

    if (strategy == ParallelStrategy::HEAD_WISE) {
        if (algorithm == AttentionAlgorithm::ONLINE_SOFTMAX) {
            attn_output = ops::mpi::attention_headwise_online_softmax(
                q, k, v, nullptr, scale,
                num_attention_heads, num_key_value_heads, comm
            );
        } else {  // STANDARD
            attn_output = ops::mpi::attention_headwise_standard(
                q, k, v, nullptr, scale,
                num_attention_heads, num_key_value_heads, comm
            );
        }
    } else {  // SEQUENCE
        if (algorithm == AttentionAlgorithm::ONLINE_SOFTMAX) {
            // 注意：需要处理全局序列长度
            size_t global_seq_len = seq_len * /* size from MPI */;
            attn_output = ops::mpi::attention_sequence_online_softmax(
                q, k, v, nullptr, scale,
                num_attention_heads, num_key_value_heads,
                global_seq_len, comm
            );
        } else {
            throw std::runtime_error("Sequence parallelism with standard attention not implemented");
        }
    }

    // 输出投影 (与旧版本相同)
    // ... (lines 166-172 from old version)
}
```

### 4. qwen3_ops_mpi_avx.h/cpp 更新

需要添加相同的枚举和函数声明/实现。

### 5. benchmark_qwen3.cpp 命令行更新

添加新的命令行选项：
```bash
--parallel-strategy [headwise|sequence]  # 并行策略
--attention-algo [standard|online_softmax]  # Attention算法
```

向后兼容映射：
- `--attention streaming` → `HEAD_WISE + ONLINE_SOFTMAX`
- `--attention standard` → `HEAD_WISE + STANDARD`

### 6. 编译和测试

```bash
# 编译
cd build && make -j8

# 测试Head-wise Standard
mpirun -np 2 ./benchmark_qwen3 \
    --parallel-strategy headwise \
    --attention-algo standard

# 测试Head-wise Online Softmax (原STREAMING)
mpirun -np 2 ./benchmark_qwen3 \
    --parallel-strategy headwise \
    --attention-algo online_softmax

# 测试Sequence Online Softmax (新实现)
mpirun -np 2 ./benchmark_qwen3 \
    --parallel-strategy sequence \
    --attention-algo online_softmax
```

### 7. 性能验证

需要测试的场景：
- [ ] Head-wise + Standard (baseline)
- [ ] Head-wise + Online Softmax (当前STREAMING)
- [ ] Sequence + Online Softmax (新实现)

预期性能排序：
```
Sequence + Online Softmax > Head-wise + Online Softmax > Head-wise + Standard
```

## 关键文件

| 文件 | 状态 | 说明 |
|------|------|------|
| `include/tensor_cpp/ops_mpi.h` | ✅ 完成 | 新函数声明 |
| `src/ops_mpi.cpp` | ✅ 完成 | 新函数实现 |
| `include/tensor_cpp/qwen3_ops_mpi.h` | 🚧 半完成 | 枚举已添加，待实现新重载 |
| `src/qwen3_ops_mpi.cpp` | ⏳ 待做 | 需要添加新重载实现 |
| `include/tensor_cpp/qwen3_ops_mpi_avx.h` | ⏳ 待做 | 需要同步枚举 |
| `src/qwen3_ops_mpi_avx.cpp` | ⏳ 待做 | 需要同步实现 |
| `tests/benchmark/benchmark_qwen3.cpp` | ⏳ 待做 | 需要更新命令行解析 |

## 下一步行动

1. 完成qwen3_ops_mpi.cpp新重载实现
2. 编译测试
3. 简单功能测试
4. Commit第一阶段工作
5. 继续剩余集成工作

---
**状态**: 进行中 (50% 完成)
**最后更新**: 2025-01-15
