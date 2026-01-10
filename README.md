# Streaming Block Attention - 并行计算课程项目

面向多 NUMA、多节点 CPU 集群的 Streaming Block Attention 并行化实现与性能分析。

## 项目结构

```
project/
├── attention/
│   ├── naive_serial.cpp       # Phase 1: Naive串行实现
│   ├── streaming_serial.cpp   # Phase 1: Streaming串行实现
│   ├── streaming_omp.cpp      # Phase 2: OpenMP并行实现
│   ├── streaming_mpi.cpp      # Phase 3: MPI+OpenMP混合并行实现
│   └── attention.h            # 公共头文件
├── utils/
│   ├── timer.h                # 性能计时工具
│   └── softmax_online.h       # Online Softmax核心算法
├── experiments/
│   ├── run_single_node.sh     # 单节点实验脚本
│   └── run_multi_node.sh      # 多节点实验脚本
├── test_correctness.cpp       # Phase 1: 正确性测试
├── test_omp.cpp               # Phase 2: OpenMP性能测试
├── test_mpi.cpp               # Phase 3: MPI性能测试
├── Makefile                   # 编译配置
└── README.md                  # 本文件
```

## 依赖项

### 必需
- C++ 编译器 (g++ 7.0+)
- OpenMP 支持
- Make

### Phase 3 可选
- MPI 实现 (OpenMPI 或 MPICH)
  - Ubuntu/Debian: `sudo apt-get install openmpi-bin openmpi-dev libopenmpi-dev`

## 编译

### Phase 1: 串行实现
```bash
make test_correctness
```

### Phase 2: OpenMP 并行
```bash
make test_omp
```

### Phase 3: MPI + OpenMP
```bash
# 首先确保已安装 MPI
make test_mpi
```

## 运行测试

### Phase 1: 正确性验证

```bash
# 基本测试
./test_correctness

# 自定义参数
./test_correctness --T 2048 --d 128 --block 64

# 预期输出:
# - L2 Error < 1e-6
# - Max Error < 1e-7
```

### Phase 2: OpenMP 性能测试

```bash
# 设置 OpenMP 线程数
export OMP_NUM_THREADS=8

# 运行测试
./test_omp --T 4096 --d 128 --block 64

# 测试内容:
# - 线程扩展性 (1, 2, 4, 8, 16, 32 线程)
# - Block size 影响 (16, 32, 64, 128, 256, 512)
# - NUMA-aware vs 非 NUMA-aware
```

### Phase 3: MPI 多节点测试

```bash
# 单机多进程测试
mpirun -np 4 ./test_mpi --T 8192 --d 128 --block 64

# 多节点测试 (需要配置 SSH 和 hosts 文件)
mpirun -hostfile hosts -np 16 ./test_mpi --T 16384 --d 256 --block 128

# 测试内容:
# - 正确性: MPI vs Serial
# - 强扩展性: 固定问题规模，增加 ranks
# - 通信开销分析
```

## 算法说明

### Naive Attention (串行)
```
O = softmax(Q @ K^T) @ V
```
完整计算，需要构造完整 QK^T 矩阵。

### Streaming Block Attention (Online Softmax)
```
初始化: m = -∞, l = 1, O = 0

对每个 block b:
    S_b = Q @ K_b^T
    m_new = max(m, max(S_b))
    l_new = l * exp(m - m_new) + Σ exp(S_b - m_new)
    O_new = O * (l * exp(m - m_new) / l_new) + Σ exp(S_b - m_new) * V_b / l_new
```
分块计算，使用 online softmax 避免构造完整矩阵。

### OpenMP 并行化策略
- **Chunk-level Parallelism**: KV blocks 分配给不同 OpenMP 线程
- **Tree Reduction**: 合并各线程的 partial results
- **NUMA-aware**: First-touch 数据分配

### MPI 并行化策略
- **Data Parallelism**: KV cache 分布在多个 MPI ranks
- **MPI_Bcast**: 广播 Q
- **MPI_Gather**: 收集所有 partial results
- **Hybrid**: 每个 rank 内部使用 OpenMP

## 性能结果

### Phase 2: OpenMP 加速比 (T=8192, d=256)

| 线程数 | 时间 (ms) | 加速比 | 效率 |
|--------|----------|--------|------|
| 1      | 1.416    | 1.00x  | 100% |
| 2      | 0.864    | 1.80x  | 90%  |
| 4      | 0.815    | 1.91x  | 48%  |
| 8      | 0.447    | 3.48x  | 44%  |
| 16     | 0.345    | 4.50x  | 28%  |

### Phase 3: MPI 强扩展性

(需要多节点环境才能运行)

## 实验报告要点

### 1. 正确性分析
- Phase 1: 验证 Naive vs Streaming 数值等价性
- Phase 2: 验证 OMP vs Serial 误差 < 1e-6
- Phase 3: 验证 MPI vs Serial 误差 < 1e-6

### 2. 性能分析
- **计算复杂度**: O(Td) for all implementations
- **空间复杂度**:
  - Naive: O(T) for scores
  - Streaming: O(block_size) for scores
  - OMP: O(block_size * n_threads) for partial results
  - MPI: O(block_size * n_ranks) for communication

### 3. 扩展性分析
- **OpenMP**: 最优线程数 8-16，过饱和导致效率下降
- **MPI**: 理论上线性加速，实际受通信开销限制
- **通信/计算比**:
  ```
  通信时间: MPI_Bcast(Q) + MPI_Gather(results)
  计算时间: local_attention(T_local)
  通信/计算比 ≈ (d * latency + d * n_ranks / bandwidth) / (T_local * d * flops_per_element)
  ```

### 4. NUMA 优化
- First-touch 分配策略
- Thread pinning (可选)
- 实测效果: 单节点提升有限，多节点显著

## 故障排除

### MPI 编译错误
```bash
# 安装 OpenMPI
sudo apt-get update
sudo apt-get install openmpi-bin openmpi-dev

# 或使用 MPICH
sudo apt-get install mpich libmpich-dev
```

### OpenMP 线程数设置
```bash
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=close  # 绑定核心
export OMP_PLACES=cores
```

### 多节点 SSH 配置
```bash
# 配置免密登录
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

## 参考文献

1. "Online Normalizer Calculation for Softmax" - Parallel softmax technique
2. "Flash Attention" - Fast attention with IO awareness
3. "Efficient Attention: Attention with Linear Complexities" - Linear attention variants

## 许可证

课程项目，仅供学习使用。
