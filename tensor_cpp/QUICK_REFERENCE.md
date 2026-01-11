# 性能测试快速参考卡

## 常用命令

### 编译
```bash
make all
```

### 快速验证
```bash
make benchmark-quick
# 或
./quick_test.sh
```

### 单独测试Attention
```bash
# Standard
OMP_NUM_THREADS=16 ./build/benchmark_attention \
    --mode standard --seq-len 1024 --iters 100 --threads 16

# Streaming
OMP_NUM_THREADS=16 ./build/benchmark_attention \
    --mode streaming --seq-len 1024 --iters 100 --threads 16 --block-size 64
```

### 单独测试Qwen3
```bash
# Prefill
OMP_NUM_THREADS=16 ./build/benchmark_qwen3 \
    --model /path/to/model.safetensors \
    --phase prefill --prompt-len 128 --iters 10 --threads 16

# Decode
OMP_NUM_THREADS=16 ./build/benchmark_qwen3 \
    --model /path/to/model.safetensors \
    --phase decode --gen-len 100 --iters 1 --threads 16
```

### 批量测试
```bash
# 线程扩展性测试
for threads in 1 2 4 8 12 16; do
    OMP_NUM_THREADS=$threads ./build/benchmark_attention \
        --mode standard --seq-len 512 --iters 20 --threads $threads
done

# 序列长度测试
for seq_len in 128 256 512 1024; do
    OMP_NUM_THREADS=16 ./build/benchmark_attention \
        --mode standard --seq-len $seq_len --iters 50 --threads 16
done
```

## 参数说明

### benchmark_attention
| 参数 | 说明 | 默认值 |
|------|------|--------|
| --mode | standard/streaming | standard |
| --seq-len | 序列长度 | 1024 |
| --hidden | 隐藏维度 | 128 |
| --heads | attention头数 | 16 |
| --iters | 迭代次数 | 100 |
| --threads | OpenMP线程数 | 16 |
| --block-size | streaming块大小 | 64 |

### benchmark_qwen3
| 参数 | 说明 | 默认值 |
|------|------|--------|
| --model | 模型文件路径 | Qwen3-0.6B路径 |
| --phase | prefill/decode | prefill |
| --prompt-len | prompt长度 | 128 |
| --gen-len | 生成长度 | 100 |
| --iters | 迭代次数 | 10 |
| --threads | OpenMP线程数 | 16 |

## 输出指标

| 指标 | 说明 |
|------|------|
| 总时间 | 所有迭代的总时间 (ms) |
| 平均时间 | 每次迭代的平均时间 (ms/iter) |
| 吞吐量 | 每秒处理的token数 (tokens/sec) |
| GFLOPS | 每秒十亿次浮点运算 |

## 课程报告关键指标

| 指标 | 计算公式 | 期望值 |
|------|----------|--------|
| 加速比 | T₁/Tₚ | >1 |
| 并行效率 | 加速比/处理器数 | 0.5-0.9 |
| 可扩展性 | 性能随处理器增长趋势 | 递增 |

## 服务器运行命令

### 单节点OpenMP
```bash
export OMP_NUM_THREADS=16
srun -p student --ntasks=1 --cpus-per-task=16 \
    env OMP_NUM_THREADS=16 ./build/benchmark_qwen3 \
    --model /path/to/model.safetensors \
    --phase prefill --prompt-len 128 --iters 10
```

### 多节点MPI (需实现MPI版本)
```bash
srun --mpi=pmix -p student \
    -N 2 --ntasks=8 --ntasks-per-node=4 \
    --cpus-per-task=2 \
    env OMP_NUM_THREADS=2 \
    ./build/benchmark_qwen3_mpi \
    --model /path/to/model.safetensors \
    --mode mpi --phase prefill --prompt-len 128
```

## 测试数据收集清单

- [ ] Attention算子线程扩展性 (1,2,4,8,12,16线程)
- [ ] Attention算子序列长度扩展性 (128,256,512,1024,2048)
- [ ] Standard vs Streaming对比
- [ ] Qwen3 Prefill性能
- [ ] Qwen3 Decode性能 (with KV cache)
- [ ] 不同规模下的最优线程数

## 故障排除

| 问题 | 解决方案 |
|------|---------|
| 编译错误 | 检查g++版本, 确认支持OpenMP |
| 运行缓慢 | 减少迭代次数, 检查CPU频率 |
| 结果异常 | 增加预热次数, 检查内存是否足够 |
| 线程不扩展 | 增加问题规模 (seq_len) |

## 文件位置

| 文件 | 路径 |
|------|------|
| 本文档 | `QUICK_REFERENCE.md` |
| 详细说明 | `BENCHMARK_README.md` |
| 测试程序 | `tests/benchmark_*.cpp` |
| 测试脚本 | `*.sh` |
| 结果输出 | `benchmark_results*/` |

---
快速参考 | 详细文档请查看 BENCHMARK_README.md
