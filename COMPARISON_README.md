# Streaming Attention vs PyTorch SDPA 性能对比

本目录包含对比测试脚本，用于测试C++ Streaming Attention和PyTorch SDPA的性能。

## 文件说明

- **`compare_attention.py`** - 简单版本对比（PyTorch vs C++串行）
- **`compare_attention_omp.py`** - 完整版本（包括OpenMP多线程测试）

## 使用方法

### 1. 安装依赖

```bash
pip install torch numpy
```

### 2. 准备C++程序

确保在项目根目录下有attention目录和相关源文件：

```bash
cd /media/song/LocalDisk/Weblearning/并行计算/final/
ls attention/
# 应该看到: streaming_serial.cpp, streaming_omp.cpp 等
```

### 3. 运行对比测试

#### 简单版本（只测试串行）

```bash
python compare_attention.py
```

#### 完整版本（包括OpenMP）

```bash
python compare_attention_omp.py
```

脚本会自动：
1. 编译C++程序（如果需要）
2. 运行PyTorch测试
3. 运行C++测试
4. 生成对比报告

## 测试配置

默认配置：
- **序列长度**: 512, 1024, 2048, 4096, 8192
- **隐藏维度**: 128
- **迭代次数**: 100
- **Block Size**: 64 (Streaming Attention)
- **OpenMP线程**: 1, 2, 4, 8, 16

可以在脚本中修改这些参数：

```python
SEQ_LENS = [512, 1024, 2048, 4096, 8192]  # 序列长度
HIDDEN_DIM = 128                          # 隐藏维度
ITERS = 100                               # 迭代次数
```

## 预期结果

### PyTorch SDPA优势

1. **高度优化的内核**
   - 使用oneDNN (Intel)、MKL等优化库
   - SIMD指令优化 (AVX2, AVX512)
   - 更好的缓存利用率

2. **单线程性能**
   - 在单线程下，PyTorch通常比朴素C++实现快
   - 典型加速比: 2-10x

### C++ Streaming Attention优势

1. **OpenMP多线程**
   - 可以充分利用多核CPU
   - 线性扩展性（在问题规模足够大时）

2. **可定制性**
   - 可以针对特定硬件优化
   - 适合教学和算法演示

3. **算法透明性**
   - 清晰的算法实现
   - 便于理解和修改

## 典型输出示例

```
=========================================================================
  Streaming Attention vs PyTorch SDPA 性能对比
=========================================================================

PyTorch版本: 2.0.0
CUDA可用: False

序列长度 | PyTorch(ms) | C++(ms) | PT-吞吐量 | C++-吞吐量 | 加速比(C++/PT)
---------|------------|---------|----------|-----------|--------------
    512  |     1.2345 |   2.4567 |   414.91 |    208.43 |      1.99x
   1024  |     3.4567 |   6.7890 |   296.29 |    150.82 |      1.96x
   2048  |    10.2345 |  20.4567 |   200.08 |    100.15 |      2.00x
   4096  |    35.6789 |  75.2345 |   114.81 |     54.44 |      2.11x
   8192  |   125.6789 | 250.4567 |    65.20 |     32.70 |      1.99x

平均加速比: 2.01x
结论: C++ Streaming Attention平均比PyTorch SDPA快 2.01x
```

## 性能分析

### 为什么PyTorch更快？

1. **优化库**
   - Intel oneDNN / MKL
   - 高度优化的汇编代码
   - 自动向量化

2. **内存访问**
   - 更好的缓存利用率
   - 内存预取
   - 数据布局优化

### 如何让C++更快？

1. **使用OpenMP多线程**
   ```bash
   OMP_NUM_THREADS=8 ./streaming_omp 2048 128 64 8
   ```

2. **编译优化**
   ```bash
   g++ -O3 -march=native -fopenmp \
       -ffast-math -funroll-loops \
       streaming_omp.cpp streaming_serial.cpp -o streaming_omp
   ```

3. **SIMD优化**
   - 使用AVX/AVX512 intrinsics
   - 手动向量化关键循环

## 故障排除

### 问题1: 找不到attention目录

```
✗ 错误: attention目录不存在
```

**解决**: 确保在正确的目录运行脚本
```bash
cd /media/song/LocalDisk/Weblearning/并行计算/final/
python compare_attention.py
```

### 问题2: 编译失败

```
✗ 编译失败: error: ...
```

**解决**: 检查g++版本和OpenMP支持
```bash
g++ --version  # 需要7.0+
echo | g++ -fopenmp -x c++ - -E - > /dev/null && echo "OpenMP OK"
```

### 问题3: PyTorch未安装

```
ModuleNotFoundError: No module named 'torch'
```

**解决**: 安装PyTorch
```bash
pip install torch
```

### 问题4: C++程序运行缓慢

**原因**: 可能是问题规模太小，或者没有使用优化编译

**解决**:
1. 使用更大的序列长度 (2048+)
2. 确保使用 `-O3 -march=native` 编译
3. 使用OpenMP多线程版本

## 课程报告建议

### 实验设计

1. **对比目标**
   - PyTorch SDPA (工业级优化)
   - C++ Streaming Attention (教学实现)

2. **测试维度**
   - 不同序列长度下的性能
   - 单线程 vs 多线程
   - 串行 vs 并行效率

3. **性能指标**
   - 执行时间
   - 吞吐量 (tokens/sec)
   - 加速比
   - 并行效率

### 报告要点

1. **算法对比**
   - Standard Attention vs Streaming Attention
   - 时间复杂度相同，但cache行为不同

2. **实现对比**
   - PyTorch使用优化库
   - C++实现算法透明

3. **性能分析**
   - PyTorch单线程优势
   - C++多线程扩展性
   - 实际应用场景选择

## 参考资源

- [PyTorch SDPA文档](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [Flash Attention论文](https://arxiv.org/abs/2205.14135)
- [Online Softmax算法](https://arxiv.org/abs/1802.09579)

---

**最后更新**: 2026年1月11日
