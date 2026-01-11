# 内存带宽测试脚本使用说明

## 文件说明

1. **test_memory_bandwidth.cpp** - 单机OpenMP带宽测试程序
2. **test_memory_bandwidth_mpi.cpp** - MPI多机带宽测试程序
3. **test_bandwidth.sh** - 单机测试脚本
4. **test_bandwidth_mpi.sh** - 多机测试脚本
5. **run_all_bandwidth_tests.sh** - 完整测试脚本（推荐）

## 使用方法

### 方法1: 完整测试（推荐）

一键运行所有测试：

```bash
./run_all_bandwidth_tests.sh
```

这会自动：
- 测试单机OpenMP带宽（1/2/4/8/16/32线程）
- 测试MPI多机带宽（1/2/4/8/16 ranks × 1/2/4/8 OMP）
- 生成所有图表和CSV文件
- 生成综合报告

### 方法2: 分别测试

#### 仅测试单机带宽：

```bash
./test_bandwidth.sh
```

#### 仅测试多机带宽：

```bash
./test_bandwidth_mpi.sh
```

### 方法3: 手动测试

#### 单机测试：

```bash
# 编译
g++ -O3 -march=native -fopenmp test_memory_bandwidth.cpp -o test_memory_bandwidth

# 测试1线程
export OMP_NUM_THREADS=1
./test_memory_bandwidth

# 测试16线程
export OMP_NUM_THREADS=16
./test_memory_bandwidth
```

#### 多机测试：

```bash
# 编译
mpicxx -O3 -march=native -fopenmp test_memory_bandwidth_mpi.cpp -o test_memory_bandwidth_mpi

# 测试4个MPI rank，每个rank 4个OMP线程
mpirun -np 4 --map-by ppr:4:node --bind-to core \
    -x OMP_NUM_THREADS=4 ./test_memory_bandwidth_mpi

# 测试16个MPI rank
mpirun -np 16 --map-by ppr:1:node --bind-to core \
    -x OMP_NUM_THREADS=1 ./test_memory_bandwidth_mpi
```

## 输出文件

测试完成后会生成：

1. **CSV数据文件**
   - `bandwidth_results.csv` - 单机测试详细数据
   - `bandwidth_mpi_results.csv` - 多机测试详细数据

2. **图表文件**
   - `bandwidth_scaling.png` - 单机带宽扩展性曲线
   - `bandwidth_mpi_scaling.png` - 多机聚合带宽扩展性
   - `bandwidth_comprehensive_report.png` - 综合对比报告

## 测试内容

### 单机测试 (test_bandwidth.sh)

测试不同OpenMP线程数下的内存带宽：
- 线程数：1, 2, 4, 8, 16, 32
- 测试操作：Copy, Scale, Add, Triad
- 数组大小：每个400MB（确保大于L3缓存）

### 多机测试 (test_bandwidth_mpi.sh)

测试不同MPI和OpenMP组合：
- MPI ranks：1, 2, 4, 8, 16
- 每rank的OMP线程：1, 2, 4, 8
- 测试操作：Triad（最常用指标）
- 每个rank独立运行，无通信，仅测试聚合带宽

## 关键指标说明

### Triad带宽

最常用的内存带宽指标，操作为：`c[i] = a[i] + scalar * b[i]`

- **读内存**：2次 (a[i], b[i])
- **写内存**：1次 (c[i])
- **总访问**：3次数组大小

### 聚合带宽

所有MPI节点的带宽总和：

```
聚合带宽 = Σ (单节点带宽)
```

理想情况下：
- 2节点 = 2 × 单节点带宽
- 4节点 = 4 × 单节点带宽
- N节点 = N × 单节点带宽

### 效率

```
效率 = (实际聚合带宽 / 理想聚合带宽) × 100%
     = (实际聚合带宽 / (N × 单节点带宽)) × 100%
```

效率 > 80%：扩展性优秀
效率 50-80%：扩展性良好
效率 < 50%：通信/同步开销较大

## 注意事项

1. **服务器配置**：如果你的服务器有特殊的MPI配置，请修改脚本中的mpirun命令

2. **内存充足**：确保每个节点有足够内存（至少2GB空闲）

3. **独占运行**：测试期间尽量独占节点，避免其他进程干扰

4. **编译选项**：
   - `-O3`：最高优化级别
   - `-march=native`：针对当前CPU架构优化
   - `-fopenmp`：启用OpenMP支持

5. **运行时间**：完整测试大约需要5-10分钟

## 结果分析

将以下文件打包发送用于分析：
- 所有 `.png` 图片文件
- 所有 `.csv` 数据文件

示例：
```bash
tar czf bandwidth_results_$(date +%Y%m%d).tar.gz *.png *.csv
```

## 故障排除

### 编译失败

```bash
# 检查编译器
g++ --version
mpicxx --version

# 检查OpenMP
g++ -fopenmp -E -x c++ /dev/null > /dev/null 2>&1 && echo "OpenMP OK"
```

### MPI运行失败

```bash
# 检查MPI配置
mpirun --version

# 测试简单的MPI程序
echo "Test" | mpirun -np 4 cat
```

### 带宽异常低

- 检查是否有其他进程占用内存
- 检查CPU频率设置（应该处于最高性能模式）
- 检查NUMA配置
