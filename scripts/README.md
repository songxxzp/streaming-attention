# 性能测试脚本使用说明

本目录包含在服务器上运行性能测试的完整脚本。

## 测试环境

- **硬件配置**: 8节点，每节点2CPU×26核 = 共416核
- **软件要求**:
  - spack (模块管理)
  - cmake
  - openmpi
  - g++ (支持C++17和OpenMP)

## 脚本列表

### 1. `build_on_server.sh`
服务器编译脚本，自动加载模块并编译所有测试程序。

```bash
./build_on_server.sh
```

**功能**:
- 自动加载spack模块 (cmake, openmpi)
- 检查编译器版本
- 清理旧构建
- 编译所有测试程序

### 2. `test_attention_scalability.sh`
Attention算子性能扩展性测试，测试串行、OpenMP和MPI+OpenMP三种模式。

```bash
./test_attention_scalability.sh [seq_len]
```

**参数**:
- `seq_len`: 序列长度（默认4096）

**测试内容**:
1. **串行版本**: 单核baseline
2. **OpenMP版本**: 1, 2, 4, 8, 16, 26, 52, 104, 208, 416线程
3. **MPI+OpenMP混合**: 1, 2, 4, 8, 16个MPI进程，每进程26个OpenMP线程

**结果输出**: `results/attention_scalability_<timestamp>/`
- `serial_log.txt`: 串行版本日志
- `omp_threads_N_log.txt`: OpenMP各线程数日志
- `mpi_ranks_N_log.txt`: MPI各进程数日志
- `summary.txt`: 性能总结报告

### 3. `test_qwen3_throughput.sh`
Qwen3模型吞吐量测试，测试Prefill和Decode两个阶段。

```bash
./test_qwen3_throughput.sh <model_path>
```

**参数**:
- `model_path`: Qwen3模型文件路径（必需）

**测试配置**:
- **Prefill长度**: 4096 tokens
- **Decode长度**: 1024 tokens
- **测试线程数**: 1, 2, 4, 8, 16, 26, 52, 104, 208, 416

**测试内容**:
1. **Prefill阶段（串行+多线程）**: 测试prompt处理吞吐量
2. **Decode阶段（串行+多线程）**: 测试token生成吞吐量（使用KV cache）

**结果输出**: `results/qwen3_throughput_<timestamp>/`
- `prefill_serial_log.txt`: Prefill串行日志
- `prefill_threads_N_log.txt`: Prefill多线程日志
- `decode_serial_log.txt`: Decode串行日志
- `decode_threads_N_log.txt`: Decode多线程日志
- `summary.txt`: 吞吐量总结报告
- `analyze_results.py`: Python分析脚本

## 使用流程

### 步骤1: 编译项目

```bash
cd /media/song/LocalDisk/Weblearning/并行计算/final/scripts
./build_on_server.sh
```

### 步骤2: 运行Attention测试

```bash
# 使用默认参数（seq_len=4096）
./test_attention_scalability.sh

# 或指定序列长度
./test_attention_scalability.sh 8192
```

### 步骤3: 运行Qwen3吞吐量测试

```bash
# 指定模型路径
./test_qwen3_throughput.sh /path/to/Qwen3-0.6B/model.safetensors
```

## 服务器运行注意事项

### 模块加载

脚本会自动加载以下模块：
```bash
spack load cmake
spack load openmpi
```

如果服务器环境不同，可能修改`build_on_server.sh`中的模块名。

### OpenMP线程数设置

脚本通过环境变量设置线程数：
```bash
export OMP_NUM_THREADS=16
```

### MPI运行

Attention测试使用原始attention项目的MPI版本：
```bash
/usr/bin/mpirun -np 16 ./streaming_mpi
```

**重要**: 使用`/usr/bin/mpirun`（系统OpenMPI），而非Anaconda的MPICH。

### SLURM作业调度

如果使用SLURM提交作业，修改脚本添加`srun`前缀：

```bash
# 单节点OpenMP
srun -p student --ntasks=1 --cpus-per-task=16 \
    env OMP_NUM_THREADS=16 ./build/benchmark_qwen3 \
    --model $MODEL_PATH --phase prefill --prompt-len 4096

# 多节点MPI+OpenMP
srun --mpi=pmix -p student \
    -N 2 --ntasks=8 --ntasks-per-node=4 \
    --cpus-per-task=26 \
    env OMP_NUM_THREADS=26 \
    /usr/bin/mpirun -np 8 ./streaming_mpi
```

## 性能指标说明

### Attention测试
- **总时间**: 完成一次attention计算的时间（ms）
- **加速比**: 相对于串行版本的性能提升倍数
- **计算公式**: `加速比 = 串行时间 / 并行时间`

### Qwen3吞吐量测试
- **Prefill吞吐量**: 每秒处理的token数（tokens/sec）
- **Decode吞吐量**: 每秒生成的token数（tokens/sec）
- **时间**: 完成指定长度处理的总时间（ms）

## 故障排除

### 编译错误
```bash
# 检查g++版本
g++ --version  # 需要7.0+

# 检查OpenMP支持
echo "#include <omp.h>" | g++ -x c++ - -fopenmp -

# 检查MPI
/usr/bin/mpirun --version
```

### 运行时错误
```bash
# 检查模型文件
ls -lh /path/to/model.safetensors

# 检查线程数设置
echo $OMP_NUM_THREADS

# 测试基本功能
./build/test_ops
```

### 性能异常
```bash
# 检查CPU频率
cat /proc/cpuinfo | grep MHz

# 检查系统负载
top

# 使用perf分析
perf stat -e cycles,instructions,cache-misses ./build/benchmark_qwen3
```

## 结果分析

### 查看结果摘要
```bash
# Attention测试
cat results/attention_scalability_*/summary.txt

# Qwen3测试
cat results/qwen3_throughput_*/summary.txt
```

### 生成图表（如果服务器有Python）
```bash
cd results/qwen3_throughput_<timestamp>
python3 analyze_results.py > chart_data.txt
```

### 课程报告所需数据

1. **加速比曲线**: 不同线程数下的性能提升
2. **并行效率**: `效率 = 加速比 / 处理器数`
3. **可扩展性分析**: 强扩展性（固定问题规模）vs 弱扩展性
4. **Prefill vs Decode**: 两个阶段的性能特征对比

## 文件结构

```
scripts/
├── README.md                          # 本文档
├── build_on_server.sh                 # 编译脚本
├── test_attention_scalability.sh      # Attention扩展性测试
└── test_qwen3_throughput.sh          # Qwen3吞吐量测试

../results/
├── attention_scalability_<timestamp>/  # Attention测试结果
│   ├── serial_log.txt
│   ├── omp_threads_*.txt
│   ├── mpi_ranks_*.txt
│   └── summary.txt
└── qwen3_throughput_<timestamp>/      # Qwen3测试结果
    ├── prefill_*.txt
    ├── decode_*.txt
    ├── analyze_results.py
    └── summary.txt
```

## 联系与支持

如有问题，请检查：
1. 服务器环境是否满足要求
2. 模块是否正确加载
3. 模型文件路径是否正确
4. 日志文件中的错误信息

---

**最后更新**: 2026年1月11日
**适用课程**: 并行计算
