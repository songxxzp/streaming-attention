# 服务器部署使用指南

本指南用于在服务器集群上部署和运行Qwen3性能测试实验。

## 📋 文件清单

```
scripts/
├── build_on_server.sh           # 服务器编译脚本
├── quickstart_server.sh         # 快速开始脚本（推荐）
├── exp1_serial_baseline.sh      # 实验1: 串行baseline
├── exp2_single_node_16threads.sh # 实验2: 单机16线程
├── exp3_mpi_parallel.sh         # 实验3: MPI并行
├── run_all_experiments.sh       # 运行全部实验
└── README_EXPERIMENTS.md        # 详细实验文档
```

## 🚀 快速开始

### 方法1: 使用快速开始脚本（推荐）

```bash
cd /media/song/LocalDisk/Weblearning/并行计算/final/tensor_cpp
bash scripts/quickstart_server.sh
```

该脚本会自动：
1. ✅ 检查服务器环境（MPI、模型文件、CPU支持）
2. 🔨 编译项目（如果需要）
3. 🧪 运行快速测试验证
4. 📊 让你选择要运行的实验

### 方法2: 手动步骤

#### 1. 编译项目

```bash
cd /media/song/LocalDisk/Weblearning/并行计算/final/tensor_cpp
bash scripts/build_on_server.sh
```

编译选项：
- `bash scripts/build_on_server.sh` - 标准编译
- `bash scripts/build_on_server.sh clean` - 清理后重新编译
- `bash scripts/build_on_server.sh clean verbose` - 详细输出

#### 2. 运行实验

**本地实验**（不需要SLURM）：

```bash
# 实验1: 串行Baseline (~5分钟)
bash scripts/exp1_serial_baseline.sh

# 实验2: 单机16线程 (~10分钟)
bash scripts/exp2_single_node_16threads.sh

# 或运行全部本地实验
bash scripts/run_all_experiments.sh  # 只会运行exp1和exp2
```

**MPI实验**（需要SLURM集群）：

```bash
# 单节点测试
srun --mpi=pmix -p student -N 1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=16 \
  bash scripts/exp3_mpi_parallel.sh

# 8节点完整测试
srun --mpi=pmix -p student -N 8 --ntasks=8 --ntasks-per-node=1 --cpus-per-task=16 \
  bash scripts/exp3_mpi_parallel.sh
```

## 📂 路径配置

所有脚本已配置为使用服务器上的路径：

- **模型路径**: `/student/2025310707/Qwen3-0.6B/model.safetensors` ✅
- **项目路径**: `/media/song/LocalDisk/Weblearning/并行计算/final/tensor_cpp`
- **结果路径**: `results/exp*` (相对于项目根目录)

## 📊 查看结果

实验完成后，结果保存在CSV文件中：

```bash
# 查看实验1结果
cat results/exp1_serial_baseline/serial_baseline_results.csv

# 查看实验2结果
cat results/exp2_single_node_16threads/single_node_16threads_results.csv

# 查看实验3结果
cat results/exp3_mpi_parallel/mpi_parallel_results.csv
```

## 🔧 环境要求

### 必需软件
- GCC/G++ (支持C++17)
- CMake (>= 3.16)
- MPI (OpenMPI或MPICH)
- OpenMP

### 硬件要求
- CPU支持AVX2指令集
- 足够的内存（建议>= 16GB）

### 服务器配置
- 8 nodes × 2 sockets × 26 CPUs
- 每节点使用16线程进行测试

## 📖 详细文档

查看完整的实验设计和说明：
```bash
cat scripts/README_EXPERIMENTS.md
```

## ⚠️ 常见问题

### 编译失败
```bash
# 检查MPI是否安装
mpicxx --version

# 如果未安装，加载MPI模块
module load openmpi  # 或 module load mpich
```

### 运行时找不到库
```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

### 内存不足
- 减少batch_size或序列长度
- 关闭其他占用内存的程序

### MPI实验失败
- 确保在SLURM环境中运行
- 检查分区名是否正确（默认: student）
- 检查节点是否可用（`sinfo`）

## 📞 支持

遇到问题时检查：
1. 编译日志: `build/` 目录
2. 实验日志: `results/exp*/` 目录下的 `.log` 文件
3. Benchmark源码: `tests/benchmark/benchmark_qwen3.cpp`
4. MPI实现: `src/ops_mpi.cpp`

---

**更新时间**: 2025-01-16
**状态**: ✅ 可用于服务器集群
