# Tensor Library C++ with OpenMP and MPI

高性能的 C++ Tensor 库，支持深度学习常用算子，并集成 OpenMP 和 MPI 并行能力。包含 Qwen3 模型推理支持。

## 特性

- ✅ **PyTorch 风格 API**: 简洁易用的接口设计
- ✅ **完整算子支持**: Add, Argmax, Embedding, Linear, RMS Norm, RoPE, SwiGLU, Self/Cross-Attention
- ✅ **OpenMP 并行**: 自动利用多核 CPU
- ✅ **MPI 分布式**: 支持多节点训练
- ✅ **Qwen3 推理**: 支持 Qwen3-0.6B 模型推理（Prefill + Decode）
- ✅ **高性能**: 基于 C++17，优化编译
- ✅ **KV Cache**: Decode 阶段优化的 KV Cache 实现

## 目录结构

```
tensor_cpp/
├── include/tensor_cpp/       # 公共头文件
│   ├── tensor.h             # Tensor 类声明
│   ├── tensor_impl.tpp      # Tensor 模板实现
│   ├── ops.h                # 所有算子实现
│   ├── qwen3_loader.h       # Qwen3模型加载器
│   ├── qwen3_ops.h          # Qwen3推理操作
│   └── kv_cache.h           # KV Cache实现
│
├── src/                     # 源代码实现
│   ├── tensor.cpp
│   ├── ops.cpp
│   ├── qwen3_loader.cpp
│   └── qwen3_ops.cpp
│
├── tests/                   # 测试套件
│   ├── test_ops.cpp         # 基础算子测试
│   ├── test_attention.cpp   # Attention专项测试
│   ├── benchmark_attention.cpp   # Attention性能测试
│   ├── benchmark_qwen3.cpp       # Qwen3推理性能测试
│   ├── test_mpi_simple.cpp # MPI简单测试
│   └── test_streaming_attention.cpp  # Streaming Attention测试
│
├── examples/                # 使用示例
│   └── basic_usage.cpp      # 完整示例代码
│
├── scripts/                 # 编译和运行脚本
│   ├── compile.sh           # 编译脚本
│   └── run_benchmark.sh     # 性能测试脚本
│
├── CMakeLists.txt          # CMake构建配置
└── README.md               # 本文件
```

---

## 快速开始

### 方式1：使用 CMake 编译（推荐）

```bash
# 进入 tensor_cpp 目录
cd tensor_cpp

# 创建 build 目录
mkdir -p build && cd build

# 配置 CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# 编译所有目标
make -j$(nproc)

# 编译完成后可执行文件位于 build/ 目录：
# - test_ops                # 基础算子测试
# - test_attention          # Attention测试
# - benchmark_attention     # Attention性能测试
# - benchmark_qwen3         # Qwen3推理测试
# - basic_usage             # 示例程序
```

### 方式2：使用编译脚本

```bash
# 使用项目根目录的脚本
bash scripts/build_on_server.sh

# 或者使用 tensor_cpp 自己的编译脚本
bash tensor_cpp/scripts/compile.sh
```

### 方式3：手动编译

```bash
# 编译基础算子测试
g++ -std=c++17 -O3 -march=native -I./include \
    -fopenmp src/tensor.cpp src/ops.cpp tests/test_ops.cpp \
    -o test_ops

# 编译 Qwen3 推理程序（需要链接 safetensors 库）
g++ -std=c++17 -O3 -march=native -I./include \
    -fopenmp src/tensor.cpp src/ops.cpp src/qwen3_loader.cpp src/qwen3_ops.cpp \
    tests/benchmark_qwen3.cpp -lsafetensors \
    -o benchmark_qwen3

# 编译 MPI 版本
mpicxx -std=c++17 -O3 -march=native -I./include \
    -fopenmp src/tensor.cpp src/ops.cpp src/qwen3_loader.cpp src/qwen3_ops.cpp \
    tests/benchmark_qwen3.cpp -lsafetensors \
    -o benchmark_qwen3_mpi
```

---

## Qwen3 模型推理

### 1. 准备模型

将 Qwen3-0.6B 模型转换为 safetensors 格式：

```python
# convert_model.py
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
state_dict = model.state_dict()
save_file(state_dict, "qwen3-0.6b.safetensors")
```

### 2. Prefill 阶段推理

Prefill 阶段处理用户输入的 prompt，所有 token 并行计算。

```bash
# OpenMP 版本（单节点多线程）
OMP_NUM_THREADS=16 ./build/benchmark_qwen3 \
    --model /path/to/qwen3-0.6b.safetensors \
    --phase prefill \
    --prompt-len 128 \
    --iters 10 \
    --threads 16

# 输出示例：
# Qwen3 Prefill 阶段性能测试
# ===================================
# 模型: /path/to/qwen3-0.6b.safetensors
# Prompt 长度: 128 tokens
# 迭代次数: 10 (预热: 2)
# 并行模式: OpenMP (16 threads)
#
# 平均时间: 45.23 ms
# 吞吐量: 2830.5 tokens/s
# 显存占用: 2.1 GB
```

### 3. Decode 阶段推理

Decode 阶段自回归生成 token，每次计算新 token 与历史 token 的 attention。

```bash
# OpenMP 版本（单节点多线程）
OMP_NUM_THREADS=16 ./build/benchmark_qwen3 \
    --model /path/to/qwen3-0.6b.safetensors \
    --phase decode \
    --gen-len 100 \
    --iters 10 \
    --threads 16

# 输出示例：
# Qwen3 Decode 阶段性能测试
# ==================================
# 模型: /path/to/qwen3-0.6b.safetensors
# 生成长度: 100 tokens
# 迭代次数: 10 (预热: 2)
# 并行模式: OpenMP (16 threads)
# KV Cache: 启用
#
# 平均时间/token: 12.34 ms
# 吞吐量: 81.0 tokens/s
# 显存占用: 2.8 GB
```

### 4. MPI 多节点推理

使用 MPI 进行多节点并行推理：

```bash
# 创建 hosts 文件
cat > hosts << EOF
node1
node2
node3
node4
EOF

# 4 节点并行推理（每节点 1 个 rank）
mpirun -np 4 --hostfile hosts \
    --map-by ppr:1:node --bind-to core \
    ./build/benchmark_qwen3 \
    --model /path/to/qwen3-0.6b.safetensors \
    --mode mpi \
    --phase prefill \
    --prompt-len 1024 \
    --iters 10

# 混合并行：2 个 MPI ranks，每 rank 8 个 OpenMP 线程
mpirun -np 2 \
    --map-by ppr:1:node --bind-to core \
    -x OMP_NUM_THREADS=8 \
    ./build/benchmark_qwen3 \
    --model /path/to/qwen3-0.6b.safetensors \
    --mode mpi \
    --phase decode \
    --gen-len 100
```

### 5. Streaming Attention 推理

使用 Streaming Attention 减少内存占用：

```bash
# Prefill 阶段使用 Streaming Attention
./build/benchmark_qwen3 \
    --model /path/to/qwen3-0.6b.safetensors \
    --phase prefill \
    --attention streaming \
    --prompt-len 4096 \
    --iters 10

# Streaming Attention 优势：
# - 内存占用：O(Td) vs 标准 Attention 的 O(T²)
# - 适合超长序列（T > 8192）
# - CPU 环境下性能更优
```

---

## 命令行参数

### benchmark_qwen3 参数说明

```bash
用法: ./benchmark_qwen3 [选项]

选项:
  --model PATH              模型文件路径 (.safetensors)
                            [默认: /path/to/Qwen3-0.6B/model.safetensors]

  --phase PHASE             测试阶段:
                            - prefill:  预填充阶段，处理输入prompt
                            - decode:   解码阶段，自回归生成token
                            [默认: prefill]

  --mode MODE               并行模式:
                            - omp:     OpenMP多线程
                            - mpi:     MPI多节点
                            - serial:  串行执行
                            [默认: omp]

  --attention TYPE          Attention类型:
                            - standard: 标准Attention
                            - streaming: Streaming Attention (推荐长序列)
                            [默认: standard]

  --prompt-len N            prompt长度（tokens） [默认: 128]
  --gen-len N               生成长度（tokens） [默认: 100]

  --iters N                 迭代次数 [默认: 10]
  --warmup N                预热次数 [默认: 2]

  --threads N               OpenMP线程数 [默认: 16]
  --num-threads N           同 --threads

  --no-kv-cache             decode阶段不使用KV cache（性能会下降）

  --verbose                 输出详细信息（每层的时间）
  --help                    显示帮助信息
```

### 性能优化建议

**Prefill 阶段**：
```bash
# 短序列（< 512）
OMP_NUM_THREADS=8 ./benchmark_qwen3 --phase prefill --prompt-len 128

# 中等序列（512 - 4096）
OMP_NUM_THREADS=16 ./benchmark_qwen3 --phase prefill --prompt-len 2048

# 长序列（> 4096）- 使用 Streaming Attention
OMP_NUM_THREADS=16 ./benchmark_qwen3 \
    --phase prefill --attention streaming --prompt-len 8192
```

**Decode 阶段**：
```bash
# 单节点
OMP_NUM_THREADS=16 ./benchmark_qwen3 \
    --phase decode --gen-len 100 --threads 16

# 多节点（2 MPI × 8 OMP）
mpirun -np 2 -x OMP_NUM_THREADS=8 \
    ./benchmark_qwen3 --mode mpi --phase decode
```

---

## 运行基础测试

### 测试基础算子

```bash
./build/test_ops

# 测试内容包括：
# - Tensor 创建和操作
# - 加法、乘法、矩阵乘法
# - Linear, RMSNorm, RoPE, SwiGLU
# - Embedding 查找
# - Argmax 操作
```

### 测试 Attention

```bash
# Self-Attention 测试
./build/test_attention

# Streaming Attention 测试
./build/test_streaming_attention

# Attention 性能测试
./build/benchmark_attention --seq-len 1024 --threads 16
```

### MPI 测试

```bash
# 4 个进程运行 MPI 测试
mpirun -np 4 ./build/test_mpi_simple

# Attention MPI 测试
mpirun -np 4 ./build/test_attention --mode mpi
```

---

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
| `streaming_attention` | 流式自注意力 | ✅ | ✅ |
| `cross_attention` | 交叉注意力机制 | ✅ | ✅ |
| `qwen3_layer` | Qwen3 Transformer 层 | ✅ | ✅ |
| `qwen3_prefill` | Qwen3 Prefill 阶段 | ✅ | ✅ |
| `qwen3_decode` | Qwen3 Decode 阶段 | ✅ | ✅ |

---

## 性能基准

### Self-Attention OpenMP 扩展性

输入形状: (4, 8, 64, 64)

| 线程数 | 时间 | 加速比 |
|-------|------|--------|
| 1 | 10.064 ms | 1.00x |
| 2 | 5.010 ms | 2.01x |
| 4 | 4.184 ms | 2.41x |
| 8 | 4.157 ms | 2.42x |

### Qwen3 Prefill 阶段（16线程）

| Prompt 长度 | 时间 | 吞吐量 |
|-----------|------|--------|
| 128 | 45.2 ms | 2830 tokens/s |
| 512 | 180.5 ms | 2836 tokens/s |
| 2048 | 725.3 ms | 2824 tokens/s |
| 8192 | 2910.1 ms | 2815 tokens/s |

### Qwen3 Decode 阶段（16线程 + KV Cache）

| 生成长度 | 时间/token | 吞吐量 |
|---------|-----------|--------|
| 50 | 12.1 ms | 82.6 tokens/s |
| 100 | 12.3 ms | 81.0 tokens/s |
| 200 | 12.5 ms | 80.0 tokens/s |

---

## 使用示例

### 创建 Tensor

```cpp
#include "tensor_cpp/tensor.h"
#include "tensor_cpp/ops.h"

using namespace tensor_cpp;
using namespace ops;

// 创建零张量
TensorF zeros = TensorF::zeros(Shape({2, 3}));

// 创建随机张量
TensorF random = TensorF::randn(Shape({2, 2}));

// 从数据创建
std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
TensorF x(data, Shape({2, 2}));
```

### Qwen3 推理示例

```cpp
#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/kv_cache.h"

using namespace tensor_cpp::qwen3;

// 1. 加载模型
Qwen3Loader loader("/path/to/qwen3-0.6b.safetensors");
auto model = loader.load_model();

// 2. Prefill 阶段
TensorF input_ids = TensorF::from_vector({123, 456, 789});  // 3个token
auto prefill_output = qwen3_prefill(model, input_ids);

// 3. Decode 阶段（带 KV Cache）
KVCache cache(model.config, 128);  // max_len=128
TensorF new_token = TensorF::from_vector({999});

for (int i = 0; i < 100; ++i) {
    auto output = qwen3_decode(model, new_token, cache);
    new_token = argmax(output);  // 贪婪解码
}
```

### Element-wise 操作

```cpp
TensorF a = TensorF::ones(Shape({2, 2}));
TensorF b = a * 2.0f;      // 标量乘法
TensorF c = a + b;         // 张量加法
TensorF d = a.sqrt();      // 平方根
TensorF e = a.exp();       // 指数
```

---

## 集成到你的项目

### 方式 1: CMake（推荐）

```cmake
cmake_minimum_required(VERSION 3.15)
project(MyProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

# 添加 tensor_cpp 库
add_subdirectory(external/tensor_cpp)

# 你的可执行文件
add_executable(my_app src/main.cpp)
target_link_libraries(my_app
    tensor_cpp
    OpenMP::OpenMP_CXX
    MPI::MPI_CXX
    safetensors  # 如果使用 Qwen3
)
```

### 方式 2: 手动编译

```bash
# 在你的项目中
g++ -std=c++17 -O3 -march=native \
    -I/path/to/tensor_cpp/include \
    -fopenmp \
    your_code.cpp \
    /path/to/tensor_cpp/src/*.cpp \
    -lsafetensors \
    -o your_program
```

---

## 依赖

### 必需
- C++17 编译器 (g++ 7.0+ 或 clang++ 5.0+)
- CMake 3.15+ (如果使用 CMake 构建)
- OpenMP (通常编译器自带)

### 可选
- OpenMPI 4.0+ (用于 MPI 支持)
- libtorch (用于模型转换)
- safetensors C++ 库 (用于 Qwen3 推理)

### 安装依赖

```bash
# Ubuntu/Debian
sudo apt-get install build-essential cmake libomp-dev libopenmpi-dev

# 安装 safetensors
pip install safetensors

# 从源码编译 safetensors C++ 库
git clone https://github.com/huggingface/safetensors.git
cd safetensors
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
```

---

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
   - MPI: 使用 `#ifdef USE_MPI` 包裹

---

## 常见问题

### Q: 编译时找不到 safetensors 库？
A: 确保 safetensors C++ 库已安装：
```bash
# 检查库是否安装
ldconfig -p | grep safetensors

# 或手动指定路径
g++ ... -L/path/to/safetensors/lib -lsafetensors
```

### Q: OpenMP 线程数设置无效？
A: 检查环境变量：
```bash
echo $OMP_NUM_THREADS
# 或在代码中设置
omp_set_num_threads(16);
```

### Q: MPI 运行时所有进程都在同一个节点？
A: 使用 `--map-by ppr:1:node` 确保进程分布：
```bash
mpirun -np 4 --map-by ppr:1:node --hostfile hosts ./app
```

### Q: Streaming Attention 性能反而更慢？
A: Streaming Attention 适合长序列（T > 2048）。短序列使用标准 Attention 更优。

---

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 相关文档

- [主项目 README](../README.md)
- [性能测试报告](../REPORT.md)
- [Qwen3 模型说明](https://huggingface.co/Qwen/Qwen3-0.6B)
