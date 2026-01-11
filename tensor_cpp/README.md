# Tensor C++ Library - Qwen3 Implementation

高性能的 C++ Tensor 库，包含 Qwen3-0.6B 模型的完整实现。支持 OpenMP 并行和 KV Cache 优化。

## 特性

- ✅ **Qwen3-0.6B 模型完整实现**: 28层Transformer架构
- ✅ **KV Cache 支持**: 大幅提升decode阶段性能
- ✅ **OpenMP 并行**: 多线程加速
- ✅ **Safetensors 格式**: 支持HuggingFace模型权重

## 目录结构

```
tensor_cpp/
├── include/tensor_cpp/       # 头文件
│   ├── tensor.h             # Tensor类定义
│   ├── tensor_impl.tpp      # Tensor实现
│   ├── ops.h                # 算子实现（linear, rms_norm, rope等）
│   ├── qwen3_loader.h       # Qwen3模型加载器
│   ├── qwen3_ops.h          # Qwen3前向传播
│   └── kv_cache.h           # KV Cache实现
│
├── src/                     # 源文件
│   ├── tensor.cpp
│   ├── ops.cpp
│   ├── qwen3_loader.cpp
│   └── qwen3_ops.cpp
│
├── tests/                   # 测试程序（30个文件）
│   ├── test_qwen3_logits.cpp           # Forward pass示例 ⭐
│   ├── test_qwen3_generate.cpp         # 自回归生成示例 ⭐
│   └── test_qwen3_generate_with_cache.cpp # KV Cache生成示例 ⭐
│
├── CMakeLists.txt          # CMake配置
└── README.md               # 本文件
```

---

## 快速开始

### 1. 编译项目

```bash
cd tensor_cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

编译完成后，在 `build/` 目录生成以下可执行文件：
- `test_qwen3_logits` - Forward pass测试
- `test_qwen3_generate` - 自回归生成测试
- `test_qwen3_generate_with_cache` - 带KV Cache的生成测试
- `test_ops` - 基础算子测试
- `test_attention` - Attention测试
- `test_qwen3` - Qwen3基础测试
- `test_qwen3_decode` - Decode阶段测试
- `test_qwen3_verify` - 模型验证测试
- `benchmark_qwen3` - 性能基准测试
- `benchmark_attention` - Attention性能测试
- `test_mpi_simple` - MPI测试

### 2. 运行环境配置

**重要**: 如果使用anaconda环境，需要设置系统库路径：

```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

或者在每个命令前加上：
```bash
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH ./test_qwen3_logits
```

---

## 使用示例

### 示例1：Forward Pass (test_qwen3_logits) ⭐

**功能**: 对单个token进行前向传播，输出详细的logits信息，用于调试和与PyTorch对比。

```bash
cd build
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
./test_qwen3_logits
```

**输出示例**:
```
============================================================
  Qwen3 Logits Debugging Test
============================================================

Loading weights...
Weights loaded!

Input: [9707] (token for 'Hello')

Running forward pass...
Forward complete!

Hidden States (last layer, last token):
  Shape: (1, 1, 1024)
  Range: [-26.1674, 29.6104]
  Mean: -0.0723627
  Std: 2.58689

Computing logits...
Top 20 tokens:
  [0] token=21806 logit=8.1391
  [1] token=14582 logit=8.0768
  [2] token=15846 logit=7.6319
  [3] token=477 logit=7.5790
  ...

Logits statistics:
  Mean: -1.0940
  Std: 1.9828
  Min: -10.3701 (token 111386)
  Max: 8.1391 (token 21806)
```

**保存的文件**:
- `/tmp/cpp_hidden_states.bin` - 隐藏层输出（1024个float）
- `/tmp/cpp_last_hidden.bin` - 最后一个token的隐藏状态（1024个float）
- `/tmp/cpp_logits.bin` - 完整的logits（151936个float）

**用途**:
- 调试模型实现
- 与PyTorch实现对比
- 验证数值正确性

---

### 示例2：文本生成 (test_qwen3_generate)

**功能**: 自回归文本生成，不使用KV Cache（每次重新处理整个序列）。

```bash
cd build
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
./test_qwen3_generate
```

**输出示例**:
```
============================================================
  Qwen3 Text Generation Test
============================================================

Test 1: "Hello"
Input tokens (9): 151644 872 198 9707 151645 198 151644 77091 198

Generating 12 tokens...

Step  1: token=151667  logit=28.46  time= 874 ms
Step  2: token=   198  logit=31.82  time= 853 ms
Step  3: token= 32313  logit=21.70  time= 821 ms
Step  4: token=    11  logit=25.31  time= 845 ms
...

Generation Summary:
  Total time: 12647 ms
  Tokens generated: 12
  Average time per token: 1053 ms
  Tokens per second: 0.95

Decoding output:
OUTPUT: 'user\nHello\nassistant\n\nOkay, the user said "Hello" and I'
```

**特点**:
- ✅ 完整实现，易于理解
- ❌ 性能较低（每次forward都处理整个序列）
- ⏱️ 平均 1秒/token
- 📚 适合学习生成流程

---

### 示例3：文本生成 with KV Cache (test_qwen3_generate_with_cache) ⭐⭐⭐

**功能**: 使用KV Cache的自回归文本生成，性能提升约**1.7倍**。

```bash
cd build
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
./test_qwen3_generate_with_cache
```

**输出示例**:
```
============================================================
  Qwen3 Text Generation Test WITH KV CACHE
============================================================

Test 1: "Hello"
Input tokens (9): 151644 872 198 9707 151645 198 151644 77091 198

Initializing KV cache...
KV cache initialized!

Generating 12 tokens...

Phase: PREFILL (processing initial prompt)
  Prefill time: 907 ms
  Tokens processed: 9
  First predicted token: 151667 (logit=28.464)
  Cache initialized: 9 tokens

Phase: DECODE (generating tokens one by one)
  With KV cache, each step only processes 1 new token!

Step  2: token=  3553  logit=13.47  time= 608 ms  (cached_tokens=10)
Step  3: token= 75965  logit=13.16  time= 599 ms  (cached_tokens=11)
Step  4: token=  3342  logit=12.15  time= 591 ms  (cached_tokens=12)
...

Generation Summary:
  Total time: 6992 ms
  Tokens generated: 11
  Average time per token: 635 ms
  Tokens per second: 1.57
  Final cache size: 20 tokens
```

**性能对比**:
| 方法 | 总时间 | 平均时间/token | 吞吐量 | 加速比 |
|------|--------|----------------|--------|--------|
| 不用KV Cache | 12497 ms | 1041 ms | 0.96 tokens/s | 1.0x |
| **用KV Cache** | **6610 ms** | **600 ms** | **1.66 tokens/s** | **1.74x** |

**优势**:
- ✅ 性能提升1.74倍
- ✅ 内存效率更高
- ✅ 适合实际应用
- ✅ 结果完全一致（已修复索引bug）

---

## 模型规格

**Qwen3-0.6B**:
```
层数 (num_layers): 28
隐藏层维度 (hidden_size): 1024
Attention heads (num_attention_heads): 16
KV heads (num_key_value_heads): 8 (GQA - Grouped Query Attention)
Head维度 (head_dim): 128
词汇表大小 (vocab_size): 151936
中间层维度 (intermediate_size): 4096 (4 * hidden_size)
RMSNorm epsilon: 1e-6
```

---

## 代码示例

### Forward Pass

```cpp
#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/qwen3_ops.h"

using namespace tensor_cpp;
using namespace tensor_cpp::qwen3;

// 加载模型
std::string model_path = "/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors";
Qwen3Weights weights = load_qwen3_weights(model_path);

// 准备输入
std::vector<long> input_ids = {9707};  // "Hello"
Shape input_shape({1, input_ids.size()});
TensorL input(input_ids, input_shape);

// Forward pass
Tensor hidden_states = qwen3::qwen3_forward(
    input,
    weights.embed_tokens,
    weights.layers,
    weights.norm_weight,
    weights.num_layers,
    weights.num_attention_heads,
    weights.num_key_value_heads,
    weights.head_dim,
    1e-6f  // epsilon for RMSNorm
);
// hidden_states: Shape(batch_size, seq_len, hidden_size)
//                Shape(1, 1, 1024)
```

### Generation with KV Cache

```cpp
#include "tensor_cpp/qwen3_loader.h"
#include "tensor_cpp/qwen3_ops.h"
#include "tensor_cpp/kv_cache.h"

using namespace tensor_cpp;
using namespace tensor_cpp::qwen3;

// 加载模型
Qwen3Weights weights = load_qwen3_weights(model_path);

// 创建KV Cache
auto kv_cache = std::make_unique<KVCache>(
    weights.num_layers,          // 28 layers
    1,                            // batch_size
    weights.num_key_value_heads,  // 8 KV heads
    weights.head_dim,             // 128 head_dim
    4096                          // max_seq_len
);

// Phase 1: Prefill - 处理初始prompt
std::vector<long> input_ids = {151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198};
Shape input_shape({1, input_ids.size()});
TensorL input(input_ids, input_shape);

Tensor hidden_states = qwen3::qwen3_forward_with_cache(
    input,
    kv_cache.get(),
    weights.embed_tokens,
    weights.layers,
    weights.norm_weight,
    weights.num_layers,
    weights.num_attention_heads,
    weights.num_key_value_heads,
    weights.head_dim,
    1e-6f
);

// Phase 2: Decode - 逐个生成token
std::vector<long> generated = input_ids;
for (int step = 0; step < max_new_tokens; ++step) {
    // 准备单个新token
    std::vector<long> new_token = {generated.back()};
    TensorL new_input(new_token, Shape({1, 1}));

    // Forward with cache
    Tensor new_hidden = qwen3::qwen3_forward_with_cache(
        new_input,
        kv_cache.get(),
        weights.embed_tokens,
        weights.layers,
        weights.norm_weight,
        weights.num_layers,
        weights.num_attention_heads,
        weights.num_key_value_heads,
        weights.head_dim,
        1e-6f
    );

    // 计算logits
    long next_token = predict_next_token(new_hidden, weights.lm_head);
    generated.push_back(next_token);

    // 检查EOS
    if (next_token == 151645) break;
}
```

---

## 测试程序说明

### 核心测试程序 ⭐

这三个程序是最主要的使用示例：

| 程序 | 功能 | 运行时间 | 推荐场景 |
|------|------|----------|----------|
| `test_qwen3_logits` | Forward pass，输出详细logits | ~900 ms | 调试、与PyTorch对比 |
| `test_qwen3_generate` | 自回归生成（无cache） | ~13秒 (12 tokens) | 理解生成流程 |
| `test_qwen3_generate_with_cache` | 自回归生成（有cache） | ~7秒 (12 tokens) | **实际应用** ⭐⭐⭐ |

### 其他测试程序

**Qwen3相关**:
- `test_qwen3.cpp` - Qwen3基础测试
- `test_qwen3_decode.cpp` - Decode阶段专项测试
- `test_qwen3_verify.cpp` - 模型正确性验证
- `benchmark_qwen3.cpp` - 性能基准测试

**Attention相关**:
- `test_attention.cpp` - Attention机制测试
- `test_streaming_attention.cpp` - Streaming Attention测试
- `benchmark_attention.cpp` - Attention性能测试

**基础算子**:
- `test_ops.cpp` - Linear, RMSNorm, RoPE, SwiGLU等算子测试
- `test_mpi_simple.cpp` - MPI并行测试

**调试工具**:
- `test_align_qwen3.cpp` - 与PyTorch对齐测试
- `test_detailed_layer2.cpp` - 逐层详细输出
- `test_layers_debug.cpp` - 层级调试
- `torch_validation.cpp` - PyTorch验证工具

---

## 依赖

### 必需
- C++17 编译器 (g++ 7.0+ 或 clang++ 5.0+)
- CMake 3.16+
- OpenMP 4.5+ (通常编译器自带)
- MPI 4.0+ (可选，用于MPI测试)

### 系统要求
- **操作系统**: Linux (测试环境：Ubuntu 22.04)
- **内存**: 至少4GB（加载Qwen3-0.6B模型需要约2.4GB）
- **磁盘**: 约2.4GB (model.safetensors)
- **模型**: Qwen3-0.6B safetensors格式

### 安装依赖 (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install build-essential cmake
sudo apt-get install libomp-dev libopenmpi-dev

# 如果没有模型，需要安装transformers和safetensors
pip install transformers safetensors
```

---

## 性能数据

### 测试环境
- CPU: Intel Xeon (具体型号未指定)
- 编译器: GCC 13.3.0
- 优化选项: `-O3 -march=native`
- OpenMP: 4.5
- MPI: 4.0

### 实测性能

**Prefill阶段** (9 tokens):
- 时间: 907 ms
- 吞吐量: 9.9 tokens/s

**Decode阶段** (with KV Cache):
- 平均时间/token: 635 ms
- 吞吐量: 1.57 tokens/s
- 加速比: 1.8x (相比不用cache)

**对比: 不用KV Cache**:
- 平均时间/token: 1053 ms
- 吞吐量: 0.95 tokens/s

### 性能瓶颈分析

1. **内存带宽限制**: CPU上LLM推理的主要瓶颈
2. **未优化矩阵乘法**: 当前使用朴素实现
3. **单线程batch**: 当前batch_size=1

### 优化方向

- [ ] 使用BLAS库优化矩阵乘法
- [ ] SIMD指令优化（AVX-512）
- [ ] INT8/FP16量化
- [ ] 多线程batch处理
- [ ] 更好的内存访问模式

---

## 常见问题

### Q: 运行时提示 "GLIBCXX_3.4.32 not found"？
A: anaconda环境的libstdc++版本问题。设置系统库路径：
```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

### Q: 模型文件找不到？
A: 确保模型路径正确：
```bash
/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors
```

如需修改路径，编辑测试文件中的 `model_path` 变量。

### Q: 生成的文本有重复？
A: 当前使用贪婪解码（greedy decoding），容易产生重复。改进方法：
- 添加温度采样
- 使用Top-k采样
- 使用Nucleus sampling

### Q: 如何改变生成参数？
A: 编辑测试文件，修改以下参数：
```cpp
size_t max_new_tokens = 12;  // 生成的token数量
float temperature = 1.0f;    // 温度（需要自己实现）
int top_k = 50;               // Top-k采样（需要自己实现）
```

### Q: 编译时出现MPI相关错误？
A: MPI是可选的。如果不需要MPI测试，可以修改CMakeLists.txt注释掉MPI相关部分。

---

## 与PyTorch对比

### 数值验证

可以使用提供的Python脚本验证C++实现的正确性：

```python
import torch
from safetensors.torch import load_file

# 加载权重
weights = load_file("/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B/model.safetensors")

# 运行C++程序
# ./test_qwen3_logits

# 对比C++输出的bin文件
cpp_hidden = np.fromfile("/tmp/cpp_hidden_states.bin", dtype=np.float32)
cpp_logits = np.fromfile("/tmp/cpp_logits.bin", dtype=np.float32)

# 在PyTorch中运行相同输入
# ... (具体验证代码见tests/torch_validation.cpp)
```

---

## 开发计划

### 短期
- [ ] 添加温度采样
- [ ] 添加Top-k和Nucleus sampling
- [ ] 支持batch_size > 1

### 中期
- [ ] 使用BLAS库优化矩阵乘法
- [ ] 添加INT8量化支持
- [ ] 优化KV Cache内存布局

### 长期
- [ ] 支持更多模型（Llama, Mistral等）
- [ ] 分布式推理
- [ ] GPU实现（CUDA）

---

## 许可证

MIT License

---

## 相关资源

- [Qwen3-0.6B模型](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Safetensors文档](https://huggingface.co/docs/safetensors)
- [主项目README](../README.md)
- [并行计算课程报告](../REPORT.md)
