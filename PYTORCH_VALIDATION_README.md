# PyTorch Validation Tests for tensor_cpp

完整的PyTorch正确性验证测试系统。

## 功能

使用PyTorch作为参考实现，验证tensor_cpp中所有算子的正确性：

1. **Attention算子**
   - `self_attention` - 自注意力 (使用F.scaled_dot_product_attention)
   - `cross_attention` - 交叉注意力
   - `streaming_attention_serial` - 流式注意力串行版

2. **基础算子**
   - `linear` - 线性层
   - `rms_norm` - RMS归一化
   - `embedding` - Embedding查找
   - `argmax` - Argmax操作
   - `swiglu` - SwiGLU激活函数

## 使用方法

### 步骤1: 生成测试数据

```bash
python3 torch_validation.py
```

这将生成18个测试用例，保存在`test_data/`目录：
- 输入数据 (.npy格式)
- PyTorch参考输出
- 测试元数据 (JSON格式)

### 步骤2: 编译C++验证程序

```bash
cd tensor_cpp
make torch-validation
```

### 步骤3: 运行C++验证程序

```bash
cd tensor_cpp
./build/torch_validation
```

C++程序将：
1. 读取test_data/中的输入数据
2. 运行tensor_cpp算子
3. 保存输出到test_data/cpp_*_output.npy

### 步骤4: 检查结果

```bash
python3 torch_validation.py --check-results
```

这将比较C++输出和PyTorch参考输出，显示：
- 每个测试的通过/失败状态
- 最大绝对误差和相对误差
- 最终汇总统计

## 示例输出

```
============================================================
Validating C++ Outputs Against PyTorch References
============================================================

✓ self_attention_1: PASSED
    Max abs error: 1.23e-06
✓ self_attention_2: PASSED
    Max abs error: 2.45e-06
✓ streaming_attention_1: PASSED
    Max abs error: 9.87e-07
...

============================================================
Validation Summary
============================================================
Total tests:  18
Passed:       18
Failed:       0
```

## 容差设置

每个测试用例都有自己的容差设置：

- **Attention算子**: rtol=1e-3, atol=1e-4
- **Linear层**: rtol=1e-4, atol=1e-5
- **RMS Norm**: rtol=1e-4, atol=1e-5
- **Embedding**: rtol=1e-5, atol=1e-6
- **Argmax**: 精确匹配（整数索引）
- **SwiGLU**: rtol=1e-5, atol=1e-6

## 工作原理

### Python侧 (torch_validation.py)

1. **PyTorchValidator类** - 使用PyTorch实现参考算子
2. **TestGenerator类** - 生成测试用例并保存为.npy文件
3. **check_results函数** - 比较C++和PyTorch输出

### C++侧 (torch_validation.cpp)

1. 读取.npy文件（简化实现）
2. 运行tensor_cpp算子
3. 保存输出为.npy文件供Python验证

## 测试覆盖

| 算子 | 测试数量 | 配置 |
|------|----------|------|
| Self-Attention | 3 | 不同batch/head/seq/dim组合 |
| Cross-Attention | 2 | 不同q_len/kv_len |
| Streaming Attention | 3 | 不同T/d组合 (512-2048) |
| Linear | 2 | 不同in/out_features |
| RMS Norm | 2 | 不同batch/seq/hidden |
| Embedding | 2 | 不同num_embeddings/embedding_dim |
| Argmax | 2 | 不同batch/vocab_size |
| SwiGLU | 2 | 不同batch/seq/hidden |
| **总计** | **18** | - |

## 故障排除

### 导入错误
```bash
pip install torch numpy
```

### numpy数组加载失败
确保.npy文件格式正确，使用numpy version >= 1.16

### C++编译错误
确保已安装OpenMP和MPI（可选）：
```bash
sudo apt install libomp-dev libopenmpi-dev
```

## 扩展新测试

要添加新的算子测试：

1. 在`PyTorchValidator`中添加参考实现
2. 在`TestGenerator`中添加生成方法
3. 在C++的`torch_validation.cpp`中添加运行器
4. 调用`generate_all_tests()`注册新测试

## 集成到CI/CD

可以轻松集成到自动化测试流程：

```bash
# 生成测试数据（只需一次）
python3 torch_validation.py

# 运行验证
cd tensor_cpp && make torch-validation && ./build/torch_validation

# 检查结果
python3 torch_validation.py --check-results || exit 1
```
