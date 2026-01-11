# PyTorch正确性验证测试 - 完整指南

本项目提供了完整的PyTorch验证测试系统，用于验证tensor_cpp中所有算子的正确性。

## 📋 验证系统概述

### 已实现的验证测试

| 算子类型 | 测试文件 | 验证方法 | 状态 |
|---------|---------|---------|------|
| **Self-Attention** | torch_validation.py | F.scaled_dot_product_attention | ✅ 已实现 |
| **Cross-Attention** | torch_validation.py | 手动实现 | ✅ 已实现 |
| **Streaming Attention** | torch_validation.py | 手动实现 | ✅ 已实现 |
| **Linear Layer** | torch_validation.py | torch.nn.functional.linear | ✅ 已实现 |
| **RMS Norm** | torch_validation.py | 自定义实现 | ✅ 已实现 |
| **Embedding** | torch_validation.py | torch.nn.functional.embedding | ✅ 已实现 |
| **Argmax** | torch_validation.py | torch.argmax | ✅ 已实现 |
| **SwiGLU** | torch_validation.py | torch.nn.functional.silu | ✅ 已实现 |

## 🚀 快速开始

### 方式1: 快速验证测试 (推荐新手)

```bash
# 步骤1: 生成测试数据
python3 quick_attention_test.py

# 步骤2: 查看生成的测试数据
ls -lh test_*.npy test_cross_*.npy

# 步骤3: C++程序加载这些数据进行计算（需要实现）
# 然后生成 cpp_self_attention_output.npy 和 cpp_cross_attention_output.npy

# 步骤4: 再次运行验证
python3 quick_attention_test.py
```

**输出示例：**
```
============================================================
  Quick Attention Validation Test
============================================================

Testing Self-Attention
============================================================
Input shape: (2, 2, 8, 16)
Reference output shape: (2, 2, 8, 16)
Reference output (first element): -0.126617
✓ Test data saved

Checking C++ Outputs
============================================================
Self-Attention:
  ✓ PASSED - Max abs error: 1.23e-06
```

### 方式2: 完整验证测试

```bash
# 一键运行所有测试
./run_torch_validation.sh

# 或手动分步执行
python3 torch_validation.py                    # 生成18个测试用例
cd tensor_cpp && make torch-validation          # 编译
./build/torch_validation                        # 运行C++测试
cd .. && python3 torch_validation.py --check-results  # 验证结果
```

## 📁 文件结构

```
.
├── torch_validation.py              # 主验证脚本（18个测试用例）
├── quick_attention_test.py          # 快速验证脚本（2个测试用例）
├── run_torch_validation.sh          # 一键运行脚本
├── PYTORCH_VALIDATION_README.md     # 详细文档
├── tensor_cpp/
│   ├── tests/
│   │   └── torch_validation.cpp    # C++验证程序
│   └── Makefile                    # 添加了torch-validation目标
└── test_data/                       # 测试数据目录（自动生成）
    ├── self_attention_1_*.npy
    ├── self_attention_1_meta.json
    ├── ...
    └── index.json
```

## 🎯 测试覆盖详情

### Attention算子测试

#### 1. Self-Attention (3个测试)
- 使用 `torch.nn.functional.scaled_dot_product_attention`
- 测试配置：
  - 小: (batch=2, heads=2, seq=8, dim=16)
  - 中: (batch=1, heads=4, seq=16, dim=32)
  - 大: (batch=4, heads=8, seq=64, dim=64)

#### 2. Cross-Attention (2个测试)
- 验证不同序列长度的交叉注意力
- 测试配置：
  - (batch=2, heads=2, q_len=8, kv_len=16, dim=16)
  - (batch=1, heads=4, q_len=32, kv_len=128, dim=32)

#### 3. Streaming Attention (3个测试)
- 单查询格式的流式注意力
- 测试配置：
  - T=512, d=64
  - T=1024, d=128
  - T=2048, d=256

### 基础算子测试

#### 4. Linear Layer (2个测试)
```python
y = xA^T + b
```
- 容差: rtol=1e-4, atol=1e-5

#### 5. RMS Norm (2个测试)
```python
output = input / sqrt(mean(input^2) + eps) * weight
```
- 容差: rtol=1e-4, atol=1e-5

#### 6. Embedding (2个测试)
```python
output = weight[indices]
```
- 容差: rtol=1e-5, atol=1e-6

#### 7. Argmax (2个测试)
```python
output = argmax(input, dim=-1)
```
- 精确匹配（整数索引）

#### 8. SwiGLU (2个测试)
```python
output = silu(gate) * x
```
- 容差: rtol=1e-5, atol=1e-6

## ✅ 验证流程

### Python侧工作流

```python
# 1. PyTorchValidator类 - 参考实现
class PyTorchValidator:
    @staticmethod
    def scaled_dot_product_attention(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)

# 2. TestGenerator类 - 生成测试用例
generator = TestGenerator()
generator.generate_all_tests()  # 生成18个测试

# 3. check_results函数 - 验证结果
check_results()  # 比较C++输出和PyTorch参考
```

### C++侧工作流

```cpp
// 1. 加载测试数据
auto q_data = load_npy_float32("test_data/self_attention_1_query.npy");
auto k_data = load_npy_float32("test_data/self_attention_1_key.npy");
auto v_data = load_npy_float32("test_data/self_attention_1_value.npy");

// 2. 创建tensor并运行算子
TensorF q(q_data, Shape({2, 2, 8, 16}));
TensorF k(k_data, Shape({2, 2, 8, 16}));
TensorF v(v_data, Shape({2, 2, 8, 16}));
TensorF output = self_attention(q, k, v, nullptr, scale);

// 3. 保存输出
save_npy_float32("test_data/cpp_self_attention_1_output.npy",
                  output.data(), output.size(), shape);
```

## 📊 容差设置说明

不同算子使用不同的容差，基于数值稳定性考虑：

| 算子类型 | 相对容差(rtol) | 绝对容差(atol) | 说明 |
|---------|---------------|---------------|------|
| Attention | 1e-3 | 1e-4 | softmax+matmul累积误差 |
| Linear | 1e-4 | 1e-5 | 单次matmul |
| RMS Norm | 1e-4 | 1e-5 | 平方根操作 |
| Embedding | 1e-5 | 1e-6 | 简单查表 |
| SwiGLU | 1e-5 | 1e-6 | SiLU激活函数 |
| Argmax | 0 | 0 | 整数精确匹配 |

## 🔧 扩展新算子

### 添加新算子验证的步骤

1. **在PyTorchValidator中添加参考实现**
```python
class PyTorchValidator:
    @staticmethod
    def your_new_operator(input1, input2):
        # 使用PyTorch实现
        return torch.some_function(input1, input2)
```

2. **在TestGenerator中添加测试生成器**
```python
def generate_your_operator_tests(self):
    validator = PyTorchValidator()
    config = {'param1': 64, 'param2': 128}

    # 生成输入
    input1 = torch.randn(...)
    input2 = torch.randn(...)

    # 计算参考
    ref = validator.your_new_operator(input1, input2)

    # 保存测试用例
    test_case = TestCase(
        name=f"your_operator_1",
        inputs={'input1': input1.numpy(), 'input2': input2.numpy()},
        reference_output=ref.numpy(),
        tolerance={'rtol': 1e-4, 'atol': 1e-5}
    )
    self.save_test_case(test_case)
```

3. **在torch_validation.cpp中添加运行器**
```cpp
void run_your_operator_test(const string& test_name, const string& data_dir) {
    // 加载输入
    auto input1 = load_npy_float32(data_dir + "/" + test_name + "_input1.npy");
    auto input2 = load_npy_float32(data_dir + "/" + test_name + "_input2.npy");

    // 运行算子
    TensorF output = your_operator(input1, input2);

    // 保存输出
    save_npy_float32(data_dir + "/cpp_" + test_name + "_output.npy", ...);
}
```

4. **在generate_all_tests中注册**
```python
def generate_all_tests(self):
    # ... 现有测试 ...
    self.generate_your_operator_tests()
```

## 🐛 故障排除

### 常见问题

**Q: ImportError: No module named 'torch'**
```bash
pip install torch numpy
```

**Q: numpy加载失败**
```bash
pip install --upgrade numpy
```

**Q: C++编译错误**
```bash
# 确保安装了必要的依赖
sudo apt install libomp-dev libopenmpi-dev
```

**Q: 测试失败但误差很小**
- 检查容差设置是否合理
- 浮点运算在不同平台可能有微小差异
- 可以适当调整容差

## 📈 CI/CD集成

示例GitHub Actions配置：

```yaml
name: PyTorch Validation

on: [push, pull_request]

jobs:
  validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install dependencies
        run: |
          pip install torch numpy
      - name: Generate test data
        run: python3 torch_validation.py
      - name: Build C++ tests
        run: |
          cd tensor_cpp
          make torch-validation
      - name: Run C++ validation
        run: |
          cd tensor_cpp
          ./build/torch_validation
      - name: Check results
        run: python3 torch_validation.py --check-results
```

## 🎓 学习资源

- [PyTorch Attention机制](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [Online Softmax算法](https://arxiv.org/abs/2002.05702)
- [Flash Attention](https://arxiv.org/abs/2205.14135)

## 📝 更新日志

- **2026-01-11**: 初始版本
  - 实现18个测试用例
  - 覆盖8类算子
  - 支持完整验证流程

## 🤝 贡献

欢迎提交新的算子验证测试！

---

**注意**: 所有测试使用固定随机种子(torch.manual_seed(42))以确保可重复性。
