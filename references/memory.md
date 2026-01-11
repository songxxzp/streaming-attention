streaming attention能节省内存吗？

这是一个**必须答得“准且克制”的问题**，因为老师/评审真正关心的是：

> *你这个 streaming attention 不是拍脑袋的，它在学术和系统工作中是有清晰来源谱系的。*

下面我会按**思想来源（而不是简单罗列论文）**来给你一个**“可写进报告的权威列表”**，并明确每一类工作**到底启发了 streaming 的哪一部分**。你不需要全部实现，但**必须知道你站在谁的肩膀上**。

---

# 一、最核心的思想源头（一定要写）

## 1. Online / Streaming Softmax（最根本）

### 代表工作

* **“Online normalizer calculation for softmax”**
* 早期数值计算与并行 softmax 工作（HPC / 数值线性代数领域）

### 核心思想

* Softmax 可以：

  * **分块计算**
  * **逐步更新 max 和 sum**
  * 不需要一次性看到全部数据

### 对你的启发

> **这是 streaming attention 的数学基础**
> 没有 online softmax，就不可能在不 materialize score matrix 的情况下做 attention。

你在报告中可以写：

> Our streaming attention relies on the online softmax formulation, which allows incremental normalization without storing the full attention score matrix.

---

## 2. FlashAttention 系列（最直接的现代启发）

### 代表工作

* **FlashAttention (Dao et al., NeurIPS 2022)**
* FlashAttention-2

### 核心思想

* Block-wise attention
* 在线 softmax
* 减少内存读写
* 不 materialize attention matrix

### 关键点（你要非常注意措辞）

你**不能**说你实现了 FlashAttention（那是 GPU kernel）
你应该说：

> We adopt the *algorithmic principle* of FlashAttention—block-wise attention with online softmax—while targeting CPU and multi-node environments.

### 对你的启发

* **算法结构**
* **memory traffic 优化思想**
* 而不是 CUDA 实现

这是**完全正当、非常标准的引用方式**。

---

# 二、CPU / 系统方向的重要启发（非常加分）

## 3. Blocked / Tiled Matrix Algorithms（HPC 传统）

### 代表思想

* Cache blocking
* Tiled GEMM
* Loop tiling

### 核心思想

* 让数据：

  * 留在 cache 里
  * 被多次复用

### 对 streaming attention 的启发

* 按 KV block 处理
* 提高 cache locality
* 减少 DRAM 流量

你可以写：

> Streaming attention follows classic blocked algorithm principles widely used in high-performance computing.

---

## 4. NUMA-aware Streaming / Data Partitioning

### 代表工作（思想层面）

* NUMA-aware graph processing
* Streaming graph analytics
* Out-of-core algorithms

### 核心思想

* 数据本地化
* 计算随数据走
* 减少 remote memory access

### 对你的启发

* KV block 绑定 NUMA
* MPI 多节点 KV partition
* 带宽扩展

这一步**和你的实验结论直接呼应**。

---

# 三、与 LLM 推理强相关的系统工作（可选但很好）

## 5. LLM 推理系统中的 Prefill 优化

### 代表方向

* CPU inference optimization
* KV cache 管理
* Prefill vs Decode 分离

### 核心思想

* Prefill 是 attention-heavy
* 可批量、可并行、可 streaming

你可以写：

> Prior work on LLM inference systems highlights the prefill phase as the primary opportunity for parallelization.

---

## 6. Memory-Bound Kernel Optimization（Roofline 思想）

### 代表思想

* Roofline model
* Memory-bound vs compute-bound

### 对你的启发

* Attention 在 CPU 上是 memory-bound
* 优化重点：

  * 减少内存流量
  * 提高带宽利用率

这是你 **“为什么 MPI 比 OMP 好” 的理论根基**。

---

# 四、你可以在报告中用的“标准归因表述”（强烈推荐）

你可以直接用这段（非常安全）：

> The design of streaming attention is inspired by several lines of prior work, including online softmax normalization, block-wise attention algorithms such as FlashAttention, and classic blocked algorithms in high-performance computing. In addition, NUMA-aware data partitioning and memory-bandwidth-oriented optimization techniques motivate our system-level design for CPU and multi-node environments.

---

# 五、一个“千万别踩的雷”（重要）

❌ 不要写：

* “We implement FlashAttention on CPU”
* “We reproduce FlashAttention”

✅ 要写：

* “Inspired by”
* “Following the algorithmic principle”
* “Adapting to CPU / MPI environment”

这一区别，**决定你是“懂的人”还是“外行抄名字”**。

---

# 六、如果你愿意，我可以继续帮你做三件事之一

1. **帮你列一个“References”列表（10 篇以内，全部合理）**
2. **帮你写“Related Work”一小节（1–1.5 页，课程作业刚好）**
3. **帮你把 streaming attention 的“思想来源 → 本工作贡献”画成一张逻辑图**

你这个问题，已经完全进入**论文级写作阶段**了。
