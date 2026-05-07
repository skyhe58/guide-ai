---
title: "数学基础"
module: "ml-basics"
difficulty: "intermediate"
interviewFrequency: "medium"
tags:
  - "数学"
  - "线性代数"
  - "概率统计"
  - "梯度下降"
  - "优化"
codeExample: "01-ml-basics/"
relatedEntries:
  - "/1-ml-basics/01-supervised-learning"
  - "/1-ml-basics/05-neural-networks"
  - "/1-ml-basics/11-loss-functions"
prerequisites:
  - "/0-prerequisites/05-numpy-basics"
estimatedTime: "50min"
toolReferences:
  - name: "Perplexity"
    usage: "搜索数学概念的直觉解释和可视化"
    link: "/7-ai-tools/7.1-efficiency/ai-search"
---

# 数学基础

> 📌 **够用即可，边做边补。** 本节不追求数学严谨性，而是提供 AI/ML 开发中"够用"的数学直觉。遇到不懂的公式时回来查阅。

## 概念说明

AI/ML 涉及的数学主要有三块：线性代数（向量/矩阵运算）、概率统计（分布/贝叶斯）、优化（梯度下降）。作为后端开发者，你不需要从零学数学，只需要理解这些概念在 AI 中的直觉含义。

## 核心原理

### 1. 线性代数

**向量**：Embedding 就是向量，一个 768 维的浮点数组。

```python
import numpy as np
# 两个文档的 Embedding 向量
doc_a = np.array([0.1, 0.3, 0.5])
doc_b = np.array([0.2, 0.4, 0.6])

# 点积：衡量两个向量的相似度
dot = np.dot(doc_a, doc_b)  # 0.02 + 0.12 + 0.30 = 0.44

# 范数：向量的"长度"
norm_a = np.linalg.norm(doc_a)  # √(0.01 + 0.09 + 0.25) ≈ 0.59

# 余弦相似度 = 点积 / (范数之积)
cos_sim = dot / (norm_a * np.linalg.norm(doc_b))
```

**矩阵乘法**：注意力机制的核心运算。

```python
# Q @ K^T → 注意力分数矩阵
Q = np.random.randn(10, 64)   # 10 个 Token 的 Query
K = np.random.randn(10, 64)   # 10 个 Token 的 Key
scores = Q @ K.T               # (10, 10) 注意力分数
```

**特征值/SVD**：PCA 降维的数学基础。直觉理解：找到数据方差最大的方向。

### 2. 概率统计

**贝叶斯定理**：

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

AI 中的应用：
- 朴素贝叶斯分类器
- LLM 生成：P(下一个 Token | 前面的 Token)
- 贝叶斯优化（超参调优）

**概率分布**：
- **正态分布**：权重初始化、噪声建模
- **均匀分布**：随机采样
- **Softmax 分布**：LLM 输出的 Token 概率分布

**交叉熵**：衡量两个概率分布的差异，分类任务的标准损失函数。

$$H(p, q) = -\sum p(x) \log q(x)$$

直觉：真实分布 p 和预测分布 q 越接近，交叉熵越小。

**最大似然估计（MLE）**：找到使观测数据概率最大的参数。LLM 预训练本质上就是最大似然估计——最大化训练数据中下一个 Token 的预测概率。

### 3. 梯度下降与优化

**梯度**：损失函数对参数的偏导数，指向损失增大最快的方向。梯度下降沿梯度反方向更新参数。

**优化器对比**：

| 优化器 | 核心思想 | 适用场景 |
|--------|----------|----------|
| **SGD** | 基础梯度下降 | 简单模型、需要精细调参 |
| **SGD + Momentum** | 加动量，加速收敛 | CNN 训练 |
| **Adam** | 自适应学习率（一阶矩+二阶矩） | **默认首选**，大多数场景 |
| **AdamW** | Adam + 权重衰减解耦 | **Transformer/LLM 训练标配** |

**学习率调度**：

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| Constant | 固定学习率 | 简单实验 |
| StepLR | 每 N 个 epoch 衰减 | CNN 训练 |
| CosineAnnealing | 余弦退火 | Transformer 训练 |
| Warmup + Cosine | 先升后降 | **LLM 训练标配** |
| ReduceLROnPlateau | 验证集不提升时衰减 | 通用 |

```python
# PyTorch 学习率调度
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
```

## 实战要点

**你需要记住的：**
- 余弦相似度 = 点积 / 范数之积（RAG 检索核心）
- 交叉熵是分类损失的标准选择
- Adam/AdamW 是默认优化器
- 学习率从 1e-3（小模型）或 1e-4（大模型微调）开始

**你不需要深入的：**
- 特征值分解的推导过程
- 贝叶斯定理的严格证明
- 优化器的数学推导（理解直觉即可）

## 常见面试题

### Q1: 为什么 LLM 训练用 AdamW 而不是 SGD？

**难度**：⭐⭐ | **频率**：🔥🔥

**标准答案**：AdamW 结合了 Adam 的自适应学习率和正确的权重衰减。Adam 为每个参数维护一阶矩（均值）和二阶矩（方差）估计，自动调整学习率——梯度大的参数学习率小，梯度小的参数学习率大。这对 Transformer 的大量参数特别重要。AdamW 修正了 Adam 中权重衰减和 L2 正则化的耦合问题。

**追问**：学习率 Warmup 的作用？（训练初期参数随机，大学习率会导致不稳定）

### Q2: 交叉熵损失的直觉含义？

**难度**：⭐⭐ | **频率**：🔥🔥🔥

**标准答案**：交叉熵衡量预测概率分布与真实分布的差异。真实标签是 one-hot 分布（正确类别概率为 1），预测是 Softmax 输出的概率分布。交叉熵 = -log(预测正确类别的概率)。预测越准确（正确类别概率越接近 1），交叉熵越小。LLM 预训练的损失就是下一个 Token 的交叉熵。

**追问**：交叉熵和 KL 散度的关系？

## 推荐工具

> 📌 以下工具可帮助你更高效地学习和实践本知识点，详见 [模块 7：AI 使用与实践](/7-ai-tools/)

| 工具 | 用途 | 详情 |
|------|------|------|
| Perplexity | 搜索数学概念的直觉解释和可视化 | [AI 搜索](/7-ai-tools/7.1-efficiency/ai-search) |

## 参考资料

- [3Blue1Brown — 线性代数的本质](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [StatQuest — 概率统计系列](https://www.youtube.com/c/joshstarmer)
- [Deep Learning Book — 数学基础章节](https://www.deeplearningbook.org/contents/linear_algebra.html)
- [Mathematics for Machine Learning（免费电子书）](https://mml-book.github.io/)
