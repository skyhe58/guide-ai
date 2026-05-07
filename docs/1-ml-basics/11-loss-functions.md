---
title: "损失函数"
module: "ml-basics"
difficulty: "intermediate"
interviewFrequency: "high"
tags:
  - "深度学习"
  - "损失函数"
  - "MSE"
  - "交叉熵"
  - "Focal Loss"
  - "面试高频"
codeExample: "01-ml-basics/"
relatedEntries:
  - "/1-ml-basics/01-supervised-learning"
  - "/1-ml-basics/09-math-foundations"
  - "/1-ml-basics/10-evaluation-tuning"
prerequisites:
  - "/1-ml-basics/01-supervised-learning"
estimatedTime: "35min"
toolReferences:
  - name: "Perplexity"
    usage: "搜索损失函数选择指南和数学推导"
    link: "/7-ai-tools/7.1-efficiency/ai-search"
---

# 损失函数

## 概念说明

**损失函数**（Loss Function）衡量模型预测值与真实值的差距，是模型训练的优化目标。训练过程就是通过梯度下降最小化损失函数。

选对损失函数 = 告诉模型"什么是好的预测"。

## 核心原理

### 1. 回归损失

| 损失函数 | 公式 | 特点 | 适用场景 |
|----------|------|------|----------|
| **MSE** | $\frac{1}{n}\sum(y-\hat{y})^2$ | 对大误差敏感（平方放大） | 通用回归 |
| **MAE** | $\frac{1}{n}\sum\|y-\hat{y}\|$ | 对异常值鲁棒 | 有异常值的回归 |
| **Huber** | MSE（小误差）+ MAE（大误差） | 兼顾两者优点 | 鲁棒回归 |

```python
import torch.nn as nn

mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()
huber_loss = nn.SmoothL1Loss()
```

### 2. 分类损失

| 损失函数 | 适用场景 | PyTorch |
|----------|----------|---------|
| **交叉熵** | 多分类（LLM 预训练） | `nn.CrossEntropyLoss()` |
| **二元交叉熵** | 二分类/多标签 | `nn.BCEWithLogitsLoss()` |
| **Focal Loss** | 类别极度不平衡 | 需手动实现 |

**交叉熵**是最重要的分类损失：

$$L = -\sum_{i} y_i \log(\hat{y}_i)$$

直觉：预测正确类别的概率越高，损失越小。LLM 预训练的损失就是下一个 Token 的交叉熵。

```python
# PyTorch CrossEntropyLoss 内含 Softmax，输入是 logits（不需要手动 softmax）
criterion = nn.CrossEntropyLoss()
logits = model(input)           # (batch, num_classes) — 原始分数
loss = criterion(logits, labels) # labels 是类别索引（不是 one-hot）
```

### 3. Focal Loss

Focal Loss 解决类别极度不平衡问题（如目标检测中背景远多于目标）：

$$FL = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

- $\gamma$（聚焦参数）：降低易分类样本的权重，让模型关注难分类样本
- $\alpha$（平衡因子）：调整正负样本的权重

```python
# Focal Loss 简化实现
def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    return (alpha * (1 - pt) ** gamma * ce_loss).mean()
```

### 4. 损失函数选择指南

| 任务 | 推荐损失函数 |
|------|-------------|
| 回归 | MSE（通用）、MAE（有异常值） |
| 二分类 | BCEWithLogitsLoss |
| 多分类 | CrossEntropyLoss |
| 多标签分类 | BCEWithLogitsLoss（每个标签独立） |
| 类别不平衡 | Focal Loss 或加权 CrossEntropy |
| LLM 预训练 | CrossEntropyLoss（下一个 Token 预测） |
| 对比学习 | InfoNCE / Triplet Loss |

## 常见面试题

### Q1: 交叉熵损失的直觉含义？为什么分类不用 MSE？

**难度**：⭐⭐⭐ | **频率**：🔥🔥🔥

**标准答案**：交叉熵衡量预测概率分布与真实分布的差异，值越小越好。分类不用 MSE 的原因：(1) MSE 对 Softmax 输出求导时梯度可能很小（学习慢）；(2) 交叉熵的梯度与预测误差成正比（学习快）；(3) 交叉熵有概率论基础（最大似然估计等价于最小化交叉熵）。

**追问**：Label Smoothing 是什么？为什么有用？（将 one-hot 标签软化为 [0.9, 0.05, 0.05]，防止模型过于自信）

### Q2: Focal Loss 解决什么问题？

**难度**：⭐⭐ | **频率**：🔥🔥

**标准答案**：Focal Loss 解决类别极度不平衡问题。标准交叉熵中，大量易分类的负样本主导了损失，模型无法关注少量难分类的正样本。Focal Loss 通过 $(1-p_t)^\gamma$ 因子降低易分类样本的权重，让模型聚焦于难分类样本。$\gamma=2$ 时，预测概率 0.9 的样本权重降低 100 倍。

**追问**：除了 Focal Loss，还有什么方法处理类别不平衡？（过采样、欠采样、加权损失、SMOTE）

## 推荐工具

> 📌 以下工具可帮助你更高效地学习和实践本知识点，详见 [模块 7：AI 使用与实践](/7-ai-tools/)

| 工具 | 用途 | 详情 |
|------|------|------|
| Perplexity | 搜索损失函数选择指南和数学推导 | [AI 搜索](/7-ai-tools/7.1-efficiency/ai-search) |

## 参考资料

- [PyTorch — Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions)
- [Focal Loss 论文](https://arxiv.org/abs/1708.02002)
- [Cross-Entropy 直觉解释 — Chris Olah](https://colah.github.io/posts/2015-09-Visual-Information/)
