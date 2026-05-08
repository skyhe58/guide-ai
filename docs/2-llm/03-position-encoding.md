---
title: "位置编码"
module: "llm"
difficulty: "advanced"
interviewFrequency: "high"
tags:
  - "LLM"
  - "RoPE"
  - "ALiBi"
  - "位置编码"
  - "面试高频"
codeExample: "02-llm/transformer/02_position_encoding.py"
relatedEntries:
  - "/2-llm/01-transformer-deep-dive"
  - "/2-llm/02-attention-mechanism"
prerequisites:
  - "/1-ml-basics/08-transformer"
estimatedTime: "40min"
toolReferences:
  - name: "Perplexity"
    usage: "搜索 RoPE 和长序列外推方案"
    link: "/7-ai-tools/7.1-efficiency/ai-search"
---

# 位置编码

## 概念说明

注意力机制是置换不变的（permutation invariant），不感知 Token 的顺序。位置编码为每个 Token 注入位置信息，使模型能区分"猫吃鱼"和"鱼吃猫"。

## 核心原理

### 1. 绝对位置编码（正弦函数）

原始 Transformer 使用固定的正弦/余弦函数：

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

特点：固定编码，不需要学习，但长序列外推能力有限。

### 2. 旋转位置编码（RoPE）

RoPE 是现代 LLM 的标配位置编码（LLaMA、Qwen、Mistral）。

核心思想：将位置信息编码为旋转角度，通过旋转 Q/K 向量注入位置信息。

```
RoPE(x, pos) = x × e^(i·pos·θ)
```

关键优势：
- Q·K 的点积自然包含**相对位置**信息
- 支持长序列外推（配合 NTK-aware / YaRN 缩放）
- 不增加额外参数

### 3. ALiBi（Attention with Linear Biases）

ALiBi 不修改 Q/K，而是在注意力分数上加线性偏置：

```
Attention(Q, K, V) = softmax(Q @ K^T / √d_k + m · distance_matrix) @ V
```

其中 m 是每个头的斜率，distance_matrix 是位置距离矩阵。

特点：实现简单，外推能力好，BLOOM 和 MPT 使用。

### 4. 方案对比

| 方案 | 类型 | 代表模型 | 外推能力 | 实现复杂度 |
|------|------|----------|----------|-----------|
| 正弦编码 | 绝对 | 原始 Transformer | 有限 | 低 |
| 可学习编码 | 绝对 | GPT-2, BERT | 无 | 低 |
| **RoPE** | 相对 | **LLaMA, Qwen, Mistral** | **较好** | 中 |
| ALiBi | 相对 | BLOOM, MPT | 好 | 低 |

### 5. 长序列外推

当推理序列长度超过训练长度时，需要外推策略：

- **NTK-aware 缩放**：调整 RoPE 的频率基数
- **YaRN**：结合 NTK 和注意力缩放
- **Dynamic NTK**：根据序列长度动态调整
- **效果**：4K 训练 → 32K+ 推理

## 代码示例

> 💻 完整可运行代码：[code-examples/02-llm/transformer/02_position_encoding.py](https://github.com/skyhe58/guide-ai/tree/main/code-examples/02-llm/transformer/02_position_encoding.py)

```python
# RoPE 频率预计算
def precompute_rope_freqs(dim, max_len=4096, base=10000.0):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len, dtype=torch.float)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)
```

## 实战要点

1. **RoPE 是现代 LLM 标配**，面试必须掌握其核心思想
2. **长序列外推**是实际部署中的常见需求
3. 不需要记忆正弦编码的具体公式，理解原理即可

## 常见面试题

### Q1: RoPE 的核心思想是什么？

**难度**：⭐⭐⭐ | **频率**：🔥🔥🔥

**标准答案**：RoPE 将位置信息编码为旋转角度，对 Q 和 K 向量进行旋转。旋转后 Q·K 的点积自然包含相对位置信息（只依赖 m-n），不需要额外的位置编码向量。优势：支持长序列外推，不增加参数。

### Q2: RoPE 和 ALiBi 的区别？

**难度**：⭐⭐⭐ | **频率**：🔥🔥

**标准答案**：RoPE 修改 Q/K 向量（旋转），ALiBi 修改注意力分数（加偏置）。RoPE 是主流方案（LLaMA/Qwen），ALiBi 实现更简单但使用较少。两者都支持长序列外推。

## 推荐工具

| 工具 | 用途 | 详情 |
|------|------|------|
| Perplexity | 搜索 RoPE 外推方案 | [AI 搜索](/7-ai-tools/7.1-efficiency/ai-search) |

## 参考资料

- [RoPE 论文 — RoFormer](https://arxiv.org/abs/2104.09864)
- [ALiBi 论文](https://arxiv.org/abs/2108.12409)
- [YaRN 论文](https://arxiv.org/abs/2309.00071)
- [Eleuther AI — RoPE 详解](https://blog.eleuther.ai/rotary-embeddings/)
