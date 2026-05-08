---
title: "注意力机制深入"
module: "llm"
difficulty: "advanced"
interviewFrequency: "high"
tags:
  - "LLM"
  - "Attention"
  - "KV Cache"
  - "Flash Attention"
  - "面试高频"
codeExample: "02-llm/transformer/01_attention_mechanism.py"
relatedEntries:
  - "/2-llm/01-transformer-deep-dive"
  - "/2-llm/03-position-encoding"
prerequisites:
  - "/1-ml-basics/08-transformer"
estimatedTime: "45min"
toolReferences:
  - name: "Perplexity"
    usage: "搜索注意力机制最新优化论文"
    link: "/7-ai-tools/7.1-efficiency/ai-search"
  - name: "Cursor"
    usage: "辅助编写注意力机制代码"
    link: "/7-ai-tools/7.1-efficiency/ai-coding"
---

# 注意力机制深入

## 概念说明

注意力机制是 Transformer 的核心组件。本节深入分析 Scaled Dot-Product Attention、Multi-Head Attention、KV Cache 和 Flash Attention，这些是理解 LLM 推理优化的关键。

## 核心原理

### 1. Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V
```

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | Q @ K^T | 计算每对 Token 的关联度 |
| 2 | / √d_k | 缩放，防止点积过大导致 Softmax 梯度消失 |
| 3 | softmax | 归一化为概率分布 |
| 4 | × V | 加权求和，得到上下文感知的表示 |

### 2. Multi-Head Attention（MHA）

将 Q/K/V 分成多个头，每个头独立计算注意力：

- **标准 MHA**：每个头有独立的 Q/K/V 投影（GPT-2/GPT-3）
- **GQA**（Grouped Query Attention）：多个 Q 头共享 K/V 头（LLaMA 2/3、Qwen2）
- **MQA**（Multi-Query Attention）：所有 Q 头共享一组 K/V（GPT-J、Falcon）

```
MHA:  Q 头数 = K 头数 = V 头数 = 32    → KV Cache 最大
GQA:  Q 头数 = 32, K/V 头数 = 8        → KV Cache 减少 4x ⭐推荐
MQA:  Q 头数 = 32, K/V 头数 = 1        → KV Cache 最小，质量略降
```

### 3. KV Cache（推理加速核心）

自回归生成时，已生成 Token 的 K/V 不变，缓存它们避免重复计算：

```
无 KV Cache: 每步重新计算所有 Token 的 K/V → O(n²)
有 KV Cache: 每步只计算新 Token 的 K/V    → O(n)
```

KV Cache 显存占用公式：
```
KV Cache 大小 = 2 × num_layers × num_kv_heads × d_k × seq_len × batch_size × dtype_size
```

7B 模型（32 层，8 KV 头，d_k=128）生成 4096 Token：
- FP16: 2 × 32 × 8 × 128 × 4096 × 2 bytes ≈ **512 MB / 请求**

### 4. Flash Attention

Flash Attention 通过 IO 感知的算法优化注意力计算：

- **核心思想**：减少 HBM（显存）读写，利用 SRAM（片上缓存）
- **分块计算**：将 Q/K/V 分成小块，在 SRAM 中完成计算
- **效果**：速度提升 2-4x，显存从 O(n²) 降到 O(n)
- **使用**：PyTorch 2.0+ 内置 `torch.nn.functional.scaled_dot_product_attention`

## 代码示例

> 💻 完整可运行代码：[code-examples/02-llm/transformer/01_attention_mechanism.py](https://github.com/skyhe58/guide-ai/tree/main/code-examples/02-llm/transformer/01_attention_mechanism.py)

```python
# Scaled Dot-Product Attention
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights @ V, weights
```

## 实战要点

1. **KV Cache 是推理优化的关键**：理解它才能理解 vLLM 的 PagedAttention
2. **GQA 是现代 LLM 标配**：减少 KV Cache 显存，几乎不损失质量
3. **Flash Attention 已内置 PyTorch**：无需手动实现，`F.scaled_dot_product_attention` 自动启用

## 常见面试题

### Q1: 为什么注意力要除以 √d_k？

**难度**：⭐⭐⭐ | **频率**：🔥🔥🔥

**标准答案**：当 d_k 较大时，Q 和 K 的点积方差为 d_k，值会很大。大的值经过 Softmax 后梯度接近 0（饱和区），导致训练困难。除以 √d_k 将方差归一化为 1，使 Softmax 输出更平滑，梯度更稳定。

### Q2: KV Cache 的原理和作用？

**难度**：⭐⭐⭐ | **频率**：🔥🔥🔥

**标准答案**：自回归生成时，每步只生成一个新 Token，但需要与所有历史 Token 计算注意力。KV Cache 缓存已计算的 K 和 V 矩阵，每步只需计算新 Token 的 Q/K/V，将计算复杂度从 O(n²) 降到 O(n)。代价是额外的显存占用。

### Q3: GQA 和 MQA 的区别？

**难度**：⭐⭐⭐ | **频率**：🔥🔥

**标准答案**：MQA（Multi-Query Attention）所有 Q 头共享一组 K/V，KV Cache 最小但质量可能下降。GQA（Grouped Query Attention）将 Q 头分组，每组共享 K/V，是 MHA 和 MQA 的折中。LLaMA 2/3 和 Qwen2 使用 GQA。

## 推荐工具

| 工具 | 用途 | 详情 |
|------|------|------|
| Perplexity | 搜索注意力优化论文 | [AI 搜索](/7-ai-tools/7.1-efficiency/ai-search) |
| Cursor | 辅助编写注意力代码 | [AI 编程辅助](/7-ai-tools/7.1-efficiency/ai-coding) |

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Flash Attention 论文](https://arxiv.org/abs/2205.14135)
- [GQA 论文](https://arxiv.org/abs/2305.13245)
- [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
