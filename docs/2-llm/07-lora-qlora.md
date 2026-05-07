---
title: "LoRA/QLoRA 微调"
module: "llm"
difficulty: "advanced"
interviewFrequency: "high"
tags:
  - "LLM"
  - "LoRA"
  - "QLoRA"
  - "微调"
  - "PEFT"
  - "面试高频"
codeExample: "02-llm/finetuning/01_lora_example.py"
relatedEntries:
  - "/2-llm/08-full-finetuning"
  - "/2-llm/09-data-preparation"
  - "/2-llm/10-finetuning-tools"
prerequisites:
  - "/2-llm/04-training-pipeline"
estimatedTime: "50min"
toolReferences:
  - name: "Perplexity"
    usage: "搜索 LoRA 最新变体和最佳实践"
    link: "/7-ai-tools/7.1-efficiency/ai-search"
  - name: "Cursor"
    usage: "辅助编写微调代码"
    link: "/7-ai-tools/7.1-efficiency/ai-coding"
---

# LoRA/QLoRA 微调

## 概念说明

LoRA（Low-Rank Adaptation）是最流行的参数高效微调方法，通过低秩分解大幅减少可训练参数。QLoRA 在此基础上加入 4-bit 量化，进一步降低显存需求。

## 核心原理

### 1. LoRA 原理

核心公式：**W' = W + ΔW = W + B × A**

- W：原始权重（冻结，不更新）
- A：低秩矩阵 (d_in, r)，高斯初始化
- B：低秩矩阵 (r, d_out)，零初始化
- r：秩（rank），通常 8-64

参数量对比（7B 模型）：
| 方法 | 可训练参数 | 占比 | 显存需求 |
|------|-----------|------|----------|
| 全参数微调 | 7B | 100% | ~56 GB |
| LoRA (r=8) | ~4M | 0.06% | ~16 GB |
| QLoRA (r=16) | ~8M | 0.1% | ~6 GB |

### 2. 秩（Rank）选择

| 秩 | 参数量 | 适用场景 |
|----|--------|----------|
| r=4 | 最少 | 简单任务、数据少 |
| r=8 | 推荐 | 通用场景 |
| r=16 | 较多 | 复杂任务（代码、数学） |
| r=64 | 接近全参数 | 追求最佳效果 |

### 3. 目标模块选择

| 策略 | 目标模块 | 效果 |
|------|----------|------|
| 最小 | q_proj, v_proj | 效果好，参数最少 |
| 推荐 | q_proj, k_proj, v_proj, o_proj | 更好效果 |
| 全面 | 注意力 + FFN 所有线性层 | 接近全参数微调 |

### 4. QLoRA 三大创新

1. **NF4 量化**：4-bit 量化基座模型，显存减少 4x
2. **双重量化**：对量化常数再量化，进一步节省 ~0.4 GB
3. **分页优化器**：GPU 显存不足时自动卸载到 CPU

### 5. LoRA 关键超参数

```python
LoraConfig(
    r=16,                    # 秩
    lora_alpha=32,           # 缩放因子（通常 = 2 × r）
    lora_dropout=0.05,       # Dropout
    target_modules=[         # 目标模块
        "q_proj", "k_proj", "v_proj", "o_proj",
    ],
    task_type="CAUSAL_LM",
)
```

## 代码示例

> 💻 LoRA 实现：[code-examples/02-llm/finetuning/01_lora_example.py](https://github.com/your-repo/tree/main/code-examples/02-llm/finetuning/01_lora_example.py)
> 💻 QLoRA 示例：[code-examples/02-llm/finetuning/02_qlora_example.py](https://github.com/your-repo/tree/main/code-examples/02-llm/finetuning/02_qlora_example.py)

## 实战要点

1. **QLoRA 是个人开发者的最佳选择**：7B 模型只需 6GB 显存
2. **推荐 r=16, alpha=32**：通用场景的最佳平衡
3. **推理时合并权重**：无额外开销，与原始模型速度相同
4. **Google Colab 免费 T4 GPU 可实践**

## 常见面试题

### Q1: LoRA 的原理是什么？

**难度**：⭐⭐⭐ | **频率**：🔥🔥🔥

**标准答案**：LoRA 通过低秩分解近似权重更新：W' = W + B×A，其中 W 冻结，只训练低秩矩阵 A(d_in, r) 和 B(r, d_out)。当 r 远小于 d_in 和 d_out 时，可训练参数量大幅减少（通常 <0.1%），效果接近全参数微调。推理时可将 LoRA 权重合并回原始权重，无额外开销。

### Q2: QLoRA 相比 LoRA 的改进？

**难度**：⭐⭐⭐ | **频率**：🔥🔥🔥

**标准答案**：QLoRA 三大创新：(1) NF4 量化基座模型到 4-bit，显存减少 4x。(2) 双重量化，对量化常数再量化。(3) 分页优化器，显存不足时卸载到 CPU。效果：7B 模型微调只需 ~6GB 显存，效果与全参数微调相当。

### Q3: LoRA 的秩 r 如何选择？

**难度**：⭐⭐ | **频率**：🔥🔥

**标准答案**：r 越大，表达能力越强但参数越多。通用场景 r=8-16 即可，复杂任务（代码、数学）用 r=16-64。alpha 通常设为 2×r。目标模块至少包含 q_proj 和 v_proj。

## 推荐工具

| 工具 | 用途 | 详情 |
|------|------|------|
| PEFT | Hugging Face 官方 LoRA 库 | [GitHub](https://github.com/huggingface/peft) |
| Unsloth | 2x 加速 LoRA 微调 | [GitHub](https://github.com/unslothai/unsloth) |
| Perplexity | 搜索 LoRA 最佳实践 | [AI 搜索](/7-ai-tools/7.1-efficiency/ai-search) |

## 参考资料

- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [QLoRA 论文](https://arxiv.org/abs/2305.14314)
- [PEFT 库文档](https://huggingface.co/docs/peft)
