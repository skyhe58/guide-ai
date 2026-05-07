---
title: "全参数微调"
module: "llm"
difficulty: "advanced"
interviewFrequency: "medium"
tags:
  - "LLM"
  - "全参数微调"
  - "Full Fine-tuning"
codeExample: "02-llm/finetuning/"
relatedEntries:
  - "/2-llm/07-lora-qlora"
  - "/2-llm/04-training-pipeline"
prerequisites:
  - "/2-llm/07-lora-qlora"
estimatedTime: "30min"
toolReferences:
  - name: "Perplexity"
    usage: "搜索全参数微调最佳实践"
    link: "/7-ai-tools/7.1-efficiency/ai-search"
---

# 全参数微调

## 概念说明

全参数微调（Full Fine-tuning）更新模型的所有参数，是效果最好但资源需求最高的微调方式。适用于数据充足、需要大幅改变模型行为的场景。

## 核心原理

### 1. 全参数微调 vs LoRA

| 维度 | 全参数微调 | LoRA/QLoRA |
|------|-----------|------------|
| 可训练参数 | 100% | 0.01-1% |
| 显存需求（7B） | ~56 GB (FP32) / ~28 GB (FP16) | ~6 GB (QLoRA) |
| 训练速度 | 慢 | 快 2-5x |
| 效果上限 | 最高 | 接近全参数 |
| 灾难性遗忘 | 风险较高 | 风险较低 |
| 适用场景 | 大幅改变模型行为 | 领域适配、指令微调 |

### 2. 适用场景

全参数微调适合：
- **领域差异大**：医疗、法律等专业领域
- **数据充足**：10K+ 高质量样本
- **资源充足**：多卡 A100/H100
- **需要最佳效果**：不接受任何质量损失

LoRA 更适合：
- **快速实验**：几小时完成微调
- **资源有限**：单卡消费级 GPU
- **多任务适配**：同一基座模型 + 多个 LoRA 适配器

### 3. 显存优化技术

| 技术 | 显存节省 | 说明 |
|------|----------|------|
| 混合精度（FP16/BF16） | 50% | 前向 FP16，梯度 FP32 |
| 梯度检查点 | 30-50% | 用计算换显存 |
| DeepSpeed ZeRO | 60-90% | 分布式优化器状态 |
| FSDP | 60-90% | PyTorch 原生分布式 |
| 梯度累积 | 间接 | 小 batch 模拟大 batch |

### 4. DeepSpeed ZeRO 阶段

```
ZeRO-1: 分片优化器状态    → 显存减少 ~4x
ZeRO-2: + 分片梯度        → 显存减少 ~8x
ZeRO-3: + 分片模型参数    → 显存减少 ~N x（N = GPU 数）
ZeRO-Offload: 卸载到 CPU  → 进一步减少 GPU 显存
```

## 代码示例

> 💻 微调代码：[code-examples/02-llm/finetuning/](https://github.com/your-repo/tree/main/code-examples/02-llm/finetuning/)

## 实战要点

1. **大多数场景 LoRA/QLoRA 足够**：全参数微调是"杀鸡用牛刀"
2. **DeepSpeed ZeRO-3 是多卡全参数微调的标配**
3. **注意灾难性遗忘**：全参数微调可能丢失预训练知识

## 常见面试题

### Q1: 全参数微调和 LoRA 如何选择？

**难度**：⭐⭐⭐ | **频率**：🔥🔥

**标准答案**：大多数场景推荐 LoRA/QLoRA：参数效率高、显存需求低、效果接近全参数。全参数微调适合：领域差异大、数据充足、资源充足、追求最佳效果。实际工程中 90%+ 的微调使用 LoRA。

### Q2: DeepSpeed ZeRO 的三个阶段？

**难度**：⭐⭐⭐ | **频率**：🔥🔥

**标准答案**：ZeRO-1 分片优化器状态，ZeRO-2 额外分片梯度，ZeRO-3 额外分片模型参数。每个阶段进一步减少单卡显存占用，代价是增加通信开销。ZeRO-3 + Offload 可以在消费级 GPU 上训练大模型。

## 推荐工具

| 工具 | 用途 | 详情 |
|------|------|------|
| DeepSpeed | 分布式训练框架 | [GitHub](https://github.com/microsoft/DeepSpeed) |
| FSDP | PyTorch 原生分布式 | [文档](https://pytorch.org/docs/stable/fsdp.html) |

## 参考资料

- [DeepSpeed ZeRO 论文](https://arxiv.org/abs/1910.02054)
- [PyTorch FSDP 教程](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
