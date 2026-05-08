---
title: "微调数据准备"
module: "llm"
difficulty: "intermediate"
interviewFrequency: "medium"
tags:
  - "LLM"
  - "数据准备"
  - "Alpaca"
  - "ShareGPT"
codeExample: "02-llm/finetuning/03_data_preparation.py"
relatedEntries:
  - "/2-llm/07-lora-qlora"
  - "/2-llm/10-finetuning-tools"
prerequisites:
  - "/2-llm/04-training-pipeline"
estimatedTime: "35min"
toolReferences:
  - name: "Perplexity"
    usage: "搜索高质量微调数据集"
    link: "/7-ai-tools/7.1-efficiency/ai-search"
---

# 微调数据准备

## 概念说明

微调数据的质量直接决定模型效果。本节介绍主流数据格式（Alpaca/ShareGPT）、数据清洗方法和质量控制策略。

## 核心原理

### 1. Alpaca 格式（单轮指令）

```json
{
    "instruction": "解释什么是机器学习",
    "input": "",
    "output": "机器学习是人工智能的一个分支..."
}
```

适用：单轮指令微调（SFT）

### 2. ShareGPT 格式（多轮对话）

```json
{
    "conversations": [
        {"role": "system", "content": "你是一个助手"},
        {"role": "user", "content": "什么是 LoRA？"},
        {"role": "assistant", "content": "LoRA 是..."},
        {"role": "user", "content": "它的优势是什么？"},
        {"role": "assistant", "content": "主要优势包括..."}
    ]
}
```

适用：多轮对话微调

### 3. 数据质量 > 数据数量

| 数据量 | 效果 | 建议 |
|--------|------|------|
| 100 条 | 有限 | 快速验证 |
| 1,000 条 | 明显 | 推荐起步量 |
| 5,000 条 | 良好 | 大多数场景足够 |
| 10,000+ 条 | 优秀 | 追求最佳效果 |

### 4. 数据清洗流程

```
原始数据 → 去空值 → 去重 → 长度过滤 → 质量评分 → 格式转换 → 训练数据
```

关键步骤：
- **去重**：完全重复 + 语义相似去重
- **长度过滤**：太短无信息，太长浪费资源
- **质量评分**：用 GPT-4 评分或人工抽检
- **多样性**：确保指令类型多样，避免模式单一

### 5. 数据来源

| 来源 | 质量 | 成本 | 适用 |
|------|------|------|------|
| 人工标注 | 最高 | 高 | 关键领域 |
| GPT-4 生成 | 高 | 中 | 快速构建 |
| 开源数据集 | 中等 | 免费 | 通用场景 |
| 爬虫清洗 | 低 | 低 | 预训练 |

## 代码示例

> 💻 数据准备脚本：[code-examples/02-llm/finetuning/03_data_preparation.py](https://github.com/skyhe58/guide-ai/tree/main/code-examples/02-llm/finetuning/03_data_preparation.py)

## 实战要点

1. **100 条高质量数据 > 10000 条低质量数据**
2. **用 GPT-4 生成种子数据，再人工校验**
3. **确保指令多样性**：避免模型只学会一种回答模式

## 常见面试题

### Q1: Alpaca 和 ShareGPT 格式的区别？

**难度**：⭐⭐ | **频率**：🔥🔥

**标准答案**：Alpaca 格式包含 instruction/input/output 三个字段，适合单轮指令微调。ShareGPT 格式包含 conversations 数组，支持多轮对话，适合对话模型微调。选择取决于目标任务：单轮用 Alpaca，多轮用 ShareGPT。

## 推荐工具

| 工具 | 用途 | 详情 |
|------|------|------|
| Argilla | 数据标注平台 | [GitHub](https://github.com/argilla-io/argilla) |
| Perplexity | 搜索开源数据集 | [AI 搜索](/7-ai-tools/7.1-efficiency/ai-search) |

## 参考资料

- [Alpaca 数据集](https://github.com/tatsu-lab/stanford_alpaca)
- [LIMA 论文（少量高质量数据）](https://arxiv.org/abs/2305.11206)
- [Hugging Face Datasets](https://huggingface.co/datasets)
