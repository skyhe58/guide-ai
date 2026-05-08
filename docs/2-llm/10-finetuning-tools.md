---
title: "微调工具"
module: "llm"
difficulty: "intermediate"
interviewFrequency: "medium"
tags:
  - "LLM"
  - "PEFT"
  - "Unsloth"
  - "LLaMA-Factory"
codeExample: "02-llm/finetuning/04_unsloth_finetune.py"
relatedEntries:
  - "/2-llm/07-lora-qlora"
  - "/2-llm/09-data-preparation"
prerequisites:
  - "/2-llm/07-lora-qlora"
estimatedTime: "35min"
toolReferences:
  - name: "Perplexity"
    usage: "搜索微调工具最新版本和教程"
    link: "/7-ai-tools/7.1-efficiency/ai-search"
---

# 微调工具

## 概念说明

LLM 微调生态已经非常成熟，从代码级别的 PEFT 库到零代码的 LLaMA-Factory，选择合适的工具可以大幅提升效率。

## 核心原理

### 工具对比

| 工具 | 类型 | 速度 | 易用性 | 适用场景 |
|------|------|------|--------|----------|
| **PEFT** | Python 库 | 1x | ⭐⭐⭐ | 代码级控制 |
| **Unsloth** | Python 库 | 2-5x | ⭐⭐⭐ | 加速微调 |
| **LLaMA-Factory** | WebUI | 1x | ⭐⭐⭐⭐⭐ | 零代码微调 |
| **Axolotl** | 配置文件 | 1x | ⭐⭐⭐⭐ | 灵活配置 |
| **TRL** | Python 库 | 1x | ⭐⭐⭐ | RLHF/DPO |

### 1. PEFT（Hugging Face 官方）

Hugging Face 官方参数高效微调库，支持 LoRA、QLoRA、Prefix Tuning 等。

```python
from peft import LoraConfig, get_peft_model
config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, config)
```

优势：官方维护，文档完善，与 transformers 深度集成。

### 2. Unsloth（加速微调）

手写 Triton 内核，2x 速度 + 60% 显存节省。

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("unsloth/Qwen2-7B-bnb-4bit")
model = FastLanguageModel.get_peft_model(model, r=16)
```

优势：速度快，显存省，支持一键导出 GGUF。

### 3. LLaMA-Factory（零代码微调）

WebUI 界面，支持 100+ 模型，零代码完成微调。

```bash
# 安装
git clone https://github.com/hiyouga/LLaMA-Factory
pip install -e ".[torch,metrics]"

# 启动 WebUI
llamafactory-cli webui
```

优势：零代码，支持模型多，内置数据集管理。

### 4. 推荐选择

```
新手入门 → LLaMA-Factory（WebUI，零代码）
快速实验 → Unsloth（速度快，显存省）
生产环境 → PEFT + TRL（代码级控制）
RLHF/DPO → TRL（Hugging Face 官方）
```

## 代码示例

> 💻 Unsloth 示例：[code-examples/02-llm/finetuning/04_unsloth_finetune.py](https://github.com/skyhe58/guide-ai/tree/main/code-examples/02-llm/finetuning/04_unsloth_finetune.py)

## 实战要点

1. **新手推荐 LLaMA-Factory**：WebUI 零代码，5 分钟开始微调
2. **追求速度用 Unsloth**：2x 加速，支持导出 GGUF
3. **生产环境用 PEFT + TRL**：代码级控制，灵活定制

## 常见面试题

### Q1: 常用的 LLM 微调工具有哪些？

**难度**：⭐⭐ | **频率**：🔥🔥

**标准答案**：PEFT（Hugging Face 官方 LoRA 库）、Unsloth（2x 加速）、LLaMA-Factory（WebUI 零代码）、TRL（RLHF/DPO 训练）、Axolotl（配置文件驱动）。推荐：新手用 LLaMA-Factory，追求速度用 Unsloth，生产环境用 PEFT。

## 推荐工具

| 工具 | 用途 | 链接 |
|------|------|------|
| PEFT | 官方 LoRA 库 | [GitHub](https://github.com/huggingface/peft) |
| Unsloth | 加速微调 | [GitHub](https://github.com/unslothai/unsloth) |
| LLaMA-Factory | WebUI 微调 | [GitHub](https://github.com/hiyouga/LLaMA-Factory) |
| TRL | RLHF/DPO | [GitHub](https://github.com/huggingface/trl) |

## 参考资料

- [PEFT 文档](https://huggingface.co/docs/peft)
- [Unsloth Wiki](https://github.com/unslothai/unsloth/wiki)
- [LLaMA-Factory 文档](https://github.com/hiyouga/LLaMA-Factory)
