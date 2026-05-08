---
title: "模块 2：大语言模型 LLM"
---

# 模块 2：大语言模型 LLM

> 本模块覆盖 LLM 原理、主流模型对比、微调技术、量化部署和 Tokenizer。预计学习时间：**3-4 周**。

## 模块概述

从 Transformer 架构深入到 LLM 的完整技术栈：理解原理 → 对比模型 → 微调定制 → 量化部署。

## 知识点导航

### LLM 原理

| 序号 | 知识点 | 难度 | 面试频率 | 代码示例 |
|:----:|--------|:----:|:--------:|:--------:|
| 01 | [Transformer 架构详解](/2-llm/01-transformer-deep-dive) | ⭐⭐⭐ | 🔥🔥🔥 | [代码](https://github.com/skyhe58/guide-ai/tree/main/code-examples/02-llm/transformer/) |
| 02 | [注意力机制深入](/2-llm/02-attention-mechanism) | ⭐⭐⭐ | 🔥🔥🔥 | [代码](https://github.com/skyhe58/guide-ai/tree/main/code-examples/02-llm/transformer/01_attention_mechanism.py) |
| 03 | [位置编码](/2-llm/03-position-encoding) | ⭐⭐⭐ | 🔥🔥🔥 | [代码](https://github.com/skyhe58/guide-ai/tree/main/code-examples/02-llm/transformer/02_position_encoding.py) |
| 04 | [训练流程](/2-llm/04-training-pipeline) | ⭐⭐⭐ | 🔥🔥🔥 | — |
| 05 | [Scaling Laws](/2-llm/05-scaling-laws) | ⭐⭐⭐ | 🔥🔥 | — |
| 06 | [主流模型对比](/2-llm/06-model-comparison) | ⭐⭐ | 🔥🔥🔥 | [代码](https://github.com/skyhe58/guide-ai/tree/main/code-examples/02-llm/milestone_projects/model_comparison/) |

### 微调技术

| 序号 | 知识点 | 难度 | 面试频率 | 代码示例 |
|:----:|--------|:----:|:--------:|:--------:|
| 07 | [LoRA/QLoRA 微调](/2-llm/07-lora-qlora) | ⭐⭐⭐ | 🔥🔥🔥 | [代码](https://github.com/skyhe58/guide-ai/tree/main/code-examples/02-llm/finetuning/) |
| 08 | [全参数微调](/2-llm/08-full-finetuning) | ⭐⭐⭐ | 🔥🔥 | — |
| 09 | [微调数据准备](/2-llm/09-data-preparation) | ⭐⭐ | 🔥🔥 | [代码](https://github.com/skyhe58/guide-ai/tree/main/code-examples/02-llm/finetuning/03_data_preparation.py) |
| 10 | [微调工具](/2-llm/10-finetuning-tools) | ⭐⭐ | 🔥🔥 | [代码](https://github.com/skyhe58/guide-ai/tree/main/code-examples/02-llm/finetuning/04_unsloth_finetune.py) |

### 量化与部署

| 序号 | 知识点 | 难度 | 面试频率 | 代码示例 |
|:----:|--------|:----:|:--------:|:--------:|
| 11 | [量化与 GGUF](/2-llm/11-quantization-gguf) | ⭐⭐ | 🔥🔥 | [代码](https://github.com/skyhe58/guide-ai/tree/main/code-examples/02-llm/deployment/03_gguf_convert.py) |
| 12 | [vLLM 推理加速](/2-llm/12-vllm-deployment) | ⭐⭐⭐ | 🔥🔥🔥 | [代码](https://github.com/skyhe58/guide-ai/tree/main/code-examples/02-llm/deployment/01_vllm_serve.py) |
| 13 | [Ollama 本地部署](/2-llm/13-ollama-local) | ⭐ | 🔥🔥 | [代码](https://github.com/skyhe58/guide-ai/tree/main/code-examples/02-llm/deployment/02_ollama_api.py) |
| 14 | [TGI 推理服务](/2-llm/14-tgi-deployment) | ⭐⭐ | 🔥 | — |

### Tokenizer

| 序号 | 知识点 | 难度 | 面试频率 | 代码示例 |
|:----:|--------|:----:|:--------:|:--------:|
| 15 | [Tokenizer](/2-llm/15-tokenizer) | ⭐⭐ | 🔥🔥 | [代码](https://github.com/skyhe58/guide-ai/tree/main/code-examples/02-llm/tokenizer/) |

## 里程碑项目

完成所有知识点后，通过里程碑项目串联所学：

### 项目 1：多模型对比评测
本地部署 Qwen2-7B，编写多模型对比评测脚本，对比延迟、吞吐量和回答质量。

→ [代码](https://github.com/skyhe58/guide-ai/tree/main/code-examples/02-llm/milestone_projects/model_comparison/)

### 项目 2：领域微调 + 部署
使用 LoRA/QLoRA 微调 Qwen2-7B，导出 GGUF，部署为 API 服务。

→ [代码](https://github.com/skyhe58/guide-ai/tree/main/code-examples/02-llm/milestone_projects/domain_finetune/)

## 快速链接

- 📋 [面试指南](/2-llm/interview) — Transformer 注意力、LoRA 原理、KV Cache、vLLM PagedAttention 等高频题
- 📝 [速查卡片](/2-llm/cheatsheet) — 核心概念和常用命令速查

## 推荐学习资源

| 资源 | 类型 | 说明 |
|------|------|------|
| [Hugging Face LLM Course](https://huggingface.co/learn/llm-course) | 在线课程 | 官方 LLM 课程，免费 |
| [fast.ai Practical Deep Learning](https://course.fast.ai/) | 在线课程 | 实践导向，Part 2 涵盖 LLM |
| [Ollama 官方文档](https://github.com/ollama/ollama) | 文档 | 本地部署入门 |
| [vLLM 文档](https://docs.vllm.ai/) | 文档 | 生产部署参考 |
| [Lilian Weng 博客](https://lilianweng.github.io/) | 博客 | 深度技术解析 |

## 下一步

完成本模块后，进入 → [模块 3：AI 应用开发](/3-ai-apps/)
