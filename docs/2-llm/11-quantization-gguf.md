---
title: "量化与 GGUF 格式"
module: "llm"
difficulty: "intermediate"
interviewFrequency: "medium"
tags:
  - "LLM"
  - "量化"
  - "GGUF"
  - "llama.cpp"
codeExample: "02-llm/deployment/03_gguf_convert.py"
relatedEntries:
  - "/2-llm/12-vllm-deployment"
  - "/2-llm/13-ollama-local"
prerequisites:
  - "/2-llm/07-lora-qlora"
estimatedTime: "35min"
toolReferences:
  - name: "Perplexity"
    usage: "搜索量化方法对比和最新进展"
    link: "/7-ai-tools/7.1-efficiency/ai-search"
---

# 量化与 GGUF 格式

## 概念说明

量化（Quantization）将模型权重从高精度（FP32/FP16）转换为低精度（INT8/INT4），大幅减少模型大小和显存需求。GGUF 是 llama.cpp 的统一模型格式，是本地部署的标准。

## 核心原理

### 1. 量化级别对比（7B 模型）

| 量化级别 | 位数 | 大小 | 质量 | 推荐 |
|----------|------|------|------|------|
| Q2_K | 2.5-bit | 2.7 GB | 低 | ❌ |
| Q3_K_M | 3.5-bit | 3.3 GB | 中低 | ❌ |
| **Q4_K_M** | **4.5-bit** | **4.1 GB** | **中高** | **⭐推荐** |
| **Q5_K_M** | **5.5-bit** | **4.8 GB** | **高** | **⭐推荐** |
| Q6_K | 6.5-bit | 5.5 GB | 很高 | 可选 |
| Q8_0 | 8-bit | 7.2 GB | 极高 | 可选 |
| F16 | 16-bit | 14 GB | 无损 | 基准 |

### 2. 量化方法

| 方法 | 工具 | 适用 | 特点 |
|------|------|------|------|
| GGUF | llama.cpp | Ollama/本地 | 通用，CPU+GPU |
| AWQ | AutoAWQ | vLLM | 激活感知，GPU |
| GPTQ | AutoGPTQ | vLLM | 经典方法，GPU |
| bitsandbytes | bitsandbytes | 训练 | QLoRA 专用 |

### 3. GGUF 格式

GGUF（GPT-Generated Unified Format）特点：
- **单文件**：权重 + 元数据 + 分词器
- **跨平台**：CPU/GPU/Metal/CUDA
- **生态丰富**：Ollama、llama.cpp、LM Studio

### 4. 转换流程

```
Hugging Face 模型 → convert_hf_to_gguf.py → GGUF (F16)
                                                ↓
                                        llama-quantize
                                                ↓
                                        GGUF (Q4_K_M)
                                                ↓
                                    Ollama / llama.cpp / LM Studio
```

## 代码示例

> 💻 GGUF 转换：[code-examples/02-llm/deployment/03_gguf_convert.py](https://github.com/skyhe58/guide-ai/tree/main/code-examples/02-llm/deployment/03_gguf_convert.py)

## 实战要点

1. **Q4_K_M 是最佳性价比**：质量损失小，大小减少 3.5x
2. **Unsloth 支持一键导出 GGUF**：微调后直接部署
3. **Ollama 直接使用 GGUF 文件**：创建 Modelfile 即可

## 常见面试题

### Q1: 模型量化的原理和作用？

**难度**：⭐⭐⭐ | **频率**：🔥🔥

**标准答案**：量化将模型权重从高精度（FP16）转为低精度（INT4/INT8），减少模型大小和显存需求。例如 7B 模型从 14GB（FP16）压缩到 4GB（Q4_K_M），可在消费级 GPU 上运行。代价是轻微的质量损失，Q4_K_M 级别几乎无感知。

### Q2: GGUF 和 AWQ 的区别？

**难度**：⭐⭐ | **频率**：🔥🔥

**标准答案**：GGUF 是 llama.cpp 的格式，支持 CPU+GPU，用于 Ollama/本地部署。AWQ 是激活感知量化，只支持 GPU，用于 vLLM 高性能推理。选择：本地部署用 GGUF，生产 GPU 推理用 AWQ。

## 推荐工具

| 工具 | 用途 | 链接 |
|------|------|------|
| llama.cpp | GGUF 转换和推理 | [GitHub](https://github.com/ggerganov/llama.cpp) |
| Ollama | 本地部署 GGUF | [官网](https://ollama.com) |

## 参考资料

- [llama.cpp 量化文档](https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md)
- [GGUF 格式规范](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [AWQ 论文](https://arxiv.org/abs/2306.00978)
