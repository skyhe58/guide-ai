---
title: "TGI 推理服务"
module: "llm"
difficulty: "intermediate"
interviewFrequency: "low"
tags:
  - "LLM"
  - "TGI"
  - "Hugging Face"
  - "推理部署"
codeExample: "02-llm/deployment/"
relatedEntries:
  - "/2-llm/12-vllm-deployment"
  - "/2-llm/13-ollama-local"
prerequisites:
  - "/2-llm/12-vllm-deployment"
estimatedTime: "30min"
toolReferences:
  - name: "Perplexity"
    usage: "搜索 TGI 配置和部署方案"
    link: "/7-ai-tools/7.1-efficiency/ai-search"
---

# TGI 推理服务

## 概念说明

Text Generation Inference（TGI）是 Hugging Face 官方的 LLM 推理服务，基于 Rust 实现，支持 Flash Attention、Continuous Batching 等优化。

## 核心原理

### 1. TGI 特点

- **Hugging Face 官方**：与 Hub 深度集成
- **Rust 实现**：高性能，低延迟
- **Docker 部署**：一行命令启动
- **OpenAI 兼容**：支持 Messages API

### 2. Docker 部署

```bash
# 基础部署
docker run --gpus all --shm-size 1g -p 8080:80 \\
    -v $PWD/data:/data \\
    ghcr.io/huggingface/text-generation-inference:latest \\
    --model-id Qwen/Qwen2-7B-Instruct \\
    --max-input-length 4096 \\
    --max-total-tokens 8192

# 量化部署
docker run --gpus all --shm-size 1g -p 8080:80 \\
    -v $PWD/data:/data \\
    ghcr.io/huggingface/text-generation-inference:latest \\
    --model-id Qwen/Qwen2-7B-Instruct \\
    --quantize awq
```

### 3. API 调用

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="tgi")
response = client.chat.completions.create(
    model="Qwen/Qwen2-7B-Instruct",
    messages=[{"role": "user", "content": "你好"}],
)
```

### 4. 部署方案对比

| 维度 | vLLM | Ollama | TGI |
|------|------|--------|-----|
| 性能 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 易用性 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| CPU 支持 | ❌ | ✅ | ❌ |
| 量化支持 | AWQ/GPTQ | GGUF | AWQ/GPTQ |
| OpenAI API | ✅ | ✅ | ✅ |
| 适用场景 | 生产（最快） | 开发/本地 | 生产（HF 生态） |

**推荐选择**：
- **开发测试**：Ollama（最简单）
- **生产部署**：vLLM（最快）或 TGI（HF 生态）
- **本地 CPU**：Ollama（唯一选择）

## 代码示例

> 💻 部署代码：[code-examples/02-llm/deployment/](https://github.com/skyhe58/guide-ai/tree/main/code-examples/02-llm/deployment/)

## 实战要点

1. **vLLM 性能最好**，是生产环境首选
2. **TGI 与 Hugging Face 生态集成最好**
3. **三者都支持 OpenAI 兼容 API**，切换成本低

## 常见面试题

### Q1: vLLM、Ollama、TGI 如何选择？

**难度**：⭐⭐⭐ | **频率**：🔥🔥

**标准答案**：Ollama 最简单，适合开发测试和本地 CPU 推理。vLLM 性能最好（PagedAttention），适合生产 GPU 推理。TGI 是 Hugging Face 官方方案，与 Hub 集成好。三者都支持 OpenAI 兼容 API，切换成本低。

## 推荐工具

| 工具 | 用途 | 链接 |
|------|------|------|
| TGI | HF 官方推理服务 | [GitHub](https://github.com/huggingface/text-generation-inference) |

## 参考资料

- [TGI 官方文档](https://huggingface.co/docs/text-generation-inference)
- [TGI GitHub](https://github.com/huggingface/text-generation-inference)
