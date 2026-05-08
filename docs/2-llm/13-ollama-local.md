---
title: "Ollama 本地部署"
module: "llm"
difficulty: "beginner"
interviewFrequency: "medium"
tags:
  - "LLM"
  - "Ollama"
  - "本地部署"
codeExample: "02-llm/deployment/02_ollama_api.py"
relatedEntries:
  - "/2-llm/12-vllm-deployment"
  - "/2-llm/14-tgi-deployment"
  - "/2-llm/11-quantization-gguf"
prerequisites:
  - "/2-llm/11-quantization-gguf"
estimatedTime: "30min"
toolReferences:
  - name: "Perplexity"
    usage: "搜索 Ollama 最新模型和配置"
    link: "/7-ai-tools/7.1-efficiency/ai-search"
---

# Ollama 本地部署

## 概念说明

Ollama 是最简单的本地 LLM 部署方案，一行命令即可运行大语言模型。支持 macOS、Linux、Windows，支持 CPU 和 GPU 推理。

## 核心原理

### 1. 安装与使用

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# 拉取并运行模型
ollama pull qwen2:7b
ollama run qwen2:7b
```

### 2. 推荐模型

| 模型 | 大小 | 最低内存 | 适用 |
|------|------|----------|------|
| qwen2:0.5b | 0.4 GB | 1 GB | 测试 |
| qwen2:7b | 4.4 GB | 8 GB | **推荐** |
| llama3.1:8b | 4.7 GB | 8 GB | 英文 |
| deepseek-coder-v2 | 8.9 GB | 16 GB | 代码 |
| qwen2:72b | 41 GB | 48 GB | 最强 |

### 3. API 调用

Ollama 提供 REST API 和 OpenAI 兼容 API：

```python
# REST API
import requests
resp = requests.post("http://localhost:11434/api/chat", json={
    "model": "qwen2:7b",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": False,
})

# OpenAI 兼容 API
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
```

### 4. Modelfile 自定义

```
FROM qwen2:7b
SYSTEM "你是一个 Python 编程助手"
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
```

```bash
ollama create my-assistant -f Modelfile
ollama run my-assistant
```

### 5. GPU 加速

```bash
# Docker GPU 模式
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 ollama/ollama

# 环境变量配置
OLLAMA_NUM_GPU=1 ollama serve        # 指定 GPU 数量
OLLAMA_HOST=0.0.0.0 ollama serve     # 允许远程访问
```

## 代码示例

> 💻 Ollama API：[code-examples/02-llm/deployment/02_ollama_api.py](https://github.com/skyhe58/guide-ai/tree/main/code-examples/02-llm/deployment/02_ollama_api.py)

## 实战要点

1. **入门首选 Ollama**：最简单的本地 LLM 方案
2. **推荐 qwen2:7b**：中文效果最好的 7B 开源模型
3. **OpenAI 兼容 API**：开发时用 Ollama，生产时切换 vLLM

## 常见面试题

### Q1: Ollama 和 vLLM 的区别？

**难度**：⭐⭐ | **频率**：🔥🔥

**标准答案**：Ollama 面向个人开发者，安装简单，支持 CPU，适合本地开发和测试。vLLM 面向生产环境，性能更高（PagedAttention），但需要 GPU。开发时用 Ollama，生产时用 vLLM。

## 推荐工具

| 工具 | 用途 | 链接 |
|------|------|------|
| Ollama | 本地 LLM 部署 | [官网](https://ollama.com) |
| Open WebUI | Ollama 的 Web 界面 | [GitHub](https://github.com/open-webui/open-webui) |

## 参考资料

- [Ollama 官方文档](https://github.com/ollama/ollama)
- [Ollama 模型库](https://ollama.com/library)
- [Ollama API 文档](https://github.com/ollama/ollama/blob/main/docs/api.md)
