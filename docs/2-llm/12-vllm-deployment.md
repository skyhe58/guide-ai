---
title: "vLLM 推理加速"
module: "llm"
difficulty: "intermediate"
interviewFrequency: "high"
tags:
  - "LLM"
  - "vLLM"
  - "PagedAttention"
  - "推理部署"
  - "面试高频"
codeExample: "02-llm/deployment/01_vllm_serve.py"
relatedEntries:
  - "/2-llm/13-ollama-local"
  - "/2-llm/14-tgi-deployment"
  - "/2-llm/02-attention-mechanism"
prerequisites:
  - "/2-llm/02-attention-mechanism"
estimatedTime: "40min"
toolReferences:
  - name: "Perplexity"
    usage: "搜索 vLLM 最新版本和配置"
    link: "/7-ai-tools/7.1-efficiency/ai-search"
---

# vLLM 推理加速

## 概念说明

vLLM 是当前最快的开源 LLM 推理引擎，核心技术是 PagedAttention。它提供 OpenAI 兼容 API，可无缝替换 OpenAI SDK。

## 核心原理

### 1. PagedAttention

借鉴操作系统虚拟内存的分页机制：

- **问题**：KV Cache 显存碎片化，利用率低（传统方案 ~50%）
- **方案**：将 KV Cache 分成固定大小的块（pages），按需分配
- **效果**：显存利用率提升到 ~95%，吞吐量提升 2-4x

```
传统方案: [请求1 KV Cache][空闲][请求2 KV Cache][空闲][空闲]
                          ↑ 碎片化，浪费显存

PagedAttention: [Page1][Page2][Page3][Page4][Page5]
                 请求1   请求1  请求2  请求2  请求1
                          ↑ 非连续存储，按需分配
```

### 2. Continuous Batching

传统 batching 等所有请求完成才处理下一批，Continuous Batching 动态调度：

- 请求完成立即释放资源
- 新请求立即加入 batch
- 吞吐量提升 10-24x

### 3. Tensor Parallelism

多卡并行推理大模型：

```bash
# 4 卡并行推理 70B 模型
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-72B-Instruct \
    --tensor-parallel-size 4
```

### 4. OpenAI 兼容 API

```python
from openai import OpenAI

# 只需修改 base_url，代码完全兼容
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="Qwen/Qwen2-7B-Instruct",
    messages=[{"role": "user", "content": "你好"}],
)
```

## 代码示例

> 💻 vLLM 服务：[code-examples/02-llm/deployment/01_vllm_serve.py](https://github.com/your-repo/tree/main/code-examples/02-llm/deployment/01_vllm_serve.py)

## 实战要点

1. **生产环境首选 vLLM**：性能最好，API 兼容
2. **Docker 部署最简单**：一行命令启动
3. **支持 AWQ/GPTQ 量化**：减少显存需求

## 常见面试题

### Q1: vLLM 的 PagedAttention 原理？

**难度**：⭐⭐⭐⭐ | **频率**：🔥🔥🔥

**标准答案**：PagedAttention 借鉴操作系统虚拟内存分页，将 KV Cache 分成固定大小的块（pages），非连续存储，按需分配。解决了传统方案 KV Cache 碎片化问题，显存利用率从 ~50% 提升到 ~95%，吞吐量提升 2-4x。

### Q2: Continuous Batching 的优势？

**难度**：⭐⭐⭐ | **频率**：🔥🔥

**标准答案**：传统 static batching 等所有请求完成才处理下一批，短请求被长请求阻塞。Continuous Batching 动态调度：请求完成立即释放，新请求立即加入，消除等待时间，吞吐量提升 10-24x。

## 推荐工具

| 工具 | 用途 | 链接 |
|------|------|------|
| vLLM | 高性能推理引擎 | [GitHub](https://github.com/vllm-project/vllm) |

## 参考资料

- [vLLM 论文](https://arxiv.org/abs/2309.06180)
- [PagedAttention 博客](https://blog.vllm.ai/2023/06/20/vllm.html)
- [vLLM 文档](https://docs.vllm.ai/)
