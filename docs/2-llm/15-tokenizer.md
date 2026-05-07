---
title: "Tokenizer"
module: "llm"
difficulty: "intermediate"
interviewFrequency: "medium"
tags:
  - "LLM"
  - "Tokenizer"
  - "BPE"
  - "SentencePiece"
codeExample: "02-llm/tokenizer/"
relatedEntries:
  - "/2-llm/01-transformer-deep-dive"
prerequisites:
  - "/2-llm/01-transformer-deep-dive"
estimatedTime: "35min"
toolReferences:
  - name: "Perplexity"
    usage: "搜索分词器对比和最新进展"
    link: "/7-ai-tools/7.1-efficiency/ai-search"
---

# Tokenizer

## 概念说明

Tokenizer（分词器）将文本转换为 Token ID 序列，是 LLM 输入输出的桥梁。分词质量直接影响模型性能和推理成本。

## 核心原理

### 1. BPE 算法（Byte Pair Encoding）

最主流的分词算法，GPT 系列、Qwen、DeepSeek 均使用。

步骤：
1. 将所有单词拆分为字符序列
2. 统计所有相邻字符对的频率
3. 合并频率最高的字符对为新 Token
4. 重复 2-3，直到达到目标词表大小

```
训练语料: ["hello", "help", "held"]
Step 0: h e l l o, h e l p, h e l d
Step 1: 合并 (h,e) → he l l o, he l p, he l d
Step 2: 合并 (he,l) → hel l o, hel p, hel d
```

### 2. SentencePiece

语言无关的分词器，直接在原始文本上训练（不需要预分词）。

- LLaMA、T5 使用
- 支持 BPE 和 Unigram 两种算法
- 对中文、日文等非空格分隔语言友好

### 3. 主流模型分词器

| 模型 | 算法 | 词表大小 | 中文效率 |
|------|------|----------|----------|
| GPT-4 | BPE (tiktoken) | ~100K | 一般 |
| LLaMA 3 | BPE (tiktoken) | 128K | 较好 |
| Qwen2 | BPE | 152K | 优秀 |
| DeepSeek V2 | BPE | 100K | 优秀 |
| BERT | WordPiece | 30K | 差 |

### 4. Token 计数经验

- 英文：~1 Token ≈ 4 字符 ≈ 0.75 单词
- 中文：~1 Token ≈ 1-2 汉字
- 代码：Token 数通常比自然语言多 50-100%

### 5. Chat 模板

对话模型使用特殊模板格式化输入：

```python
# Qwen2 Chat 模板
tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": "你好"},
    ],
    tokenize=False,
    add_generation_prompt=True,
)
# → <|im_start|>system\n你是助手<|im_end|>\n<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n
```

## 代码示例

> 💻 BPE 实现：[code-examples/02-llm/tokenizer/01_bpe_basics.py](https://github.com/your-repo/tree/main/code-examples/02-llm/tokenizer/01_bpe_basics.py)
> 💻 HF Tokenizers：[code-examples/02-llm/tokenizer/02_hf_tokenizers.py](https://github.com/your-repo/tree/main/code-examples/02-llm/tokenizer/02_hf_tokenizers.py)

## 实战要点

1. **中文场景选 Qwen2 分词器**：中文 Token 效率最高
2. **Token 数影响成本和速度**：精简 prompt 可省钱
3. **Chat 模板很重要**：错误的模板会导致模型输出异常

## 常见面试题

### Q1: BPE 算法的原理？

**难度**：⭐⭐⭐ | **频率**：🔥🔥

**标准答案**：BPE 从字符级别开始，迭代合并频率最高的相邻 Token 对，直到达到目标词表大小。它是子词分词方法，平衡了字符级（词表小但序列长）和词级（词表大但 OOV 多）的优缺点。GPT 系列使用 Byte-level BPE。

### Q2: 为什么不同模型的 Token 数不同？

**难度**：⭐⭐ | **频率**：🔥🔥

**标准答案**：不同模型使用不同的分词器和词表。词表越大、训练语料中某语言占比越高，该语言的 Token 效率越高。例如 Qwen2 词表 152K 且中文训练数据多，中文 Token 效率优于 GPT-4。

## 推荐工具

| 工具 | 用途 | 链接 |
|------|------|------|
| tiktoken | OpenAI 分词器 | [GitHub](https://github.com/openai/tiktoken) |
| tokenizers | HF 高性能分词器 | [GitHub](https://github.com/huggingface/tokenizers) |

## 参考资料

- [Hugging Face Tokenizers 文档](https://huggingface.co/docs/tokenizers)
- [BPE 原始论文](https://arxiv.org/abs/1508.07909)
- [SentencePiece](https://github.com/google/sentencepiece)
