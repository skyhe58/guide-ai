---
title: "模块 2 速查卡片"
---

# 模块 2：大语言模型 LLM — 速查卡片

## Transformer 核心公式

```python
# 缩放点积注意力
Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V

# 多头注意力
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)

# Decoder Block (Pre-Norm, 现代 LLM)
x = x + CausalSelfAttention(RMSNorm(x))
x = x + SwiGLU_FFN(RMSNorm(x))
```

## 注意力变体

```
MHA:  Q 头=32, K/V 头=32  → 标准多头（GPT-2/3）
GQA:  Q 头=32, K/V 头=8   → 分组查询（LLaMA 2/3, Qwen2）⭐推荐
MQA:  Q 头=32, K/V 头=1   → 多查询（Falcon）
```

## 位置编码

```python
# 正弦编码（原始 Transformer）
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

# RoPE（现代 LLM 标配）
RoPE(x, pos) = x × e^(i·pos·θ)
# Q·K 点积自然包含相对位置信息
```

## LoRA 微调

```python
# LoRA 核心公式
W' = W + B @ A × (alpha / rank)
# W: 冻结原始权重
# A: (d_in, r) 高斯初始化
# B: (r, d_out) 零初始化

# PEFT 配置
from peft import LoraConfig
config = LoraConfig(
    r=16,                              # 秩
    lora_alpha=32,                     # 缩放因子 = 2×r
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
```

## QLoRA 配置

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

## 数据格式

```json
// Alpaca 格式（单轮）
{"instruction": "...", "input": "", "output": "..."}

// ShareGPT 格式（多轮）
{"conversations": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
]}
```

## Ollama 常用命令

```bash
# 安装
curl -fsSL https://ollama.com/install.sh | sh

# 模型管理
ollama pull qwen2:7b          # 下载模型
ollama list                    # 列出模型
ollama run qwen2:7b            # 交互对话
ollama run qwen2:7b "你好"     # 单次提问
ollama serve                   # 启动服务

# Docker GPU
docker run -d --gpus=all -v ollama:/root/.ollama \
    -p 11434:11434 ollama/ollama
```

## vLLM 部署

```bash
# Docker 部署
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2-7B-Instruct

# pip 部署
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-7B-Instruct --port 8000

# 多卡部署
--tensor-parallel-size 4

# 量化部署
--quantization awq
```

## OpenAI 兼容 API

```python
from openai import OpenAI

# Ollama
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# vLLM
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

# TGI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="tgi")

# 调用方式完全相同
response = client.chat.completions.create(
    model="qwen2:7b",
    messages=[{"role": "user", "content": "你好"}],
)
```

## GGUF 转换

```bash
# HF → GGUF
python convert_hf_to_gguf.py ./model --outfile model-f16.gguf --outtype f16

# 量化
./llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M

# 导入 Ollama
# Modelfile: FROM ./model-q4_k_m.gguf
ollama create my-model -f Modelfile
```

## 量化级别速查

```
Q4_K_M → 4.1 GB (7B) → 最佳性价比 ⭐
Q5_K_M → 4.8 GB (7B) → 高质量 ⭐
Q8_0   → 7.2 GB (7B) → 接近无损
F16    → 14 GB  (7B) → 无损基准
```

## 显存估算

```
全参数微调 (FP32): 参数量 × 16 bytes (权重+梯度+优化器)
LoRA (FP16):       参数量 × 2 bytes + LoRA 参数
QLoRA (NF4):       参数量 × 0.5 bytes + LoRA 参数

7B 模型:
  全参数 FP32: ~56 GB
  LoRA FP16:   ~16 GB
  QLoRA NF4:   ~6 GB  ← 消费级 GPU 可用
```

## 模型选型速查

```
中文本地部署 → Qwen2.5-7B
英文本地部署 → LLaMA 3.1 8B
最强 API    → GPT-4o / Claude 3.5 Sonnet
性价比 API  → DeepSeek V2.5
代码生成    → Claude 3.5 Sonnet / DeepSeek Coder
```

## 训练三阶段

```
预训练 (TB 数据, $100K+)
  → 基座模型 (Base)
    → SFT 指令微调 (10K-100K 条, $100-$10K)
      → Chat 模型
        → RLHF/DPO 对齐 (偏好数据, $1K-$100K)
          → 对齐模型 (Aligned)
```
