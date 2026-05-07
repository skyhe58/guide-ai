---
title: "AI 知识图谱"
description: "AI 知识点关联图谱和常见面试追问路径"
---

# 🧠 AI 知识图谱

> 本页展示 AI 知识点之间的关联关系和面试中常见的追问路径。面试官通常会沿着知识链深入追问，提前了解追问路径有助于准备更充分。

---

## 核心知识图谱

```mermaid
graph TB
    subgraph foundations["基础层"]
        LA["线性代数<br/>向量/矩阵/SVD"]
        PROB["概率统计<br/>贝叶斯/分布/MLE"]
        PY["Python 基础<br/>asyncio/类型注解"]
        NP["NumPy/Pandas<br/>向量化运算"]
    end

    subgraph ml["机器学习层"]
        SL["监督学习<br/>回归/分类"]
        USL["无监督学习<br/>聚类/降维"]
        LOSS["损失函数<br/>MSE/交叉熵"]
        OPT["优化算法<br/>SGD/Adam"]
        EVAL["评估指标<br/>F1/AUC/mAP"]
        REG["正则化<br/>L1/L2/Dropout"]
    end

    subgraph dl["深度学习层"]
        NN["神经网络<br/>MLP/反向传播"]
        CNN["CNN<br/>卷积/池化/ResNet"]
        RNN["RNN/LSTM<br/>序列建模"]
        TF["Transformer<br/>自注意力机制"]
        ATT["注意力机制<br/>Multi-Head/KV Cache"]
        PE["位置编码<br/>RoPE/ALiBi"]
    end

    subgraph llm["大语言模型层"]
        PRE["预训练<br/>Pre-training"]
        SFT["指令微调<br/>SFT/RLHF/DPO"]
        LORA["LoRA/QLoRA<br/>参数高效微调"]
        QUANT["量化<br/>GGUF/INT4/INT8"]
        TOK["Tokenizer<br/>BPE/SentencePiece"]
        SCALE["Scaling Laws<br/>规模定律"]
    end

    subgraph apps["应用层"]
        PE2["Prompt Engineering<br/>CoT/Few-shot"]
        RAG["RAG 架构<br/>检索增强生成"]
        EMB["Embedding<br/>向量嵌入"]
        VDB["向量数据库<br/>Chroma/Pinecone"]
        AGENT["AI Agent<br/>ReAct/Multi-Agent"]
        FC["Function Calling<br/>工具调用"]
        FW["框架<br/>LangChain/LangGraph"]
    end

    subgraph eng["工程层"]
        VLLM["vLLM<br/>PagedAttention"]
        SERVE["模型服务化<br/>API 网关"]
        GPU["GPU 优化<br/>显存/混合精度"]
        MLOPS["MLOps<br/>流水线/实验追踪"]
        MON["监控<br/>Prometheus/Grafana"]
        COST["成本优化<br/>缓存/模型选择"]
    end

    subgraph frontier["前沿层"]
        MCP["MCP 协议<br/>Agent 通信"]
        SEC["AI 安全<br/>Prompt Injection"]
        MULTI["多模态<br/>GPT-4V/LLaVA"]
        VIBE["Vibe Coding<br/>AI 编程"]
    end

    %% 基础 → ML
    LA --> SL
    LA --> USL
    PROB --> SL
    PROB --> LOSS
    NP --> EVAL

    %% ML → DL
    SL --> NN
    LOSS --> NN
    OPT --> NN
    NN --> CNN
    NN --> RNN
    RNN --> TF
    CNN --> TF

    %% DL → LLM
    TF --> ATT
    TF --> PE
    ATT --> PRE
    PRE --> SFT
    PRE --> SCALE
    SFT --> LORA
    LORA --> QUANT
    TF --> TOK

    %% LLM → 应用
    PRE --> PE2
    TOK --> EMB
    EMB --> VDB
    VDB --> RAG
    PE2 --> RAG
    PE2 --> AGENT
    FC --> AGENT
    RAG --> FW
    AGENT --> FW

    %% 应用 → 工程
    ATT --> VLLM
    QUANT --> VLLM
    VLLM --> SERVE
    SERVE --> GPU
    SERVE --> MON
    FW --> MLOPS
    SERVE --> COST

    %% 工程 → 前沿
    AGENT --> MCP
    PE2 --> SEC
    TF --> MULTI
    FW --> VIBE

    style TF fill:#fff3e0,stroke:#f57c00
    style RAG fill:#e3f2fd,stroke:#1976d2
    style AGENT fill:#e8f5e9,stroke:#388e3c
    style VLLM fill:#fce4ec,stroke:#c62828
```

---

## 高频追问路径

面试官通常会沿着以下路径深入追问。每个节点链接到对应模块的面试题。

### 路径 1：Transformer → 推理优化

```mermaid
graph LR
    A["Transformer 架构<br/>⭐⭐⭐"]
    B["注意力机制<br/>⭐⭐⭐"]
    C["KV Cache<br/>⭐⭐⭐⭐"]
    D["PagedAttention<br/>⭐⭐⭐⭐"]
    E["vLLM 部署<br/>⭐⭐⭐⭐"]

    A -->|"Q: 注意力怎么算？"| B
    B -->|"Q: 推理时如何加速？"| C
    C -->|"Q: 内存如何管理？"| D
    D -->|"Q: 生产环境怎么部署？"| E

    style A fill:#fff3e0
    style E fill:#fce4ec
```

- [Transformer 架构](/2-llm/interview#q1) → [注意力机制](/2-llm/interview#q1) → [KV Cache](/2-llm/interview#q5) → [vLLM PagedAttention](/5-ai-engineering/interview#q3)

### 路径 2：RAG → 优化 → 评估

```mermaid
graph LR
    A["RAG 架构<br/>⭐⭐⭐"]
    B["向量数据库选型<br/>⭐⭐⭐"]
    C["检索策略<br/>⭐⭐⭐"]
    D["Rerank 重排序<br/>⭐⭐⭐"]
    E["RAG 评估<br/>⭐⭐⭐"]

    A -->|"Q: 向量库怎么选？"| B
    B -->|"Q: 检索效果不好怎么办？"| C
    C -->|"Q: 如何提升精度？"| D
    D -->|"Q: 如何评估效果？"| E

    style A fill:#e3f2fd
```

- [RAG 架构](/3-ai-apps/interview#q1) → [向量数据库](/3-ai-apps/interview#q2) → [RAG 优化](/3-ai-apps/interview#q4) → [RAG 评估](/3-ai-apps/interview#q9)

### 路径 3：微调 → 量化 → 部署

```mermaid
graph LR
    A["LoRA 原理<br/>⭐⭐⭐"]
    B["QLoRA 量化微调<br/>⭐⭐⭐"]
    C["GGUF 量化<br/>⭐⭐⭐"]
    D["Ollama 部署<br/>⭐⭐"]
    E["vLLM 生产部署<br/>⭐⭐⭐⭐"]

    A -->|"Q: 显存不够怎么办？"| B
    B -->|"Q: 如何导出模型？"| C
    C -->|"Q: 本地怎么跑？"| D
    C -->|"Q: 生产环境呢？"| E

    style A fill:#e8f5e9
```

- [LoRA 原理](/2-llm/interview#q3) → [量化](/2-llm/interview#q7) → [部署方案对比](/2-llm/interview#q10)

### 路径 4：Agent → 安全 → 监控

```mermaid
graph LR
    A["Agent 架构<br/>⭐⭐⭐"]
    B["Function Calling<br/>⭐⭐"]
    C["Multi-Agent<br/>⭐⭐⭐⭐"]
    D["Prompt Injection<br/>⭐⭐⭐"]
    E["生产监控<br/>⭐⭐⭐"]

    A -->|"Q: 工具怎么调用？"| B
    A -->|"Q: 多 Agent 怎么协作？"| C
    A -->|"Q: 安全问题怎么处理？"| D
    D -->|"Q: 生产环境怎么监控？"| E

    style A fill:#e8f5e9
```

- [Agent 架构](/3-ai-apps/interview#q5) → [Multi-Agent](/3-ai-apps/interview#q7) → [Prompt Injection](/6-ai-frontier/interview#q1) → [生产监控](/5-ai-engineering/interview#q11)

### 路径 5：ML 基础 → 工程化

```mermaid
graph LR
    A["偏差-方差<br/>⭐⭐⭐"]
    B["正则化<br/>⭐⭐⭐"]
    C["交叉验证<br/>⭐⭐"]
    D["超参调优<br/>⭐⭐⭐"]
    E["MLOps 流水线<br/>⭐⭐⭐"]

    A -->|"Q: 过拟合怎么解决？"| B
    B -->|"Q: 如何验证效果？"| C
    C -->|"Q: 如何自动调参？"| D
    D -->|"Q: 如何自动化？"| E

    style A fill:#fff3e0
```

- [偏差-方差](/1-ml-basics/interview#q1) → [正则化](/1-ml-basics/interview#q3) → [交叉验证](/1-ml-basics/interview#q4) → [MLOps](/5-ai-engineering/interview#q1)

---

## 知识点关联矩阵

| 知识点 | 强关联 | 弱关联 |
|--------|--------|--------|
| Transformer | 注意力机制、位置编码、KV Cache | CNN、RNN、Scaling Laws |
| RAG | Embedding、向量数据库、Rerank | Prompt Engineering、LangChain |
| LoRA | QLoRA、PEFT、微调数据 | 量化、部署 |
| Agent | Function Calling、ReAct、记忆 | MCP、LangGraph |
| vLLM | PagedAttention、KV Cache、GPU | Tensor Parallelism、负载均衡 |
| MLOps | 实验追踪、模型注册、CI/CD | 监控、成本优化 |

---

## 💡 使用建议

1. **面试前**：沿着追问路径练习，确保每条路径上的知识点都能流畅回答
2. **模拟面试**：让同学从路径起点开始提问，逐步深入
3. **查漏补缺**：如果某条路径中断，说明该知识点需要加强
4. **建立联系**：理解知识点之间的关联，面试时能自然地引导话题
