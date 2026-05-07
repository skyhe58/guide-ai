---
title: "按公司分类面试索引"
description: "按公司类型索引面试重点、面试特点和推荐复习策略"
---

# 🏢 按公司类型分类面试索引

> 不同类型的公司对 AI 岗位的考察侧重点不同。根据目标公司类型，调整复习策略和重点。

---

## 大厂 AI Lab

> 代表：字节跳动 AI Lab、阿里达摩院、腾讯 AI Lab、百度 PaddlePaddle、华为诺亚方舟、Google DeepMind、Meta FAIR、OpenAI

### 面试特点

- **原理深度要求高**：不仅要会用，还要理解底层原理和数学推导
- **算法手撕**：可能要求手写 Attention、反向传播、LoRA 等核心算法
- **系统设计**：大规模分布式训练、推理服务架构设计
- **论文阅读**：可能问到最新论文的理解和改进思路
- **面试轮次多**：通常 4-6 轮，包含算法题 + 系统设计 + 项目深挖

### 重点模块

| 优先级 | 模块 | 重点内容 | 链接 |
|:------:|------|----------|------|
| P0 | 模块 1 | 深度学习原理、反向传播、Transformer | [面试题](/1-ml-basics/interview) |
| P0 | 模块 2 | Transformer 架构、注意力机制、KV Cache、Scaling Laws | [面试题](/2-llm/interview) |
| P1 | 模块 5 | 分布式训练、GPU 优化、推理加速 | [面试题](/5-ai-engineering/interview) |
| P1 | 模块 3 | RAG 架构设计、Agent 系统设计 | [面试题](/3-ai-apps/interview) |
| P2 | 模块 6 | AI 安全、MCP 协议 | [面试题](/6-ai-frontier/interview) |

### 推荐复习策略

1. 深入理解 Transformer 每一层的计算过程，能手推 Attention 公式
2. 掌握 LoRA 的数学原理（低秩分解），不只是会调参
3. 理解 vLLM PagedAttention 的内存管理机制
4. 准备 1-2 个有深度的项目，能讲清楚技术选型和优化过程
5. 关注最新论文（如 DeepSeek-V3 MoE 架构、Flash Attention 2）

### 高频题索引

| 题目 | 难度 | 链接 |
|------|:----:|------|
| Transformer 自注意力机制原理 | ⭐⭐⭐ | [查看](/2-llm/interview#q1) |
| 反向传播算法推导 | ⭐⭐⭐⭐ | [查看](/1-ml-basics/interview#q7) |
| KV Cache 与 PagedAttention | ⭐⭐⭐⭐ | [查看](/2-llm/interview#q5) |
| RLHF 与 DPO 对比 | ⭐⭐⭐⭐ | [查看](/2-llm/interview#q6) |
| GPU 显存优化策略 | ⭐⭐⭐ | [查看](/5-ai-engineering/interview#q5) |
| vLLM 推理加速原理 | ⭐⭐⭐⭐ | [查看](/5-ai-engineering/interview#q3) |
| Scaling Laws | ⭐⭐⭐ | [查看](/2-llm/interview#q8) |
| 分布式训练（DeepSpeed ZeRO） | ⭐⭐⭐⭐ | [查看](/5-ai-engineering/interview#q6) |

---

## AI 创业公司

> 代表：月之暗面（Kimi）、智谱 AI、MiniMax、零一万物、Cohere、Anthropic 合作伙伴、各类 AI SaaS 创业公司

### 面试特点

- **实战能力优先**：更看重能不能快速落地产品，而非理论深度
- **全栈能力**：可能需要从模型选型到部署上线全流程负责
- **快速迭代**：关注工程效率和成本控制
- **项目经验**：重点考察实际项目经验和问题解决能力
- **面试轮次少**：通常 2-4 轮，节奏快

### 重点模块

| 优先级 | 模块 | 重点内容 | 链接 |
|:------:|------|----------|------|
| P0 | 模块 3 | RAG 实战、Agent 开发、框架使用 | [面试题](/3-ai-apps/interview) |
| P0 | 模块 5 | 模型服务化、成本优化、监控 | [面试题](/5-ai-engineering/interview) |
| P1 | 模块 2 | 模型选型、微调实战、部署方案 | [面试题](/2-llm/interview) |
| P1 | 模块 0 | Python 工程能力、FastAPI | [面试题](/0-prerequisites/interview) |
| P2 | 模块 6 | MCP 协议、AI 安全基础 | [面试题](/6-ai-frontier/interview) |

### 推荐复习策略

1. 准备完整的 RAG 项目经验，能讲清楚从文档加载到生成的全链路
2. 熟悉 LangChain/LangGraph 框架的实际使用和踩坑经验
3. 掌握成本优化策略（模型选择、缓存、Token 计费）
4. 了解快速原型开发流程（Ollama 本地开发 → vLLM 生产部署）
5. 准备"如何从 0 到 1 搭建 AI 应用"的系统性回答

### 高频题索引

| 题目 | 难度 | 链接 |
|------|:----:|------|
| RAG 完整架构和工作流程 | ⭐⭐⭐ | [查看](/3-ai-apps/interview#q1) |
| 向量数据库选型 | ⭐⭐⭐ | [查看](/3-ai-apps/interview#q2) |
| Agent 架构设计 | ⭐⭐⭐ | [查看](/3-ai-apps/interview#q5) |
| LangChain vs LangGraph 选型 | ⭐⭐⭐ | [查看](/3-ai-apps/interview#q8) |
| MLOps 流水线设计 | ⭐⭐⭐ | [查看](/5-ai-engineering/interview#q1) |
| 成本优化策略 | ⭐⭐⭐ | [查看](/5-ai-engineering/interview#q10) |
| 部署方案对比 | ⭐⭐⭐ | [查看](/2-llm/interview#q10) |
| LoRA 微调实战 | ⭐⭐⭐ | [查看](/2-llm/interview#q3) |

---

## 传统企业 AI 部门

> 代表：银行/保险 AI 中心、制造业智能化部门、零售/电商 AI 团队、医疗/教育 AI 应用、政府/国企数字化转型

### 面试特点

- **落地方案优先**：关注 AI 如何解决实际业务问题
- **稳定性要求高**：生产环境稳定性、安全合规、数据隐私
- **技术选型保守**：倾向成熟方案，不追求最新技术
- **跨部门沟通**：需要向非技术人员解释 AI 能力和局限
- **面试风格**：技术面 + 业务面 + HR 面，注重综合素质

### 重点模块

| 优先级 | 模块 | 重点内容 | 链接 |
|:------:|------|----------|------|
| P0 | 模块 3 | RAG 知识库、智能客服、Prompt Engineering | [面试题](/3-ai-apps/interview) |
| P0 | 模块 5 | 生产监控、日志管理、告警策略 | [面试题](/5-ai-engineering/interview) |
| P1 | 模块 7 | AI 工具选型、产品设计思路 | [查看](/7-ai-tools/) |
| P1 | 模块 2 | 模型选型（成本/安全/私有化部署） | [面试题](/2-llm/interview) |
| P2 | 模块 6 | AI 安全、Prompt Injection 防御 | [面试题](/6-ai-frontier/interview) |

### 推荐复习策略

1. 准备"AI 如何解决业务问题"的案例（如智能客服降低人工成本）
2. 熟悉私有化部署方案（Ollama/vLLM 本地部署，数据不出企业）
3. 了解数据安全和合规要求（数据脱敏、审计日志、权限控制）
4. 掌握成本估算能力（Token 计费、GPU 租赁成本、ROI 分析）
5. 准备向非技术人员解释 AI 能力边界的话术

### 高频题索引

| 题目 | 难度 | 链接 |
|------|:----:|------|
| RAG 完整架构和工作流程 | ⭐⭐⭐ | [查看](/3-ai-apps/interview#q1) |
| Prompt Engineering 技巧 | ⭐⭐ | [查看](/3-ai-apps/interview#q10) |
| Ollama 本地部署方案 | ⭐⭐ | [查看](/2-llm/interview#q10) |
| 生产监控与告警 | ⭐⭐⭐ | [查看](/5-ai-engineering/interview#q11) |
| Prompt Injection 防御 | ⭐⭐⭐ | [查看](/6-ai-frontier/interview#q1) |
| 成本优化策略 | ⭐⭐⭐ | [查看](/5-ai-engineering/interview#q10) |
| asyncio 事件循环原理 | ⭐⭐⭐ | [查看](/0-prerequisites/interview#q1) |

---

## 公司类型对比速查

| 维度 | 大厂 AI Lab | AI 创业公司 | 传统企业 AI 部门 |
|------|:----------:|:----------:|:---------------:|
| 原理深度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| 实战能力 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 系统设计 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 工程能力 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 业务理解 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 面试轮次 | 4-6 轮 | 2-4 轮 | 3-4 轮 |
| 薪资范围 | 最高 | 高（含期权） | 中等（稳定） |
| 核心模块 | 模块 1+2 | 模块 3+5 | 模块 3+5+7 |
