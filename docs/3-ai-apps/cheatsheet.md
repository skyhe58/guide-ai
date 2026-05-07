---
title: "模块 3 速查卡片"
module: "ai-apps"
description: "AI 应用开发核心概念和常用 API 速查"
---

# 模块 3 速查卡片

## Prompt Engineering 速查

| 技巧 | 说明 | 示例 |
|------|------|------|
| 角色设定 | 在 System Prompt 中定义角色 | "你是一个资深 Python 工程师" |
| 输出格式 | 明确指定输出格式 | "请以 JSON 格式输出" |
| 分隔符 | 用标记分隔不同内容 | `<context>...</context>` |
| Few-shot | 提供示例引导输出 | "示例：输入→输出" |
| CoT | 让模型分步推理 | "请一步一步思考" |
| Self-Consistency | 多次采样取多数 | temperature=0.7, n=5 |

## RAG 流程速查

```
文档加载 → 文档切分 → Embedding → 向量存储 → 检索 → Rerank → 生成
  PDF       chunk_size   BGE-M3     Chroma     Top-K   BGE-Reranker  GPT-4o
  Markdown  overlap      OpenAI     Pinecone   MMR     Cohere        Claude
  HTML      语义切分     M3E        FAISS      混合    Cross-Encoder  Qwen
```

### 关键参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| chunk_size | 500-1000 | 切分块大小（tokens） |
| chunk_overlap | 50-200 | 切分重叠区域 |
| top_k | 3-5 | 检索返回文档数 |
| temperature | 0-0.3 | 生成温度（RAG 场景低温） |
| similarity_threshold | 0.7+ | 相似度过滤阈值 |

### Embedding 模型选型

| 模型 | 维度 | 中文 | 成本 | 推荐场景 |
|------|------|------|------|---------|
| text-embedding-3-small | 1536 | 一般 | API 付费 | 通用英文 |
| text-embedding-3-large | 3072 | 一般 | API 付费 | 高精度 |
| BGE-M3 | 1024 | 优秀 | 免费开源 | 中文场景 |
| M3E-base | 768 | 优秀 | 免费开源 | 轻量中文 |

### 向量数据库选型

| 数据库 | 类型 | 规模 | 适用场景 |
|--------|------|------|---------|
| Chroma | 本地 | <100 万 | 开发/小规模 |
| Pinecone | 云服务 | 无限 | 生产环境 |
| FAISS | 库 | 十亿级 | 高性能离线 |
| Milvus | 分布式 | 十亿级 | 企业级 |

## Agent 模式速查

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| Function Calling | LLM API 原生工具调用 | 简单工具调用 |
| ReAct | 推理-行动循环 | 多步骤推理 |
| Supervisor | 协调者 + Worker | 多 Agent 协作 |
| Sequential | 顺序管道 | 流水线处理 |
| Debate | 多 Agent 讨论 | 多角度分析 |

### Agent 记忆类型

| 类型 | 存储 | 生命周期 | 实现 |
|------|------|---------|------|
| 短期记忆 | Context Window | 单次对话 | 消息列表 |
| 长期记忆 | 向量数据库 | 跨对话 | Embedding + 检索 |
| 工作记忆 | 内存/缓存 | 单次任务 | 键值存储 |

## 框架速查

### LangChain LCEL

```python
# 基础 Chain
chain = prompt | llm | parser
result = chain.invoke({"question": "..."})

# 流式输出
async for chunk in chain.astream({"question": "..."}):
    print(chunk, end="")

# 批量处理
results = chain.batch([{"question": "Q1"}, {"question": "Q2"}])
```

### LangGraph

```python
# 构建图
graph = StateGraph(State)
graph.add_node("retrieve", retrieve_fn)
graph.add_node("generate", generate_fn)
graph.add_conditional_edges("route", route_fn, {"query": "retrieve", "chat": "generate"})
graph.set_entry_point("route")
app = graph.compile(checkpointer=MemorySaver())

# 执行
result = app.invoke({"messages": [("human", "...")]}, config={"configurable": {"thread_id": "1"}})
```

### LlamaIndex

```python
# 基础 RAG
documents = SimpleDirectoryReader("data/").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("...")

# 对话模式
chat_engine = index.as_chat_engine()
response = chat_engine.chat("...")
```

## 评估指标速查

| 指标 | 维度 | 计算方式 | 需要标准答案 |
|------|------|---------|-------------|
| Faithfulness | 生成质量 | 有支持的声明 / 总声明 | ❌ |
| Answer Relevancy | 生成质量 | 反向生成问题相似度 | ❌ |
| Context Precision | 检索质量 | 相关上下文 / 总上下文 | ✅ |
| Context Recall | 检索质量 | 被覆盖的声明 / 标准答案声明 | ✅ |

### 评估框架对比

| 框架 | 定位 | pytest 集成 | 适用场景 |
|------|------|------------|---------|
| RAGAS | RAG 专用 | ❌ | RAG 系统评估 |
| DeepEval | 通用 LLM | ✅ | 各类 LLM 应用 |
| LangSmith | 追踪+评估 | ❌ | LangChain 生态 |

## 常用命令速查

```bash
# 安装依赖
pip install langchain langchain-openai langchain-community
pip install langgraph
pip install llama-index
pip install ragas deepeval
pip install chromadb

# 启动 Ollama
docker compose -f docker/docker-compose.yml up -d ollama
docker exec guide-ai-ollama ollama pull qwen2

# 启动 Chroma
docker compose -f docker/docker-compose.yml up -d chroma

# 运行评估
deepeval test run test_rag.py
python -m ragas.evaluate
```

## 环境变量速查

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# LangSmith
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="ls-..."
export LANGCHAIN_PROJECT="my-project"

# Ollama
export OLLAMA_HOST="http://localhost:11434"
```
