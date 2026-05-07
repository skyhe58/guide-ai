---
title: "模块 3 面试指南"
module: "ai-apps"
description: "AI 应用开发高频面试题 — RAG/Agent/框架/评估"
---

# 模块 3 面试指南

> 本指南覆盖 AI 应用开发领域的高频面试题，适用于 AI 应用工程师、LLM 工程师岗位。
> 难度标注：⭐ 初级 ⭐⭐ 中级 ⭐⭐⭐ 高级 ⭐⭐⭐⭐ 专家
> 频率标注：🔥 偶尔 🔥🔥 经常 🔥🔥🔥 几乎必问

---

## Q1: 请描述 RAG 的完整架构和工作流程

**难度**：⭐⭐⭐ | **频率**：🔥🔥🔥

**答题思路**：整体架构 → 离线索引流程 → 在线查询流程 → 关键组件

**标准答案**：RAG 分为离线索引和在线查询两个阶段。离线索引：文档加载（PDF/Markdown/HTML）→ 文档切分（RecursiveCharacterTextSplitter，chunk_size 通常 500-1000）→ Embedding 向量化（text-embedding-3-small 或 BGE-M3）→ 存入向量数据库（Chroma/Pinecone）。在线查询：用户问题 → Embedding → 向量检索 Top-K → 可选 Rerank 重排序 → 将检索文档注入 Prompt → LLM 生成回答。关键优化点：切分策略（语义切分优于固定长度）、检索策略（混合检索 = 向量 + BM25）、Rerank（BGE-Reranker 提升精度）、查询改写（HyDE 假设文档嵌入）。

**追问**：
- 文档切分的 chunk_size 如何选择？（根据模型 context window 和文档类型，通常 500-1000 tokens）
- 如何处理跨 chunk 的信息？（增加 overlap、父子节点关联、滑动窗口）
- RAG 和 Fine-tuning 如何选择？（RAG 适合知识更新频繁的场景，Fine-tuning 适合固定领域知识）

---

## Q2: 向量数据库如何选型？Chroma、Pinecone、FAISS、Milvus 各有什么特点？

**难度**：⭐⭐⭐ | **频率**：🔥🔥🔥

**答题思路**：各数据库特点 → 适用场景 → 选型建议

**标准答案**：Chroma——轻量级本地数据库，零配置，适合开发和小规模应用（<100 万向量）。Pinecone——全托管云服务，自动扩缩容，适合生产环境，按用量付费。FAISS——Facebook 开源的向量检索库（非数据库），性能极高，适合大规模离线检索，需要自行管理持久化。Milvus——分布式向量数据库，支持十亿级向量，适合企业级大规模应用。选型建议：开发阶段用 Chroma，中小规模生产用 Pinecone，大规模且有运维能力用 Milvus，纯检索性能优先用 FAISS。

**追问**：
- 向量数据库的索引算法有哪些？（HNSW、IVF、PQ、Flat）
- 如何评估向量检索的质量？（Recall@K、MRR、NDCG）
- 向量数据库如何做水平扩展？（分片、副本、读写分离）

---

## Q3: AI Agent 的 ReAct 模式是什么？如何实现？

**难度**：⭐⭐⭐ | **频率**：🔥🔥🔥

**答题思路**：ReAct 定义 → 推理-行动循环 → 实现方式

**标准答案**：ReAct（Reasoning + Acting）是一种让 LLM 交替进行推理和行动的 Agent 模式。工作流程：(1) Thought——LLM 分析当前状态，决定下一步行动；(2) Action——选择并调用工具（搜索、计算、API 等）；(3) Observation——获取工具返回结果；(4) 重复 1-3 直到得出最终答案。实现方式：在 Prompt 中定义可用工具和 ReAct 格式，LLM 按格式输出 Thought/Action/Observation，代码解析输出并执行工具调用。LangChain 的 `create_react_agent` 和 LangGraph 都支持 ReAct 模式。

**追问**：
- ReAct 和 Function Calling 有什么区别？（ReAct 是 Prompt 驱动，Function Calling 是 API 原生支持）
- Agent 陷入无限循环怎么办？（设置最大迭代次数、超时机制、循环检测）
- 如何提升 Agent 的工具选择准确率？（优化工具描述、Few-shot 示例、工具路由）

---

## Q4: Embedding 模型如何选择？有哪些关键考量因素？

**难度**：⭐⭐⭐ | **频率**：🔥🔥🔥

**答题思路**：选型维度 → 主流模型对比 → 实际建议

**标准答案**：选型维度：(1) 语言支持——中文场景优先 BGE-M3 或 M3E，英文用 text-embedding-3-small；(2) 维度——高维（1536/3072）精度高但存储大，低维（384/768）效率高；(3) 性能——MTEB 排行榜评分；(4) 成本——API 调用费用 vs 本地部署成本；(5) 延迟——在线服务对延迟敏感。推荐：通用场景用 OpenAI text-embedding-3-small（性价比高），中文场景用 BGE-M3（开源免费），离线批量用本地部署的 BGE-large-zh。

**追问**：
- Embedding 模型的维度对检索质量有多大影响？（通常 768 维以上差异不大）
- 如何评估 Embedding 模型的质量？（MTEB 基准、下游任务评估）
- 不同 Embedding 模型的向量可以混用吗？（不可以，必须用同一模型）

---

## Q5: LangChain 和 LangGraph 有什么区别？什么时候用哪个？

**难度**：⭐⭐⭐ | **频率**：🔥🔥🔥

**答题思路**：核心区别 → 适用场景 → 组合使用

**标准答案**：LangChain 的 LCEL 是线性管道（Prompt → LLM → Parser），适合简单的顺序执行场景。LangGraph 是图状态机，支持条件分支、循环迭代、人机协作和状态持久化，适合复杂的多步骤工作流。选型：简单 RAG 问答用 LCEL；需要条件路由、重试逻辑的用 LangGraph；Multi-Agent 协作必须用 LangGraph。两者可以混用——LangGraph 节点内部调用 LCEL Chain。LangGraph 是 LangChain 团队推荐的 Agent 构建方式。

**追问**：
- LangGraph 的 State 设计有什么最佳实践？
- 如何实现 Human-in-the-Loop？（interrupt_before/after + Checkpointer）
- LangGraph 和 LlamaIndex 可以一起用吗？（可以，LlamaIndex 做检索，LangGraph 做编排）

---

## Q6: RAG 系统的评估指标有哪些？如何系统化评估？

**难度**：⭐⭐⭐⭐ | **频率**：🔥🔥🔥

**答题思路**：四大核心指标 → 评估方法 → 优化方向

**标准答案**：RAG 评估分检索质量和生成质量。检索质量：Context Precision（检索精确度）和 Context Recall（检索召回率）。生成质量：Faithfulness（忠实度，回答是否忠于上下文）和 Answer Relevancy（答案相关性）。评估方法：使用 RAGAS 或 DeepEval 框架自动化评估，构建 50-200 条评估数据集，定期运行。优化方向：Faithfulness 低→优化 Prompt 减少幻觉；Relevancy 低→改进查询理解；Precision 低→增加 Rerank；Recall 低→增大 Top-K 或查询改写。

**追问**：
- 用 LLM 评估 LLM 可靠吗？（70-80% 与人工一致，存在偏差需校准）
- 如何构建高质量评估数据集？（专家标注 + LLM 辅助 + 生产日志采样）
- RAGAS 和 DeepEval 如何选择？（纯 RAG 用 RAGAS，通用评估用 DeepEval）

---

## Q7: 如何优化 RAG 系统的检索质量？

**难度**：⭐⭐⭐⭐ | **频率**：🔥🔥

**答题思路**：问题诊断 → 优化策略 → 效果验证

**标准答案**：检索优化策略：(1) 切分优化——语义切分替代固定长度，增加 overlap，保留文档结构；(2) Embedding 优化——选择适合领域的模型，Fine-tune Embedding 模型；(3) 检索策略——混合检索（向量 + BM25），Multi-Query 多角度检索；(4) Rerank——用 BGE-Reranker 或 Cohere Rerank 对初检结果重排序；(5) 查询改写——HyDE（假设文档嵌入）、Step-back Prompting；(6) 元数据过滤——利用文档元数据（时间、来源、类别）缩小检索范围。

**追问**：
- HyDE 的原理是什么？（先让 LLM 生成假设答案，用假设答案的 Embedding 检索）
- 混合检索如何融合向量和 BM25 的结果？（RRF 倒数排名融合、加权融合）
- 检索优化的效果如何量化？（Recall@K、MRR、A/B 测试）

---

## Q8: Multi-Agent 系统如何设计？有哪些常见架构模式？

**难度**：⭐⭐⭐⭐ | **频率**：🔥🔥

**答题思路**：架构模式 → 通信机制 → 实际案例

**标准答案**：三种主要架构：(1) Supervisor 模式——一个协调者 Agent 分配任务给多个 Worker Agent，最常用；(2) Hierarchical 模式——多层级管理，适合复杂组织结构；(3) Debate 模式——多个 Agent 讨论协商，适合需要多角度分析的场景。通信机制：消息传递（结构化消息）、共享状态（LangGraph State）、工具调用（Agent 间互相调用）。实际案例：代码审查系统（Coder + Reviewer + Tester）、研究助手（Researcher + Writer + Editor）。

**追问**：
- Agent 间如何避免无限来回？（最大轮次、收敛检测、超时机制）
- 如何保证 Multi-Agent 系统的可观测性？（LangSmith 追踪、结构化日志）
- MCP 协议在 Multi-Agent 中的作用？（标准化 Agent 间通信）

---

## Q9: Prompt Engineering 有哪些进阶技巧？

**难度**：⭐⭐⭐ | **频率**：🔥🔥

**答题思路**：核心技巧 → 实际应用 → 评估迭代

**标准答案**：进阶技巧：(1) Chain-of-Thought——让 LLM 分步推理，提升复杂问题准确率；(2) Few-shot 示例设计——选择多样性高、与查询相似的示例；(3) 输出格式控制——用 JSON Schema 或 Pydantic 约束输出格式；(4) System Prompt 设计——明确角色、能力边界、输出要求和安全规则；(5) 分隔符使用——用 XML 标签或 Markdown 分隔不同内容区域；(6) Self-Consistency——多次采样取多数投票，提升可靠性。

**追问**：
- Zero-shot CoT 和 Few-shot CoT 有什么区别？
- 如何做 Prompt 的 A/B 测试？（LangSmith 评估数据集对比）
- Prompt Injection 如何防御？（输入过滤、输出检测、角色隔离）

---

## Q10: 如何设计一个生产级的 RAG 系统？

**难度**：⭐⭐⭐⭐ | **频率**：🔥🔥🔥

**答题思路**：架构设计 → 关键组件 → 运维监控

**标准答案**：生产级 RAG 架构：(1) 数据层——文档管道（加载→清洗→切分→Embedding→入库），支持增量更新；(2) 检索层——混合检索（向量 + BM25）+ Rerank，支持元数据过滤；(3) 生成层——Prompt 模板管理、流式输出、引用溯源；(4) 缓存层——语义缓存减少重复调用，Redis 缓存热点查询；(5) 监控层——LangSmith 追踪、延迟/错误率/Token 成本监控；(6) 评估层——定期自动评估（RAGAS）+ 用户反馈收集。关键设计：多租户隔离、权限控制、敏感信息过滤、Fallback 机制。

**追问**：
- 如何处理知识库的增量更新？（增量索引、版本管理、过期文档清理）
- 如何控制 RAG 系统的成本？（缓存、模型级联、采样评估）
- 如何保证回答的安全性？（输出过滤、敏感词检测、引用验证）

---

## Q11: LlamaIndex 和 LangChain 在 RAG 场景下如何选择？

**难度**：⭐⭐⭐ | **频率**：🔥🔥

**答题思路**：定位差异 → 各自优势 → 组合方案

**标准答案**：LlamaIndex 专注数据索引和检索，提供更丰富的索引类型（向量/树/关键词/知识图谱）和查询优化（子问题分解、路由查询），数据连接器生态更丰富（160+）。LangChain 是全栈框架，Agent 和 Chain 能力更强。选型：纯 RAG 知识库用 LlamaIndex 更高效；需要 Agent + RAG 的复杂应用用 LangChain/LangGraph；两者可集成——LlamaIndex 做索引检索，LangChain 做编排。

**追问**：
- LlamaIndex 的 TreeIndex 适合什么场景？（长文档摘要、层级问答）
- 如何将 LlamaIndex 集成到 LangChain 中？（`index.as_retriever()` 转为 LangChain Retriever）

---

## 面试准备建议

1. **RAG 是重中之重**：几乎每次 AI 应用面试都会问 RAG 架构，务必能画出完整流程图
2. **动手实践**：跑通至少一个完整的 RAG 项目，理解每个环节的细节
3. **框架不是重点**：面试官更关心你对原理的理解，而非框架 API 的记忆
4. **准备系统设计**：高级岗位会问"如何设计一个生产级 RAG 系统"，准备完整方案
5. **关注评估**：能说清楚如何评估 RAG 质量是加分项
