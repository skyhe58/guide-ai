"""
RAG 优化技巧 — 查询改写 / HyDE / 多步检索 / 上下文压缩

知识点：查询改写（Query Rewriting）、HyDE（假设文档嵌入）、
       查询分解（Query Decomposition）、上下文压缩（Context Compression）、
       多步检索（Iterative Retrieval）、RAG 评估指标

Python 版本：3.11+
依赖：标准库（默认模式）
最后验证：2024-12-01

外部服务（可选）：
  无外部服务依赖，所有示例使用模拟数据
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any

# ============================================================
# 1. 基础工具
# ============================================================

class VectorOps:
    """向量运算工具。"""

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na > 0 and nb > 0 else 0.0

    @staticmethod
    def normalize(a: list[float]) -> list[float]:
        n = math.sqrt(sum(x * x for x in a))
        return [x / n for x in a] if n > 0 else a


class SimpleEmbedding:
    """简单 Embedding 模型。"""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        vec = [(b - 128) / 128.0 for b in h]
        while len(vec) < self.dim:
            h = hashlib.sha256(h).digest()
            vec.extend((b - 128) / 128.0 for b in h)
        vec = vec[:self.dim]
        kw_map = {"RAG": 0, "检索": 1, "向量": 2, "优化": 3, "LLM": 4,
                   "查询": 5, "改写": 6, "HyDE": 7, "压缩": 8, "Rerank": 9}
        for kw, idx in kw_map.items():
            if kw in text and idx < self.dim:
                vec[idx] += 0.5
        return VectorOps.normalize(vec)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


# ============================================================
# 2. 模拟 LLM
# ============================================================

class SimulatedLLM:
    """模拟 LLM 用于查询改写、HyDE 等。

    使用规则引擎模拟 LLM 的行为，实际生产中替换为真实 LLM API。
    """

    def __init__(self, model_name: str = "simulated-qwen2"):
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        """模拟 LLM 生成。"""
        if "改写" in prompt or "rewrite" in prompt.lower():
            return self._simulate_query_rewrite(prompt)
        elif "假设" in prompt or "hyde" in prompt.lower() or "回答" in prompt:
            return self._simulate_hyde(prompt)
        elif "分解" in prompt or "decompose" in prompt.lower():
            return self._simulate_decomposition(prompt)
        elif "压缩" in prompt or "提取" in prompt:
            return self._simulate_compression(prompt)
        else:
            return "这是一个模拟的 LLM 回答。"

    def _simulate_query_rewrite(self, prompt: str) -> str:
        """模拟查询改写。"""
        rewrites = {
            "RAG 怎么搞": '["RAG 系统架构设计方法", "RAG 检索增强生成实现步骤"]',
            "向量数据库咋选": '["向量数据库选型对比 Chroma Milvus FAISS", "向量数据库生产环境部署方案"]',
            "检索效果不好怎么办": '["RAG 检索优化策略", "混合检索和 Rerank 提升检索准确率"]',
        }
        for key, value in rewrites.items():
            if key in prompt:
                return value
        return '["优化后的查询1", "优化后的查询2"]'

    def _simulate_hyde(self, prompt: str) -> str:
        """模拟 HyDE 假设答案生成。"""
        if "RAG" in prompt and "优化" in prompt:
            return ("RAG 系统优化可以从多个环节入手：首先是查询优化，包括查询改写和 HyDE；"
                    "其次是检索优化，使用混合检索结合向量和 BM25；然后是 Rerank 重排序，"
                    "使用交叉编码器对候选文档精排；最后是上下文压缩，提取最相关的文档片段。")
        elif "向量数据库" in prompt:
            return ("选择向量数据库需要考虑数据规模、部署方式和性能需求。"
                    "Chroma 适合开发阶段，FAISS 适合高性能场景，"
                    "Milvus 适合企业级生产环境，Pinecone 适合免运维需求。")
        return "这是一个关于该主题的假设性回答，包含了相关的技术细节和实践建议。"

    def _simulate_decomposition(self, prompt: str) -> str:
        """模拟查询分解。"""
        if "对比" in prompt and "数据库" in prompt:
            return ('["Chroma 向量数据库的性能特点和适用场景", '
                    '"Milvus 向量数据库的性能特点和适用场景", '
                    '"Chroma 和 Milvus 的部署成本对比"]')
        return '["子问题1", "子问题2", "子问题3"]'

    def _simulate_compression(self, prompt: str) -> str:
        """模拟上下文压缩。"""
        # 提取文档中与查询相关的句子
        if "文档：" in prompt:
            doc_start = prompt.index("文档：") + 3
            document = prompt[doc_start:].strip()
            sentences = re.split(r"[。！？]", document)
            # 返回前 2 个非空句子作为压缩结果
            relevant = [s.strip() for s in sentences if s.strip()][:2]
            return "。".join(relevant) + "。" if relevant else document[:100]
        return "压缩后的相关内容。"


# ============================================================
# 3. 简单向量检索器
# ============================================================

@dataclass
class SearchResult:
    """搜索结果。"""
    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class SimpleRetriever:
    """简单向量检索器。"""

    def __init__(self, embedding: SimpleEmbedding | None = None):
        self.embedding = embedding or SimpleEmbedding()
        self.documents: list[str] = []
        self.doc_ids: list[str] = []
        self.embeddings: list[list[float]] = []

    def add_documents(self, documents: list[str], doc_ids: list[str] | None = None) -> None:
        if doc_ids is None:
            doc_ids = [f"doc_{len(self.documents) + i}" for i in range(len(documents))]
        self.documents.extend(documents)
        self.doc_ids.extend(doc_ids)
        self.embeddings.extend(self.embedding.embed_batch(documents))

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        query_emb = self.embedding.embed(query)
        scored = []
        for i, emb in enumerate(self.embeddings):
            sim = VectorOps.cosine_similarity(query_emb, emb)
            scored.append((i, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            SearchResult(doc_id=self.doc_ids[i], text=self.documents[i], score=s)
            for i, s in scored[:k]
        ]


# ============================================================
# 4. 查询改写（Query Rewriting）
# ============================================================

class QueryRewriter:
    """查询改写器。

    用 LLM 将用户的口语化查询改写为更精确的检索查询。
    这是最简单且高效的 RAG 优化手段。
    """

    def __init__(self, llm: SimulatedLLM | None = None):
        self.llm = llm or SimulatedLLM()

    def rewrite(self, query: str) -> list[str]:
        """将原始查询改写为多个精确查询。"""
        prompt = f"""你是一个搜索查询优化专家。
将用户的口语化查询改写为更适合知识库检索的查询。

原始查询：{query}

改写要求：
1. 补充隐含的上下文信息
2. 使用更精确的技术术语
3. 去除口语化表达
4. 输出 JSON 列表格式

改写后的查询："""

        response = self.llm.generate(prompt)
        # 解析 JSON 列表
        try:
            import json
            queries = json.loads(response)
            return queries if isinstance(queries, list) else [response]
        except (json.JSONDecodeError, ValueError):
            return [response]

    def rewrite_with_context(self, query: str, chat_history: list[str]) -> str:
        """结合对话历史改写查询（多轮对话场景）。"""
        history_text = "\n".join(chat_history[-3:])  # 最近 3 轮
        prompt = f"""结合对话历史，将最新查询改写为独立的检索查询。

对话历史：
{history_text}

最新查询：{query}

改写后的独立查询："""

        return self.llm.generate(prompt)


# ============================================================
# 5. HyDE（假设文档嵌入）
# ============================================================

class HyDERetriever:
    """HyDE 检索器。

    Hypothetical Document Embeddings：
    1. 让 LLM 根据查询生成一个"假设答案"
    2. 用假设答案的 Embedding 去检索（而非原始查询）
    3. 假设答案与真实文档在向量空间中更接近
    """

    def __init__(
        self,
        retriever: SimpleRetriever,
        llm: SimulatedLLM | None = None,
    ):
        self.retriever = retriever
        self.llm = llm or SimulatedLLM()

    def search(self, query: str, k: int = 5) -> tuple[list[SearchResult], str]:
        """HyDE 检索。

        Returns:
            (检索结果, 假设答案)
        """
        # Step 1: 生成假设答案
        hyde_prompt = f"""请回答以下问题。即使你不确定，也请给出一个合理的答案。

问题：{query}

答案："""
        hypothetical_answer = self.llm.generate(hyde_prompt)

        # Step 2: 用假设答案检索
        results = self.retriever.search(hypothetical_answer, k=k)

        return results, hypothetical_answer


# ============================================================
# 6. 查询分解（Query Decomposition）
# ============================================================

class QueryDecomposer:
    """查询分解器。

    将复杂查询拆分为多个简单子查询，分别检索后合并结果。
    适合对比类、多方面分析类问题。
    """

    def __init__(self, llm: SimulatedLLM | None = None):
        self.llm = llm or SimulatedLLM()

    def decompose(self, query: str) -> list[str]:
        """将复杂查询分解为子查询。"""
        prompt = f"""将以下复杂问题分解为 2-4 个简单的子问题：

问题：{query}

子问题列表（JSON）："""

        response = self.llm.generate(prompt)
        try:
            import json
            sub_queries = json.loads(response)
            return sub_queries if isinstance(sub_queries, list) else [query]
        except (json.JSONDecodeError, ValueError):
            return [query]

    def search_decomposed(
        self,
        query: str,
        retriever: SimpleRetriever,
        k_per_query: int = 3,
        final_k: int = 5,
    ) -> list[SearchResult]:
        """分解查询后分别检索，合并去重。"""
        sub_queries = self.decompose(query)
        all_results: dict[str, SearchResult] = {}

        for sub_query in sub_queries:
            results = retriever.search(sub_query, k=k_per_query)
            for r in results:
                if r.doc_id not in all_results or r.score > all_results[r.doc_id].score:
                    all_results[r.doc_id] = r

        sorted_results = sorted(all_results.values(), key=lambda x: x.score, reverse=True)
        return sorted_results[:final_k]


# ============================================================
# 7. 上下文压缩（Context Compression）
# ============================================================

class ContextCompressor:
    """上下文压缩器。

    从检索到的文档中提取与查询最相关的部分，
    减少送入 LLM 的 Token 数量，提升生成质量。
    """

    def __init__(self, llm: SimulatedLLM | None = None):
        self.llm = llm or SimulatedLLM()

    def compress(self, query: str, documents: list[str]) -> list[str]:
        """压缩文档，提取相关内容。"""
        compressed = []
        for doc in documents:
            prompt = f"""给定查询和文档，提取文档中与查询最相关的内容。
只保留直接回答查询的信息，去除无关内容。

查询：{query}
文档：{doc}

相关内容："""
            result = self.llm.generate(prompt)
            compressed.append(result)
        return compressed

    def compress_by_sentences(
        self,
        query: str,
        documents: list[str],
        embedding: SimpleEmbedding | None = None,
        top_sentences: int = 3,
    ) -> list[str]:
        """基于句子级相似度的压缩（不依赖 LLM）。"""
        emb = embedding or SimpleEmbedding()
        query_emb = emb.embed(query)
        compressed = []

        for doc in documents:
            sentences = re.split(r"[。！？.!?]", doc)
            sentences = [s.strip() for s in sentences if s.strip()]

            if not sentences:
                compressed.append(doc)
                continue

            # 计算每个句子与查询的相似度
            scored = []
            for sent in sentences:
                sent_emb = emb.embed(sent)
                sim = VectorOps.cosine_similarity(query_emb, sent_emb)
                scored.append((sent, sim))

            scored.sort(key=lambda x: x[1], reverse=True)
            top = scored[:top_sentences]
            # 按原始顺序排列
            top_texts = [s for s, _ in top]
            compressed.append("。".join(top_texts) + "。")

        return compressed


# ============================================================
# 8. 完整 RAG 优化管道
# ============================================================

class OptimizedRAGPipeline:
    """完整的 RAG 优化管道。

    集成查询改写、HyDE、混合检索、Rerank、上下文压缩。
    """

    def __init__(
        self,
        retriever: SimpleRetriever,
        llm: SimulatedLLM | None = None,
        embedding: SimpleEmbedding | None = None,
    ):
        self.retriever = retriever
        self.llm = llm or SimulatedLLM()
        self.embedding = embedding or SimpleEmbedding()
        self.query_rewriter = QueryRewriter(self.llm)
        self.hyde_retriever = HyDERetriever(retriever, self.llm)
        self.compressor = ContextCompressor(self.llm)

    def run(
        self,
        query: str,
        use_rewrite: bool = True,
        use_hyde: bool = False,
        use_compression: bool = True,
        k: int = 5,
    ) -> dict[str, Any]:
        """运行完整 RAG 优化管道。"""
        pipeline_log: list[str] = []

        # Step 1: 查询改写
        search_query = query
        if use_rewrite:
            rewritten = self.query_rewriter.rewrite(query)
            search_query = rewritten[0] if rewritten else query
            pipeline_log.append(f"查询改写: '{query}' → '{search_query}'")

        # Step 2: 检索
        if use_hyde:
            results, hyde_answer = self.hyde_retriever.search(search_query, k=k)
            pipeline_log.append(f"HyDE 假设答案: '{hyde_answer[:50]}...'")
        else:
            results = self.retriever.search(search_query, k=k)
            pipeline_log.append(f"向量检索: Top-{k}")

        # Step 3: 上下文压缩
        documents = [r.text for r in results]
        if use_compression:
            documents = self.compressor.compress_by_sentences(
                query, documents, self.embedding, top_sentences=2,
            )
            pipeline_log.append("上下文压缩: 句子级相似度过滤")

        # Step 4: 生成（模拟）
        context = "\n\n".join(documents[:3])
        answer = f"基于检索到的 {len(results)} 个文档，关于'{query}'的回答：{context[:100]}..."
        pipeline_log.append("LLM 生成: 基于压缩后的上下文")

        return {
            "query": query,
            "answer": answer,
            "sources": [r.doc_id for r in results],
            "pipeline_log": pipeline_log,
            "num_documents": len(results),
        }


# ============================================================
# 9. 测试数据
# ============================================================

KNOWLEDGE_BASE = [
    "RAG 系统通过检索外部知识库增强 LLM 的生成能力，减少幻觉问题",
    "混合检索结合向量检索和 BM25 关键词检索，通过 RRF 融合提升准确率",
    "Rerank 重排序使用交叉编码器对候选文档精排，提升检索精度 5-15%",
    "查询改写用 LLM 将口语化查询优化为精确检索查询，提升检索召回率",
    "HyDE 让 LLM 先生成假设答案，用假设答案的 Embedding 检索效果更好",
    "上下文压缩提取文档中与查询最相关的部分，减少 LLM 输入 Token 消耗",
    "向量数据库 Chroma 适合开发阶段，Milvus 适合企业级生产环境",
    "Embedding 模型将文本转为高维向量，BGE 是中文场景推荐模型",
    "文档切分使用 RecursiveCharacterTextSplitter 按分隔符层级递归切分",
    "多步检索对复杂问题进行多轮检索，逐步补充缺失信息",
    "FAISS 是 Meta 开源的高性能向量搜索库，纯内存操作速度最快",
    "查询分解将复杂问题拆分为多个简单子查询，分别检索后合并结果",
]


def _build_retriever() -> SimpleRetriever:
    """构建检索器。"""
    retriever = SimpleRetriever()
    doc_ids = [f"doc_{i}" for i in range(len(KNOWLEDGE_BASE))]
    retriever.add_documents(KNOWLEDGE_BASE, doc_ids)
    return retriever


# ============================================================
# 10. 演示函数
# ============================================================

def demo_query_rewriting() -> None:
    """演示查询改写。"""
    print("\n" + "=" * 60)
    print("1. 查询改写（Query Rewriting）")
    print("=" * 60)

    rewriter = QueryRewriter()
    retriever = _build_retriever()

    queries = [
        "RAG 怎么搞",
        "向量数据库咋选",
        "检索效果不好怎么办",
    ]

    for query in queries:
        rewritten = rewriter.rewrite(query)
        print(f"\n  原始查询: '{query}'")
        print(f"  改写结果: {rewritten}")

        # 对比改写前后的检索效果
        original_results = retriever.search(query, k=3)
        if rewritten:
            rewritten_results = retriever.search(rewritten[0], k=3)
            print(f"  原始检索 Top-1: {original_results[0].text[:40]}... (score={original_results[0].score:.4f})")
            print(f"  改写检索 Top-1: {rewritten_results[0].text[:40]}... (score={rewritten_results[0].score:.4f})")


def demo_hyde() -> None:
    """演示 HyDE 假设文档嵌入。"""
    print("\n" + "=" * 60)
    print("2. HyDE（假设文档嵌入）")
    print("=" * 60)

    retriever = _build_retriever()
    hyde = HyDERetriever(retriever)

    query = "如何优化 RAG 系统？"
    print(f"  🔍 查询: '{query}'")

    # 普通检索
    normal_results = retriever.search(query, k=3)
    print(f"\n  普通检索 Top-3:")
    for i, r in enumerate(normal_results):
        print(f"    {i+1}. (score={r.score:.4f}) {r.text[:50]}...")

    # HyDE 检索
    hyde_results, hypothesis = hyde.search(query, k=3)
    print(f"\n  HyDE 假设答案: '{hypothesis[:60]}...'")
    print(f"  HyDE 检索 Top-3:")
    for i, r in enumerate(hyde_results):
        print(f"    {i+1}. (score={r.score:.4f}) {r.text[:50]}...")


def demo_query_decomposition() -> None:
    """演示查询分解。"""
    print("\n" + "=" * 60)
    print("3. 查询分解（Query Decomposition）")
    print("=" * 60)

    decomposer = QueryDecomposer()
    retriever = _build_retriever()

    query = "对比 Chroma 和 Milvus 向量数据库的性能和成本"
    print(f"  🔍 原始查询: '{query}'")

    sub_queries = decomposer.decompose(query)
    print(f"  分解为 {len(sub_queries)} 个子查询:")
    for i, sq in enumerate(sub_queries):
        print(f"    {i+1}. {sq}")

    # 分解检索
    results = decomposer.search_decomposed(query, retriever, k_per_query=3, final_k=5)
    print(f"\n  分解检索结果 Top-5:")
    for i, r in enumerate(results):
        print(f"    {i+1}. (score={r.score:.4f}) {r.text[:50]}...")


def demo_context_compression() -> None:
    """演示上下文压缩。"""
    print("\n" + "=" * 60)
    print("4. 上下文压缩（Context Compression）")
    print("=" * 60)

    compressor = ContextCompressor()
    query = "RAG 检索优化"

    documents = [
        "RAG 系统通过检索外部知识库增强 LLM 的生成能力。混合检索结合向量和 BM25 提升准确率。Rerank 使用交叉编码器精排。Python 是最流行的编程语言之一。",
        "查询改写将口语化查询优化为精确检索查询。HyDE 生成假设答案用于检索。上下文压缩提取相关内容。今天天气很好适合散步。",
    ]

    print(f"  🔍 查询: '{query}'")

    # LLM 压缩
    compressed_llm = compressor.compress(query, documents)
    print(f"\n  LLM 压缩结果:")
    for i, (orig, comp) in enumerate(zip(documents, compressed_llm)):
        print(f"    文档 {i+1} 原始: {len(orig)} 字符 → 压缩后: {len(comp)} 字符")
        print(f"    压缩内容: {comp[:60]}...")

    # 句子级压缩
    compressed_sent = compressor.compress_by_sentences(query, documents, top_sentences=2)
    print(f"\n  句子级压缩结果:")
    for i, comp in enumerate(compressed_sent):
        print(f"    文档 {i+1}: {comp[:60]}...")


def demo_full_pipeline() -> None:
    """演示完整 RAG 优化管道。"""
    print("\n" + "=" * 60)
    print("5. 完整 RAG 优化管道")
    print("=" * 60)

    retriever = _build_retriever()
    pipeline = OptimizedRAGPipeline(retriever)

    query = "RAG 怎么搞"
    print(f"  🔍 查询: '{query}'")

    # 基础模式
    print(f"\n  --- 基础模式（无优化）---")
    result_basic = pipeline.run(query, use_rewrite=False, use_hyde=False, use_compression=False)
    for log in result_basic["pipeline_log"]:
        print(f"    {log}")
    print(f"    来源: {result_basic['sources'][:3]}")

    # 查询改写 + 压缩
    print(f"\n  --- 优化模式（改写 + 压缩）---")
    result_opt = pipeline.run(query, use_rewrite=True, use_hyde=False, use_compression=True)
    for log in result_opt["pipeline_log"]:
        print(f"    {log}")
    print(f"    来源: {result_opt['sources'][:3]}")

    # HyDE + 压缩
    print(f"\n  --- HyDE 模式 ---")
    result_hyde = pipeline.run(query, use_rewrite=False, use_hyde=True, use_compression=True)
    for log in result_hyde["pipeline_log"]:
        print(f"    {log}")
    print(f"    来源: {result_hyde['sources'][:3]}")


def demo_optimization_comparison() -> None:
    """对比不同优化策略的效果。"""
    print("\n" + "=" * 60)
    print("6. 优化策略效果对比")
    print("=" * 60)

    retriever = _build_retriever()
    pipeline = OptimizedRAGPipeline(retriever)

    configs = [
        ("无优化", {"use_rewrite": False, "use_hyde": False, "use_compression": False}),
        ("查询改写", {"use_rewrite": True, "use_hyde": False, "use_compression": False}),
        ("HyDE", {"use_rewrite": False, "use_hyde": True, "use_compression": False}),
        ("改写+压缩", {"use_rewrite": True, "use_hyde": False, "use_compression": True}),
        ("全部优化", {"use_rewrite": True, "use_hyde": False, "use_compression": True}),
    ]

    queries = ["RAG 怎么搞", "检索效果不好怎么办"]

    for query in queries:
        print(f"\n  🔍 查询: '{query}'")
        print(f"  {'策略':<12} {'文档数':>6} {'管道步骤':>8}")
        print(f"  {'-' * 30}")
        for name, config in configs:
            result = pipeline.run(query, **config)
            print(f"  {name:<12} {result['num_documents']:>6} {len(result['pipeline_log']):>8}")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行所有 RAG 优化演示。"""
    print("🐍 RAG 优化技巧 — 查询改写 / HyDE / 多步检索 / 上下文压缩")
    print("=" * 60)

    demo_query_rewriting()
    demo_hyde()
    demo_query_decomposition()
    demo_context_compression()
    demo_full_pipeline()
    demo_optimization_comparison()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. 优先做混合检索 + Rerank（投入产出比最高）")
    print("  2. 查询改写是低成本高回报的优化手段")
    print("  3. HyDE 适合短查询场景，假设答案与真实文档更接近")
    print("  4. 上下文压缩减少 Token 消耗，提升生成质量")
    print("  5. 查询分解适合复杂的对比类、多方面分析类问题")
    print("  6. 不要一次性加所有优化，逐步添加并评估效果")


if __name__ == "__main__":
    main()
