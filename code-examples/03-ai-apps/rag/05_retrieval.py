"""
检索策略示例 — 相似度检索 / MMR / 混合检索 / BM25

知识点：向量相似度检索、BM25 稀疏检索、混合检索（RRF 融合）、
       MMR 多样性检索、检索评估指标、检索策略对比

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
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


# ============================================================
# 1. 向量运算 + Embedding 工具
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
    """简单 Embedding 模型（确定性哈希 + 关键词信号）。"""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        vec = [(b - 128) / 128.0 for b in h]
        while len(vec) < self.dim:
            h = hashlib.sha256(h).digest()
            vec.extend((b - 128) / 128.0 for b in h)
        vec = vec[:self.dim]
        kw_map = {"RAG": 0, "检索": 1, "向量": 2, "数据库": 3, "LLM": 4,
                   "Embedding": 5, "切分": 6, "Rerank": 7, "BM25": 8, "混合": 9}
        for kw, idx in kw_map.items():
            if kw in text and idx < self.dim:
                vec[idx] += 0.5
        return VectorOps.normalize(vec)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


# ============================================================
# 2. 搜索结果数据模型
# ============================================================

@dataclass
class SearchResult:
    """搜索结果。"""
    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""  # 来源：vector / bm25 / hybrid / mmr


# ============================================================
# 3. 向量相似度检索
# ============================================================

class VectorRetriever:
    """向量相似度检索器。

    基于 Embedding 余弦相似度的语义检索。
    优势：理解语义，"部署"和"上线"能匹配。
    劣势：对精确关键词不敏感。
    """

    def __init__(self, embedding_model: SimpleEmbedding | None = None):
        self.model = embedding_model or SimpleEmbedding()
        self.documents: list[str] = []
        self.doc_ids: list[str] = []
        self.embeddings: list[list[float]] = []
        self.metadatas: list[dict[str, Any]] = []

    def add_documents(
        self,
        documents: list[str],
        doc_ids: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """添加文档。"""
        if doc_ids is None:
            doc_ids = [f"doc_{len(self.documents) + i}" for i in range(len(documents))]
        if metadatas is None:
            metadatas = [{} for _ in documents]

        embeddings = self.model.embed_batch(documents)
        self.documents.extend(documents)
        self.doc_ids.extend(doc_ids)
        self.embeddings.extend(embeddings)
        self.metadatas.extend(metadatas)

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """向量相似度搜索。"""
        query_emb = self.model.embed(query)
        scored = []
        for i, doc_emb in enumerate(self.embeddings):
            sim = VectorOps.cosine_similarity(query_emb, doc_emb)
            scored.append((i, sim))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            SearchResult(
                doc_id=self.doc_ids[idx],
                text=self.documents[idx],
                score=score,
                metadata=self.metadatas[idx],
                source="vector",
            )
            for idx, score in scored[:k]
        ]


# ============================================================
# 4. BM25 稀疏检索
# ============================================================

class BM25Retriever:
    """BM25 关键词检索器。

    基于词频和逆文档频率的经典检索算法。
    优势：精确关键词匹配，速度快。
    劣势：不理解语义。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1  # 词频饱和参数
        self.b = b     # 文档长度归一化参数
        self.documents: list[str] = []
        self.doc_ids: list[str] = []
        self.metadatas: list[dict[str, Any]] = []
        self._tokenized_docs: list[list[str]] = []
        self._doc_freqs: Counter[str] = Counter()
        self._avg_doc_len: float = 0.0

    def add_documents(
        self,
        documents: list[str],
        doc_ids: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """添加文档。"""
        if doc_ids is None:
            doc_ids = [f"doc_{len(self.documents) + i}" for i in range(len(documents))]
        if metadatas is None:
            metadatas = [{} for _ in documents]

        self.documents.extend(documents)
        self.doc_ids.extend(doc_ids)
        self.metadatas.extend(metadatas)

        for doc in documents:
            tokens = self._tokenize(doc)
            self._tokenized_docs.append(tokens)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self._doc_freqs[token] += 1

        total_len = sum(len(t) for t in self._tokenized_docs)
        self._avg_doc_len = total_len / max(len(self._tokenized_docs), 1)

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """BM25 搜索。"""
        query_tokens = self._tokenize(query)
        n_docs = len(self._tokenized_docs)
        scores: list[tuple[int, float]] = []

        for i, doc_tokens in enumerate(self._tokenized_docs):
            score = 0.0
            doc_len = len(doc_tokens)
            token_counts = Counter(doc_tokens)

            for qt in query_tokens:
                if qt not in token_counts:
                    continue
                tf = token_counts[qt]
                df = self._doc_freqs.get(qt, 0)
                # IDF
                idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
                # TF 归一化
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_len / max(self._avg_doc_len, 1))
                )
                score += idf * tf_norm

            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        return [
            SearchResult(
                doc_id=self.doc_ids[idx],
                text=self.documents[idx],
                score=score,
                metadata=self.metadatas[idx],
                source="bm25",
            )
            for idx, score in scores[:k]
        ]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """简单中文分词（按字符 + 英文单词）。"""
        tokens: list[str] = []
        # 提取英文单词
        english_words = re.findall(r"[a-zA-Z]+", text)
        tokens.extend(w.lower() for w in english_words)
        # 中文按字符分词（简化版，生产用 jieba）
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
        # 按 bigram 分词
        for i in range(len(chinese_chars) - 1):
            tokens.append(chinese_chars[i] + chinese_chars[i + 1])
        tokens.extend(chinese_chars)
        return tokens


# ============================================================
# 5. 混合检索（RRF 融合）
# ============================================================

class HybridRetriever:
    """混合检索器：向量检索 + BM25 + RRF 融合。

    结合语义理解和精确匹配的优势，是生产环境的最佳实践。
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
        vector_weight: float = 0.6,
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k
        self.vector_weight = vector_weight

    def search(self, query: str, k: int = 5, fetch_k: int = 20) -> list[SearchResult]:
        """混合检索 + RRF 融合。"""
        # 两路检索
        vector_results = self.vector_retriever.search(query, k=fetch_k)
        bm25_results = self.bm25_retriever.search(query, k=fetch_k)

        # RRF 融合
        fused_scores: dict[str, float] = {}
        doc_map: dict[str, SearchResult] = {}

        for rank, result in enumerate(vector_results):
            rrf_score = self.vector_weight / (self.rrf_k + rank + 1)
            fused_scores[result.doc_id] = fused_scores.get(result.doc_id, 0) + rrf_score
            doc_map[result.doc_id] = result

        bm25_weight = 1 - self.vector_weight
        for rank, result in enumerate(bm25_results):
            rrf_score = bm25_weight / (self.rrf_k + rank + 1)
            fused_scores[result.doc_id] = fused_scores.get(result.doc_id, 0) + rrf_score
            if result.doc_id not in doc_map:
                doc_map[result.doc_id] = result

        # 按融合分数排序
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

        return [
            SearchResult(
                doc_id=doc_id,
                text=doc_map[doc_id].text,
                score=fused_scores[doc_id],
                metadata=doc_map[doc_id].metadata,
                source="hybrid",
            )
            for doc_id in sorted_ids[:k]
        ]


# ============================================================
# 6. MMR 多样性检索
# ============================================================

class MMRRetriever:
    """MMR（Maximal Marginal Relevance）多样性检索器。

    在相关性和多样性之间取得平衡，避免返回高度重复的结果。
    """

    def __init__(self, vector_retriever: VectorRetriever, lambda_mult: float = 0.5):
        self.vector_retriever = vector_retriever
        self.lambda_mult = lambda_mult  # 0=最大多样性, 1=最大相关性

    def search(self, query: str, k: int = 5, fetch_k: int = 20) -> list[SearchResult]:
        """MMR 搜索。"""
        # 先获取候选集
        candidates = self.vector_retriever.search(query, k=fetch_k)
        if not candidates:
            return []

        query_emb = self.vector_retriever.model.embed(query)
        candidate_embs = [
            self.vector_retriever.model.embed(c.text) for c in candidates
        ]

        selected: list[int] = []
        remaining = list(range(len(candidates)))

        for _ in range(min(k, len(candidates))):
            best_idx = -1
            best_score = -float("inf")

            for idx in remaining:
                # 与查询的相关性
                relevance = VectorOps.cosine_similarity(query_emb, candidate_embs[idx])

                # 与已选文档的最大相似度
                max_sim_to_selected = 0.0
                for sel_idx in selected:
                    sim = VectorOps.cosine_similarity(candidate_embs[idx], candidate_embs[sel_idx])
                    max_sim_to_selected = max(max_sim_to_selected, sim)

                # MMR 分数
                mmr_score = (
                    self.lambda_mult * relevance
                    - (1 - self.lambda_mult) * max_sim_to_selected
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                remaining.remove(best_idx)

        return [
            SearchResult(
                doc_id=candidates[idx].doc_id,
                text=candidates[idx].text,
                score=candidates[idx].score,
                metadata=candidates[idx].metadata,
                source="mmr",
            )
            for idx in selected
        ]


# ============================================================
# 7. 检索评估工具
# ============================================================

class RetrievalEvaluator:
    """检索效果评估工具。"""

    @staticmethod
    def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
        """Recall@K：Top-K 中包含多少相关文档。"""
        retrieved_set = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        return len(retrieved_set & relevant_set) / max(len(relevant_set), 1)

    @staticmethod
    def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
        """MRR（Mean Reciprocal Rank）：第一个相关文档的排名倒数。"""
        relevant_set = set(relevant_ids)
        for rank, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                return 1.0 / (rank + 1)
        return 0.0

    @staticmethod
    def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
        """Precision@K：Top-K 中相关文档的比例。"""
        retrieved_set = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        return len(retrieved_set & relevant_set) / max(k, 1)


# ============================================================
# 8. 测试数据
# ============================================================

KNOWLEDGE_BASE = [
    ("doc_0", "RAG 系统通过检索外部知识库增强 LLM 的生成能力，减少幻觉", {"topic": "RAG"}),
    ("doc_1", "向量数据库 Chroma 适合开发阶段，支持内存和持久化模式", {"topic": "vector_db"}),
    ("doc_2", "BM25 是基于词频的经典检索算法，擅长精确关键词匹配", {"topic": "retrieval"}),
    ("doc_3", "混合检索结合向量检索和 BM25，通过 RRF 融合两路结果", {"topic": "retrieval"}),
    ("doc_4", "Embedding 模型将文本转为高维向量，BGE 是中文推荐模型", {"topic": "embedding"}),
    ("doc_5", "Rerank 重排序使用交叉编码器对检索结果精排，提升 5-15% 准确率", {"topic": "rerank"}),
    ("doc_6", "文档切分使用 RecursiveCharacterTextSplitter 按分隔符层级递归", {"topic": "splitting"}),
    ("doc_7", "FAISS 是 Meta 开源的高性能向量搜索库，纯内存操作速度最快", {"topic": "vector_db"}),
    ("doc_8", "Milvus 支持分布式部署和百亿级数据，适合企业级生产环境", {"topic": "vector_db"}),
    ("doc_9", "MMR 在相关性和多样性之间取得平衡，避免返回重复结果", {"topic": "retrieval"}),
    ("doc_10", "查询改写用 LLM 将口语化查询优化为精确检索查询", {"topic": "optimization"}),
    ("doc_11", "HyDE 让 LLM 先生成假设答案，用假设答案的 Embedding 检索", {"topic": "optimization"}),
]


# ============================================================
# 9. 演示函数
# ============================================================

def _build_retrievers() -> tuple[VectorRetriever, BM25Retriever]:
    """构建检索器并加载数据。"""
    model = SimpleEmbedding(dim=64)
    vector_ret = VectorRetriever(model)
    bm25_ret = BM25Retriever()

    docs = [text for _, text, _ in KNOWLEDGE_BASE]
    ids = [doc_id for doc_id, _, _ in KNOWLEDGE_BASE]
    metas = [meta for _, _, meta in KNOWLEDGE_BASE]

    vector_ret.add_documents(docs, ids, metas)
    bm25_ret.add_documents(docs, ids, metas)

    return vector_ret, bm25_ret


def demo_vector_search() -> None:
    """演示向量相似度检索。"""
    print("\n" + "=" * 60)
    print("1. 向量相似度检索")
    print("=" * 60)

    vector_ret, _ = _build_retrievers()
    query = "如何选择向量数据库？"
    results = vector_ret.search(query, k=5)

    print(f"  🔍 查询: '{query}'")
    for i, r in enumerate(results):
        print(f"    Top-{i+1}: (score={r.score:.4f}) {r.text[:50]}...")


def demo_bm25_search() -> None:
    """演示 BM25 检索。"""
    print("\n" + "=" * 60)
    print("2. BM25 关键词检索")
    print("=" * 60)

    _, bm25_ret = _build_retrievers()
    query = "BM25 检索算法"
    results = bm25_ret.search(query, k=5)

    print(f"  🔍 查询: '{query}'")
    for i, r in enumerate(results):
        print(f"    Top-{i+1}: (score={r.score:.4f}) {r.text[:50]}...")


def demo_hybrid_search() -> None:
    """演示混合检索。"""
    print("\n" + "=" * 60)
    print("3. 混合检索（向量 + BM25 + RRF）⭐ 生产推荐")
    print("=" * 60)

    vector_ret, bm25_ret = _build_retrievers()
    hybrid_ret = HybridRetriever(vector_ret, bm25_ret, vector_weight=0.6)

    query = "RAG 检索优化方法"
    results = hybrid_ret.search(query, k=5)

    print(f"  🔍 查询: '{query}'")
    print(f"  融合权重: 向量=0.6, BM25=0.4, RRF k=60")
    for i, r in enumerate(results):
        print(f"    Top-{i+1}: (score={r.score:.6f}) {r.text[:50]}...")


def demo_mmr_search() -> None:
    """演示 MMR 多样性检索。"""
    print("\n" + "=" * 60)
    print("4. MMR 多样性检索")
    print("=" * 60)

    vector_ret, _ = _build_retrievers()

    query = "向量数据库"
    print(f"  🔍 查询: '{query}'")

    # 普通检索
    normal_results = vector_ret.search(query, k=5)
    print(f"\n  普通检索 Top-5:")
    for i, r in enumerate(normal_results):
        print(f"    {i+1}. {r.text[:50]}...")

    # MMR 检索（lambda=0.5）
    mmr_ret = MMRRetriever(vector_ret, lambda_mult=0.5)
    mmr_results = mmr_ret.search(query, k=5, fetch_k=10)
    print(f"\n  MMR 检索 Top-5 (lambda=0.5):")
    for i, r in enumerate(mmr_results):
        print(f"    {i+1}. {r.text[:50]}...")


def demo_strategy_comparison() -> None:
    """对比不同检索策略。"""
    print("\n" + "=" * 60)
    print("5. 检索策略对比")
    print("=" * 60)

    vector_ret, bm25_ret = _build_retrievers()
    hybrid_ret = HybridRetriever(vector_ret, bm25_ret)
    mmr_ret = MMRRetriever(vector_ret, lambda_mult=0.5)

    queries = ["RAG 系统设计", "BM25 关键词匹配", "向量数据库选型"]

    for query in queries:
        print(f"\n  🔍 查询: '{query}'")
        strategies = {
            "向量检索": vector_ret.search(query, k=3),
            "BM25": bm25_ret.search(query, k=3),
            "混合检索": hybrid_ret.search(query, k=3),
            "MMR": mmr_ret.search(query, k=3),
        }
        for name, results in strategies.items():
            top1 = results[0].text[:40] if results else "无结果"
            print(f"    {name:<8}: {top1}...")


def demo_retrieval_evaluation() -> None:
    """演示检索评估。"""
    print("\n" + "=" * 60)
    print("6. 检索效果评估")
    print("=" * 60)

    vector_ret, bm25_ret = _build_retrievers()
    hybrid_ret = HybridRetriever(vector_ret, bm25_ret)
    evaluator = RetrievalEvaluator()

    # 评估数据：查询 → 相关文档 ID
    eval_data = [
        ("向量数据库选型", ["doc_1", "doc_7", "doc_8"]),
        ("检索策略对比", ["doc_2", "doc_3", "doc_9"]),
        ("RAG 优化技巧", ["doc_10", "doc_11", "doc_5"]),
    ]

    print(f"  {'查询':<16} {'策略':<10} {'Recall@5':>10} {'MRR':>8} {'P@3':>8}")
    print(f"  {'-' * 56}")

    for query, relevant_ids in eval_data:
        for name, retriever in [("向量", vector_ret), ("BM25", bm25_ret), ("混合", hybrid_ret)]:
            results = retriever.search(query, k=5)
            retrieved_ids = [r.doc_id for r in results]
            recall = evaluator.recall_at_k(retrieved_ids, relevant_ids, 5)
            mrr = evaluator.mrr(retrieved_ids, relevant_ids)
            precision = evaluator.precision_at_k(retrieved_ids, relevant_ids, 3)
            print(f"  {query:<16} {name:<10} {recall:>10.2f} {mrr:>8.2f} {precision:>8.2f}")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行所有检索策略演示。"""
    print("🐍 检索策略示例 — 相似度 / MMR / 混合检索 / BM25")
    print("=" * 60)

    demo_vector_search()
    demo_bm25_search()
    demo_hybrid_search()
    demo_mmr_search()
    demo_strategy_comparison()
    demo_retrieval_evaluation()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. 生产环境用混合检索（向量 + BM25 + RRF 融合）")
    print("  2. MMR 用于需要多样性的场景，lambda=0.5 是好的起点")
    print("  3. BM25 中文需要分词（jieba），英文按空格分词")
    print("  4. 先检索 20-50 个候选，再通过 Rerank 精排到 Top-5")
    print("  5. 用 Recall@K、MRR、Precision@K 评估检索效果")
    print("  6. 不同场景最优策略不同，需要 A/B 测试验证")


if __name__ == "__main__":
    main()
