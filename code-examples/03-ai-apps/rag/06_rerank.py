"""
Rerank 重排序示例 — 模拟交叉编码器 / 多策略重排序

知识点：交叉编码器原理、Rerank 流程、多策略重排序、
       LLM as Reranker、Rerank 评估、与检索的配合

Python 版本：3.11+
依赖：标准库（默认模式）、sentence-transformers（本地模型模式）
最后验证：2024-12-01

外部服务（可选）：
  sentence-transformers 本地模型
  安装：pip install sentence-transformers
  模型：BAAI/bge-reranker-v2-m3
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
# 1. 数据模型
# ============================================================

@dataclass
class RerankResult:
    """重排序结果。"""
    doc_id: str
    text: str
    original_rank: int
    rerank_score: float
    new_rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================
# 2. 模拟交叉编码器
# ============================================================

class SimulatedCrossEncoder:
    """模拟交叉编码器（Cross-Encoder）。

    真实的交叉编码器将 query 和 document 拼接后送入 BERT/Transformer，
    通过全注意力机制计算相关性分数。

    这里使用关键词重叠 + 语义信号模拟交叉编码器的行为。
    """

    def __init__(self, model_name: str = "simulated-bge-reranker"):
        self.model_name = model_name
        # 语义关联词典（模拟交叉注意力捕获的语义关系）
        self._semantic_relations: dict[str, set[str]] = {
            "RAG": {"检索", "增强", "生成", "知识库", "文档"},
            "检索": {"搜索", "查询", "匹配", "召回", "向量"},
            "向量": {"Embedding", "嵌入", "维度", "相似度"},
            "数据库": {"存储", "索引", "查询", "Chroma", "Milvus", "FAISS"},
            "Rerank": {"重排序", "精排", "交叉编码器", "排序"},
            "优化": {"改进", "提升", "调优", "策略"},
            "LLM": {"大模型", "语言模型", "GPT", "生成"},
            "部署": {"上线", "服务", "推理", "API"},
            "切分": {"分块", "Chunk", "拆分", "分割"},
            "BM25": {"关键词", "词频", "TF-IDF", "稀疏"},
        }

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """预测 query-document 对的相关性分数。

        Args:
            pairs: [(query, document), ...] 查询-文档对列表

        Returns:
            相关性分数列表，范围 [0, 1]
        """
        return [self._compute_relevance(q, d) for q, d in pairs]

    def _compute_relevance(self, query: str, document: str) -> float:
        """计算查询-文档相关性（模拟交叉编码器）。"""
        score = 0.0

        # 1. 直接关键词匹配（权重 0.4）
        query_tokens = set(self._tokenize(query))
        doc_tokens = set(self._tokenize(document))
        if query_tokens:
            direct_overlap = len(query_tokens & doc_tokens) / len(query_tokens)
            score += 0.4 * direct_overlap

        # 2. 语义关联匹配（权重 0.3）— 模拟交叉注意力
        semantic_score = 0.0
        query_keywords = self._extract_keywords(query)
        doc_keywords = self._extract_keywords(document)

        for qk in query_keywords:
            related = self._semantic_relations.get(qk, set())
            for dk in doc_keywords:
                if dk in related or qk in self._semantic_relations.get(dk, set()):
                    semantic_score += 0.2
        semantic_score = min(semantic_score, 1.0)
        score += 0.3 * semantic_score

        # 3. 文档质量信号（权重 0.15）
        doc_length_score = min(len(document) / 200, 1.0)  # 适中长度得分高
        score += 0.15 * doc_length_score

        # 4. 查询覆盖度（权重 0.15）
        query_words = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+", query)
        covered = sum(1 for w in query_words if w.lower() in document.lower())
        coverage = covered / max(len(query_words), 1)
        score += 0.15 * coverage

        return min(max(score, 0.0), 1.0)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """简单分词。"""
        tokens = re.findall(r"[\u4e00-\u9fff]", text)
        tokens.extend(w.lower() for w in re.findall(r"[a-zA-Z]+", text))
        return tokens

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """提取关键词。"""
        keywords = re.findall(r"[a-zA-Z]{2,}", text)
        # 中文关键词（简单提取）
        cn_keywords = ["RAG", "检索", "向量", "数据库", "Rerank", "优化",
                        "LLM", "部署", "切分", "BM25", "Embedding", "混合"]
        for kw in cn_keywords:
            if kw in text:
                keywords.append(kw)
        return keywords


# ============================================================
# 3. Rerank 管道
# ============================================================

class RerankPipeline:
    """Rerank 重排序管道。

    支持多种重排序策略：
    - 交叉编码器重排序
    - 关键词加权重排序
    - LLM 重排序（模拟）
    - 组合重排序
    """

    def __init__(self, cross_encoder: SimulatedCrossEncoder | None = None):
        self.cross_encoder = cross_encoder or SimulatedCrossEncoder()

    def rerank_cross_encoder(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_n: int = 5,
    ) -> list[RerankResult]:
        """使用交叉编码器重排序。"""
        pairs = [(query, doc["text"]) for doc in documents]
        scores = self.cross_encoder.predict(pairs)

        results = [
            RerankResult(
                doc_id=doc.get("id", f"doc_{i}"),
                text=doc["text"],
                original_rank=i,
                rerank_score=score,
                metadata=doc.get("metadata", {}),
            )
            for i, (doc, score) in enumerate(zip(documents, scores))
        ]

        results.sort(key=lambda x: x.rerank_score, reverse=True)
        for i, r in enumerate(results):
            r.new_rank = i

        return results[:top_n]

    def rerank_keyword_boost(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_n: int = 5,
        boost_weight: float = 0.3,
    ) -> list[RerankResult]:
        """关键词加权重排序。

        在原始分数基础上，对包含查询关键词的文档加分。
        """
        query_keywords = set(re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+", query.lower()))

        results = []
        for i, doc in enumerate(documents):
            original_score = doc.get("score", 0.5)
            text_lower = doc["text"].lower()

            # 关键词匹配加分
            keyword_hits = sum(1 for kw in query_keywords if kw in text_lower)
            keyword_boost = keyword_hits / max(len(query_keywords), 1) * boost_weight

            final_score = original_score * (1 - boost_weight) + keyword_boost

            results.append(RerankResult(
                doc_id=doc.get("id", f"doc_{i}"),
                text=doc["text"],
                original_rank=i,
                rerank_score=final_score,
                metadata=doc.get("metadata", {}),
            ))

        results.sort(key=lambda x: x.rerank_score, reverse=True)
        for i, r in enumerate(results):
            r.new_rank = i

        return results[:top_n]

    def rerank_llm_simulated(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_n: int = 5,
    ) -> list[RerankResult]:
        """模拟 LLM as Reranker。

        实际中用 Prompt 让 LLM 判断文档相关性打分。
        这里用更复杂的评分逻辑模拟 LLM 的判断能力。
        """
        results = []
        for i, doc in enumerate(documents):
            # 模拟 LLM 的多维度评估
            relevance = self._llm_relevance_score(query, doc["text"])
            quality = self._llm_quality_score(doc["text"])
            specificity = self._llm_specificity_score(query, doc["text"])

            # LLM 综合评分
            final_score = 0.5 * relevance + 0.3 * quality + 0.2 * specificity

            results.append(RerankResult(
                doc_id=doc.get("id", f"doc_{i}"),
                text=doc["text"],
                original_rank=i,
                rerank_score=final_score,
                metadata={**doc.get("metadata", {}), "llm_scores": {
                    "relevance": round(relevance, 3),
                    "quality": round(quality, 3),
                    "specificity": round(specificity, 3),
                }},
            ))

        results.sort(key=lambda x: x.rerank_score, reverse=True)
        for i, r in enumerate(results):
            r.new_rank = i

        return results[:top_n]

    @staticmethod
    def _llm_relevance_score(query: str, document: str) -> float:
        """模拟 LLM 相关性评分。"""
        query_words = set(re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+", query.lower()))
        doc_words = set(re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+", document.lower()))
        overlap = len(query_words & doc_words)
        return min(overlap / max(len(query_words), 1), 1.0)

    @staticmethod
    def _llm_quality_score(document: str) -> float:
        """模拟 LLM 文档质量评分。"""
        length_score = min(len(document) / 100, 1.0)
        has_structure = 0.2 if any(c in document for c in "，。、；") else 0.0
        return min(length_score + has_structure, 1.0)

    @staticmethod
    def _llm_specificity_score(query: str, document: str) -> float:
        """模拟 LLM 具体性评分。"""
        specific_terms = re.findall(r"[A-Z][a-zA-Z]+|[\d]+", document)
        return min(len(specific_terms) / 10, 1.0)


# ============================================================
# 4. 测试数据
# ============================================================

QUERY = "如何优化 RAG 系统的检索效果？"

CANDIDATE_DOCUMENTS = [
    {"id": "d1", "text": "RAG 系统通过检索外部知识库增强 LLM 的生成能力，是 AI 应用的核心架构", "score": 0.85},
    {"id": "d2", "text": "混合检索结合向量检索和 BM25，通过 RRF 融合提升检索准确率 10-30%", "score": 0.78},
    {"id": "d3", "text": "Rerank 重排序使用交叉编码器对候选文档精排，提升检索精度 5-15%", "score": 0.75},
    {"id": "d4", "text": "查询改写用 LLM 将口语化查询优化为精确检索查询，提升检索召回率", "score": 0.72},
    {"id": "d5", "text": "HyDE 让 LLM 先生成假设答案，用假设答案的 Embedding 检索效果更好", "score": 0.70},
    {"id": "d6", "text": "Python 异步编程使用 asyncio 库实现并发操作", "score": 0.65},
    {"id": "d7", "text": "上下文压缩提取文档中与查询最相关的部分，减少 LLM 输入 Token", "score": 0.68},
    {"id": "d8", "text": "向量数据库 Chroma 适合开发阶段，Milvus 适合生产环境", "score": 0.60},
    {"id": "d9", "text": "文档切分使用 RecursiveCharacterTextSplitter 按分隔符层级递归切分", "score": 0.55},
    {"id": "d10", "text": "今天天气很好，适合出去散步", "score": 0.30},
]


# ============================================================
# 5. 演示函数
# ============================================================

def demo_cross_encoder_rerank() -> None:
    """演示交叉编码器重排序。"""
    print("\n" + "=" * 60)
    print("1. 交叉编码器重排序")
    print("=" * 60)

    pipeline = RerankPipeline()
    results = pipeline.rerank_cross_encoder(QUERY, CANDIDATE_DOCUMENTS, top_n=5)

    print(f"  🔍 查询: '{QUERY}'")
    print(f"  候选文档: {len(CANDIDATE_DOCUMENTS)} 个 → 精排 Top-5")
    print()
    for r in results:
        rank_change = r.original_rank - r.new_rank
        arrow = "↑" if rank_change > 0 else ("↓" if rank_change < 0 else "→")
        print(f"  #{r.new_rank+1} (原#{r.original_rank+1} {arrow}) "
              f"score={r.rerank_score:.4f} | {r.text[:45]}...")


def demo_keyword_boost_rerank() -> None:
    """演示关键词加权重排序。"""
    print("\n" + "=" * 60)
    print("2. 关键词加权重排序")
    print("=" * 60)

    pipeline = RerankPipeline()
    results = pipeline.rerank_keyword_boost(QUERY, CANDIDATE_DOCUMENTS, top_n=5)

    print(f"  🔍 查询: '{QUERY}'")
    for r in results:
        print(f"  #{r.new_rank+1} score={r.rerank_score:.4f} | {r.text[:45]}...")


def demo_llm_rerank() -> None:
    """演示 LLM as Reranker。"""
    print("\n" + "=" * 60)
    print("3. LLM as Reranker（模拟）")
    print("=" * 60)

    pipeline = RerankPipeline()
    results = pipeline.rerank_llm_simulated(QUERY, CANDIDATE_DOCUMENTS, top_n=5)

    print(f"  🔍 查询: '{QUERY}'")
    for r in results:
        llm_scores = r.metadata.get("llm_scores", {})
        print(f"  #{r.new_rank+1} score={r.rerank_score:.4f} "
              f"(rel={llm_scores.get('relevance', 0):.2f}, "
              f"qual={llm_scores.get('quality', 0):.2f}, "
              f"spec={llm_scores.get('specificity', 0):.2f}) "
              f"| {r.text[:35]}...")


def demo_rerank_comparison() -> None:
    """对比不同重排序策略。"""
    print("\n" + "=" * 60)
    print("4. 重排序策略对比")
    print("=" * 60)

    pipeline = RerankPipeline()

    strategies = {
        "原始排序": None,
        "交叉编码器": pipeline.rerank_cross_encoder(QUERY, CANDIDATE_DOCUMENTS, top_n=5),
        "关键词加权": pipeline.rerank_keyword_boost(QUERY, CANDIDATE_DOCUMENTS, top_n=5),
        "LLM Rerank": pipeline.rerank_llm_simulated(QUERY, CANDIDATE_DOCUMENTS, top_n=5),
    }

    print(f"  🔍 查询: '{QUERY}'")
    print(f"\n  {'策略':<14} {'Top-1':>40} {'Top-2':>40}")
    print(f"  {'-' * 98}")

    # 原始排序
    top1 = CANDIDATE_DOCUMENTS[0]["text"][:35]
    top2 = CANDIDATE_DOCUMENTS[1]["text"][:35]
    print(f"  {'原始排序':<14} {top1:>40} {top2:>40}")

    for name, results in strategies.items():
        if results is None:
            continue
        top1 = results[0].text[:35] if len(results) > 0 else "N/A"
        top2 = results[1].text[:35] if len(results) > 1 else "N/A"
        print(f"  {name:<14} {top1:>40} {top2:>40}")


def demo_rerank_with_threshold() -> None:
    """演示带阈值的重排序。"""
    print("\n" + "=" * 60)
    print("5. 带阈值过滤的重排序")
    print("=" * 60)

    pipeline = RerankPipeline()
    all_results = pipeline.rerank_cross_encoder(QUERY, CANDIDATE_DOCUMENTS, top_n=10)

    thresholds = [0.3, 0.4, 0.5]
    for threshold in thresholds:
        filtered = [r for r in all_results if r.rerank_score >= threshold]
        print(f"\n  阈值 >= {threshold}: 保留 {len(filtered)}/{len(all_results)} 个文档")
        for r in filtered[:3]:
            print(f"    score={r.rerank_score:.4f} | {r.text[:50]}...")


def demo_rerank_model_comparison() -> None:
    """演示 Rerank 模型选型对比。"""
    print("\n" + "=" * 60)
    print("6. Rerank 模型选型对比")
    print("=" * 60)

    models = [
        {"name": "Cohere Rerank v3", "type": "API", "chinese": "良好",
         "cost": "$1/1K次", "latency": "100ms", "accuracy": "优秀"},
        {"name": "BGE-Reranker-v2-m3", "type": "本地", "chinese": "优秀",
         "cost": "免费", "latency": "50ms", "accuracy": "优秀"},
        {"name": "BGE-Reranker-large", "type": "本地", "chinese": "优秀",
         "cost": "免费", "latency": "30ms", "accuracy": "良好"},
        {"name": "Jina Reranker v2", "type": "API/本地", "chinese": "良好",
         "cost": "免费/付费", "latency": "80ms", "accuracy": "良好"},
        {"name": "LLM as Reranker", "type": "API/本地", "chinese": "优秀",
         "cost": "较高", "latency": "500ms", "accuracy": "最佳"},
    ]

    print(f"  {'模型':<25} {'类型':<10} {'中文':>6} {'成本':>10} {'延迟':>8} {'效果':>6}")
    print(f"  {'-' * 70}")
    for m in models:
        print(f"  {m['name']:<25} {m['type']:<10} {m['chinese']:>6} "
              f"{m['cost']:>10} {m['latency']:>8} {m['accuracy']:>6}")

    print("\n  💡 中文场景推荐: BGE-Reranker-v2-m3（本地部署，免费，中文效果最佳）")


# ============================================================
# 主入口
# ============================================================

def main(server_mode: bool = False) -> None:
    """运行所有 Rerank 演示。"""
    print("🐍 Rerank 重排序示例 — 交叉编码器 / 多策略重排序")
    print("=" * 60)

    demo_cross_encoder_rerank()
    demo_keyword_boost_rerank()
    demo_llm_rerank()
    demo_rerank_comparison()
    demo_rerank_with_threshold()
    demo_rerank_model_comparison()

    if server_mode:
        print("\n" + "=" * 60)
        print("7. 服务模式 — 使用真实 Rerank 模型")
        print("=" * 60)
        print("  💡 安装: pip install sentence-transformers")
        print("  💡 使用:")
        print("     from sentence_transformers import CrossEncoder")
        print("     model = CrossEncoder('BAAI/bge-reranker-v2-m3')")
        print("     scores = model.predict([['查询', '文档1'], ['查询', '文档2']])")

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. Rerank 是 RAG 的精排阶段，通常提升 5-15% 准确率")
    print("  2. 交叉编码器比双塔模型精度高，但速度慢")
    print("  3. 候选数量控制在 20-50，Rerank 后取 Top-3~5")
    print("  4. 中文场景推荐 BGE-Reranker-v2-m3")
    print("  5. 设置相关性阈值过滤不相关文档")
    print("  6. Rerank 不能替代好的初步检索")

    if not server_mode:
        print("\n💡 要使用真实模型: python 06_rerank.py server")


if __name__ == "__main__":
    is_server = len(sys.argv) > 1 and sys.argv[1].lower() == "server"
    main(server_mode=is_server)
