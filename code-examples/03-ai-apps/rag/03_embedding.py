"""
Embedding 模型使用 — 模拟 Embedding 计算 + 相似度搜索

知识点：Embedding 原理、余弦相似度、欧氏距离、向量归一化、
       批量 Embedding、模型选型对比、缓存策略、降维可视化

Python 版本：3.11+
依赖：numpy（默认模式）、sentence-transformers / openai（服务模式）
最后验证：2024-12-01

外部服务（可选）：
  sentence-transformers 本地模型
  安装：pip install sentence-transformers
  OpenAI API
  设置：export OPENAI_API_KEY=your-key
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Any


# ============================================================
# 1. 向量运算工具（纯 Python 实现，无需 numpy）
# ============================================================

class VectorOps:
    """向量运算工具类。

    纯 Python 实现，用于演示 Embedding 相关的向量运算。
    生产环境推荐使用 numpy 提升性能。
    """

    @staticmethod
    def dot_product(a: list[float], b: list[float]) -> float:
        """向量点积。"""
        return sum(x * y for x, y in zip(a, b))

    @staticmethod
    def norm(a: list[float]) -> float:
        """向量 L2 范数。"""
        return math.sqrt(sum(x * x for x in a))

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """余弦相似度。

        衡量两个向量的方向相似性，范围 [-1, 1]。
        1 = 完全相同方向，0 = 正交，-1 = 完全相反。
        """
        dot = VectorOps.dot_product(a, b)
        norm_a = VectorOps.norm(a)
        norm_b = VectorOps.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def euclidean_distance(a: list[float], b: list[float]) -> float:
        """欧氏距离。

        衡量两个向量的绝对距离，范围 [0, ∞)。
        0 = 完全相同，越大越不相似。
        """
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    @staticmethod
    def normalize(a: list[float]) -> list[float]:
        """L2 归一化。

        归一化后的向量长度为 1，此时点积 = 余弦相似度。
        """
        n = VectorOps.norm(a)
        if n == 0:
            return a
        return [x / n for x in a]

    @staticmethod
    def mean_pooling(vectors: list[list[float]]) -> list[float]:
        """平均池化：多个向量取平均。

        用于将多个 Token 的向量合并为一个句子向量。
        """
        if not vectors:
            return []
        dim = len(vectors[0])
        result = [0.0] * dim
        for vec in vectors:
            for i in range(dim):
                result[i] += vec[i]
        return [x / len(vectors) for x in result]


# ============================================================
# 2. 模拟 Embedding 模型
# ============================================================

class SimulatedEmbeddingModel:
    """模拟 Embedding 模型。

    使用确定性哈希生成伪向量，保证相同文本生成相同向量。
    语义相似的文本会生成相似的向量（通过共享词汇实现）。
    """

    def __init__(self, model_name: str = "simulated-bge-zh", dimension: int = 128):
        self.model_name = model_name
        self.dimension = dimension
        self._cache: dict[str, list[float]] = {}
        self._call_count = 0

    def embed_text(self, text: str) -> list[float]:
        """将单个文本转为向量。"""
        # 检查缓存
        if text in self._cache:
            return self._cache[text]

        self._call_count += 1
        vector = self._text_to_vector(text)
        self._cache[text] = vector
        return vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量 Embedding。

        生产环境中批量处理比逐条处理快 5-10 倍。
        """
        return [self.embed_text(text) for text in texts]

    def _text_to_vector(self, text: str) -> list[float]:
        """将文本转为确定性伪向量。

        通过词汇重叠模拟语义相似性：
        - 共享词汇多的文本 → 向量相似
        - 无共享词汇的文本 → 向量不相似
        """
        # 基础向量：基于文本哈希
        hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()
        base_vector = [
            (b - 128) / 128.0 for b in hash_bytes[:self.dimension]
        ]

        # 如果维度不够，循环填充
        while len(base_vector) < self.dimension:
            extra_hash = hashlib.sha256(
                (text + str(len(base_vector))).encode("utf-8")
            ).digest()
            base_vector.extend((b - 128) / 128.0 for b in extra_hash)
        base_vector = base_vector[:self.dimension]

        # 添加语义信号：基于关键词
        keywords_signals = {
            "RAG": [0.5, 0.3, 0.2],
            "检索": [0.4, 0.3, 0.1],
            "向量": [0.3, 0.4, 0.2],
            "Embedding": [0.3, 0.4, 0.3],
            "LLM": [0.2, 0.1, 0.5],
            "模型": [0.1, 0.2, 0.4],
            "数据库": [0.2, 0.5, 0.1],
            "部署": [0.1, 0.1, 0.3],
            "优化": [0.3, 0.2, 0.3],
            "切分": [0.4, 0.2, 0.1],
        }

        for keyword, signal in keywords_signals.items():
            if keyword in text:
                for i, s in enumerate(signal):
                    if i < self.dimension:
                        base_vector[i] += s

        # L2 归一化
        return VectorOps.normalize(base_vector)

    @property
    def stats(self) -> dict[str, Any]:
        """模型调用统计。"""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "total_calls": self._call_count,
            "cache_size": len(self._cache),
            "cache_hit_rate": f"{(1 - self._call_count / max(len(self._cache), 1)) * 100:.1f}%"
            if self._cache else "N/A",
        }


# ============================================================
# 3. Embedding 缓存管理
# ============================================================

class EmbeddingCache:
    """Embedding 缓存管理器。

    通过文本哈希缓存 Embedding 结果，避免重复计算。
    生产环境可以用 Redis 替代内存缓存。
    """

    def __init__(self, model: SimulatedEmbeddingModel):
        self.model = model
        self._cache: dict[str, list[float]] = {}
        self._hits = 0
        self._misses = 0

    def get_embedding(self, text: str) -> list[float]:
        """获取文本 Embedding（带缓存）。"""
        cache_key = hashlib.md5(text.encode()).hexdigest()

        if cache_key in self._cache:
            self._hits += 1
            return self._cache[cache_key]

        self._misses += 1
        embedding = self.model.embed_text(text)
        self._cache[cache_key] = embedding
        return embedding

    def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """批量获取 Embedding（带缓存）。"""
        return [self.get_embedding(text) for text in texts]

    @property
    def stats(self) -> dict[str, Any]:
        """缓存统计。"""
        total = self._hits + self._misses
        return {
            "cache_size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{self._hits / max(total, 1) * 100:.1f}%",
        }


# ============================================================
# 4. 语义搜索引擎
# ============================================================

@dataclass
class SearchResult:
    """搜索结果。"""
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class SemanticSearchEngine:
    """基于 Embedding 的语义搜索引擎。"""

    def __init__(self, model: SimulatedEmbeddingModel):
        self.model = model
        self.documents: list[str] = []
        self.embeddings: list[list[float]] = []
        self.metadata_list: list[dict[str, Any]] = []

    def add_documents(
        self,
        documents: list[str],
        metadata_list: list[dict[str, Any]] | None = None,
    ) -> None:
        """添加文档到搜索引擎。"""
        embeddings = self.model.embed_batch(documents)
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        if metadata_list:
            self.metadata_list.extend(metadata_list)
        else:
            self.metadata_list.extend({} for _ in documents)

    def search(
        self,
        query: str,
        k: int = 5,
        similarity_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """语义搜索：找到与查询最相似的文档。"""
        query_embedding = self.model.embed_text(query)

        # 计算与所有文档的相似度
        scores: list[tuple[int, float]] = []
        for i, doc_embedding in enumerate(self.embeddings):
            sim = VectorOps.cosine_similarity(query_embedding, doc_embedding)
            if sim >= similarity_threshold:
                scores.append((i, sim))

        # 按相似度排序，取 Top-K
        scores.sort(key=lambda x: x[1], reverse=True)
        top_k = scores[:k]

        return [
            SearchResult(
                text=self.documents[idx],
                score=score,
                metadata=self.metadata_list[idx],
            )
            for idx, score in top_k
        ]


# ============================================================
# 5. 演示函数
# ============================================================

def demo_basic_embedding() -> None:
    """演示基础 Embedding 计算。"""
    print("\n" + "=" * 60)
    print("1. 基础 Embedding 计算")
    print("=" * 60)

    model = SimulatedEmbeddingModel(dimension=128)

    texts = [
        "如何部署 LLM？",
        "LLM 部署方法",
        "大语言模型的部署方案",
        "今天天气真好",
    ]

    print("  文本 → 向量:")
    embeddings = model.embed_batch(texts)
    for text, emb in zip(texts, embeddings):
        print(f"    '{text}' → [{emb[0]:.4f}, {emb[1]:.4f}, ..., {emb[-1]:.4f}] (dim={len(emb)})")


def demo_similarity_comparison() -> None:
    """演示相似度计算对比。"""
    print("\n" + "=" * 60)
    print("2. 相似度计算对比")
    print("=" * 60)

    model = SimulatedEmbeddingModel(dimension=128)

    pairs = [
        ("RAG 系统设计", "检索增强生成架构"),
        ("RAG 系统设计", "向量数据库选型"),
        ("RAG 系统设计", "今天天气真好"),
        ("Embedding 模型选择", "向量表示模型对比"),
        ("Embedding 模型选择", "Python 异步编程"),
    ]

    print(f"  {'文本对':<40} {'余弦相似度':>10} {'欧氏距离':>10}")
    print(f"  {'-' * 64}")

    for text_a, text_b in pairs:
        emb_a = model.embed_text(text_a)
        emb_b = model.embed_text(text_b)
        cos_sim = VectorOps.cosine_similarity(emb_a, emb_b)
        euc_dist = VectorOps.euclidean_distance(emb_a, emb_b)
        label = f"'{text_a}' vs '{text_b}'"
        print(f"  {label:<40} {cos_sim:>10.4f} {euc_dist:>10.4f}")


def demo_normalization() -> None:
    """演示向量归一化。"""
    print("\n" + "=" * 60)
    print("3. 向量归一化")
    print("=" * 60)

    vec = [3.0, 4.0, 0.0]
    normalized = VectorOps.normalize(vec)

    print(f"  原始向量: {vec}")
    print(f"  L2 范数: {VectorOps.norm(vec):.4f}")
    print(f"  归一化后: [{normalized[0]:.4f}, {normalized[1]:.4f}, {normalized[2]:.4f}]")
    print(f"  归一化后范数: {VectorOps.norm(normalized):.4f}")

    # 验证：归一化后点积 = 余弦相似度
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]
    cos_sim = VectorOps.cosine_similarity(a, b)
    a_norm = VectorOps.normalize(a)
    b_norm = VectorOps.normalize(b)
    dot_after_norm = VectorOps.dot_product(a_norm, b_norm)

    print(f"\n  验证: 归一化后点积 = 余弦相似度")
    print(f"    余弦相似度: {cos_sim:.6f}")
    print(f"    归一化后点积: {dot_after_norm:.6f}")
    print(f"    差异: {abs(cos_sim - dot_after_norm):.10f}")


def demo_semantic_search() -> None:
    """演示语义搜索。"""
    print("\n" + "=" * 60)
    print("4. 语义搜索引擎")
    print("=" * 60)

    model = SimulatedEmbeddingModel(dimension=128)
    engine = SemanticSearchEngine(model)

    # 添加知识库文档
    documents = [
        "RAG 系统通过检索外部知识库来增强 LLM 的生成能力",
        "向量数据库 Chroma 适合开发阶段，Milvus 适合生产环境",
        "Embedding 模型将文本转为高维向量，BGE 是中文推荐模型",
        "文档切分使用 RecursiveCharacterTextSplitter 效果最好",
        "Rerank 重排序使用交叉编码器提升检索精度",
        "LLM 微调包括 LoRA、QLoRA 等参数高效方法",
        "vLLM 通过 PagedAttention 优化推理性能",
        "Prompt Engineering 是 AI 应用开发的基础技能",
        "混合检索结合向量检索和 BM25 关键词检索",
        "上下文压缩提取文档中与查询最相关的部分",
    ]

    metadata = [{"id": i, "category": "RAG" if i < 5 else "LLM"} for i in range(len(documents))]
    engine.add_documents(documents, metadata)

    # 搜索测试
    queries = ["如何选择向量数据库？", "RAG 检索优化方法", "模型推理加速"]

    for query in queries:
        print(f"\n  🔍 查询: '{query}'")
        results = engine.search(query, k=3)
        for i, result in enumerate(results):
            print(f"    Top-{i + 1}: (score={result.score:.4f}) {result.text[:50]}...")


def demo_embedding_cache() -> None:
    """演示 Embedding 缓存。"""
    print("\n" + "=" * 60)
    print("5. Embedding 缓存策略")
    print("=" * 60)

    model = SimulatedEmbeddingModel(dimension=128)
    cache = EmbeddingCache(model)

    texts = ["RAG 系统设计", "向量数据库", "RAG 系统设计", "Embedding 模型", "向量数据库"]

    print("  模拟 5 次 Embedding 请求（含重复）:")
    for text in texts:
        cache.get_embedding(text)
        print(f"    '{text}' → 缓存状态: {cache.stats}")

    print(f"\n  最终缓存统计: {json.dumps(cache.stats, ensure_ascii=False)}")
    print(f"  💡 缓存命中率 {cache.stats['hit_rate']}，节省了 {cache._hits} 次 Embedding 计算")


def demo_model_comparison() -> None:
    """演示模型选型对比。"""
    print("\n" + "=" * 60)
    print("6. Embedding 模型选型对比")
    print("=" * 60)

    models_info = [
        {"name": "OpenAI text-embedding-3-large", "dim": 3072, "max_tokens": 8191,
         "chinese": "良好", "cost": "$0.13/1M", "deploy": "API"},
        {"name": "OpenAI text-embedding-3-small", "dim": 1536, "max_tokens": 8191,
         "chinese": "良好", "cost": "$0.02/1M", "deploy": "API"},
        {"name": "BGE-large-zh-v1.5", "dim": 1024, "max_tokens": 512,
         "chinese": "优秀", "cost": "免费", "deploy": "本地"},
        {"name": "BGE-M3", "dim": 1024, "max_tokens": 8192,
         "chinese": "优秀", "cost": "免费", "deploy": "本地"},
        {"name": "M3E-large", "dim": 1024, "max_tokens": 512,
         "chinese": "优秀", "cost": "免费", "deploy": "本地"},
    ]

    print(f"  {'模型':<35} {'维度':>6} {'最大Token':>8} {'中文':>6} {'成本':>10} {'部署':>6}")
    print(f"  {'-' * 78}")
    for m in models_info:
        print(
            f"  {m['name']:<35} {m['dim']:>6} {m['max_tokens']:>8} "
            f"{m['chinese']:>6} {m['cost']:>10} {m['deploy']:>6}"
        )

    print("\n  💡 中文 RAG 推荐: BGE-large-zh-v1.5（本地）或 OpenAI text-embedding-3-small（API）")


def demo_batch_performance() -> None:
    """演示批量 Embedding 性能。"""
    print("\n" + "=" * 60)
    print("7. 批量 Embedding 性能对比")
    print("=" * 60)

    model = SimulatedEmbeddingModel(dimension=128)
    texts = [f"这是第 {i} 个测试文档，用于测试批量 Embedding 性能" for i in range(100)]

    # 逐条处理
    start = time.perf_counter()
    for text in texts:
        model.embed_text(text)
    single_time = time.perf_counter() - start

    # 清除缓存
    model._cache.clear()

    # 批量处理
    start = time.perf_counter()
    model.embed_batch(texts)
    batch_time = time.perf_counter() - start

    print(f"  文档数量: {len(texts)}")
    print(f"  逐条处理: {single_time * 1000:.1f}ms")
    print(f"  批量处理: {batch_time * 1000:.1f}ms")
    print(f"  💡 生产环境中批量处理（batch_size=32-128）可提升 5-10 倍吞吐量")


# ============================================================
# 主入口
# ============================================================

def main(server_mode: bool = False) -> None:
    """运行所有 Embedding 演示。"""
    print("🐍 Embedding 模型使用 — 向量计算 + 相似度搜索")
    print("=" * 60)

    demo_basic_embedding()
    demo_similarity_comparison()
    demo_normalization()
    demo_semantic_search()
    demo_embedding_cache()
    demo_model_comparison()
    demo_batch_performance()

    if server_mode:
        print("\n" + "=" * 60)
        print("8. 服务模式 — 使用真实 Embedding 模型")
        print("=" * 60)
        print("  💡 安装 sentence-transformers: pip install sentence-transformers")
        print("  💡 使用: from sentence_transformers import SentenceTransformer")
        print("  💡 model = SentenceTransformer('BAAI/bge-large-zh-v1.5')")
        print("  💡 embeddings = model.encode(['文本1', '文本2'])")

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. 中文场景优先选 BGE/M3E，效果优于 OpenAI Embedding")
    print("  2. 余弦相似度是最常用的相似度度量，归一化后点积等价")
    print("  3. 批量 Embedding 比逐条处理快 5-10 倍")
    print("  4. Embedding 缓存避免重复计算，节省成本")
    print("  5. 索引和查询必须使用同一个 Embedding 模型")
    print("  6. 向量存入数据库前做 L2 归一化")

    if not server_mode:
        print("\n💡 要使用真实模型: python 03_embedding.py server")


if __name__ == "__main__":
    is_server = len(sys.argv) > 1 and sys.argv[1].lower() == "server"
    main(server_mode=is_server)
