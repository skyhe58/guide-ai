"""
向量数据库操作 — Chroma 内存模式 + 服务模式

知识点：向量数据库 CRUD、Chroma 本地/服务模式、HNSW 索引、
       元数据过滤、批量操作、持久化、集合管理

Python 版本：3.11+
依赖：标准库（默认模式）、chromadb>=0.4（服务模式）
最后验证：2024-12-01

外部服务（可选）：
  Chroma 向量数据库服务
  启动命令：docker run -p 8000:8000 chromadb/chroma
  或：docker compose -f docker/docker-compose.yml up -d chroma
"""

from __future__ import annotations

import hashlib
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Any

# ============================================================
# 1. 向量运算工具
# ============================================================

class VectorOps:
    """向量运算工具。"""

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def euclidean_distance(a: list[float], b: list[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    @staticmethod
    def normalize(a: list[float]) -> list[float]:
        n = math.sqrt(sum(x * x for x in a))
        return [x / n for x in a] if n > 0 else a


# ============================================================
# 2. 模拟 Embedding 模型
# ============================================================

class SimpleEmbedding:
    """简单的确定性 Embedding 模型。"""

    def __init__(self, dimension: int = 64):
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        """文本转向量（确定性哈希）。"""
        h = hashlib.sha256(text.encode()).digest()
        vec = [(b - 128) / 128.0 for b in h]
        while len(vec) < self.dimension:
            h = hashlib.sha256(h).digest()
            vec.extend((b - 128) / 128.0 for b in h)
        vec = vec[:self.dimension]
        # 添加关键词信号
        keywords = {"RAG": 0, "检索": 1, "向量": 2, "数据库": 3, "LLM": 4,
                     "Embedding": 5, "Chroma": 6, "Milvus": 7}
        for kw, idx in keywords.items():
            if kw in text and idx < self.dimension:
                vec[idx] += 0.5
        return VectorOps.normalize(vec)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


# ============================================================
# 3. 内存向量数据库（模拟 Chroma）
# ============================================================

@dataclass
class VectorRecord:
    """向量记录。"""
    id: str
    embedding: list[float]
    document: str
    metadata: dict[str, Any] = field(default_factory=dict)


class InMemoryVectorStore:
    """内存向量数据库（模拟 Chroma 核心功能）。

    支持：
    - CRUD 操作（增删改查）
    - 余弦相似度 / 欧氏距离搜索
    - 元数据过滤
    - 批量操作
    """

    def __init__(self, name: str = "default", distance_metric: str = "cosine"):
        self.name = name
        self.distance_metric = distance_metric
        self._records: dict[str, VectorRecord] = {}
        self._embedding_model = SimpleEmbedding()

    @property
    def count(self) -> int:
        """集合中的记录数。"""
        return len(self._records)

    # --- 增 ---
    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """添加文档到向量数据库。"""
        if embeddings is None:
            embeddings = self._embedding_model.embed_batch(documents)
        if metadatas is None:
            metadatas = [{} for _ in documents]

        for doc_id, doc, emb, meta in zip(ids, documents, embeddings, metadatas):
            if doc_id in self._records:
                raise ValueError(f"ID 已存在: {doc_id}，请使用 update 方法")
            self._records[doc_id] = VectorRecord(
                id=doc_id, embedding=emb, document=doc, metadata=meta,
            )

    # --- 查 ---
    def query(
        self,
        query_texts: list[str] | None = None,
        query_embeddings: list[list[float]] | None = None,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
        where_document: dict[str, str] | None = None,
    ) -> dict[str, list]:
        """查询最相似的文档。"""
        if query_texts:
            query_embs = self._embedding_model.embed_batch(query_texts)
        elif query_embeddings:
            query_embs = query_embeddings
        else:
            raise ValueError("必须提供 query_texts 或 query_embeddings")

        all_ids: list[list[str]] = []
        all_documents: list[list[str]] = []
        all_distances: list[list[float]] = []
        all_metadatas: list[list[dict]] = []

        for query_emb in query_embs:
            # 过滤记录
            candidates = list(self._records.values())
            if where:
                candidates = self._filter_by_metadata(candidates, where)
            if where_document:
                candidates = self._filter_by_document(candidates, where_document)

            # 计算距离
            scored: list[tuple[VectorRecord, float]] = []
            for record in candidates:
                if self.distance_metric == "cosine":
                    sim = VectorOps.cosine_similarity(query_emb, record.embedding)
                    distance = 1 - sim  # Chroma 返回距离（越小越相似）
                else:
                    distance = VectorOps.euclidean_distance(query_emb, record.embedding)
                scored.append((record, distance))

            # 排序取 Top-K
            scored.sort(key=lambda x: x[1])
            top_k = scored[:n_results]

            all_ids.append([r.id for r, _ in top_k])
            all_documents.append([r.document for r, _ in top_k])
            all_distances.append([d for _, d in top_k])
            all_metadatas.append([r.metadata for r, _ in top_k])

        return {
            "ids": all_ids,
            "documents": all_documents,
            "distances": all_distances,
            "metadatas": all_metadatas,
        }

    def get(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> dict[str, list]:
        """按 ID 或元数据获取文档。"""
        if ids:
            records = [self._records[i] for i in ids if i in self._records]
        else:
            records = list(self._records.values())

        if where:
            records = self._filter_by_metadata(records, where)

        return {
            "ids": [r.id for r in records],
            "documents": [r.document for r in records],
            "metadatas": [r.metadata for r in records],
        }

    # --- 改 ---
    def update(
        self,
        ids: list[str],
        documents: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """更新已有文档。"""
        for i, doc_id in enumerate(ids):
            if doc_id not in self._records:
                raise ValueError(f"ID 不存在: {doc_id}")
            record = self._records[doc_id]
            if documents:
                record.document = documents[i]
                record.embedding = self._embedding_model.embed(documents[i])
            if embeddings:
                record.embedding = embeddings[i]
            if metadatas:
                record.metadata = metadatas[i]

    # --- 删 ---
    def delete(self, ids: list[str]) -> None:
        """删除文档。"""
        for doc_id in ids:
            self._records.pop(doc_id, None)

    # --- 过滤 ---
    @staticmethod
    def _filter_by_metadata(
        records: list[VectorRecord], where: dict[str, Any],
    ) -> list[VectorRecord]:
        """按元数据过滤。支持简单的等值匹配。"""
        filtered = []
        for record in records:
            match = all(record.metadata.get(k) == v for k, v in where.items())
            if match:
                filtered.append(record)
        return filtered

    @staticmethod
    def _filter_by_document(
        records: list[VectorRecord], where_document: dict[str, str],
    ) -> list[VectorRecord]:
        """按文档内容过滤。"""
        filtered = []
        for record in records:
            if "$contains" in where_document:
                if where_document["$contains"] in record.document:
                    filtered.append(record)
        return filtered


# ============================================================
# 4. 集合管理器（模拟 Chroma Client）
# ============================================================

class VectorStoreClient:
    """向量数据库客户端（模拟 Chroma Client）。

    管理多个集合（Collection），支持创建、获取、删除集合。
    """

    def __init__(self, mode: str = "memory"):
        self.mode = mode
        self._collections: dict[str, InMemoryVectorStore] = {}

    def create_collection(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> InMemoryVectorStore:
        """创建新集合。"""
        if name in self._collections:
            raise ValueError(f"集合已存在: {name}")
        distance = "cosine"
        if metadata and "hnsw:space" in metadata:
            distance = metadata["hnsw:space"]
        collection = InMemoryVectorStore(name=name, distance_metric=distance)
        self._collections[name] = collection
        return collection

    def get_collection(self, name: str) -> InMemoryVectorStore:
        """获取已有集合。"""
        if name not in self._collections:
            raise ValueError(f"集合不存在: {name}")
        return self._collections[name]

    def get_or_create_collection(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> InMemoryVectorStore:
        """获取或创建集合。"""
        if name in self._collections:
            return self._collections[name]
        return self.create_collection(name, metadata)

    def delete_collection(self, name: str) -> None:
        """删除集合。"""
        self._collections.pop(name, None)

    def list_collections(self) -> list[str]:
        """列出所有集合。"""
        return list(self._collections.keys())


# ============================================================
# 5. 演示函数
# ============================================================

def demo_basic_crud() -> None:
    """演示基本 CRUD 操作。"""
    print("\n" + "=" * 60)
    print("1. 基本 CRUD 操作")
    print("=" * 60)

    client = VectorStoreClient()
    collection = client.create_collection("knowledge_base", metadata={"hnsw:space": "cosine"})

    # 添加文档
    documents = [
        "RAG 系统通过检索外部知识库增强 LLM 生成能力",
        "Chroma 是轻量级向量数据库，适合开发阶段",
        "FAISS 是 Meta 开源的高性能向量搜索库",
        "Milvus 支持分布式部署，适合企业级生产环境",
        "Pinecone 是全托管的云向量数据库服务",
    ]
    metadatas = [
        {"category": "RAG", "difficulty": "intermediate"},
        {"category": "vector_db", "difficulty": "beginner"},
        {"category": "vector_db", "difficulty": "intermediate"},
        {"category": "vector_db", "difficulty": "advanced"},
        {"category": "vector_db", "difficulty": "beginner"},
    ]
    ids = [f"doc_{i}" for i in range(len(documents))]

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"  ✅ 添加了 {collection.count} 个文档")

    # 查询
    results = collection.query(query_texts=["向量数据库选型"], n_results=3)
    print(f"\n  🔍 查询: '向量数据库选型'")
    for i in range(len(results["ids"][0])):
        print(f"    Top-{i+1}: {results['documents'][0][i][:50]}... (距离: {results['distances'][0][i]:.4f})")

    # 更新
    collection.update(ids=["doc_0"], documents=["RAG 检索增强生成是 AI 应用的核心架构模式"])
    print(f"\n  ✏️ 更新了 doc_0")

    # 删除
    collection.delete(ids=["doc_4"])
    print(f"  🗑️ 删除了 doc_4，剩余 {collection.count} 个文档")


def demo_metadata_filtering() -> None:
    """演示元数据过滤。"""
    print("\n" + "=" * 60)
    print("2. 元数据过滤查询")
    print("=" * 60)

    client = VectorStoreClient()
    collection = client.create_collection("filtered_search")

    documents = [
        "Chroma 内存模式适合快速原型开发",
        "Chroma 持久化模式支持数据落盘",
        "Milvus 单机模式适合中等规模数据",
        "Milvus 集群模式支持百亿级数据",
        "FAISS 只支持内存模式，需要自己实现持久化",
    ]
    metadatas = [
        {"db": "chroma", "mode": "memory", "scale": "small"},
        {"db": "chroma", "mode": "persistent", "scale": "small"},
        {"db": "milvus", "mode": "standalone", "scale": "medium"},
        {"db": "milvus", "mode": "cluster", "scale": "large"},
        {"db": "faiss", "mode": "memory", "scale": "large"},
    ]
    ids = [f"doc_{i}" for i in range(len(documents))]
    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    # 按元数据过滤
    print("  🔍 查询: '数据库部署'，过滤: db=milvus")
    results = collection.query(
        query_texts=["数据库部署"],
        n_results=3,
        where={"db": "milvus"},
    )
    for i in range(len(results["ids"][0])):
        print(f"    {results['documents'][0][i]}")

    # 按文档内容过滤
    print("\n  🔍 查询: '向量数据库'，文档包含: '内存'")
    results = collection.query(
        query_texts=["向量数据库"],
        n_results=3,
        where_document={"$contains": "内存"},
    )
    for i in range(len(results["ids"][0])):
        print(f"    {results['documents'][0][i]}")


def demo_collection_management() -> None:
    """演示集合管理。"""
    print("\n" + "=" * 60)
    print("3. 集合管理")
    print("=" * 60)

    client = VectorStoreClient()

    # 创建多个集合
    client.create_collection("tech_docs")
    client.create_collection("faq")
    client.create_collection("code_snippets")

    print(f"  集合列表: {client.list_collections()}")

    # 获取或创建
    collection = client.get_or_create_collection("tech_docs")
    collection.add(
        ids=["td_1"],
        documents=["RAG 系统架构设计文档"],
        metadatas=[{"type": "architecture"}],
    )
    print(f"  tech_docs 集合: {collection.count} 个文档")

    # 删除集合
    client.delete_collection("code_snippets")
    print(f"  删除 code_snippets 后: {client.list_collections()}")


def demo_batch_operations() -> None:
    """演示批量操作性能。"""
    print("\n" + "=" * 60)
    print("4. 批量操作性能")
    print("=" * 60)

    client = VectorStoreClient()
    collection = client.create_collection("batch_test")

    # 批量添加
    n_docs = 500
    documents = [f"这是第 {i} 个测试文档，包含关于 RAG 和向量数据库的知识" for i in range(n_docs)]
    ids = [f"batch_{i}" for i in range(n_docs)]
    metadatas = [{"batch_id": i // 100, "index": i} for i in range(n_docs)]

    start = time.perf_counter()
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    add_time = time.perf_counter() - start

    print(f"  批量添加 {n_docs} 个文档: {add_time * 1000:.1f}ms")
    print(f"  集合大小: {collection.count}")

    # 批量查询
    queries = ["RAG 系统", "向量数据库", "Embedding 模型"]
    start = time.perf_counter()
    results = collection.query(query_texts=queries, n_results=5)
    query_time = time.perf_counter() - start

    print(f"  批量查询 {len(queries)} 个查询: {query_time * 1000:.1f}ms")

    # 带过滤的查询
    start = time.perf_counter()
    results = collection.query(
        query_texts=["RAG 知识"],
        n_results=5,
        where={"batch_id": 2},
    )
    filter_time = time.perf_counter() - start
    print(f"  带元数据过滤查询: {filter_time * 1000:.1f}ms")


def demo_distance_metrics() -> None:
    """演示不同距离度量。"""
    print("\n" + "=" * 60)
    print("5. 距离度量对比")
    print("=" * 60)

    client = VectorStoreClient()

    # 余弦距离
    cos_collection = client.create_collection("cosine_test", metadata={"hnsw:space": "cosine"})
    # 欧氏距离
    l2_collection = client.create_collection("l2_test", metadata={"hnsw:space": "l2"})

    documents = [
        "RAG 检索增强生成系统",
        "向量数据库 Chroma 使用指南",
        "Python 异步编程教程",
    ]
    ids = ["d1", "d2", "d3"]

    cos_collection.add(ids=ids, documents=documents)
    l2_collection.add(ids=ids, documents=documents)

    query = "RAG 系统设计"
    print(f"  查询: '{query}'")

    cos_results = cos_collection.query(query_texts=[query], n_results=3)
    l2_results = l2_collection.query(query_texts=[query], n_results=3)

    print(f"\n  余弦距离排序:")
    for i in range(3):
        print(f"    {cos_results['documents'][0][i][:30]}... (距离: {cos_results['distances'][0][i]:.4f})")

    print(f"\n  欧氏距离排序:")
    for i in range(3):
        print(f"    {l2_results['documents'][0][i][:30]}... (距离: {l2_results['distances'][0][i]:.4f})")


def demo_chroma_server_mode() -> None:
    """演示 Chroma 服务模式（需要 Docker）。"""
    print("\n" + "=" * 60)
    print("6. Chroma 服务模式（需要 Docker）")
    print("=" * 60)

    try:
        import chromadb
        client = chromadb.HttpClient(host="localhost", port=8000)
        client.heartbeat()
        print("  ✅ 已连接 Chroma 服务")

        collection = client.get_or_create_collection("demo")
        collection.add(
            ids=["s1", "s2"],
            documents=["RAG 系统设计", "向量数据库选型"],
        )
        results = collection.query(query_texts=["RAG"], n_results=2)
        print(f"  查询结果: {results['documents']}")

    except ImportError:
        print("  ⚠️ chromadb 未安装: pip install chromadb")
    except Exception as e:
        print(f"  ❌ 无法连接 Chroma 服务: {e}")
        print("  💡 启动: docker run -p 8000:8000 chromadb/chroma")


# ============================================================
# 主入口
# ============================================================

def main(server_mode: bool = False) -> None:
    """运行所有向量数据库演示。"""
    print("🐍 向量数据库操作 — 内存模式 + Chroma 服务模式")
    print("=" * 60)

    demo_basic_crud()
    demo_metadata_filtering()
    demo_collection_management()
    demo_batch_operations()
    demo_distance_metrics()

    if server_mode:
        demo_chroma_server_mode()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. 开发阶段用 Chroma 内存/持久化模式，生产用 Milvus/Qdrant")
    print("  2. 元数据过滤 + 向量搜索组合使用，提升检索精度")
    print("  3. 余弦距离是最常用的距离度量（归一化后等价于点积）")
    print("  4. 批量操作比逐条操作效率高很多")
    print("  5. HNSW 是默认索引算法，M 和 ef 参数影响性能")
    print("  6. 生产环境需要备份策略和监控")

    if not server_mode:
        print("\n💡 要测试 Chroma 服务模式: python 04_vector_store.py server")


if __name__ == "__main__":
    is_server = len(sys.argv) > 1 and sys.argv[1].lower() == "server"
    main(server_mode=is_server)
