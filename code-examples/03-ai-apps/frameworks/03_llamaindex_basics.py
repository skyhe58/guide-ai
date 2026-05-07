"""
LlamaIndex 基础 — 索引/查询引擎/数据连接器模拟

知识点：LlamaIndex 核心概念模拟实现，包括 Document、Node、
       VectorStoreIndex、TreeIndex、KeywordTableIndex、
       QueryEngine、ChatEngine、数据连接器

Python 版本：3.11+
依赖：标准库（默认模式）
最后验证：2024-12-01
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ============================================================
# 1. Document 与 Node — 数据抽象
# ============================================================

@dataclass
class Document:
    """文档对象 — LlamaIndex 的原始数据抽象。"""
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = hashlib.md5(self.text[:100].encode()).hexdigest()[:8]


@dataclass
class TextNode:
    """文本节点 — Document 切分后的基本单位。"""
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    node_id: str = ""
    parent_id: str = ""
    prev_node_id: str = ""
    next_node_id: str = ""
    embedding: list[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.node_id:
            self.node_id = hashlib.md5(self.text[:50].encode()).hexdigest()[:8]

    def get_content(self) -> str:
        return self.text


# ============================================================
# 2. Node Parser — 文档切分器
# ============================================================

class SentenceSplitter:
    """句子切分器 — 按句子边界切分文档。"""

    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def get_nodes_from_documents(self, documents: list[Document]) -> list[TextNode]:
        """将文档切分为节点。"""
        all_nodes: list[TextNode] = []
        for doc in documents:
            nodes = self._split_document(doc)
            # 建立前后关系
            for i, node in enumerate(nodes):
                node.parent_id = doc.doc_id
                if i > 0:
                    node.prev_node_id = nodes[i - 1].node_id
                if i < len(nodes) - 1:
                    node.next_node_id = nodes[i + 1].node_id
            all_nodes.extend(nodes)
        return all_nodes

    def _split_document(self, doc: Document) -> list[TextNode]:
        """切分单个文档。"""
        text = doc.text
        nodes = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            # 尝试在句子边界切分
            if end < len(text):
                for sep in ["。", ".", "\n", "，", ","]:
                    last_sep = text[start:end].rfind(sep)
                    if last_sep > self.chunk_size // 2:
                        end = start + last_sep + 1
                        break
            chunk = text[start:end].strip()
            if chunk:
                nodes.append(TextNode(
                    text=chunk,
                    metadata={**doc.metadata, "chunk_index": len(nodes)},
                ))
            start = end - self.chunk_overlap if end < len(text) else end
        return nodes


# ============================================================
# 3. Embedding — 向量嵌入模拟
# ============================================================

class MockEmbedding:
    """模拟 Embedding 模型。"""

    def __init__(self, model_name: str = "text-embedding-3-small", dim: int = 8):
        self.model_name = model_name
        self.dim = dim
        self.call_count = 0

    def get_text_embedding(self, text: str) -> list[float]:
        """生成文本的模拟 Embedding 向量。"""
        self.call_count += 1
        # 基于文本哈希生成伪向量
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        vector = []
        for i in range(self.dim):
            val = ((hash_val >> (i * 4)) & 0xF) / 15.0 - 0.5
            vector.append(round(val, 4))
        # 归一化
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [round(v / norm, 4) for v in vector]
        return vector

    def get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """批量生成 Embedding。"""
        return [self.get_text_embedding(t) for t in texts]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算余弦相似度。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ============================================================
# 4. Index — 索引类型
# ============================================================

class BaseIndex(ABC):
    """索引基类。"""

    def __init__(self, nodes: list[TextNode]):
        self.nodes = nodes

    @abstractmethod
    def query(self, query_str: str, top_k: int = 3) -> list[tuple[TextNode, float]]:
        """查询索引，返回 (节点, 分数) 列表。"""

    def as_query_engine(self, **kwargs) -> "QueryEngine":
        """转换为查询引擎。"""
        return QueryEngine(self, **kwargs)

    def as_chat_engine(self, **kwargs) -> "ChatEngine":
        """转换为对话引擎。"""
        return ChatEngine(self, **kwargs)


class VectorStoreIndex(BaseIndex):
    """向量索引 — 基于向量相似度检索。"""

    def __init__(self, nodes: list[TextNode], embed_model: MockEmbedding | None = None):
        super().__init__(nodes)
        self.embed_model = embed_model or MockEmbedding()
        # 为所有节点生成 Embedding
        for node in self.nodes:
            if not node.embedding:
                node.embedding = self.embed_model.get_text_embedding(node.text)

    @classmethod
    def from_documents(cls, documents: list[Document], **kwargs) -> "VectorStoreIndex":
        """从文档列表创建向量索引。"""
        splitter = SentenceSplitter()
        nodes = splitter.get_nodes_from_documents(documents)
        return cls(nodes, **kwargs)

    def query(self, query_str: str, top_k: int = 3) -> list[tuple[TextNode, float]]:
        """向量相似度检索。"""
        query_embedding = self.embed_model.get_text_embedding(query_str)
        scored = []
        for node in self.nodes:
            score = cosine_similarity(query_embedding, node.embedding)
            scored.append((node, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class TreeIndex(BaseIndex):
    """树索引 — 层级摘要结构。"""

    def __init__(self, nodes: list[TextNode]):
        super().__init__(nodes)
        self.summaries: dict[str, str] = {}
        self._build_tree()

    def _build_tree(self) -> None:
        """构建树状摘要。"""
        # 为每个节点生成摘要
        for node in self.nodes:
            self.summaries[node.node_id] = node.text[:50] + "..."
        # 生成根摘要
        all_text = " ".join(n.text[:30] for n in self.nodes)
        self.summaries["root"] = f"文档包含 {len(self.nodes)} 个片段: {all_text[:100]}..."

    def query(self, query_str: str, top_k: int = 3) -> list[tuple[TextNode, float]]:
        """自顶向下遍历查询。"""
        # 简化实现：关键词匹配
        scored = []
        query_words = set(query_str.lower().split())
        for node in self.nodes:
            node_words = set(node.text.lower().split())
            score = len(query_words & node_words) / max(len(query_words), 1)
            scored.append((node, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class KeywordTableIndex(BaseIndex):
    """关键词索引 — 倒排索引结构。"""

    def __init__(self, nodes: list[TextNode]):
        super().__init__(nodes)
        self.keyword_table: dict[str, list[int]] = {}
        self._build_index()

    def _build_index(self) -> None:
        """构建关键词倒排索引。"""
        for i, node in enumerate(self.nodes):
            # 提取关键词（简化：按空格分词）
            words = set(re.findall(r"[\w\u4e00-\u9fff]+", node.text.lower()))
            for word in words:
                if len(word) > 1:  # 过滤单字符
                    if word not in self.keyword_table:
                        self.keyword_table[word] = []
                    self.keyword_table[word].append(i)

    def query(self, query_str: str, top_k: int = 3) -> list[tuple[TextNode, float]]:
        """关键词匹配查询。"""
        query_words = set(re.findall(r"[\w\u4e00-\u9fff]+", query_str.lower()))
        node_scores: dict[int, int] = {}
        for word in query_words:
            if word in self.keyword_table:
                for idx in self.keyword_table[word]:
                    node_scores[idx] = node_scores.get(idx, 0) + 1
        scored = [(self.nodes[idx], score / max(len(query_words), 1))
                  for idx, score in node_scores.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ============================================================
# 5. Query Engine — 查询引擎
# ============================================================

class QueryEngine:
    """查询引擎 — LlamaIndex 的核心查询接口。"""

    def __init__(self, index: BaseIndex, response_mode: str = "compact"):
        self.index = index
        self.response_mode = response_mode

    def query(self, query_str: str) -> "QueryResponse":
        """执行查询。"""
        # 检索相关节点
        results = self.index.query(query_str, top_k=3)
        # 合成回答
        context = "\n".join(node.text for node, _ in results)
        if self.response_mode == "compact":
            answer = f"基于检索到的 {len(results)} 个文档片段：{context[:200]}..."
        else:
            answer = context
        source_nodes = [node for node, _ in results]
        scores = [score for _, score in results]
        return QueryResponse(response=answer, source_nodes=source_nodes, scores=scores)


@dataclass
class QueryResponse:
    """查询响应。"""
    response: str
    source_nodes: list[TextNode] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    def __str__(self) -> str:
        return self.response


# ============================================================
# 6. Chat Engine — 对话引擎
# ============================================================

class ChatEngine:
    """对话引擎 — 在查询引擎基础上增加对话历史。"""

    def __init__(self, index: BaseIndex, mode: str = "condense_question"):
        self.index = index
        self.mode = mode
        self.chat_history: list[dict[str, str]] = []

    def chat(self, message: str) -> str:
        """对话接口。"""
        self.chat_history.append({"role": "user", "content": message})
        # condense_question 模式：将多轮对话压缩为单个查询
        if self.mode == "condense_question" and len(self.chat_history) > 2:
            condensed = f"{message} (上下文: {self.chat_history[-3]['content'][:30]}...)"
        else:
            condensed = message
        # 查询
        engine = QueryEngine(self.index)
        response = engine.query(condensed)
        answer = response.response
        self.chat_history.append({"role": "assistant", "content": answer})
        return answer

    def reset(self) -> None:
        """重置对话历史。"""
        self.chat_history.clear()


# ============================================================
# 7. Data Connectors — 数据连接器
# ============================================================

class SimpleDirectoryReader:
    """简单目录读取器 — 模拟从目录加载文档。"""

    def __init__(self, input_dir: str = "data/"):
        self.input_dir = input_dir

    def load_data(self) -> list[Document]:
        """加载目录中的文档（模拟）。"""
        # 模拟加载文件
        mock_files = {
            "rag_intro.md": "RAG（检索增强生成）是一种结合检索和生成的 AI 架构。它通过从外部知识库中检索相关文档，将检索到的信息作为上下文注入到 LLM 的 Prompt 中，从而生成更准确、更有依据的回答。RAG 的核心优势在于减少 LLM 的幻觉，提供可溯源的回答。",
            "langchain.md": "LangChain 是目前最流行的 LLM 应用开发框架。它提供了 Prompt Template、Output Parser、Memory、Retriever、Agent 等标准化组件，帮助开发者快速构建基于大语言模型的应用。LangChain 的核心理念是链式调用（LCEL），使用管道符将组件串联。",
            "vector_db.md": "向量数据库是专门用于存储和检索向量嵌入的数据库。常见的向量数据库包括 Chroma（本地轻量级）、Pinecone（云服务）、FAISS（Facebook 开源）和 Milvus（分布式）。向量数据库通过近似最近邻搜索（ANN）实现高效的语义检索。",
            "agent.md": "AI Agent 是具备自主决策和工具调用能力的智能体。Agent 能够根据用户需求，自动选择合适的工具（如搜索、计算、数据库查询），执行操作并返回结果。常见的 Agent 模式包括 ReAct（推理-行动循环）和 Multi-Agent（多智能体协作）。",
            "embedding.md": "Embedding 模型将文本映射为高维向量，使得语义相似的文本在向量空间中距离更近。常用的 Embedding 模型包括 OpenAI text-embedding-3-small、BGE-M3（中文优化）和 M3E。选择 Embedding 模型时需要考虑维度、性能、中文支持和成本。",
        }
        documents = []
        for filename, content in mock_files.items():
            documents.append(Document(
                text=content,
                metadata={"source": filename, "file_type": filename.split(".")[-1]},
            ))
        print(f"  [数据连接器] 从 '{self.input_dir}' 加载了 {len(documents)} 个文档")
        return documents


# ============================================================
# 演示函数
# ============================================================

def demo_document_and_node() -> None:
    """演示 Document 和 Node。"""
    print("\n" + "=" * 60)
    print("1. Document 与 Node — 数据抽象")
    print("=" * 60)

    reader = SimpleDirectoryReader("data/")
    docs = reader.load_data()
    splitter = SentenceSplitter(chunk_size=100, chunk_overlap=20)
    nodes = splitter.get_nodes_from_documents(docs)
    print(f"  文档数: {len(docs)}, 节点数: {len(nodes)}")
    for i, node in enumerate(nodes[:3]):
        print(f"  Node[{i}]: '{node.text[:40]}...' (parent: {node.parent_id})")


def demo_vector_index() -> None:
    """演示 VectorStoreIndex。"""
    print("\n" + "=" * 60)
    print("2. VectorStoreIndex — 向量索引")
    print("=" * 60)

    docs = SimpleDirectoryReader().load_data()
    index = VectorStoreIndex.from_documents(docs)
    engine = index.as_query_engine()
    response = engine.query("什么是 RAG 检索增强生成？")
    print(f"  查询: '什么是 RAG 检索增强生成？'")
    print(f"  回答: {response.response[:80]}...")
    print(f"  来源节点数: {len(response.source_nodes)}")


def demo_tree_index() -> None:
    """演示 TreeIndex。"""
    print("\n" + "=" * 60)
    print("3. TreeIndex — 树索引")
    print("=" * 60)

    docs = SimpleDirectoryReader().load_data()
    splitter = SentenceSplitter(chunk_size=150)
    nodes = splitter.get_nodes_from_documents(docs)
    index = TreeIndex(nodes)
    results = index.query("向量数据库有哪些？", top_k=2)
    print(f"  查询: '向量数据库有哪些？'")
    for node, score in results:
        print(f"  [{score:.2f}] {node.text[:50]}...")


def demo_keyword_index() -> None:
    """演示 KeywordTableIndex。"""
    print("\n" + "=" * 60)
    print("4. KeywordTableIndex — 关键词索引")
    print("=" * 60)

    docs = SimpleDirectoryReader().load_data()
    splitter = SentenceSplitter(chunk_size=150)
    nodes = splitter.get_nodes_from_documents(docs)
    index = KeywordTableIndex(nodes)
    print(f"  关键词表大小: {len(index.keyword_table)}")
    results = index.query("Agent 工具调用", top_k=2)
    print(f"  查询: 'Agent 工具调用'")
    for node, score in results:
        print(f"  [{score:.2f}] {node.text[:50]}...")


def demo_chat_engine() -> None:
    """演示 ChatEngine。"""
    print("\n" + "=" * 60)
    print("5. ChatEngine — 对话引擎")
    print("=" * 60)

    docs = SimpleDirectoryReader().load_data()
    index = VectorStoreIndex.from_documents(docs)
    chat = index.as_chat_engine()
    questions = ["什么是 RAG？", "它和 Agent 有什么区别？", "推荐哪个向量数据库？"]
    for q in questions:
        answer = chat.chat(q)
        print(f"  Q: {q}")
        print(f"  A: {answer[:60]}...")
    print(f"  对话轮数: {len(chat.chat_history) // 2}")


def demo_index_comparison() -> None:
    """演示不同索引类型对比。"""
    print("\n" + "=" * 60)
    print("6. 索引类型对比")
    print("=" * 60)

    docs = SimpleDirectoryReader().load_data()
    splitter = SentenceSplitter(chunk_size=150)
    nodes = splitter.get_nodes_from_documents(docs)

    query = "Embedding 模型选择"
    print(f"  查询: '{query}'")

    # 向量索引
    vec_index = VectorStoreIndex(nodes)
    vec_results = vec_index.query(query, top_k=1)
    print(f"  VectorIndex: [{vec_results[0][1]:.3f}] {vec_results[0][0].text[:40]}...")

    # 树索引
    tree_index = TreeIndex(nodes)
    tree_results = tree_index.query(query, top_k=1)
    print(f"  TreeIndex:   [{tree_results[0][1]:.3f}] {tree_results[0][0].text[:40]}...")

    # 关键词索引
    kw_index = KeywordTableIndex(nodes)
    kw_results = kw_index.query(query, top_k=1)
    if kw_results:
        print(f"  KeywordIndex:[{kw_results[0][1]:.3f}] {kw_results[0][0].text[:40]}...")
    else:
        print(f"  KeywordIndex: 无匹配结果")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行所有 LlamaIndex 基础演示。"""
    print("LlamaIndex 基础 — 索引/查询引擎/数据连接器模拟")
    print("=" * 60)

    demo_document_and_node()
    demo_vector_index()
    demo_tree_index()
    demo_keyword_index()
    demo_chat_engine()
    demo_index_comparison()

    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("\n关键要点:")
    print("  1. Document 是原始文档，Node 是切分后的检索基本单位")
    print("  2. VectorStoreIndex 是最常用的索引，基于向量相似度检索")
    print("  3. TreeIndex 适合长文档摘要，KeywordTableIndex 适合精确匹配")
    print("  4. QueryEngine 是单次查询接口，ChatEngine 支持多轮对话")
    print("  5. Node 之间的前后关系支持上下文感知的检索")
    print("  6. 90% 场景用 VectorStoreIndex 就够了")


if __name__ == "__main__":
    main()
