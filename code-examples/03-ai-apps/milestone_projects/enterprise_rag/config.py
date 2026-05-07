"""
企业级 RAG 知识库 — 配置文件

Python 版本：3.11+
最后验证：2024-12-01
"""

from dataclasses import dataclass, field


@dataclass
class EmbeddingConfig:
    """Embedding 模型配置。"""
    model_name: str = "text-embedding-3-small"
    dimension: int = 1536
    batch_size: int = 100


@dataclass
class VectorStoreConfig:
    """向量数据库配置。"""
    provider: str = "chroma"  # chroma / pinecone / faiss
    collection_name: str = "enterprise_rag"
    host: str = "localhost"
    port: int = 8000


@dataclass
class LLMConfig:
    """LLM 配置。"""
    provider: str = "ollama"  # openai / ollama
    model_name: str = "qwen2"
    temperature: float = 0.1
    max_tokens: int = 1024
    base_url: str = "http://localhost:11434"


@dataclass
class ChunkingConfig:
    """文档切分配置。"""
    chunk_size: int = 500
    chunk_overlap: int = 100
    separator: str = "\n\n"


@dataclass
class RetrievalConfig:
    """检索配置。"""
    top_k: int = 5
    similarity_threshold: float = 0.7
    use_rerank: bool = True
    rerank_top_k: int = 3


@dataclass
class RAGConfig:
    """RAG 系统总配置。"""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    data_dir: str = "data/"
    log_level: str = "INFO"
