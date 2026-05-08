"""
typing 模块基础 — 类型注解、泛型、Protocol、TypedDict

知识点：基础类型注解、typing 核心类型（Optional/Union/Literal）、
       泛型（Generic/TypeVar）、Protocol 结构化子类型、
       TypedDict 类型化字典、类型别名

Python 版本：3.11+
依赖：标准库
最后验证：2024-12-01
"""

from __future__ import annotations

from typing import (
    Any,
    Generic,
    Literal,
    Protocol,
    TypeAlias,
    TypedDict,
    TypeVar,
    runtime_checkable,
)

# ============================================================
# 1. 基础类型注解
# ============================================================

def demo_basic_annotations() -> None:
    """演示基础类型注解语法。"""
    print("\n" + "=" * 60)
    print("1. 基础类型注解")
    print("=" * 60)

    # 变量注解（Python 3.6+）
    model_name: str = "qwen2-7b"
    temperature: float = 0.7
    max_tokens: int = 2048
    is_streaming: bool = False
    tags: list[str] = ["LLM", "中文", "开源"]
    config: dict[str, Any] = {"model": model_name, "temp": temperature}

    print(f"  模型: {model_name}")
    print(f"  配置: {config}")
    print(f"  标签: {tags}")

    # 函数注解
    def format_prompt(
        system: str,
        user: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """格式化 Prompt — 参数和返回值都有类型注解。"""
        parts = [f"[System] {system}", f"[User] {user}"]
        if history:
            for msg in history:
                parts.insert(-1, f"[{msg['role']}] {msg['content']}")
        return "\n".join(parts)

    result = format_prompt(
        system="你是一个 AI 助手",
        user="什么是 RAG？",
    )
    print(f"\n  格式化 Prompt:\n  {result}")


# ============================================================
# 2. typing 核心类型
# ============================================================

# 类型别名 — 提高复杂类型的可读性
Embedding: TypeAlias = list[float]
DocumentChunk: TypeAlias = dict[str, Any]
SearchResults: TypeAlias = list[tuple[DocumentChunk, float]]  # (文档, 相似度分数)

# Literal — 限定取值范围
Difficulty = Literal["beginner", "intermediate", "advanced"]
ModelProvider = Literal["openai", "ollama", "vllm", "tgi"]


def demo_typing_core() -> None:
    """演示 typing 模块核心类型。"""
    print("\n" + "=" * 60)
    print("2. typing 核心类型")
    print("=" * 60)

    # Optional — 可选值（X | None）
    def find_model(name: str) -> dict[str, Any] | None:
        """查找模型，不存在返回 None。"""
        models = {"qwen2": {"size": "7B"}, "llama3": {"size": "8B"}}
        return models.get(name)

    result = find_model("qwen2")
    print(f"  找到模型: {result}")
    result = find_model("gpt5")
    print(f"  未找到: {result}")

    # Literal — 限定取值
    def filter_entries(difficulty: Difficulty) -> list[str]:
        """按难度筛选知识条目。"""
        entries = {
            "beginner": ["NumPy 基础", "Pandas 入门"],
            "intermediate": ["Transformer 架构", "LoRA 微调"],
            "advanced": ["vLLM PagedAttention", "Flash Attention"],
        }
        return entries.get(difficulty, [])

    print(f"\n  中级知识点: {filter_entries('intermediate')}")

    # 类型别名
    embedding: Embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    print(f"  Embedding 维度: {len(embedding)}")


# ============================================================
# 3. 泛型（Generic）
# ============================================================

T = TypeVar("T")
K = TypeVar("K", str, int)  # 约束 K 只能是 str 或 int


class LRUCache(Generic[T]):
    """泛型 LRU 缓存 — 可缓存任意类型。

    AI 应用中的典型用途：
    - 缓存 Embedding 结果，避免重复计算
    - 缓存 LLM 推理结果，减少 API 调用
    - 缓存向量检索结果，降低数据库压力
    """

    def __init__(self, max_size: int = 100):
        self._store: dict[str, T] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> T | None:
        """获取缓存值。"""
        if key in self._store:
            self.hits += 1
            return self._store[key]
        self.misses += 1
        return None

    def set(self, key: str, value: T) -> None:
        """设置缓存值（超出容量时淘汰最早的）。"""
        if len(self._store) >= self.max_size:
            oldest = next(iter(self._store))
            del self._store[oldest]
        self._store[key] = value

    @property
    def hit_rate(self) -> float:
        """缓存命中率。"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def __repr__(self) -> str:
        return (f"LRUCache(size={len(self._store)}/{self.max_size}, "
                f"hit_rate={self.hit_rate:.1%})")


def demo_generics() -> None:
    """演示泛型的使用。"""
    print("\n" + "=" * 60)
    print("3. 泛型（Generic）")
    print("=" * 60)

    # Embedding 缓存
    embed_cache: LRUCache[list[float]] = LRUCache(max_size=3)
    embed_cache.set("什么是RAG", [0.1, 0.2, 0.3])
    embed_cache.set("LoRA原理", [0.4, 0.5, 0.6])

    result = embed_cache.get("什么是RAG")
    _ = embed_cache.get("不存在的key")
    print(f"  Embedding 缓存: {embed_cache}")
    print(f"  查询结果: {result}")

    # 字符串缓存
    response_cache: LRUCache[str] = LRUCache(max_size=5)
    response_cache.set("q1", "RAG 是检索增强生成...")
    print(f"  响应缓存: {response_cache}")


# ============================================================
# 4. Protocol — 结构化子类型
# ============================================================

@runtime_checkable
class Embedder(Protocol):
    """Embedding 模型协议。

    任何实现了 embed 方法的类都满足此协议，
    无需显式继承 — 这就是 Python 的鸭子类型。
    """
    def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class Retriever(Protocol):
    """检索器协议。"""
    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]: ...


class MockOpenAIEmbedder:
    """模拟 OpenAI Embedding。"""
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1 * i for i in range(5)] for _ in texts]


class MockLocalEmbedder:
    """模拟本地 BGE Embedding。"""
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.2 * i for i in range(5)] for _ in texts]


def build_index(embedder: Embedder, documents: list[str]) -> list[list[float]]:
    """使用任何满足 Embedder 协议的对象构建索引。"""
    return embedder.embed(documents)


def demo_protocol() -> None:
    """演示 Protocol 结构化子类型。"""
    print("\n" + "=" * 60)
    print("4. Protocol — 结构化子类型")
    print("=" * 60)

    docs = ["什么是 Transformer", "LoRA 微调原理"]

    # 两个类都满足 Embedder 协议，无需继承
    openai_emb = MockOpenAIEmbedder()
    local_emb = MockLocalEmbedder()

    print(f"  OpenAI 满足 Embedder 协议: {isinstance(openai_emb, Embedder)}")
    print(f"  Local 满足 Embedder 协议: {isinstance(local_emb, Embedder)}")

    # 可以互换使用
    vectors1 = build_index(openai_emb, docs)
    vectors2 = build_index(local_emb, docs)
    print(f"  OpenAI 向量: {vectors1[0][:3]}...")
    print(f"  Local 向量: {vectors2[0][:3]}...")


# ============================================================
# 5. TypedDict — 类型化字典
# ============================================================

class LLMResponse(TypedDict):
    """LLM API 响应类型。"""
    model: str
    response: str
    total_duration: int
    eval_count: int


class RAGResult(TypedDict):
    """RAG 检索结果类型。"""
    query: str
    answer: str
    sources: list[str]
    confidence: float


def format_rag_result(result: RAGResult) -> str:
    """格式化 RAG 结果 — IDE 能自动补全字典键名。"""
    sources_str = ", ".join(result["sources"])
    return (f"问题: {result['query']}\n"
            f"回答: {result['answer']}\n"
            f"来源: {sources_str}\n"
            f"置信度: {result['confidence']:.2f}")


def demo_typed_dict() -> None:
    """演示 TypedDict。"""
    print("\n" + "=" * 60)
    print("5. TypedDict — 类型化字典")
    print("=" * 60)

    # TypedDict 在运行时就是普通 dict
    rag_result: RAGResult = {
        "query": "什么是 RAG？",
        "answer": "RAG 是检索增强生成，结合了信息检索和文本生成",
        "sources": ["langchain-docs", "arxiv-2005.11401"],
        "confidence": 0.92,
    }

    formatted = format_rag_result(rag_result)
    print(f"\n{formatted}")

    # TypedDict 本质是 dict
    print(f"\n  类型: {type(rag_result)}")
    print(f"  是 dict: {isinstance(rag_result, dict)}")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行所有演示。"""
    print("🐍 typing 模块基础 — 类型注解、泛型、Protocol、TypedDict")
    print("=" * 60)

    demo_basic_annotations()
    demo_typing_core()
    demo_generics()
    demo_protocol()
    demo_typed_dict()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. 类型注解不影响运行时，但能提升代码质量和 IDE 体验")
    print("  2. Python 3.10+ 用 X | None 替代 Optional[X]")
    print("  3. Generic 实现类型安全的通用数据结构")
    print("  4. Protocol 是鸭子类型的类型化版本，无需显式继承")
    print("  5. TypedDict 为字典添加键名和值类型约束")


if __name__ == "__main__":
    main()
