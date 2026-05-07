"""
Pydantic 数据验证 — BaseModel、Field、验证器、序列化

知识点：Pydantic v2 BaseModel、Field 约束、
       field_validator/model_validator、
       模型序列化（model_dump/model_dump_json）、
       AI 应用中的数据模型设计

Python 版本：3.11+
依赖：pydantic>=2.5
最后验证：2024-12-01
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================
# 1. 基础 Pydantic 模型
# ============================================================

class ChatMessage(BaseModel):
    """聊天消息模型 — 自动类型验证和转换。"""
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=10000, description="消息内容")


class LLMConfig(BaseModel):
    """LLM 推理配置模型。"""
    model: str = Field(default="qwen2", description="模型名称")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=32768)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = False


class EmbeddingRequest(BaseModel):
    """Embedding 请求模型。"""
    texts: list[str] = Field(min_length=1, description="待编码的文本列表")
    model: Literal["bge-m3", "openai", "m3e"] = "bge-m3"
    normalize: bool = True
    dimensions: int | None = Field(default=None, ge=64, le=4096)


def demo_basic_models() -> None:
    """演示基础 Pydantic 模型。"""
    print("\n" + "=" * 60)
    print("1. 基础 Pydantic 模型")
    print("=" * 60)

    # 正常创建
    msg = ChatMessage(role="user", content="什么是 RAG？")
    print(f"  消息: {msg}")
    print(f"  字典: {msg.model_dump()}")

    config = LLMConfig(model="qwen2-7b", temperature=0.3)
    print(f"\n  LLM 配置: {config}")
    print(f"  JSON: {config.model_dump_json(indent=2)}")

    # 自动类型转换
    config2 = LLMConfig(temperature="0.5", max_tokens="1024")  # type: ignore
    print(f"\n  自动转换: temperature={config2.temperature} (float)")

    # 验证失败
    print("\n  --- 验证失败示例 ---")
    try:
        LLMConfig(temperature=3.0)  # 超出范围 [0, 2]
    except Exception as e:
        print(f"  ❌ 验证失败: {e}")

    try:
        ChatMessage(role="invalid", content="test")  # role 不在 Literal 范围内
    except Exception as e:
        print(f"  ❌ 验证失败: {e}")


# ============================================================
# 2. 自定义验证器
# ============================================================

class RAGQuery(BaseModel):
    """RAG 查询模型 — 带自定义验证逻辑。"""
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=100)
    score_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    filters: dict[str, Any] = Field(default_factory=dict)
    rerank: bool = True

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        """验证查询不能为纯空白。"""
        stripped = v.strip()
        if not stripped:
            raise ValueError("查询内容不能为空白")
        return stripped

    @field_validator("filters")
    @classmethod
    def validate_filters(cls, v: dict[str, Any]) -> dict[str, Any]:
        """验证过滤条件的键名合法性。"""
        allowed_keys = {"module", "difficulty", "tags", "date_range"}
        invalid = set(v.keys()) - allowed_keys
        if invalid:
            raise ValueError(f"不支持的过滤条件: {invalid}")
        return v

    @model_validator(mode="after")
    def check_rerank_requires_enough_results(self) -> "RAGQuery":
        """模型级验证：开启 rerank 时 top_k 不能太小。"""
        if self.rerank and self.top_k < 3:
            raise ValueError("开启 rerank 时 top_k 至少为 3")
        return self


def demo_validators() -> None:
    """演示自定义验证器。"""
    print("\n" + "=" * 60)
    print("2. 自定义验证器")
    print("=" * 60)

    # 正常查询
    query = RAGQuery(
        query="  如何使用 LangGraph 构建 Agent？  ",
        top_k=10,
        filters={"module": "ai-apps", "difficulty": "intermediate"},
    )
    print(f"  查询（已 strip）: '{query.query}'")
    print(f"  过滤条件: {query.filters}")

    # 验证失败示例
    print("\n  --- 验证失败示例 ---")
    test_cases = [
        {"query": "   ", "desc": "纯空白查询"},
        {"query": "test", "filters": {"invalid_key": "x"}, "desc": "非法过滤条件"},
        {"query": "test", "rerank": True, "top_k": 1, "desc": "rerank + top_k 太小"},
    ]

    for case in test_cases:
        desc = case.pop("desc")
        try:
            RAGQuery(**case)
        except Exception as e:
            # 只取第一行错误信息
            err_msg = str(e).split("\n")[0]
            print(f"  ❌ {desc}: {err_msg}")


# ============================================================
# 3. 模型嵌套与序列化
# ============================================================

class DocumentChunk(BaseModel):
    """文档块模型。"""
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0


class RAGResponse(BaseModel):
    """RAG 完整响应模型 — 嵌套模型。"""
    query: str
    answer: str
    chunks: list[DocumentChunk] = Field(default_factory=list)
    model: str = "qwen2"
    latency_ms: float = 0.0

    @property
    def source_count(self) -> int:
        """引用的文档块数量。"""
        return len(self.chunks)


def demo_nested_models() -> None:
    """演示模型嵌套和序列化。"""
    print("\n" + "=" * 60)
    print("3. 模型嵌套与序列化")
    print("=" * 60)

    response = RAGResponse(
        query="什么是 LoRA？",
        answer="LoRA 是一种参数高效的微调方法...",
        chunks=[
            DocumentChunk(
                content="LoRA 通过低秩分解减少可训练参数...",
                metadata={"source": "arxiv-2106.09685", "page": 3},
                score=0.95,
            ),
            DocumentChunk(
                content="与全参数微调相比，LoRA 只需训练 0.1% 的参数...",
                metadata={"source": "huggingface-docs"},
                score=0.88,
            ),
        ],
        model="qwen2-7b",
        latency_ms=234.5,
    )

    # 序列化为字典
    print(f"  引用文档数: {response.source_count}")
    print(f"  延迟: {response.latency_ms}ms")

    # 序列化为 JSON
    json_str = response.model_dump_json(indent=2)
    print(f"\n  JSON 输出（前 200 字符）:")
    print(f"  {json_str[:200]}...")

    # 从字典反序列化
    data = response.model_dump()
    restored = RAGResponse.model_validate(data)
    print(f"\n  反序列化成功: {restored.query}")

    # 部分序列化（排除某些字段）
    partial = response.model_dump(exclude={"chunks"}, exclude_none=True)
    print(f"  部分序列化: {partial}")


# ============================================================
# 4. 实战模式 — FastAPI 风格的请求/响应模型
# ============================================================

class PredictRequest(BaseModel):
    """推理请求模型（FastAPI 会自动用这个做请求验证）。"""
    model_config = {"json_schema_extra": {
        "examples": [{"prompt": "什么是 RAG？", "model": "qwen2"}]
    }}

    prompt: str = Field(min_length=1, max_length=5000, description="用户提示词")
    model: str = Field(default="qwen2", description="模型名称")
    config: LLMConfig = Field(default_factory=LLMConfig)
    history: list[ChatMessage] = Field(default_factory=list)


class PredictResponse(BaseModel):
    """推理响应模型。"""
    answer: str
    model: str
    usage: dict[str, int] = Field(default_factory=lambda: {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0
    })
    latency_ms: float


def demo_fastapi_models() -> None:
    """演示 FastAPI 风格的请求/响应模型。"""
    print("\n" + "=" * 60)
    print("4. 实战模式 — FastAPI 请求/响应模型")
    print("=" * 60)

    # 模拟 FastAPI 接收到的请求 JSON
    request_json = {
        "prompt": "解释 Transformer 的自注意力机制",
        "model": "qwen2-7b",
        "config": {"temperature": 0.3, "max_tokens": 500},
        "history": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮你的？"},
        ],
    }

    # Pydantic 自动验证和解析
    request = PredictRequest.model_validate(request_json)
    print(f"  请求模型: {request.model}")
    print(f"  温度: {request.config.temperature}")
    print(f"  历史消息数: {len(request.history)}")

    # 构造响应
    response = PredictResponse(
        answer="自注意力机制允许模型在处理每个 token 时关注输入序列中的所有位置...",
        model=request.model,
        usage={"prompt_tokens": 45, "completion_tokens": 120, "total_tokens": 165},
        latency_ms=456.7,
    )
    print(f"\n  响应 JSON Schema:")
    schema = PredictResponse.model_json_schema()
    print(f"  字段: {list(schema.get('properties', {}).keys())}")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行所有演示。"""
    print("🐍 Pydantic 数据验证 — BaseModel、Field、验证器、序列化")
    print("=" * 60)

    demo_basic_models()
    demo_validators()
    demo_nested_models()
    demo_fastapi_models()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. Pydantic BaseModel 自动做类型验证和转换")
    print("  2. Field() 添加约束（min_length、ge、le 等）")
    print("  3. @field_validator 实现字段级自定义验证")
    print("  4. @model_validator 实现跨字段验证")
    print("  5. model_dump() / model_dump_json() 灵活序列化")
    print("  6. FastAPI 原生集成 Pydantic，请求/响应自动验证")


if __name__ == "__main__":
    main()
