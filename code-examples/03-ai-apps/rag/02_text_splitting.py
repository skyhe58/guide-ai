"""
文档切分 — 多种切分策略对比

知识点：RecursiveCharacterTextSplitter、语义切分、Token 切分、
       Markdown 标题切分、代码切分、切分参数调优、切分质量评估

Python 版本：3.11+
依赖：标准库（默认模式）、langchain-text-splitters / tiktoken（服务模式）
最后验证：2024-12-01

外部服务（可选）：
  无外部服务依赖，所有示例使用模拟数据
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ============================================================
# 1. Document 数据模型
# ============================================================

@dataclass
class Document:
    """统一的文档数据模型。"""
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.page_content[:60].replace("\n", " ")
        return f"Document('{preview}...', meta={self.metadata})"

    def __len__(self) -> int:
        return len(self.page_content)


# ============================================================
# 2. 基础切分器接口
# ============================================================

class BaseTextSplitter:
    """文本切分器基类。"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        """将文本切分为多个块。"""
        raise NotImplementedError

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """将 Document 列表切分为更小的 Document 列表。"""
        result: list[Document] = []
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                new_metadata = {**doc.metadata, "chunk_index": i, "total_chunks": len(chunks)}
                result.append(Document(page_content=chunk, metadata=new_metadata))
        return result


# ============================================================
# 3. 字符切分器（最简单）
# ============================================================

class CharacterTextSplitter(BaseTextSplitter):
    """按固定字符数切分。

    最简单的切分方式，不考虑语义边界。
    适合快速原型，不推荐生产使用。
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, separator: str = "\n"):
        super().__init__(chunk_size, chunk_overlap)
        self.separator = separator

    def split_text(self, text: str) -> list[str]:
        """按固定字符数切分文本。"""
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            # 尝试在分隔符处切分
            if end < len(text) and self.separator:
                last_sep = chunk.rfind(self.separator)
                if last_sep > self.chunk_size // 2:
                    end = start + last_sep + len(self.separator)
                    chunk = text[start:end]
            chunks.append(chunk.strip())
            start = end - self.chunk_overlap
        return [c for c in chunks if c]


# ============================================================
# 4. 递归字符切分器（推荐默认）
# ============================================================

class RecursiveCharacterTextSplitter(BaseTextSplitter):
    """递归字符切分器（模拟 LangChain 实现）。

    按分隔符层级递归切分：
    1. 先尝试按段落（\\n\\n）切分
    2. 段落太长则按行（\\n）切分
    3. 行太长则按句子（。！？）切分
    4. 最后按字符切分（兜底）

    这是最常用的切分器，适合 90% 的场景。
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ):
        super().__init__(chunk_size, chunk_overlap)
        # 默认分隔符层级（中文优化）
        self.separators = separators or ["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", " ", ""]

    def split_text(self, text: str) -> list[str]:
        """递归切分文本。"""
        final_chunks: list[str] = []
        self._split_recursive(text, self.separators, final_chunks)
        return [c.strip() for c in final_chunks if c.strip()]

    def _split_recursive(self, text: str, separators: list[str], chunks: list[str]) -> None:
        """递归切分逻辑。"""
        if len(text) <= self.chunk_size:
            chunks.append(text)
            return

        # 找到合适的分隔符
        separator = separators[-1]  # 默认用最后一个（空字符串）
        for sep in separators:
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                break

        # 按分隔符切分
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        # 合并小块
        current_chunk: list[str] = []
        current_length = 0

        for split in splits:
            split_with_sep = split + separator if separator else split
            if current_length + len(split_with_sep) > self.chunk_size and current_chunk:
                merged = separator.join(current_chunk) if separator else "".join(current_chunk)
                if len(merged) > self.chunk_size:
                    # 当前块太大，用下一级分隔符继续切分
                    remaining_seps = separators[separators.index(separator) + 1:] if separator in separators else [""]
                    if remaining_seps:
                        self._split_recursive(merged, remaining_seps, chunks)
                    else:
                        chunks.append(merged[:self.chunk_size])
                else:
                    chunks.append(merged)
                # 处理重叠
                overlap_text = separator.join(current_chunk[-2:]) if len(current_chunk) > 1 else ""
                current_chunk = []
                current_length = 0
                if overlap_text and len(overlap_text) <= self.chunk_overlap:
                    current_chunk = [overlap_text]
                    current_length = len(overlap_text)

            current_chunk.append(split)
            current_length += len(split_with_sep)

        # 处理剩余内容
        if current_chunk:
            merged = separator.join(current_chunk) if separator else "".join(current_chunk)
            chunks.append(merged)


# ============================================================
# 5. Markdown 标题切分器
# ============================================================

class MarkdownHeaderTextSplitter:
    """按 Markdown 标题层级切分。

    保留标题层级信息作为元数据，适合技术文档。
    """

    def __init__(self, headers_to_split_on: list[tuple[str, str]] | None = None):
        self.headers_to_split_on = headers_to_split_on or [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]

    def split_text(self, text: str) -> list[Document]:
        """按标题切分 Markdown 文本。"""
        lines = text.split("\n")
        documents: list[Document] = []
        current_headers: dict[str, str] = {}
        current_content: list[str] = []

        for line in lines:
            is_header = False
            for marker, header_name in sorted(self.headers_to_split_on, key=lambda x: len(x[0]), reverse=True):
                if line.startswith(marker + " "):
                    # 保存之前的内容
                    if current_content:
                        content = "\n".join(current_content).strip()
                        if content:
                            documents.append(Document(
                                page_content=content,
                                metadata={**current_headers},
                            ))
                    # 更新标题层级
                    header_text = line[len(marker) + 1:].strip()
                    current_headers[header_name] = header_text
                    # 清除更低层级的标题
                    header_levels = [h[1] for h in self.headers_to_split_on]
                    current_idx = header_levels.index(header_name)
                    for lower_header in header_levels[current_idx + 1:]:
                        current_headers.pop(lower_header, None)
                    current_content = []
                    is_header = True
                    break

            if not is_header:
                current_content.append(line)

        # 保存最后一段
        if current_content:
            content = "\n".join(current_content).strip()
            if content:
                documents.append(Document(page_content=content, metadata={**current_headers}))

        return documents


# ============================================================
# 6. Token 切分器（模拟）
# ============================================================

class SimpleTokenTextSplitter(BaseTextSplitter):
    """基于 Token 数量的切分器（简化版）。

    使用简单的 Token 估算（中文 1 字 ≈ 1.5 token，英文 1 词 ≈ 1.3 token）。
    生产环境推荐使用 tiktoken 精确计算。
    """

    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 30):
        super().__init__(chunk_size, chunk_overlap)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """估算文本的 Token 数量。"""
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        english_words = len(re.findall(r"[a-zA-Z]+", text))
        return int(chinese_chars * 1.5 + english_words * 1.3)

    def split_text(self, text: str) -> list[str]:
        """按 Token 数量切分文本。"""
        sentences = re.split(r"(?<=[。！？.!?\n])", text)
        chunks: list[str] = []
        current_chunk: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_tokens = self._estimate_tokens(sentence)

            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunks.append("".join(current_chunk))
                # 保留重叠部分
                overlap_chunk: list[str] = []
                overlap_tokens = 0
                for s in reversed(current_chunk):
                    s_tokens = self._estimate_tokens(s)
                    if overlap_tokens + s_tokens > self.chunk_overlap:
                        break
                    overlap_chunk.insert(0, s)
                    overlap_tokens += s_tokens
                current_chunk = overlap_chunk
                current_tokens = overlap_tokens

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        if current_chunk:
            chunks.append("".join(current_chunk))

        return [c for c in chunks if c.strip()]


# ============================================================
# 7. 切分质量评估
# ============================================================

class ChunkQualityEvaluator:
    """切分质量评估工具。"""

    @staticmethod
    def evaluate(chunks: list[str], original_text: str) -> dict[str, Any]:
        """评估切分质量。"""
        if not chunks:
            return {"error": "没有切分结果"}

        lengths = [len(c) for c in chunks]
        avg_length = sum(lengths) / len(lengths)
        min_length = min(lengths)
        max_length = max(lengths)

        # 检查信息丢失
        total_chunk_chars = sum(lengths)
        original_chars = len(original_text.strip())

        # 检查是否有截断的句子（以非标点结尾）
        truncated_count = sum(
            1 for c in chunks
            if c and c[-1] not in "。！？.!?\n"
        )

        return {
            "chunk_count": len(chunks),
            "avg_length": round(avg_length, 1),
            "min_length": min_length,
            "max_length": max_length,
            "length_std": round(
                (sum((l - avg_length) ** 2 for l in lengths) / len(lengths)) ** 0.5, 1
            ),
            "truncated_chunks": truncated_count,
            "truncated_ratio": round(truncated_count / len(chunks), 2),
            "coverage": round(total_chunk_chars / max(original_chars, 1), 2),
        }


# ============================================================
# 8. 演示函数
# ============================================================

# 测试用长文本
SAMPLE_TEXT = """RAG（Retrieval-Augmented Generation）系统设计指南

RAG 是一种将检索和生成结合的技术架构。它通过从外部知识库检索相关文档，将检索结果作为上下文输入大语言模型，从而生成更准确、更有依据的回答。

文档加载是 RAG 系统的第一步。负责将各种格式的文档转换为统一的文本格式。常见的文档格式包括 PDF、Markdown、HTML、Word、代码文件等。加载器的核心职责包括解析文档格式、提取文本内容、保留文档结构信息、提取元数据。

文档切分是 RAG 系统中将长文档拆分为适合 Embedding 和检索的小块的过程。切分质量直接决定了检索的精度和生成的质量。切分太大会引入噪声，切分太小会丢失上下文。

Embedding 模型将文本转换为高维向量，使得语义相似的文本在向量空间中距离更近。常用的 Embedding 模型包括 OpenAI text-embedding-3、BGE-large-zh、M3E-large 等。选择 Embedding 模型需要考虑语言支持、维度、速度和成本。

向量数据库是专门用于存储和检索高维向量的数据库系统。主流的向量数据库包括 Chroma、FAISS、Milvus、Pinecone 等。选择向量数据库需要考虑数据规模、部署方式、性能需求和运维能力。

检索策略决定了如何从向量数据库中找到最相关的文档。常见的检索策略包括向量相似度检索、BM25 关键词检索、混合检索和 MMR 多样性检索。生产环境推荐使用混合检索加 Rerank 的组合。

Rerank 重排序是在初步检索之后对候选文档重新打分排序的过程。它使用交叉编码器捕获查询和文档之间的细粒度语义交互，显著提升检索精度。常用的 Rerank 模型包括 Cohere Rerank 和 BGE-Reranker。"""


def demo_character_splitter() -> None:
    """演示字符切分器。"""
    print("\n" + "=" * 60)
    print("1. 字符切分器（CharacterTextSplitter）")
    print("=" * 60)

    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    chunks = splitter.split_text(SAMPLE_TEXT)

    print(f"  参数: chunk_size=200, chunk_overlap=30")
    print(f"  切分结果: {len(chunks)} 个块")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n  Chunk {i}: ({len(chunk)} 字符)")
        print(f"    {chunk[:80]}...")


def demo_recursive_splitter() -> None:
    """演示递归字符切分器。"""
    print("\n" + "=" * 60)
    print("2. 递归字符切分器（RecursiveCharacterTextSplitter）⭐ 推荐")
    print("=" * 60)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )
    chunks = splitter.split_text(SAMPLE_TEXT)

    print(f"  参数: chunk_size=200, chunk_overlap=30")
    print(f"  分隔符层级: \\n\\n → \\n → 。 → 空格 → 字符")
    print(f"  切分结果: {len(chunks)} 个块")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n  Chunk {i}: ({len(chunk)} 字符)")
        print(f"    {chunk[:100]}...")


def demo_markdown_splitter() -> None:
    """演示 Markdown 标题切分器。"""
    print("\n" + "=" * 60)
    print("3. Markdown 标题切分器")
    print("=" * 60)

    md_text = """# RAG 系统设计

## 1. 文档加载

文档加载是 RAG 的第一步，支持 PDF、Markdown、HTML 等格式。

### 1.1 PDF 加载

使用 PyPDFLoader 加载 PDF 文件，每页生成一个 Document。

### 1.2 Markdown 加载

使用 UnstructuredMarkdownLoader 加载 Markdown 文件。

## 2. 文档切分

文档切分将长文档拆分为适合 Embedding 的小块。

### 2.1 递归切分

RecursiveCharacterTextSplitter 是最常用的切分器。

## 3. 向量检索

基于 Embedding 相似度的语义检索。"""

    splitter = MarkdownHeaderTextSplitter()
    documents = splitter.split_text(md_text)

    print(f"  切分结果: {len(documents)} 个文档块")
    for doc in documents:
        print(f"\n  📝 元数据: {doc.metadata}")
        print(f"     内容: {doc.page_content[:80]}...")


def demo_token_splitter() -> None:
    """演示 Token 切分器。"""
    print("\n" + "=" * 60)
    print("4. Token 切分器")
    print("=" * 60)

    splitter = SimpleTokenTextSplitter(chunk_size=100, chunk_overlap=15)
    chunks = splitter.split_text(SAMPLE_TEXT)

    print(f"  参数: chunk_size=100 tokens, chunk_overlap=15 tokens")
    print(f"  切分结果: {len(chunks)} 个块")
    for i, chunk in enumerate(chunks[:3]):
        est_tokens = splitter._estimate_tokens(chunk)
        print(f"\n  Chunk {i}: (~{est_tokens} tokens, {len(chunk)} 字符)")
        print(f"    {chunk[:80]}...")


def demo_strategy_comparison() -> None:
    """对比不同切分策略。"""
    print("\n" + "=" * 60)
    print("5. 切分策略对比")
    print("=" * 60)

    evaluator = ChunkQualityEvaluator()

    strategies = {
        "字符切分": CharacterTextSplitter(chunk_size=200, chunk_overlap=30),
        "递归字符切分": RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30),
        "Token 切分": SimpleTokenTextSplitter(chunk_size=100, chunk_overlap=15),
    }

    print(f"  {'策略':<14} {'块数':>4} {'平均长度':>8} {'最小':>6} {'最大':>6} {'截断比':>8}")
    print(f"  {'-' * 52}")

    for name, splitter in strategies.items():
        chunks = splitter.split_text(SAMPLE_TEXT)
        metrics = evaluator.evaluate(chunks, SAMPLE_TEXT)
        print(
            f"  {name:<14} {metrics['chunk_count']:>4} "
            f"{metrics['avg_length']:>8.1f} {metrics['min_length']:>6} "
            f"{metrics['max_length']:>6} {metrics['truncated_ratio']:>8.0%}"
        )


def demo_chunk_size_impact() -> None:
    """演示 chunk_size 对切分效果的影响。"""
    print("\n" + "=" * 60)
    print("6. chunk_size 对切分效果的影响")
    print("=" * 60)

    evaluator = ChunkQualityEvaluator()
    sizes = [100, 200, 300, 500, 800]

    print(f"  {'chunk_size':>10} {'块数':>6} {'平均长度':>8} {'截断比':>8}")
    print(f"  {'-' * 36}")

    for size in sizes:
        splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=int(size * 0.1))
        chunks = splitter.split_text(SAMPLE_TEXT)
        metrics = evaluator.evaluate(chunks, SAMPLE_TEXT)
        print(
            f"  {size:>10} {metrics['chunk_count']:>6} "
            f"{metrics['avg_length']:>8.1f} {metrics['truncated_ratio']:>8.0%}"
        )

    print("\n  💡 建议: QA 场景用 256-512，摘要场景用 512-1024")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行所有切分策略演示。"""
    print("🐍 文档切分 — 多种切分策略对比")
    print("=" * 60)

    demo_character_splitter()
    demo_recursive_splitter()
    demo_markdown_splitter()
    demo_token_splitter()
    demo_strategy_comparison()
    demo_chunk_size_impact()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. 默认使用 RecursiveCharacterTextSplitter（分隔符层级递归）")
    print("  2. 中文文档添加中文标点分隔符（。！？；）")
    print("  3. chunk_size 根据场景调整：QA=256-512, 摘要=512-1024")
    print("  4. chunk_overlap 设为 chunk_size 的 10-20%")
    print("  5. Markdown 文档用 MarkdownHeaderTextSplitter 保留结构")
    print("  6. 切分后用评估工具检查质量（截断比、长度分布）")


if __name__ == "__main__":
    main()
