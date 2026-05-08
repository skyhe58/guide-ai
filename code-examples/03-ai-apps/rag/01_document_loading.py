"""
文档加载 — PDF / Markdown / HTML / 代码文件加载器

知识点：多格式文档加载、元数据提取、文本清洗、自定义加载器、
       加载管道设计、惰性加载、增量加载策略

Python 版本：3.11+
依赖：标准库（默认模式）、langchain-community / PyPDF2 / beautifulsoup4（服务模式）
最后验证：2024-12-01

外部服务（可选）：
  无外部服务依赖，所有示例使用模拟数据
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ============================================================
# 1. Document 数据模型（模拟 LangChain Document）
# ============================================================

@dataclass
class Document:
    """统一的文档数据模型。

    模拟 LangChain 的 Document 对象，包含文本内容和元数据。
    所有加载器的输出都是 Document 列表。
    """
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        content_preview = self.page_content[:80].replace("\n", " ")
        return f"Document(content='{content_preview}...', metadata={self.metadata})"

    def __len__(self) -> int:
        return len(self.page_content)


# ============================================================
# 2. 基础加载器接口
# ============================================================

class BaseLoader:
    """文档加载器基类。

    所有加载器都继承此基类，实现 load() 和 lazy_load() 方法。
    """

    def load(self) -> list[Document]:
        """加载所有文档到内存。"""
        return list(self.lazy_load())

    def lazy_load(self) -> Generator[Document, None, None]:
        """惰性加载文档，逐条 yield。

        大文件场景推荐使用 lazy_load 避免内存溢出。
        """
        raise NotImplementedError("子类必须实现 lazy_load 方法")


# ============================================================
# 3. PDF 文档加载器（模拟）
# ============================================================

class SimulatedPDFLoader(BaseLoader):
    """模拟 PDF 文档加载器。

    模拟 PyPDFLoader 的行为：按页加载 PDF，每页生成一个 Document。
    实际生产中使用 langchain_community.document_loaders.PyPDFLoader。
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        # 模拟 PDF 内容（实际中由 PyPDF2 解析）
        self._simulated_pages = [
            "第一章：RAG 系统概述\n\n"
            "RAG（Retrieval-Augmented Generation）是一种将检索和生成结合的技术架构。"
            "它通过从外部知识库检索相关文档，将检索结果作为上下文输入大语言模型，"
            "从而生成更准确、更有依据的回答。\n\n"
            "RAG 的核心优势在于：\n"
            "1. 减少幻觉：基于真实文档生成，而非纯粹依赖模型记忆\n"
            "2. 知识可更新：只需更新知识库，无需重新训练模型\n"
            "3. 可追溯：可以标注答案来源，提升可信度",

            "第二章：文档加载与预处理\n\n"
            "文档加载是 RAG 系统的第一步，负责将各种格式的文档转换为统一的文本格式。"
            "常见的文档格式包括 PDF、Markdown、HTML、Word、代码文件等。\n\n"
            "加载器的核心职责：\n"
            "1. 解析文档格式，提取文本内容\n"
            "2. 保留文档结构信息（标题、段落、表格）\n"
            "3. 提取元数据（文件名、页码、作者、创建时间）\n"
            "4. 处理编码问题（UTF-8、GBK 等）",

            "第三章：向量化与检索\n\n"
            "文档经过切分后，需要通过 Embedding 模型转换为高维向量，"
            "存入向量数据库。用户查询时，将查询文本也转为向量，"
            "通过近似最近邻（ANN）搜索找到最相关的文档片段。\n\n"
            "常用的 Embedding 模型：\n"
            "- OpenAI text-embedding-3-small/large\n"
            "- BGE-large-zh-v1.5（中文推荐）\n"
            "- M3E-large（中文推荐）",
        ]

    def lazy_load(self) -> Generator[Document, None, None]:
        """按页加载 PDF 文档。"""
        file_name = os.path.basename(self.file_path)
        for page_num, content in enumerate(self._simulated_pages):
            yield Document(
                page_content=content,
                metadata={
                    "source": self.file_path,
                    "file_name": file_name,
                    "page": page_num,
                    "total_pages": len(self._simulated_pages),
                    "file_type": "pdf",
                },
            )


# ============================================================
# 4. Markdown 文档加载器
# ============================================================

class MarkdownLoader(BaseLoader):
    """Markdown 文档加载器。

    支持两种模式：
    - single: 整个文件作为一个 Document
    - elements: 按标题拆分为多个 Document（保留结构信息）
    """

    def __init__(self, file_path: str, mode: str = "elements"):
        self.file_path = file_path
        self.mode = mode
        # 模拟 Markdown 内容
        self._simulated_content = """# RAG 系统设计指南

## 1. 架构概述

RAG 系统由三个核心组件组成：文档处理管道、向量检索引擎、LLM 生成模块。

### 1.1 文档处理管道

文档处理管道负责将原始文档转换为可检索的向量表示：
- **文档加载**：支持 PDF、Markdown、HTML 等多种格式
- **文档切分**：将长文档拆分为适合 Embedding 的小块
- **向量化**：通过 Embedding 模型将文本转为向量

## 2. 检索策略

### 2.1 向量检索

基于 Embedding 相似度的语义检索，能理解同义词和语义关系。

### 2.2 混合检索

结合向量检索和 BM25 关键词检索，兼顾语义理解和精确匹配。

## 3. 优化技巧

- 查询改写：用 LLM 优化用户查询
- Rerank：用交叉编码器对检索结果重排序
- 上下文压缩：提取文档中最相关的部分
"""

    def lazy_load(self) -> Generator[Document, None, None]:
        """加载 Markdown 文档。"""
        file_name = os.path.basename(self.file_path)

        if self.mode == "single":
            yield Document(
                page_content=self._simulated_content,
                metadata={
                    "source": self.file_path,
                    "file_name": file_name,
                    "file_type": "markdown",
                },
            )
        else:
            # elements 模式：按标题拆分
            sections = self._split_by_headers(self._simulated_content)
            for i, (header, content) in enumerate(sections):
                yield Document(
                    page_content=content.strip(),
                    metadata={
                        "source": self.file_path,
                        "file_name": file_name,
                        "file_type": "markdown",
                        "section": header,
                        "section_index": i,
                    },
                )

    @staticmethod
    def _split_by_headers(content: str) -> list[tuple[str, str]]:
        """按 Markdown 标题拆分内容。"""
        sections: list[tuple[str, str]] = []
        current_header = "Introduction"
        current_content: list[str] = []

        for line in content.split("\n"):
            if line.startswith("#"):
                if current_content:
                    sections.append((current_header, "\n".join(current_content)))
                current_header = line.lstrip("#").strip()
                current_content = [line]
            else:
                current_content.append(line)

        if current_content:
            sections.append((current_header, "\n".join(current_content)))

        return sections


# ============================================================
# 5. HTML 文档加载器
# ============================================================

class SimpleHTMLLoader(BaseLoader):
    """简易 HTML 文档加载器。

    使用正则表达式去除 HTML 标签，提取纯文本。
    生产环境推荐使用 BeautifulSoup 或 Unstructured。
    """

    def __init__(self, file_path: str, encoding: str = "utf-8"):
        self.file_path = file_path
        self.encoding = encoding
        # 模拟 HTML 内容
        self._simulated_html = """
<html>
<head><title>向量数据库选型指南</title></head>
<body>
<nav>首页 | 文档 | API</nav>
<main>
<h1>向量数据库选型指南</h1>
<p>向量数据库是 RAG 系统的核心组件，负责存储和检索高维向量。</p>
<h2>主流向量数据库对比</h2>
<p>Chroma：轻量级，适合开发和小规模应用。</p>
<p>FAISS：Meta 开源，纯内存操作，速度最快。</p>
<p>Milvus：分布式架构，支持百亿级数据。</p>
<p>Pinecone：全托管云服务，免运维。</p>
<h2>选型建议</h2>
<p>开发阶段用 Chroma，生产环境根据数据规模选择 Milvus 或 Pinecone。</p>
</main>
<footer>版权所有 2024</footer>
</body>
</html>"""

    def lazy_load(self) -> Generator[Document, None, None]:
        """加载 HTML 文档，去除标签提取纯文本。"""
        # 去除 script 和 style 标签
        text = re.sub(r"<script[^>]*>.*?</script>", "", self._simulated_html, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        # 去除导航和页脚（简单规则）
        text = re.sub(r"<nav[^>]*>.*?</nav>", "", text, flags=re.DOTALL)
        text = re.sub(r"<footer[^>]*>.*?</footer>", "", text, flags=re.DOTALL)
        # 提取 title
        title_match = re.search(r"<title>(.*?)</title>", self._simulated_html)
        title = title_match.group(1) if title_match else "Unknown"
        # 去除所有 HTML 标签
        text = re.sub(r"<[^>]+>", "\n", text)
        # 清理多余空白
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        yield Document(
            page_content=text,
            metadata={
                "source": self.file_path,
                "title": title,
                "file_type": "html",
            },
        )


# ============================================================
# 6. 代码文件加载器
# ============================================================

class PythonCodeLoader(BaseLoader):
    """Python 代码文件加载器。

    按函数和类拆分代码文件，每个函数/类生成一个 Document。
    保留完整的函数签名和文档字符串。
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        # 模拟 Python 代码内容
        self._simulated_code = '''
class VectorStore:
    """向量存储基类。"""

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.vectors = []
        self.documents = []

    def add_documents(self, documents: list, embeddings: list) -> None:
        """添加文档和对应的向量。"""
        self.vectors.extend(embeddings)
        self.documents.extend(documents)

    def similarity_search(self, query_vector: list, k: int = 5) -> list:
        """基于余弦相似度的向量检索。"""
        import numpy as np
        query = np.array(query_vector)
        scores = []
        for vec in self.vectors:
            v = np.array(vec)
            sim = np.dot(query, v) / (np.linalg.norm(query) * np.linalg.norm(v))
            scores.append(sim)
        top_k = np.argsort(scores)[-k:][::-1]
        return [self.documents[i] for i in top_k]


def create_embeddings(texts: list[str], model: str = "bge-large-zh") -> list[list[float]]:
    """批量生成文本 Embedding。"""
    import numpy as np
    # 模拟 Embedding 生成
    return [np.random.randn(1536).tolist() for _ in texts]


def chunk_documents(documents: list, chunk_size: int = 500) -> list:
    """将文档切分为固定大小的块。"""
    chunks = []
    for doc in documents:
        text = doc if isinstance(doc, str) else doc.page_content
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])
    return chunks
'''

    def lazy_load(self) -> Generator[Document, None, None]:
        """按函数/类拆分代码文件。"""
        file_name = os.path.basename(self.file_path)
        blocks = self._extract_code_blocks(self._simulated_code)

        for block_type, name, code in blocks:
            yield Document(
                page_content=code.strip(),
                metadata={
                    "source": self.file_path,
                    "file_name": file_name,
                    "file_type": "python",
                    "block_type": block_type,
                    "block_name": name,
                },
            )

    @staticmethod
    def _extract_code_blocks(code: str) -> list[tuple[str, str, str]]:
        """提取代码中的函数和类定义。"""
        blocks: list[tuple[str, str, str]] = []
        lines = code.split("\n")
        current_block: list[str] = []
        current_type = ""
        current_name = ""
        indent_level = 0

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("class ") or stripped.startswith("def "):
                # 保存上一个块
                if current_block and current_name:
                    blocks.append((current_type, current_name, "\n".join(current_block)))
                # 开始新块
                if stripped.startswith("class "):
                    current_type = "class"
                    current_name = stripped.split("(")[0].replace("class ", "").rstrip(":")
                else:
                    current_type = "function"
                    current_name = stripped.split("(")[0].replace("def ", "")
                current_block = [line]
                indent_level = len(line) - len(line.lstrip())
            elif current_block:
                current_block.append(line)

        # 保存最后一个块
        if current_block and current_name:
            blocks.append((current_type, current_name, "\n".join(current_block)))

        return blocks


# ============================================================
# 7. 文档加载管道（工厂模式）
# ============================================================

class DocumentLoadingPipeline:
    """文档加载管道。

    根据文件扩展名自动选择合适的加载器，
    支持批量加载、文本清洗、元数据补充。
    """

    # 文件扩展名 → 加载器映射
    LOADER_MAP: dict[str, type[BaseLoader]] = {
        ".pdf": SimulatedPDFLoader,
        ".md": MarkdownLoader,
        ".html": SimpleHTMLLoader,
        ".htm": SimpleHTMLLoader,
        ".py": PythonCodeLoader,
    }

    def __init__(self, clean_text: bool = True):
        self.clean_text = clean_text
        self._loaded_hashes: set[str] = set()  # 用于增量加载

    def load_file(self, file_path: str) -> list[Document]:
        """加载单个文件。"""
        ext = Path(file_path).suffix.lower()
        loader_class = self.LOADER_MAP.get(ext)

        if loader_class is None:
            print(f"  ⚠️ 不支持的文件格式: {ext}，跳过 {file_path}")
            return []

        try:
            loader = loader_class(file_path)
            documents = loader.load()

            if self.clean_text:
                documents = [self._clean_document(doc) for doc in documents]

            return documents
        except Exception as e:
            print(f"  ❌ 加载失败: {file_path}, 错误: {e}")
            return []

    def load_directory(self, dir_path: str, glob_pattern: str = "*") -> list[Document]:
        """批量加载目录下的文件。"""
        all_documents: list[Document] = []
        supported_extensions = set(self.LOADER_MAP.keys())

        # 模拟目录文件列表
        simulated_files = [
            f"{dir_path}/report.pdf",
            f"{dir_path}/guide.md",
            f"{dir_path}/api-docs.html",
            f"{dir_path}/vector_store.py",
            f"{dir_path}/image.png",  # 不支持的格式
        ]

        print(f"  📁 扫描目录: {dir_path}")
        for file_path in simulated_files:
            ext = Path(file_path).suffix.lower()
            if ext in supported_extensions:
                docs = self.load_file(file_path)
                all_documents.extend(docs)
                print(f"    ✅ {os.path.basename(file_path)}: 加载 {len(docs)} 个文档块")
            else:
                print(f"    ⏭️ {os.path.basename(file_path)}: 不支持的格式，跳过")

        return all_documents

    def load_incremental(self, file_path: str) -> list[Document] | None:
        """增量加载：只加载新增或修改的文件。"""
        # 计算文件内容哈希（模拟）
        file_hash = hashlib.md5(file_path.encode()).hexdigest()

        if file_hash in self._loaded_hashes:
            print(f"  ⏭️ 文件未变更，跳过: {file_path}")
            return None

        self._loaded_hashes.add(file_hash)
        return self.load_file(file_path)

    @staticmethod
    def _clean_document(doc: Document) -> Document:
        """清洗文档文本。"""
        text = doc.page_content
        # 去除多余空白行
        text = re.sub(r"\n{3,}", "\n\n", text)
        # 去除行首行尾空白
        text = "\n".join(line.strip() for line in text.split("\n"))
        # 去除特殊控制字符
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        doc.page_content = text.strip()
        return doc


# ============================================================
# 8. 演示函数
# ============================================================

def demo_pdf_loading() -> None:
    """演示 PDF 文档加载。"""
    print("\n" + "=" * 60)
    print("1. PDF 文档加载")
    print("=" * 60)

    loader = SimulatedPDFLoader("knowledge_base/rag_guide.pdf")
    documents = loader.load()

    print(f"  加载了 {len(documents)} 页 PDF 文档")
    for doc in documents:
        print(f"\n  📄 第 {doc.metadata['page']} 页:")
        print(f"     内容预览: {doc.page_content[:80]}...")
        print(f"     元数据: {doc.metadata}")


def demo_markdown_loading() -> None:
    """演示 Markdown 文档加载。"""
    print("\n" + "=" * 60)
    print("2. Markdown 文档加载")
    print("=" * 60)

    # elements 模式：按标题拆分
    loader = MarkdownLoader("docs/rag_design.md", mode="elements")
    documents = loader.load()

    print(f"  加载了 {len(documents)} 个章节")
    for doc in documents:
        print(f"\n  📝 章节: {doc.metadata.get('section', 'N/A')}")
        print(f"     内容预览: {doc.page_content[:80]}...")


def demo_html_loading() -> None:
    """演示 HTML 文档加载。"""
    print("\n" + "=" * 60)
    print("3. HTML 文档加载")
    print("=" * 60)

    loader = SimpleHTMLLoader("web/vector_db_guide.html")
    documents = loader.load()

    print(f"  加载了 {len(documents)} 个文档")
    for doc in documents:
        print(f"\n  🌐 标题: {doc.metadata.get('title', 'N/A')}")
        print(f"     内容预览: {doc.page_content[:120]}...")
        print(f"     去除了导航栏和页脚噪声")


def demo_code_loading() -> None:
    """演示代码文件加载。"""
    print("\n" + "=" * 60)
    print("4. Python 代码文件加载")
    print("=" * 60)

    loader = PythonCodeLoader("src/vector_store.py")
    documents = loader.load()

    print(f"  加载了 {len(documents)} 个代码块")
    for doc in documents:
        print(f"\n  🐍 {doc.metadata['block_type']}: {doc.metadata['block_name']}")
        print(f"     代码预览: {doc.page_content[:80]}...")


def demo_pipeline() -> None:
    """演示文档加载管道。"""
    print("\n" + "=" * 60)
    print("5. 文档加载管道（自动选择加载器）")
    print("=" * 60)

    pipeline = DocumentLoadingPipeline(clean_text=True)
    documents = pipeline.load_directory("knowledge_base")

    print(f"\n  📊 加载统计:")
    print(f"     总文档块数: {len(documents)}")

    # 按文件类型统计
    type_counts: dict[str, int] = {}
    for doc in documents:
        ft = doc.metadata.get("file_type", "unknown")
        type_counts[ft] = type_counts.get(ft, 0) + 1

    for ft, count in type_counts.items():
        print(f"     {ft}: {count} 个文档块")


def demo_incremental_loading() -> None:
    """演示增量加载。"""
    print("\n" + "=" * 60)
    print("6. 增量加载（只加载变更文件）")
    print("=" * 60)

    pipeline = DocumentLoadingPipeline()

    # 第一次加载
    print("  第一次加载:")
    docs1 = pipeline.load_incremental("knowledge_base/report.pdf")
    print(f"    结果: 加载了 {len(docs1) if docs1 else 0} 个文档块")

    # 第二次加载同一文件（应该跳过）
    print("\n  第二次加载同一文件:")
    docs2 = pipeline.load_incremental("knowledge_base/report.pdf")
    print(f"    结果: {'跳过（文件未变更）' if docs2 is None else f'加载了 {len(docs2)} 个文档块'}")

    # 加载新文件
    print("\n  加载新文件:")
    docs3 = pipeline.load_incremental("knowledge_base/guide.md")
    print(f"    结果: 加载了 {len(docs3) if docs3 else 0} 个文档块")


def demo_metadata_enrichment() -> None:
    """演示元数据补充策略。"""
    print("\n" + "=" * 60)
    print("7. 元数据补充策略")
    print("=" * 60)

    loader = SimulatedPDFLoader("knowledge_base/rag_guide.pdf")
    documents = loader.load()

    # 补充额外元数据
    for doc in documents:
        doc.metadata.update({
            "department": "AI 研发部",
            "category": "技术文档",
            "language": "zh-CN",
            "char_count": len(doc.page_content),
            "content_hash": hashlib.md5(doc.page_content.encode()).hexdigest()[:8],
        })

    print("  补充元数据后的文档:")
    for doc in documents[:2]:
        print(f"\n  📄 {doc.metadata.get('file_name', 'N/A')} - 第 {doc.metadata['page']} 页")
        meta_display = json.dumps(doc.metadata, ensure_ascii=False, indent=4)
        print(f"     元数据:\n{meta_display}")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行所有文档加载演示。"""
    print("🐍 文档加载 — PDF / Markdown / HTML / 代码文件加载器")
    print("=" * 60)

    demo_pdf_loading()
    demo_markdown_loading()
    demo_html_loading()
    demo_code_loading()
    demo_pipeline()
    demo_incremental_loading()
    demo_metadata_enrichment()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. 不同格式用不同加载器：PDF→PyPDFLoader, MD→MarkdownLoader, HTML→BSHTMLLoader")
    print("  2. 保留元数据（文件名、页码、标题）对检索排序至关重要")
    print("  3. 大文件用 lazy_load() 惰性加载，避免内存溢出")
    print("  4. 加载管道用工厂模式，根据扩展名自动选择加载器")
    print("  5. 增量加载通过文件哈希判断变更，避免重复处理")
    print("  6. 文本清洗去除噪声（多余空白、特殊字符、页眉页脚）")


if __name__ == "__main__":
    main()
