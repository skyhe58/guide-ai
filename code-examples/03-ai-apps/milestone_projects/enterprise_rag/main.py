"""
企业级 RAG 知识库 — 文档加载→切分→向量化→检索→生成→评估完整链路

知识点：完整的 RAG 流水线实现，包括文档加载、智能切分、Embedding 向量化、
       向量存储与检索、Rerank 重排序、LLM 生成、质量评估

Python 版本：3.11+
依赖：标准库（默认模式）、chromadb + ollama（服务模式）
最后验证：2024-12-01

外部服务（可选）：
  Ollama: docker compose -f docker/docker-compose.yml up -d ollama
  Chroma: docker compose -f docker/docker-compose.yml up -d chroma
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any

# ============================================================
# 1. 文档加载器
# ============================================================

@dataclass
class Document:
    """文档对象。"""
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = hashlib.md5(self.content[:100].encode()).hexdigest()[:8]


class DocumentLoader:
    """文档加载器 — 支持多种格式。"""

    def __init__(self, data_dir: str = "data/"):
        self.data_dir = data_dir

    def load_all(self) -> list[Document]:
        """加载所有文档（模拟）。"""
        # 模拟企业知识库文档
        documents = [
            Document("公司简介：我们是一家专注于 AI 技术的科技公司，成立于 2020 年。主要业务包括自然语言处理、计算机视觉和智能推荐系统。公司总部位于北京，在上海和深圳设有研发中心。", {"source": "company_intro.md", "category": "公司信息"}),
            Document("产品文档：智能客服系统 v3.0 支持多轮对话、意图识别、知识库问答和工单创建。系统基于 RAG 架构，使用 Qwen2 作为基础模型，Chroma 作为向量数据库。日均处理 10 万+ 用户咨询。", {"source": "product_doc.md", "category": "产品文档"}),
            Document("技术架构：系统采用微服务架构，核心服务包括 API 网关（FastAPI）、检索服务（Chroma + BGE-M3）、生成服务（vLLM + Qwen2）、对话管理服务（Redis）。所有服务通过 Docker Compose 编排。", {"source": "tech_arch.md", "category": "技术文档"}),
            Document("员工手册：新员工入职流程包括 HR 面谈、IT 设备领取、部门介绍、导师分配。试用期为 3 个月，转正需通过部门评审。年假制度：工作满 1 年享有 5 天年假，每增加 1 年增加 1 天。", {"source": "employee_handbook.md", "category": "人事制度"}),
            Document("API 文档：智能客服 API 接口说明。POST /api/chat 发送用户消息，返回 AI 回答。请求参数：message（字符串）、session_id（会话 ID）、user_id（用户 ID）。返回：answer（回答）、sources（来源）、confidence（置信度）。", {"source": "api_doc.md", "category": "技术文档"}),
            Document("常见问题 FAQ：Q1 如何重置密码？A1 访问 SSO 登录页面，点击忘记密码，输入邮箱验证后重置。Q2 如何申请 VPN？A2 在 IT 服务台提交 VPN 申请工单，审批通过后 IT 会发送配置文件。Q3 如何报销差旅费？A3 在 OA 系统提交报销申请，附上发票和行程单。", {"source": "faq.md", "category": "常见问题"}),
            Document("数据安全规范：所有用户数据必须加密存储（AES-256）。API 调用需要 Bearer Token 认证。敏感数据（身份证、银行卡）需要脱敏处理。日志中不得记录用户密码和 Token。数据保留期限为 3 年，到期自动清理。", {"source": "security_policy.md", "category": "安全规范"}),
            Document("模型部署指南：推荐使用 vLLM 部署 Qwen2-7B 模型。最低配置：NVIDIA A10 GPU（24GB 显存）。启动命令：python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2-7B-Instruct --port 8080。支持 OpenAI 兼容 API。", {"source": "deployment_guide.md", "category": "技术文档"}),
        ]
        print(f"  [文档加载] 加载了 {len(documents)} 个文档")
        return documents


# ============================================================
# 2. 文档切分器
# ============================================================

@dataclass
class TextChunk:
    """文本块。"""
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""
    embedding: list[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = hashlib.md5(self.text[:50].encode()).hexdigest()[:8]


class TextSplitter:
    """智能文本切分器。"""

    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: list[Document]) -> list[TextChunk]:
        """切分所有文档。"""
        chunks = []
        for doc in documents:
            doc_chunks = self._split_text(doc.content, doc.metadata)
            chunks.extend(doc_chunks)
        print(f"  [文档切分] {len(documents)} 个文档 → {len(chunks)} 个文本块")
        return chunks

    def _split_text(self, text: str, metadata: dict) -> list[TextChunk]:
        """切分单个文本。"""
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            # 在句子边界切分
            if end < len(text):
                for sep in ["。", ".", "\n", "；"]:
                    pos = text[start:end].rfind(sep)
                    if pos > self.chunk_size // 2:
                        end = start + pos + 1
                        break
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(TextChunk(
                    text=chunk_text,
                    metadata={**metadata, "chunk_index": idx},
                ))
                idx += 1
            start = end - self.chunk_overlap if end < len(text) else end
        return chunks


# ============================================================
# 3. Embedding 服务
# ============================================================

class EmbeddingService:
    """Embedding 向量化服务（模拟）。"""

    def __init__(self, model_name: str = "bge-m3", dim: int = 8):
        self.model_name = model_name
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        """生成文本 Embedding。"""
        h = int(hashlib.md5(text.encode()).hexdigest(), 16)
        vec = [((h >> (i * 4)) & 0xF) / 15.0 - 0.5 for i in range(self.dim)]
        norm = math.sqrt(sum(v * v for v in vec))
        return [round(v / norm, 4) for v in vec] if norm > 0 else vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量 Embedding。"""
        return [self.embed(t) for t in texts]


def cosine_sim(a: list[float], b: list[float]) -> float:
    """余弦相似度。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


# ============================================================
# 4. 向量存储
# ============================================================

class VectorStore:
    """向量存储（内存模拟）。"""

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.chunks: list[TextChunk] = []

    def add_chunks(self, chunks: list[TextChunk]) -> None:
        """添加文本块并生成 Embedding。"""
        for chunk in chunks:
            chunk.embedding = self.embedding_service.embed(chunk.text)
        self.chunks.extend(chunks)
        print(f"  [向量存储] 存入 {len(chunks)} 个向量")

    def search(self, query: str, top_k: int = 5) -> list[tuple[TextChunk, float]]:
        """向量相似度检索。"""
        query_emb = self.embedding_service.embed(query)
        scored = [(chunk, cosine_sim(query_emb, chunk.embedding)) for chunk in self.chunks]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ============================================================
# 5. Rerank 重排序
# ============================================================

class Reranker:
    """重排序器（模拟交叉编码器）。"""

    def rerank(self, query: str, chunks: list[tuple[TextChunk, float]], top_k: int = 3) -> list[tuple[TextChunk, float]]:
        """对检索结果重排序。"""
        reranked = []
        query_words = set(query.lower().split())
        for chunk, orig_score in chunks:
            chunk_words = set(chunk.text.lower().split())
            # 模拟交叉编码器评分：词汇重叠 + 原始分数
            overlap = len(query_words & chunk_words) / max(len(query_words), 1)
            new_score = orig_score * 0.4 + overlap * 0.6
            reranked.append((chunk, round(new_score, 4)))
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]


# ============================================================
# 6. LLM 生成服务
# ============================================================

class LLMService:
    """LLM 生成服务（模拟）。"""

    def __init__(self, model_name: str = "qwen2"):
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        """生成回答。"""
        # 模拟基于上下文的回答生成
        if "上下文" in prompt or "context" in prompt.lower():
            # 提取上下文中的关键信息
            lines = prompt.split("\n")
            context_lines = [l for l in lines if len(l) > 20 and "问题" not in l and "上下文" not in l]
            if context_lines:
                return f"根据知识库信息：{context_lines[0][:100]}"
        return "抱歉，我在知识库中没有找到相关信息。请尝试换一种方式提问。"


# ============================================================
# 7. RAG 流水线
# ============================================================

class RAGPipeline:
    """RAG 完整流水线。"""

    def __init__(self):
        self.loader = DocumentLoader()
        self.splitter = TextSplitter(chunk_size=200, chunk_overlap=50)
        self.embedding = EmbeddingService()
        self.vector_store = VectorStore(self.embedding)
        self.reranker = Reranker()
        self.llm = LLMService()
        self.is_indexed = False

    def index(self) -> None:
        """索引流程：加载→切分→向量化→存储。"""
        print("\n  === 索引流程 ===")
        documents = self.loader.load_all()
        chunks = self.splitter.split_documents(documents)
        self.vector_store.add_chunks(chunks)
        self.is_indexed = True
        print(f"  [索引完成] 共 {len(self.vector_store.chunks)} 个向量")

    def query(self, question: str, verbose: bool = True) -> dict:
        """查询流程：检索→重排序→生成。"""
        if not self.is_indexed:
            self.index()

        start_time = time.time()

        # 1. 向量检索
        search_results = self.vector_store.search(question, top_k=5)
        if verbose:
            print(f"\n  [检索] 查询: '{question}'")
            for chunk, score in search_results[:3]:
                print(f"    [{score:.3f}] {chunk.text[:50]}... ({chunk.metadata.get('source', '')})")

        # 2. Rerank 重排序
        reranked = self.reranker.rerank(question, search_results, top_k=3)
        if verbose:
            print(f"  [Rerank] 重排序后:")
            for chunk, score in reranked:
                print(f"    [{score:.3f}] {chunk.text[:50]}...")

        # 3. 构建 Prompt
        context = "\n".join(f"[{i+1}] {c.text}" for i, (c, _) in enumerate(reranked))
        prompt = f"""基于以下上下文回答用户问题。如果上下文不包含答案，请说"我不确定"。

上下文：
{context}

问题：{question}

回答："""

        # 4. LLM 生成
        answer = self.llm.generate(prompt)
        latency = (time.time() - start_time) * 1000

        if verbose:
            print(f"  [生成] {answer[:80]}...")
            print(f"  [耗时] {latency:.1f}ms")

        return {
            "question": question,
            "answer": answer,
            "sources": [{"text": c.text[:60], "source": c.metadata.get("source", ""), "score": s}
                       for c, s in reranked],
            "latency_ms": round(latency, 2),
        }


# ============================================================
# 8. 评估模块
# ============================================================

class RAGEvaluator:
    """RAG 评估器。"""

    def evaluate(self, results: list[dict], ground_truths: list[str]) -> dict:
        """评估 RAG 系统质量。"""
        scores = {"faithfulness": [], "relevancy": [], "precision": []}

        for result, gt in zip(results, ground_truths):
            answer = result["answer"]
            sources = [s["text"] for s in result["sources"]]
            context = " ".join(sources)

            # 忠实度
            a_words = set(answer.lower().split())
            c_words = set(context.lower().split())
            faith = len(a_words & c_words) / max(len(a_words), 1)
            scores["faithfulness"].append(min(faith * 2, 1.0))

            # 相关性
            q_words = set(result["question"].lower().split())
            rel = len(q_words & a_words) / max(len(q_words), 1)
            scores["relevancy"].append(min(rel * 3, 1.0))

            # 精确度
            gt_words = set(gt.lower().split())
            prec = len(c_words & gt_words) / max(len(c_words), 1)
            scores["precision"].append(min(prec * 3, 1.0))

        import statistics
        return {name: round(statistics.mean(vals), 4) for name, vals in scores.items()}


# ============================================================
# 演示
# ============================================================

def main() -> None:
    """运行企业级 RAG 知识库演示。"""
    print("企业级 RAG 知识库 — 完整链路演示")
    print("=" * 60)

    # 初始化 RAG 流水线
    rag = RAGPipeline()
    rag.index()

    # 测试查询
    questions = [
        "公司的主要业务是什么？",
        "智能客服系统使用什么模型？",
        "如何重置密码？",
        "数据安全有什么要求？",
        "如何部署 Qwen2 模型？",
    ]

    ground_truths = [
        "自然语言处理、计算机视觉和智能推荐系统",
        "Qwen2 基础模型，Chroma 向量数据库",
        "访问 SSO 登录页面，点击忘记密码，输入邮箱验证后重置",
        "AES-256 加密存储，Bearer Token 认证，敏感数据脱敏",
        "使用 vLLM 部署，需要 NVIDIA A10 GPU",
    ]

    results = []
    for q in questions:
        result = rag.query(q)
        results.append(result)

    # 评估
    print("\n" + "=" * 60)
    print("RAG 系统评估")
    print("=" * 60)
    evaluator = RAGEvaluator()
    eval_result = evaluator.evaluate(results, ground_truths)
    for name, score in eval_result.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {name:20s} {bar} {score:.4f}")

    print("\n" + "=" * 60)
    print("演示完成！")
    print("\n关键要点:")
    print("  1. 完整 RAG 链路：加载→切分→向量化→检索→Rerank→生成")
    print("  2. 智能切分在句子边界断开，保持语义完整性")
    print("  3. Rerank 重排序显著提升检索精度")
    print("  4. 评估模块量化 RAG 系统质量")
    print("  5. 生产环境需要增量索引、缓存和监控")


if __name__ == "__main__":
    main()
