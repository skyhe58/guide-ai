"""
RAGAS 评估模拟 — Faithfulness/Answer Relevancy/Context Precision/Recall

知识点：RAG 评估四大核心指标的计算原理和实现，包括声明提取、
       相似度计算、评估报告生成、评估数据集管理

Python 版本：3.11+
依赖：标准库（默认模式）
最后验证：2024-12-01
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass, field
from typing import Any

# ============================================================
# 1. 基础工具函数
# ============================================================

def simple_embedding(text: str, dim: int = 8) -> list[float]:
    """生成简单的模拟 Embedding 向量。"""
    hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
    vector = [((hash_val >> (i * 4)) & 0xF) / 15.0 - 0.5 for i in range(dim)]
    norm = math.sqrt(sum(v * v for v in vector))
    return [round(v / norm, 4) for v in vector] if norm > 0 else vector


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算余弦相似度。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0


def extract_claims(text: str) -> list[str]:
    """从文本中提取声明（简化实现：按句号分割）。"""
    separators = ["。", ".", "；", ";", "！", "!"]
    claims = [text]
    for sep in separators:
        new_claims = []
        for claim in claims:
            new_claims.extend(claim.split(sep))
        claims = new_claims
    return [c.strip() for c in claims if len(c.strip()) > 5]


def text_overlap_score(text_a: str, text_b: str) -> float:
    """计算两段文本的词汇重叠度。"""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    overlap = len(words_a & words_b)
    return overlap / max(len(words_a), len(words_b))


# ============================================================
# 2. 评估样本数据结构
# ============================================================

@dataclass
class EvalSample:
    """评估样本。"""
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricResult:
    """单个指标的评估结果。"""
    name: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"{self.name}: {self.score:.4f}"


@dataclass
class EvalResult:
    """完整评估结果。"""
    metrics: dict[str, float] = field(default_factory=dict)
    sample_results: list[dict[str, Any]] = field(default_factory=list)

    def __repr__(self) -> str:
        return json.dumps(self.metrics, indent=2)


# ============================================================
# 3. Faithfulness — 忠实度
# ============================================================

class FaithfulnessMetric:
    """忠实度指标 — 回答是否忠于检索到的上下文。

    计算方法：
    1. 将回答拆分为多个独立声明（claims）
    2. 逐一检查每个声明是否能在上下文中找到支持
    3. Faithfulness = 有支持的声明数 / 总声明数
    """

    def __init__(self, threshold: float = 0.5):
        self.name = "faithfulness"
        self.threshold = threshold

    def evaluate(self, sample: EvalSample) -> MetricResult:
        """评估单个样本的忠实度。"""
        # 提取回答中的声明
        claims = extract_claims(sample.answer)
        if not claims:
            return MetricResult(self.name, 1.0, {"claims": 0, "supported": 0})

        # 合并所有上下文
        context_text = " ".join(sample.contexts)

        # 检查每个声明是否有上下文支持
        supported_count = 0
        claim_details = []
        for claim in claims:
            # 计算声明与上下文的重叠度
            overlap = text_overlap_score(claim, context_text)
            is_supported = overlap > 0.15  # 阈值
            if is_supported:
                supported_count += 1
            claim_details.append({
                "claim": claim[:50],
                "supported": is_supported,
                "overlap": round(overlap, 4),
            })

        score = supported_count / len(claims)
        return MetricResult(self.name, round(score, 4), {
            "total_claims": len(claims),
            "supported_claims": supported_count,
            "claim_details": claim_details,
        })


# ============================================================
# 4. Answer Relevancy — 答案相关性
# ============================================================

class AnswerRelevancyMetric:
    """答案相关性指标 — 回答是否与问题相关。

    计算方法：
    1. 根据回答反向生成 N 个问题
    2. 计算生成问题与原始问题的余弦相似度
    3. Answer Relevancy = mean(similarities)
    """

    def __init__(self, n_generated: int = 3):
        self.name = "answer_relevancy"
        self.n_generated = n_generated

    def evaluate(self, sample: EvalSample) -> MetricResult:
        """评估单个样本的答案相关性。"""
        # 模拟反向生成问题
        generated_questions = self._generate_questions(sample.answer)

        # 计算与原始问题的相似度
        original_emb = simple_embedding(sample.question)
        similarities = []
        for gen_q in generated_questions:
            gen_emb = simple_embedding(gen_q)
            sim = cosine_similarity(original_emb, gen_emb)
            similarities.append(sim)

        # 同时考虑词汇重叠
        text_sims = [text_overlap_score(sample.question, gq) for gq in generated_questions]
        combined = [(s + t) / 2 for s, t in zip(similarities, text_sims)]

        score = statistics.mean(combined) if combined else 0.0
        return MetricResult(self.name, round(max(0, min(1, score)), 4), {
            "generated_questions": generated_questions,
            "similarities": [round(s, 4) for s in combined],
        })

    def _generate_questions(self, answer: str) -> list[str]:
        """模拟根据回答反向生成问题。"""
        # 简化实现：从回答中提取关键词构造问题
        words = answer.split()[:10]
        questions = []
        for i in range(min(self.n_generated, 3)):
            key_words = " ".join(words[i * 3:(i + 1) * 3]) if words else answer[:20]
            questions.append(f"什么是 {key_words}？")
        return questions


# ============================================================
# 5. Context Precision — 上下文精确度
# ============================================================

class ContextPrecisionMetric:
    """上下文精确度指标 — 检索到的文档有多少是相关的。

    计算方法：
    1. 对每个检索到的上下文，判断是否与问题相关
    2. Context Precision = 相关上下文数 / 总上下文数
    3. 考虑排序位置（排名靠前的相关文档权重更高）
    """

    def __init__(self):
        self.name = "context_precision"

    def evaluate(self, sample: EvalSample) -> MetricResult:
        """评估上下文精确度。"""
        if not sample.contexts:
            return MetricResult(self.name, 0.0, {"total": 0, "relevant": 0})

        # 判断每个上下文是否与问题/标准答案相关
        relevance_scores = []
        reference = sample.ground_truth if sample.ground_truth else sample.question
        for ctx in sample.contexts:
            score = text_overlap_score(ctx, reference)
            is_relevant = score > 0.1
            relevance_scores.append({"context": ctx[:40], "relevant": is_relevant, "score": round(score, 4)})

        relevant_count = sum(1 for r in relevance_scores if r["relevant"])
        # 加权精确度（排名靠前的权重更高）
        weighted_sum = 0.0
        cumulative_relevant = 0
        for i, r in enumerate(relevance_scores):
            if r["relevant"]:
                cumulative_relevant += 1
                weighted_sum += cumulative_relevant / (i + 1)
        precision = weighted_sum / max(relevant_count, 1)

        return MetricResult(self.name, round(min(precision, 1.0), 4), {
            "total_contexts": len(sample.contexts),
            "relevant_contexts": relevant_count,
            "details": relevance_scores,
        })


# ============================================================
# 6. Context Recall — 上下文召回率
# ============================================================

class ContextRecallMetric:
    """上下文召回率指标 — 标准答案中的信息有多少被上下文覆盖。

    计算方法：
    1. 将标准答案拆分为多个声明
    2. 检查每个声明是否能在上下文中找到
    3. Context Recall = 被覆盖的声明数 / 总声明数
    """

    def __init__(self):
        self.name = "context_recall"

    def evaluate(self, sample: EvalSample) -> MetricResult:
        """评估上下文召回率。"""
        if not sample.ground_truth:
            return MetricResult(self.name, 0.0, {"message": "需要标准答案"})

        # 提取标准答案中的声明
        gt_claims = extract_claims(sample.ground_truth)
        if not gt_claims:
            return MetricResult(self.name, 1.0, {"claims": 0})

        # 合并上下文
        context_text = " ".join(sample.contexts)

        # 检查每个声明是否被上下文覆盖
        covered_count = 0
        claim_details = []
        for claim in gt_claims:
            overlap = text_overlap_score(claim, context_text)
            is_covered = overlap > 0.1
            if is_covered:
                covered_count += 1
            claim_details.append({"claim": claim[:50], "covered": is_covered, "overlap": round(overlap, 4)})

        score = covered_count / len(gt_claims)
        return MetricResult(self.name, round(score, 4), {
            "total_claims": len(gt_claims),
            "covered_claims": covered_count,
            "details": claim_details,
        })


# ============================================================
# 7. RAGAS 评估器
# ============================================================

class RAGASEvaluator:
    """RAGAS 评估器 — 组合四大指标进行评估。"""

    def __init__(self, metrics: list | None = None):
        self.metrics = metrics or [
            FaithfulnessMetric(),
            AnswerRelevancyMetric(),
            ContextPrecisionMetric(),
            ContextRecallMetric(),
        ]

    def evaluate(self, samples: list[EvalSample]) -> EvalResult:
        """评估一组样本。"""
        all_scores: dict[str, list[float]] = {m.name: [] for m in self.metrics}
        sample_results = []

        for sample in samples:
            sample_scores = {}
            for metric in self.metrics:
                result = metric.evaluate(sample)
                sample_scores[metric.name] = result.score
                all_scores[metric.name].append(result.score)
            sample_results.append({
                "question": sample.question[:40],
                "scores": sample_scores,
            })

        # 计算平均分
        avg_scores = {name: round(statistics.mean(scores), 4) for name, scores in all_scores.items()}
        return EvalResult(metrics=avg_scores, sample_results=sample_results)


# ============================================================
# 演示函数
# ============================================================

def create_eval_dataset() -> list[EvalSample]:
    """创建评估数据集。"""
    return [
        EvalSample(
            question="什么是 RAG？",
            answer="RAG 是检索增强生成，它通过从知识库检索相关文档来增强 LLM 的回答质量，减少幻觉。",
            contexts=[
                "RAG（Retrieval-Augmented Generation）是检索增强生成架构，结合检索和生成两个阶段。",
                "RAG 通过从外部知识库检索相关文档，将信息注入 Prompt 中，提升回答准确性。",
                "Python 是一种编程语言，支持多种编程范式。",
            ],
            ground_truth="RAG 是检索增强生成，结合检索和生成，从知识库检索文档增强 LLM 回答，减少幻觉。",
        ),
        EvalSample(
            question="LangChain 有哪些核心组件？",
            answer="LangChain 的核心组件包括 Prompt Template、LLM、Output Parser、Memory 和 Agent。",
            contexts=[
                "LangChain 提供 Prompt Template 用于构建提示词模板。",
                "LangChain 的 Memory 组件管理对话历史。",
                "LangChain Agent 支持工具调用和自主决策。",
            ],
            ground_truth="LangChain 核心组件包括 Prompt Template、Output Parser、Memory、Retriever、Agent 和 Chain。",
        ),
        EvalSample(
            question="向量数据库有哪些选择？",
            answer="常见的向量数据库包括 Chroma、Pinecone、FAISS 和 Milvus，各有不同的适用场景。",
            contexts=[
                "Chroma 是轻量级本地向量数据库，适合开发和小规模应用。",
                "Pinecone 是云端向量数据库服务，适合生产环境。",
                "FAISS 是 Facebook 开源的向量检索库，性能优秀。",
            ],
            ground_truth="常见向量数据库有 Chroma（本地）、Pinecone（云服务）、FAISS（高性能）和 Milvus（分布式）。",
        ),
        EvalSample(
            question="什么是 Embedding？",
            answer="Embedding 是将文本映射为高维向量的技术，使语义相似的文本在向量空间中距离更近。",
            contexts=[
                "Embedding 模型将文本转换为固定维度的向量表示。",
                "常用 Embedding 模型包括 OpenAI text-embedding-3-small 和 BGE-M3。",
            ],
            ground_truth="Embedding 是将文本映射为向量的技术，语义相似的文本向量距离更近。",
        ),
    ]


def demo_faithfulness() -> None:
    """演示 Faithfulness 评估。"""
    print("\n" + "=" * 60)
    print("1. Faithfulness — 忠实度评估")
    print("=" * 60)

    metric = FaithfulnessMetric()
    samples = create_eval_dataset()
    for sample in samples[:2]:
        result = metric.evaluate(sample)
        print(f"  Q: {sample.question}")
        print(f"  Faithfulness: {result.score:.4f} ({result.details['supported_claims']}/{result.details['total_claims']} 声明有支持)")


def demo_answer_relevancy() -> None:
    """演示 Answer Relevancy 评估。"""
    print("\n" + "=" * 60)
    print("2. Answer Relevancy — 答案相关性评估")
    print("=" * 60)

    metric = AnswerRelevancyMetric()
    samples = create_eval_dataset()
    for sample in samples[:2]:
        result = metric.evaluate(sample)
        print(f"  Q: {sample.question}")
        print(f"  Relevancy: {result.score:.4f}")


def demo_context_precision() -> None:
    """演示 Context Precision 评估。"""
    print("\n" + "=" * 60)
    print("3. Context Precision — 上下文精确度评估")
    print("=" * 60)

    metric = ContextPrecisionMetric()
    samples = create_eval_dataset()
    for sample in samples[:2]:
        result = metric.evaluate(sample)
        print(f"  Q: {sample.question}")
        print(f"  Precision: {result.score:.4f} ({result.details['relevant_contexts']}/{result.details['total_contexts']} 相关)")


def demo_context_recall() -> None:
    """演示 Context Recall 评估。"""
    print("\n" + "=" * 60)
    print("4. Context Recall — 上下文召回率评估")
    print("=" * 60)

    metric = ContextRecallMetric()
    samples = create_eval_dataset()
    for sample in samples[:2]:
        result = metric.evaluate(sample)
        print(f"  Q: {sample.question}")
        print(f"  Recall: {result.score:.4f} ({result.details.get('covered_claims', 0)}/{result.details.get('total_claims', 0)} 声明被覆盖)")


def demo_full_evaluation() -> None:
    """演示完整 RAGAS 评估。"""
    print("\n" + "=" * 60)
    print("5. 完整 RAGAS 评估")
    print("=" * 60)

    evaluator = RAGASEvaluator()
    samples = create_eval_dataset()
    result = evaluator.evaluate(samples)

    print(f"  评估结果（{len(samples)} 个样本）:")
    for name, score in result.metrics.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"    {name:25s} {bar} {score:.4f}")

    print(f"\n  各样本详情:")
    for sr in result.sample_results:
        scores_str = " | ".join(f"{k}: {v:.2f}" for k, v in sr["scores"].items())
        print(f"    {sr['question']:40s} → {scores_str}")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行所有 RAGAS 评估演示。"""
    print("RAGAS 评估模拟 — Faithfulness/Relevancy/Precision/Recall")
    print("=" * 60)

    demo_faithfulness()
    demo_answer_relevancy()
    demo_context_precision()
    demo_context_recall()
    demo_full_evaluation()

    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("\n关键要点:")
    print("  1. Faithfulness 检查回答是否忠于上下文（减少幻觉）")
    print("  2. Answer Relevancy 检查回答是否与问题相关（避免答非所问）")
    print("  3. Context Precision 检查检索精准度（减少噪声文档）")
    print("  4. Context Recall 检查检索完整度（覆盖标准答案信息）")
    print("  5. 四个指标互补，需要综合评估 RAG 系统质量")
    print("  6. 实际使用 RAGAS 库时，评估基于 LLM-as-Judge")


if __name__ == "__main__":
    main()
