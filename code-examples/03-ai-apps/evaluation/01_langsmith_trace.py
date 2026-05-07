"""
LangSmith 追踪模拟 — Trace/Run 分析/Prompt 版本管理/在线评估

知识点：LangSmith 核心功能模拟，包括 Trace 追踪、Run 分析、
       延迟统计、Token 计费、Callback Handler、评估数据集管理

Python 版本：3.11+
依赖：标准库（默认模式）
最后验证：2024-12-01
"""

from __future__ import annotations

import json
import random
import statistics
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


# ============================================================
# 1. Run — 单次执行记录
# ============================================================

class RunType(Enum):
    """Run 类型。"""
    LLM = "llm"
    CHAIN = "chain"
    TOOL = "tool"
    RETRIEVER = "retriever"
    EMBEDDING = "embedding"


class RunStatus(Enum):
    """Run 状态。"""
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"


@dataclass
class TokenUsage:
    """Token 使用统计。"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def estimated_cost(self) -> float:
        """估算成本（基于 GPT-4o 定价）。"""
        input_cost = self.prompt_tokens * 2.5 / 1_000_000
        output_cost = self.completion_tokens * 10.0 / 1_000_000
        return round(input_cost + output_cost, 6)


@dataclass
class Run:
    """Run — Trace 中的单个执行步骤。"""
    run_id: str = ""
    parent_run_id: str = ""
    run_type: RunType = RunType.CHAIN
    name: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    status: RunStatus = RunStatus.PENDING
    start_time: float = 0.0
    end_time: float = 0.0
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    error: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.run_id:
            self.run_id = str(uuid.uuid4())[:8]
        if not self.start_time:
            self.start_time = time.time()

    @property
    def latency_ms(self) -> float:
        """计算延迟（毫秒）。"""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0

    def complete(self, outputs: dict | None = None, error: str = "") -> None:
        """标记 Run 完成。"""
        self.end_time = time.time()
        if error:
            self.status = RunStatus.ERROR
            self.error = error
        else:
            self.status = RunStatus.SUCCESS
            if outputs:
                self.outputs = outputs


# ============================================================
# 2. Trace — 完整请求链路
# ============================================================

@dataclass
class Trace:
    """Trace — 一次完整的用户请求链路。"""
    trace_id: str = ""
    project: str = "default"
    runs: list[Run] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.trace_id:
            self.trace_id = str(uuid.uuid4())[:8]

    def add_run(self, run: Run) -> None:
        """添加 Run。"""
        self.runs.append(run)

    @property
    def total_latency_ms(self) -> float:
        """总延迟。"""
        return sum(r.latency_ms for r in self.runs)

    @property
    def total_tokens(self) -> int:
        """总 Token 消耗。"""
        return sum(r.token_usage.total_tokens for r in self.runs)

    @property
    def total_cost(self) -> float:
        """总成本。"""
        return sum(r.token_usage.estimated_cost for r in self.runs)

    @property
    def has_error(self) -> bool:
        """是否有错误。"""
        return any(r.status == RunStatus.ERROR for r in self.runs)

    def get_summary(self) -> dict:
        """获取 Trace 摘要。"""
        return {
            "trace_id": self.trace_id,
            "project": self.project,
            "total_runs": len(self.runs),
            "latency_ms": round(self.total_latency_ms, 2),
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 6),
            "has_error": self.has_error,
            "run_types": [r.run_type.value for r in self.runs],
        }


# ============================================================
# 3. Tracer — 追踪器
# ============================================================

class LangSmithTracer:
    """LangSmith 追踪器 — 记录和管理 Trace。"""

    def __init__(self, project: str = "default", api_key: str = "ls-mock"):
        self.project = project
        self.api_key = api_key
        self.traces: list[Trace] = []
        self._current_trace: Trace | None = None

    def start_trace(self, tags: list[str] | None = None) -> Trace:
        """开始新的 Trace。"""
        trace = Trace(project=self.project, tags=tags or [])
        self._current_trace = trace
        self.traces.append(trace)
        return trace

    def start_run(self, name: str, run_type: RunType, inputs: dict | None = None,
                  parent_run_id: str = "", tags: list[str] | None = None) -> Run:
        """开始新的 Run。"""
        run = Run(
            run_type=run_type,
            name=name,
            inputs=inputs or {},
            parent_run_id=parent_run_id,
            tags=tags or [],
        )
        if self._current_trace:
            self._current_trace.add_run(run)
        return run

    def end_run(self, run: Run, outputs: dict | None = None,
                token_usage: TokenUsage | None = None, error: str = "") -> None:
        """结束 Run。"""
        run.complete(outputs, error)
        if token_usage:
            run.token_usage = token_usage

    def get_analytics(self) -> dict:
        """获取分析数据。"""
        if not self.traces:
            return {"message": "暂无数据"}

        latencies = [t.total_latency_ms for t in self.traces]
        tokens = [t.total_tokens for t in self.traces]
        costs = [t.total_cost for t in self.traces]
        errors = sum(1 for t in self.traces if t.has_error)

        return {
            "total_traces": len(self.traces),
            "error_rate": round(errors / len(self.traces), 4),
            "latency": {
                "p50": round(statistics.median(latencies), 2),
                "p95": round(sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0], 2),
                "avg": round(statistics.mean(latencies), 2),
            },
            "tokens": {
                "total": sum(tokens),
                "avg_per_trace": round(statistics.mean(tokens), 1),
            },
            "cost": {
                "total": round(sum(costs), 4),
                "avg_per_trace": round(statistics.mean(costs), 6),
            },
        }


# ============================================================
# 4. Callback Handler — 回调处理器
# ============================================================

class LangSmithCallbackHandler:
    """LangSmith 回调处理器 — 自动追踪 LangChain 调用。"""

    def __init__(self, tracer: LangSmithTracer):
        self.tracer = tracer
        self._run_stack: list[Run] = []

    def on_chain_start(self, chain_name: str, inputs: dict) -> None:
        """Chain 开始。"""
        trace = self.tracer.start_trace(tags=["chain"])
        run = self.tracer.start_run(chain_name, RunType.CHAIN, inputs)
        self._run_stack.append(run)
        print(f"  [Trace] Chain '{chain_name}' 开始")

    def on_llm_start(self, model_name: str, prompt: str) -> None:
        """LLM 调用开始。"""
        parent_id = self._run_stack[-1].run_id if self._run_stack else ""
        run = self.tracer.start_run(model_name, RunType.LLM,
                                     {"prompt": prompt[:100]}, parent_id)
        self._run_stack.append(run)
        print(f"  [Trace] LLM '{model_name}' 开始调用")

    def on_llm_end(self, output: str, prompt_tokens: int, completion_tokens: int) -> None:
        """LLM 调用结束。"""
        if self._run_stack:
            run = self._run_stack.pop()
            usage = TokenUsage(prompt_tokens, completion_tokens,
                             prompt_tokens + completion_tokens)
            self.tracer.end_run(run, {"output": output[:100]}, usage)
            print(f"  [Trace] LLM 完成, {usage.total_tokens} tokens, ${usage.estimated_cost:.6f}")

    def on_chain_end(self, output: Any) -> None:
        """Chain 结束。"""
        if self._run_stack:
            run = self._run_stack.pop()
            self.tracer.end_run(run, {"output": str(output)[:100]})
            print(f"  [Trace] Chain 完成, 延迟: {run.latency_ms:.1f}ms")


# ============================================================
# 5. Prompt 版本管理
# ============================================================

@dataclass
class PromptVersion:
    """Prompt 版本。"""
    version: int
    template: str
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    metrics: dict[str, float] = field(default_factory=dict)


class PromptHub:
    """Prompt 版本管理中心。"""

    def __init__(self):
        self.prompts: dict[str, list[PromptVersion]] = {}

    def push(self, name: str, template: str, tags: list[str] | None = None) -> PromptVersion:
        """推送新版本。"""
        if name not in self.prompts:
            self.prompts[name] = []
        version = len(self.prompts[name]) + 1
        pv = PromptVersion(version=version, template=template, tags=tags or [])
        self.prompts[name].append(pv)
        return pv

    def pull(self, name: str, version: int | None = None) -> PromptVersion | None:
        """拉取指定版本（默认最新）。"""
        if name not in self.prompts:
            return None
        versions = self.prompts[name]
        if version:
            return versions[version - 1] if version <= len(versions) else None
        return versions[-1]

    def list_versions(self, name: str) -> list[dict]:
        """列出所有版本。"""
        if name not in self.prompts:
            return []
        return [{"version": pv.version, "tags": pv.tags, "template": pv.template[:50]}
                for pv in self.prompts[name]]


# ============================================================
# 6. 评估数据集
# ============================================================

@dataclass
class EvalExample:
    """评估样本。"""
    input_text: str
    expected_output: str
    metadata: dict[str, Any] = field(default_factory=dict)


class EvalDataset:
    """评估数据集。"""

    def __init__(self, name: str):
        self.name = name
        self.examples: list[EvalExample] = []

    def add_example(self, input_text: str, expected_output: str, **metadata) -> None:
        """添加评估样本。"""
        self.examples.append(EvalExample(input_text, expected_output, metadata))

    def run_evaluation(self, predict_fn: Callable[[str], str]) -> dict:
        """运行评估。"""
        results = []
        for ex in self.examples:
            actual = predict_fn(ex.input_text)
            # 简单的字符串匹配评分
            score = self._compute_score(actual, ex.expected_output)
            results.append({"input": ex.input_text[:30], "score": score})

        avg_score = statistics.mean(r["score"] for r in results) if results else 0
        return {
            "dataset": self.name,
            "total_examples": len(self.examples),
            "avg_score": round(avg_score, 4),
            "results": results,
        }

    @staticmethod
    def _compute_score(actual: str, expected: str) -> float:
        """计算评分（简化的关键词覆盖率）。"""
        expected_words = set(expected.lower().split())
        actual_words = set(actual.lower().split())
        if not expected_words:
            return 0.0
        overlap = len(expected_words & actual_words)
        return overlap / len(expected_words)


# ============================================================
# 演示函数
# ============================================================

def demo_basic_tracing() -> None:
    """演示基础追踪。"""
    print("\n" + "=" * 60)
    print("1. 基础追踪 — Trace 和 Run")
    print("=" * 60)

    tracer = LangSmithTracer(project="demo")
    trace = tracer.start_trace(tags=["rag", "demo"])

    # 模拟 RAG 链路
    chain_run = tracer.start_run("rag_chain", RunType.CHAIN, {"query": "什么是 RAG？"})
    time.sleep(0.01)

    retriever_run = tracer.start_run("vector_retriever", RunType.RETRIEVER,
                                      {"query": "什么是 RAG？"}, chain_run.run_id)
    time.sleep(0.005)
    tracer.end_run(retriever_run, {"documents": ["doc1", "doc2"]})

    llm_run = tracer.start_run("gpt-4o", RunType.LLM,
                                {"prompt": "基于上下文回答..."}, chain_run.run_id)
    time.sleep(0.02)
    tracer.end_run(llm_run, {"output": "RAG 是检索增强生成..."},
                   TokenUsage(80, 50, 130))

    tracer.end_run(chain_run, {"answer": "RAG 是检索增强生成..."})

    print(f"  Trace 摘要: {json.dumps(trace.get_summary(), ensure_ascii=False, indent=2)}")


def demo_callback_handler() -> None:
    """演示 Callback Handler。"""
    print("\n" + "=" * 60)
    print("2. Callback Handler — 自动追踪")
    print("=" * 60)

    tracer = LangSmithTracer(project="callback-demo")
    handler = LangSmithCallbackHandler(tracer)

    handler.on_chain_start("qa_chain", {"question": "什么是 LangChain？"})
    time.sleep(0.01)
    handler.on_llm_start("gpt-4o", "请回答关于 LangChain 的问题...")
    time.sleep(0.02)
    handler.on_llm_end("LangChain 是 LLM 应用框架...", 60, 40)
    handler.on_chain_end("LangChain 是 LLM 应用框架...")


def demo_analytics() -> None:
    """演示分析功能。"""
    print("\n" + "=" * 60)
    print("3. 分析功能 — 延迟/Token/成本统计")
    print("=" * 60)

    tracer = LangSmithTracer(project="analytics-demo")

    # 模拟多次请求
    for i in range(20):
        trace = tracer.start_trace()
        run = tracer.start_run(f"query_{i}", RunType.LLM, {"query": f"问题 {i}"})
        time.sleep(random.uniform(0.005, 0.03))
        tokens = random.randint(50, 200)
        tracer.end_run(run, {"output": f"回答 {i}"},
                       TokenUsage(tokens // 2, tokens // 2, tokens))

    analytics = tracer.get_analytics()
    print(f"  分析结果:")
    print(f"    总请求数: {analytics['total_traces']}")
    print(f"    错误率: {analytics['error_rate']:.2%}")
    print(f"    延迟 P50: {analytics['latency']['p50']:.1f}ms")
    print(f"    延迟 P95: {analytics['latency']['p95']:.1f}ms")
    print(f"    总 Token: {analytics['tokens']['total']}")
    print(f"    总成本: ${analytics['cost']['total']:.4f}")


def demo_prompt_hub() -> None:
    """演示 Prompt 版本管理。"""
    print("\n" + "=" * 60)
    print("4. Prompt Hub — 版本管理")
    print("=" * 60)

    hub = PromptHub()
    hub.push("rag_qa", "基于以下上下文回答问题：\n{context}\n问题：{question}", ["v1"])
    hub.push("rag_qa", "你是一个知识助手。请基于提供的上下文准确回答。\n上下文：{context}\n问题：{question}\n要求：如果上下文不包含答案，请说不知道。", ["v2", "production"])
    hub.push("rag_qa", "# 角色\n你是企业知识库助手。\n# 上下文\n{context}\n# 问题\n{question}\n# 要求\n1. 只基于上下文回答\n2. 标注信息来源", ["v3", "staging"])

    print(f"  所有版本:")
    for v in hub.list_versions("rag_qa"):
        print(f"    v{v['version']} {v['tags']}: {v['template'][:40]}...")

    latest = hub.pull("rag_qa")
    print(f"  最新版本: v{latest.version if latest else 'N/A'}")


def demo_evaluation() -> None:
    """演示评估数据集。"""
    print("\n" + "=" * 60)
    print("5. 评估数据集 — 自动化评估")
    print("=" * 60)

    dataset = EvalDataset("rag_qa_eval")
    dataset.add_example("什么是 RAG？", "RAG 是检索增强生成，结合检索和生成的架构")
    dataset.add_example("LangChain 是什么？", "LangChain 是 LLM 应用开发框架")
    dataset.add_example("向量数据库有哪些？", "常见向量数据库包括 Chroma Pinecone FAISS Milvus")
    dataset.add_example("什么是 Agent？", "Agent 是具备工具调用能力的智能体")
    dataset.add_example("Embedding 是什么？", "Embedding 是将文本映射为向量的技术")

    def mock_predict(question: str) -> str:
        answers = {
            "RAG": "RAG 是检索增强生成架构",
            "LangChain": "LangChain 是一个 LLM 框架",
            "向量": "向量数据库有 Chroma 和 FAISS",
            "Agent": "Agent 是智能体，可以调用工具",
            "Embedding": "Embedding 将文本转为向量",
        }
        for key, answer in answers.items():
            if key in question:
                return answer
        return "我不确定这个问题的答案。"

    result = dataset.run_evaluation(mock_predict)
    print(f"  评估结果:")
    print(f"    数据集: {result['dataset']}")
    print(f"    样本数: {result['total_examples']}")
    print(f"    平均分: {result['avg_score']:.4f}")
    for r in result["results"]:
        print(f"    [{r['score']:.2f}] {r['input']}...")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行所有 LangSmith 追踪演示。"""
    print("LangSmith 追踪模拟 — Trace/Run/Prompt 管理/评估")
    print("=" * 60)

    demo_basic_tracing()
    demo_callback_handler()
    demo_analytics()
    demo_prompt_hub()
    demo_evaluation()

    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("\n关键要点:")
    print("  1. Trace 是完整请求链路，Run 是单个执行步骤")
    print("  2. Callback Handler 自动追踪 LangChain 调用")
    print("  3. 分析功能提供 P50/P95 延迟、Token 消耗和成本统计")
    print("  4. Prompt Hub 管理 Prompt 版本，支持 A/B 测试")
    print("  5. 评估数据集实现自动化质量评估")
    print("  6. 生产环境设置采样率，避免全量追踪影响性能")


if __name__ == "__main__":
    main()
