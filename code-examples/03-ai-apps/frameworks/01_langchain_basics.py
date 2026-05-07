"""
LangChain 基础 — Chain/Prompt Template/Output Parser/Memory/Retriever/Agent 模拟

知识点：LangChain 核心组件模拟实现，包括 Prompt Template、Output Parser、
       LCEL Chain、Memory 管理、Retriever 检索、Agent 工具调用、Callback 回调

Python 版本：3.11+
依赖：标准库（默认模式）
最后验证：2024-12-01
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


# ============================================================
# 1. Prompt Template — 提示词模板
# ============================================================

class PromptTemplate:
    """提示词模板 — 支持变量插入和格式化。"""

    def __init__(self, template: str, input_variables: list[str] | None = None):
        self.template = template
        # 自动提取模板中的变量名
        self.input_variables = input_variables or self._extract_variables()

    def _extract_variables(self) -> list[str]:
        """从模板字符串中提取 {variable} 格式的变量。"""
        return re.findall(r"\{(\w+)\}", self.template)

    def format(self, **kwargs: Any) -> str:
        """渲染模板，将变量替换为实际值。"""
        missing = set(self.input_variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"缺少模板变量: {missing}")
        return self.template.format(**kwargs)

    def __repr__(self) -> str:
        return f"PromptTemplate(variables={self.input_variables})"


class ChatPromptTemplate:
    """对话提示词模板 — 支持多角色消息（system/human/ai）。"""

    def __init__(self, messages: list[tuple[str, str]]):
        self.message_templates = messages

    @classmethod
    def from_messages(cls, messages: list[tuple[str, str]]) -> "ChatPromptTemplate":
        """从消息列表创建模板。"""
        return cls(messages)

    def format_messages(self, **kwargs: Any) -> list[dict[str, str]]:
        """渲染所有消息模板。"""
        result = []
        for role, template in self.message_templates:
            content = template.format(**kwargs)
            result.append({"role": role, "content": content})
        return result

    def __repr__(self) -> str:
        roles = [r for r, _ in self.message_templates]
        return f"ChatPromptTemplate(roles={roles})"


class FewShotPromptTemplate:
    """Few-shot 提示词模板 — 动态插入示例。"""

    def __init__(
        self,
        examples: list[dict[str, str]],
        example_template: str,
        prefix: str = "",
        suffix: str = "",
    ):
        self.examples = examples
        self.example_template = example_template
        self.prefix = prefix
        self.suffix = suffix

    def format(self, **kwargs: Any) -> str:
        """渲染 Few-shot 模板。"""
        # 格式化每个示例
        formatted_examples = []
        for ex in self.examples:
            formatted_examples.append(self.example_template.format(**ex))
        examples_str = "\n\n".join(formatted_examples)
        # 组合完整 Prompt
        parts = [self.prefix, examples_str, self.suffix.format(**kwargs)]
        return "\n\n".join(p for p in parts if p)


# ============================================================
# 2. Output Parser — 输出解析器
# ============================================================

class BaseOutputParser(ABC):
    """输出解析器基类。"""

    @abstractmethod
    def parse(self, text: str) -> Any:
        """解析 LLM 输出文本。"""

    def get_format_instructions(self) -> str:
        """返回格式说明，注入到 Prompt 中。"""
        return ""


class StrOutputParser(BaseOutputParser):
    """字符串输出解析器 — 直接返回文本。"""

    def parse(self, text: str) -> str:
        return text.strip()


class JsonOutputParser(BaseOutputParser):
    """JSON 输出解析器 — 将文本解析为 JSON 对象。"""

    def parse(self, text: str) -> dict:
        # 尝试提取 JSON 块
        json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw_text": text, "parse_error": True}

    def get_format_instructions(self) -> str:
        return "请以 JSON 格式输出，用 ```json ``` 包裹。"


class CommaSeparatedListParser(BaseOutputParser):
    """逗号分隔列表解析器。"""

    def parse(self, text: str) -> list[str]:
        return [item.strip() for item in text.split(",") if item.strip()]

    def get_format_instructions(self) -> str:
        return "请以逗号分隔的列表格式输出。"


# ============================================================
# 3. LLM 模拟 — 模拟语言模型调用
# ============================================================

class MockLLM:
    """模拟 LLM — 根据输入返回预设回答。"""

    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature
        self.call_count = 0
        self.total_tokens = 0

    def invoke(self, prompt: str | list[dict]) -> str:
        """模拟 LLM 调用。"""
        self.call_count += 1
        if isinstance(prompt, list):
            # ChatModel 模式
            last_msg = prompt[-1]["content"] if prompt else ""
            input_text = last_msg
        else:
            input_text = prompt

        # 模拟 token 计数
        input_tokens = len(input_text) // 2
        output_tokens = 50
        self.total_tokens += input_tokens + output_tokens

        # 根据关键词返回模拟回答
        if "RAG" in input_text.upper():
            return "RAG（检索增强生成）是一种结合检索和生成的 AI 架构，通过从知识库中检索相关文档来增强 LLM 的回答质量。"
        elif "Agent" in input_text:
            return "AI Agent 是具备自主决策和工具调用能力的智能体，能够根据用户需求自动选择和使用工具完成任务。"
        elif "LangChain" in input_text:
            return "LangChain 是最流行的 LLM 应用开发框架，提供 Chain、Prompt Template、Memory 等标准化组件。"
        else:
            return f"这是关于 '{input_text[:30]}...' 的模拟回答。"

    def get_usage(self) -> dict:
        """获取使用统计。"""
        return {
            "model": self.model_name,
            "call_count": self.call_count,
            "total_tokens": self.total_tokens,
            "estimated_cost": self.total_tokens * 0.00003,
        }


# ============================================================
# 4. Chain — LCEL 链式调用模拟
# ============================================================

class Runnable(ABC):
    """可运行组件基类 — 模拟 LCEL 的 Runnable 接口。"""

    @abstractmethod
    def invoke(self, input_data: Any) -> Any:
        """执行组件。"""

    def __or__(self, other: "Runnable") -> "RunnableSequence":
        """管道符 | 重载，实现 LCEL 链式调用。"""
        return RunnableSequence(steps=[self, other])


class RunnableSequence(Runnable):
    """顺序执行链 — 模拟 LCEL 的管道。"""

    def __init__(self, steps: list[Runnable]):
        self.steps = steps

    def invoke(self, input_data: Any) -> Any:
        """按顺序执行所有步骤。"""
        result = input_data
        for step in self.steps:
            result = step.invoke(result)
        return result

    def __or__(self, other: Runnable) -> "RunnableSequence":
        return RunnableSequence(steps=self.steps + [other])


class PromptRunnable(Runnable):
    """Prompt 可运行组件。"""

    def __init__(self, template: ChatPromptTemplate):
        self.template = template

    def invoke(self, input_data: dict) -> list[dict]:
        return self.template.format_messages(**input_data)


class LLMRunnable(Runnable):
    """LLM 可运行组件。"""

    def __init__(self, llm: MockLLM):
        self.llm = llm

    def invoke(self, input_data: Any) -> str:
        return self.llm.invoke(input_data)


class ParserRunnable(Runnable):
    """Parser 可运行组件。"""

    def __init__(self, parser: BaseOutputParser):
        self.parser = parser

    def invoke(self, input_data: str) -> Any:
        return self.parser.parse(input_data)


# ============================================================
# 5. Memory — 记忆管理
# ============================================================

class ConversationBufferMemory:
    """对话缓冲记忆 — 保存完整对话历史。"""

    def __init__(self):
        self.messages: list[dict[str, str]] = []

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "human", "content": content})

    def add_ai_message(self, content: str) -> None:
        self.messages.append({"role": "ai", "content": content})

    def get_history(self) -> list[dict[str, str]]:
        return list(self.messages)

    def clear(self) -> None:
        self.messages.clear()

    @property
    def token_count(self) -> int:
        return sum(len(m["content"]) // 2 for m in self.messages)


class ConversationWindowMemory:
    """滑动窗口记忆 — 只保留最近 K 轮对话。"""

    def __init__(self, k: int = 5):
        self.k = k
        self.messages: list[dict[str, str]] = []

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "human", "content": content})
        self._trim()

    def add_ai_message(self, content: str) -> None:
        self.messages.append({"role": "ai", "content": content})
        self._trim()

    def _trim(self) -> None:
        """保留最近 K 轮（2K 条消息）。"""
        max_messages = self.k * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def get_history(self) -> list[dict[str, str]]:
        return list(self.messages)


class ConversationSummaryMemory:
    """摘要记忆 — 超过阈值时压缩为摘要。"""

    def __init__(self, llm: MockLLM, max_tokens: int = 500):
        self.llm = llm
        self.max_tokens = max_tokens
        self.summary: str = ""
        self.recent_messages: list[dict[str, str]] = []

    def add_user_message(self, content: str) -> None:
        self.recent_messages.append({"role": "human", "content": content})
        self._maybe_summarize()

    def add_ai_message(self, content: str) -> None:
        self.recent_messages.append({"role": "ai", "content": content})
        self._maybe_summarize()

    def _maybe_summarize(self) -> None:
        """当消息过多时，压缩为摘要。"""
        token_count = sum(len(m["content"]) // 2 for m in self.recent_messages)
        if token_count > self.max_tokens and len(self.recent_messages) > 4:
            old = self.recent_messages[:-4]
            old_text = " ".join(m["content"] for m in old)
            self.summary = f"[摘要] 之前讨论了: {old_text[:100]}..."
            self.recent_messages = self.recent_messages[-4:]

    def get_history(self) -> list[dict[str, str]]:
        result = []
        if self.summary:
            result.append({"role": "system", "content": self.summary})
        result.extend(self.recent_messages)
        return result


# ============================================================
# 6. Retriever — 检索器
# ============================================================

@dataclass
class Document:
    """文档对象。"""
    page_content: str
    metadata: dict = field(default_factory=dict)


class SimpleRetriever:
    """简单检索器 — 基于关键词匹配的模拟检索。"""

    def __init__(self, documents: list[Document]):
        self.documents = documents

    def retrieve(self, query: str, top_k: int = 3) -> list[Document]:
        """根据查询检索相关文档。"""
        scored = []
        query_terms = set(query.lower().split())
        for doc in self.documents:
            doc_terms = set(doc.page_content.lower().split())
            score = len(query_terms & doc_terms)
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]


# ============================================================
# 7. Agent — 智能体模拟
# ============================================================

@dataclass
class Tool:
    """工具定义。"""
    name: str
    description: str
    func: Callable[..., str]


class SimpleAgent:
    """简单 Agent — 根据用户输入选择工具并执行。"""

    def __init__(self, llm: MockLLM, tools: list[Tool]):
        self.llm = llm
        self.tools = {t.name: t for t in tools}

    def run(self, query: str) -> str:
        """执行 Agent 推理循环。"""
        # 模拟工具选择
        tool_name = self._select_tool(query)
        if tool_name and tool_name in self.tools:
            tool = self.tools[tool_name]
            tool_result = tool.func(query)
            # 用工具结果生成最终回答
            prompt = f"工具 {tool_name} 返回: {tool_result}\n请基于此回答: {query}"
            return self.llm.invoke(prompt)
        return self.llm.invoke(query)

    def _select_tool(self, query: str) -> str | None:
        """模拟工具选择逻辑。"""
        query_lower = query.lower()
        for name, tool in self.tools.items():
            keywords = tool.description.lower().split()
            if any(kw in query_lower for kw in keywords):
                return name
        return None


# ============================================================
# 8. Callback — 回调机制
# ============================================================

class CallbackHandler:
    """回调处理器 — 记录链路执行的全生命周期。"""

    def __init__(self, name: str = "default"):
        self.name = name
        self.logs: list[dict] = []

    def on_llm_start(self, prompt: str) -> None:
        self.logs.append({"event": "llm_start", "prompt_len": len(prompt), "time": time.time()})
        print(f"  [Callback] LLM 开始调用，Prompt 长度: {len(prompt)}")

    def on_llm_end(self, output: str, latency_ms: float) -> None:
        self.logs.append({"event": "llm_end", "output_len": len(output), "latency_ms": latency_ms})
        print(f"  [Callback] LLM 调用完成，耗时: {latency_ms:.1f}ms")

    def on_chain_start(self, chain_name: str) -> None:
        self.logs.append({"event": "chain_start", "chain": chain_name})
        print(f"  [Callback] Chain '{chain_name}' 开始执行")

    def on_chain_end(self, chain_name: str) -> None:
        self.logs.append({"event": "chain_end", "chain": chain_name})
        print(f"  [Callback] Chain '{chain_name}' 执行完成")

    def on_chain_error(self, chain_name: str, error: str) -> None:
        self.logs.append({"event": "chain_error", "chain": chain_name, "error": error})
        print(f"  [Callback] Chain '{chain_name}' 出错: {error}")

    def on_tool_start(self, tool_name: str) -> None:
        self.logs.append({"event": "tool_start", "tool": tool_name})
        print(f"  [Callback] 工具 '{tool_name}' 开始调用")

    def get_summary(self) -> dict:
        """获取回调摘要。"""
        return {
            "total_events": len(self.logs),
            "llm_calls": sum(1 for l in self.logs if l["event"] == "llm_end"),
            "errors": sum(1 for l in self.logs if l["event"] == "chain_error"),
        }


# ============================================================
# 演示函数
# ============================================================

def demo_prompt_template() -> None:
    """演示 Prompt Template。"""
    print("\n" + "=" * 60)
    print("1. Prompt Template — 提示词模板")
    print("=" * 60)

    # 基础模板
    pt = PromptTemplate("请用{language}解释{concept}的核心原理")
    result = pt.format(language="中文", concept="RAG")
    print(f"  基础模板: {result}")

    # Chat 模板
    chat_pt = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}专家"),
        ("human", "{question}"),
    ])
    messages = chat_pt.format_messages(role="AI", question="什么是 LangChain？")
    print(f"  Chat 模板: {messages}")

    # Few-shot 模板
    few_shot = FewShotPromptTemplate(
        examples=[
            {"input": "什么是 RAG？", "output": "RAG 是检索增强生成..."},
            {"input": "什么是 Agent？", "output": "Agent 是智能体..."},
        ],
        example_template="问: {input}\n答: {output}",
        prefix="以下是一些示例：",
        suffix="问: {question}\n答:",
    )
    result = few_shot.format(question="什么是 LangChain？")
    print(f"  Few-shot 模板:\n{result}")


def demo_output_parser() -> None:
    """演示 Output Parser。"""
    print("\n" + "=" * 60)
    print("2. Output Parser — 输出解析器")
    print("=" * 60)

    # 字符串解析
    str_parser = StrOutputParser()
    print(f"  StrParser: '{str_parser.parse('  Hello World  ')}'")

    # JSON 解析
    json_parser = JsonOutputParser()
    json_text = '```json\n{"name": "RAG", "type": "架构"}\n```'
    print(f"  JsonParser: {json_parser.parse(json_text)}")

    # 列表解析
    list_parser = CommaSeparatedListParser()
    print(f"  ListParser: {list_parser.parse('RAG, Agent, LangChain, LangGraph')}")


def demo_lcel_chain() -> None:
    """演示 LCEL 链式调用。"""
    print("\n" + "=" * 60)
    print("3. LCEL Chain — 链式调用（管道符 |）")
    print("=" * 60)

    llm = MockLLM()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}专家"),
        ("human", "{question}"),
    ])

    # 构建 LCEL Chain: prompt | llm | parser
    chain = PromptRunnable(prompt) | LLMRunnable(llm) | ParserRunnable(StrOutputParser())
    result = chain.invoke({"role": "AI", "question": "什么是 RAG？"})
    print(f"  Chain 输出: {result}")
    print(f"  LLM 使用统计: {llm.get_usage()}")


def demo_memory() -> None:
    """演示 Memory 记忆管理。"""
    print("\n" + "=" * 60)
    print("4. Memory — 记忆管理")
    print("=" * 60)

    # Buffer Memory
    buffer = ConversationBufferMemory()
    buffer.add_user_message("什么是 RAG？")
    buffer.add_ai_message("RAG 是检索增强生成...")
    buffer.add_user_message("它有什么优势？")
    buffer.add_ai_message("RAG 的优势包括减少幻觉...")
    print(f"  Buffer Memory: {len(buffer.get_history())} 条消息, {buffer.token_count} tokens")

    # Window Memory
    window = ConversationWindowMemory(k=2)
    for i in range(5):
        window.add_user_message(f"问题 {i}")
        window.add_ai_message(f"回答 {i}")
    print(f"  Window Memory (k=2): {len(window.get_history())} 条消息")

    # Summary Memory
    llm = MockLLM()
    summary = ConversationSummaryMemory(llm, max_tokens=50)
    for i in range(10):
        summary.add_user_message(f"这是第 {i} 个很长的问题，包含很多内容")
        summary.add_ai_message(f"这是第 {i} 个很长的回答，包含很多内容")
    history = summary.get_history()
    print(f"  Summary Memory: {len(history)} 条消息")
    if summary.summary:
        print(f"  摘要: {summary.summary[:60]}...")


def demo_retriever() -> None:
    """演示 Retriever 检索器。"""
    print("\n" + "=" * 60)
    print("5. Retriever — 检索器")
    print("=" * 60)

    docs = [
        Document("RAG 是检索增强生成架构，结合检索和生成", {"source": "rag.md"}),
        Document("LangChain 是 LLM 应用开发框架", {"source": "langchain.md"}),
        Document("向量数据库用于存储和检索 Embedding 向量", {"source": "vector.md"}),
        Document("Agent 是具备工具调用能力的智能体", {"source": "agent.md"}),
        Document("Prompt Engineering 是设计提示词的技术", {"source": "prompt.md"}),
    ]

    retriever = SimpleRetriever(docs)
    results = retriever.retrieve("RAG 检索增强生成", top_k=2)
    print(f"  查询: 'RAG 检索增强生成'")
    for i, doc in enumerate(results):
        print(f"  [{i+1}] {doc.page_content[:40]}... (来源: {doc.metadata.get('source', 'N/A')})")


def demo_agent() -> None:
    """演示 Agent 智能体。"""
    print("\n" + "=" * 60)
    print("6. Agent — 智能体")
    print("=" * 60)

    llm = MockLLM()
    tools = [
        Tool("search", "搜索 查询 检索", lambda q: f"搜索结果: 找到 3 篇关于 '{q[:20]}' 的文档"),
        Tool("calculator", "计算 数学 加减", lambda q: "计算结果: 42"),
        Tool("weather", "天气 温度 气象", lambda q: "天气: 北京 25°C 晴"),
    ]

    agent = SimpleAgent(llm, tools)
    queries = ["搜索 RAG 相关论文", "计算 1+1 等于多少", "今天天气怎么样", "什么是 LangChain"]
    for q in queries:
        result = agent.run(q)
        print(f"  Q: {q}")
        print(f"  A: {result[:60]}...")


def demo_callback() -> None:
    """演示 Callback 回调机制。"""
    print("\n" + "=" * 60)
    print("7. Callback — 回调机制")
    print("=" * 60)

    cb = CallbackHandler("demo")
    cb.on_chain_start("rag_chain")
    cb.on_llm_start("请回答关于 RAG 的问题...")
    time.sleep(0.01)
    cb.on_llm_end("RAG 是检索增强生成...", latency_ms=120.5)
    cb.on_tool_start("vector_search")
    cb.on_chain_end("rag_chain")
    print(f"  回调摘要: {cb.get_summary()}")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行所有 LangChain 基础演示。"""
    print("LangChain 基础 — Chain/Prompt/OutputParser/Memory/Retriever/Agent 模拟")
    print("=" * 60)

    demo_prompt_template()
    demo_output_parser()
    demo_lcel_chain()
    demo_memory()
    demo_retriever()
    demo_agent()
    demo_callback()

    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("\n关键要点:")
    print("  1. Prompt Template 将提示词参数化，支持变量插入和多角色消息")
    print("  2. Output Parser 将 LLM 自由文本输出解析为结构化数据")
    print("  3. LCEL 用 | 管道符串联组件，天然支持流式和异步")
    print("  4. Memory 管理对话历史，Buffer/Window/Summary 三种策略")
    print("  5. Retriever 从知识库检索相关文档，是 RAG 的核心")
    print("  6. Agent 让 LLM 自主选择和调用工具")
    print("  7. Callback 提供全链路追踪，用于调试和监控")


if __name__ == "__main__":
    main()
