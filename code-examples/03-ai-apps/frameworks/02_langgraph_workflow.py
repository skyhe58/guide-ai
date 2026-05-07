"""
LangGraph 工作流 — 图状态机/节点/条件路由/Human-in-the-Loop 模拟

知识点：LangGraph 核心概念模拟实现，包括 StateGraph、Node、Edge、
       条件路由、Human-in-the-Loop、持久化、子图

Python 版本：3.11+
依赖：标准库（默认模式）
最后验证：2024-12-01
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


# ============================================================
# 1. State — 状态定义
# ============================================================

@dataclass
class GraphState:
    """图状态 — 在节点间传递的数据结构。"""
    messages: list[dict[str, str]] = field(default_factory=list)
    current_node: str = ""
    intent: str = ""
    context: list[str] = field(default_factory=list)
    tool_results: dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    is_complete: bool = False
    human_approval: str | None = None  # "approved" / "rejected" / None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str) -> None:
        """添加消息（模拟 add_messages reducer）。"""
        self.messages.append({"role": role, "content": content})

    def get_last_message(self) -> str:
        """获取最后一条消息内容。"""
        return self.messages[-1]["content"] if self.messages else ""

    def to_dict(self) -> dict:
        """序列化为字典（用于持久化）。"""
        return {
            "messages": self.messages,
            "current_node": self.current_node,
            "intent": self.intent,
            "context": self.context,
            "retry_count": self.retry_count,
            "is_complete": self.is_complete,
            "human_approval": self.human_approval,
        }


# ============================================================
# 2. Node — 节点定义
# ============================================================

class NodeResult:
    """节点执行结果。"""

    def __init__(self, updates: dict[str, Any] | None = None):
        self.updates = updates or {}


class BaseNode(ABC):
    """节点基类。"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def execute(self, state: GraphState) -> NodeResult:
        """执行节点逻辑，返回状态更新。"""

    def __repr__(self) -> str:
        return f"Node({self.name})"


class IntentClassifierNode(BaseNode):
    """意图识别节点 — 分析用户输入的意图。"""

    def __init__(self):
        super().__init__("intent_classifier")
        # 意图关键词映射
        self.intent_map = {
            "query": ["什么", "如何", "为什么", "解释", "介绍", "查询"],
            "action": ["执行", "运行", "创建", "删除", "修改", "操作"],
            "sensitive": ["删除数据", "修改权限", "重置密码", "转账"],
            "chat": ["你好", "谢谢", "再见", "聊天"],
        }

    def execute(self, state: GraphState) -> NodeResult:
        """识别用户意图。"""
        user_msg = state.get_last_message().lower()
        detected_intent = "chat"  # 默认闲聊

        for intent, keywords in self.intent_map.items():
            if any(kw in user_msg for kw in keywords):
                detected_intent = intent
                break

        print(f"  [意图识别] 用户消息: '{user_msg[:30]}...' → 意图: {detected_intent}")
        return NodeResult({"intent": detected_intent})


class RAGRetrievalNode(BaseNode):
    """RAG 检索节点 — 从知识库检索相关文档。"""

    def __init__(self):
        super().__init__("rag_retrieval")
        # 模拟知识库
        self.knowledge_base = [
            "RAG（检索增强生成）是结合检索和生成的 AI 架构模式。",
            "LangChain 是最流行的 LLM 应用开发框架，提供标准化组件。",
            "向量数据库（如 Chroma、Pinecone）用于存储和检索 Embedding。",
            "Agent 是具备自主决策和工具调用能力的 AI 系统。",
            "LangGraph 是基于图的 LLM 应用编排框架。",
            "Prompt Engineering 是设计和优化 AI 输入提示词的技术。",
        ]

    def execute(self, state: GraphState) -> NodeResult:
        """检索相关文档。"""
        query = state.get_last_message()
        # 模拟向量检索（关键词匹配）
        results = []
        for doc in self.knowledge_base:
            if any(word in doc for word in query[:20].split()):
                results.append(doc)
        if not results:
            results = [self.knowledge_base[0]]  # 默认返回第一条

        print(f"  [RAG 检索] 检索到 {len(results)} 条相关文档")
        return NodeResult({"context": results[:3]})


class RelevanceCheckNode(BaseNode):
    """相关性检查节点 — 检查检索结果是否相关。"""

    def __init__(self):
        super().__init__("relevance_check")

    def execute(self, state: GraphState) -> NodeResult:
        """检查检索到的上下文是否与问题相关。"""
        query = state.get_last_message()
        context = state.context
        # 模拟相关性评分
        is_relevant = len(context) > 0 and any(
            any(w in doc for w in query[:15].split()) for doc in context
        )
        print(f"  [相关性检查] 相关: {is_relevant}")
        return NodeResult({"metadata": {**state.metadata, "is_relevant": is_relevant}})


class QueryRewriteNode(BaseNode):
    """查询改写节点 — 优化查询表述。"""

    def __init__(self):
        super().__init__("query_rewrite")

    def execute(self, state: GraphState) -> NodeResult:
        """改写查询以提升检索质量。"""
        original = state.get_last_message()
        rewritten = f"[改写] {original} 的详细解释和核心原理"
        print(f"  [查询改写] '{original[:20]}...' → '{rewritten[:30]}...'")
        state.add_message("system", rewritten)
        return NodeResult({"retry_count": state.retry_count + 1})


class ToolExecutionNode(BaseNode):
    """工具执行节点 — 调用外部工具。"""

    def __init__(self):
        super().__init__("tool_execution")
        self.tools = {
            "search": lambda q: f"搜索结果: 找到关于 '{q[:15]}' 的 5 篇文档",
            "calculator": lambda q: "计算结果: 42",
            "database": lambda q: "数据库查询: 返回 10 条记录",
        }

    def execute(self, state: GraphState) -> NodeResult:
        """执行工具调用。"""
        query = state.get_last_message()
        # 模拟工具选择
        tool_name = "search"
        result = self.tools[tool_name](query)
        print(f"  [工具执行] 调用 '{tool_name}': {result}")
        return NodeResult({"tool_results": {**state.tool_results, tool_name: result}})


class HumanApprovalNode(BaseNode):
    """人工审批节点 — 模拟 Human-in-the-Loop。"""

    def __init__(self, auto_approve: bool = True):
        super().__init__("human_approval")
        self.auto_approve = auto_approve

    def execute(self, state: GraphState) -> NodeResult:
        """等待人工审批（模拟）。"""
        if self.auto_approve:
            print("  [人工审批] 自动审批通过（模拟）")
            return NodeResult({"human_approval": "approved"})
        else:
            print("  [人工审批] 等待人工审核...")
            print("  [人工审批] 审批拒绝（模拟）")
            return NodeResult({"human_approval": "rejected"})


class GenerateResponseNode(BaseNode):
    """生成回答节点 — 基于上下文生成回答。"""

    def __init__(self):
        super().__init__("generate_response")

    def execute(self, state: GraphState) -> NodeResult:
        """生成最终回答。"""
        context = state.context
        query = state.get_last_message()

        if context:
            answer = f"基于检索到的 {len(context)} 条文档：{context[0][:50]}..."
        elif state.tool_results:
            tool_info = json.dumps(state.tool_results, ensure_ascii=False)
            answer = f"工具调用结果：{tool_info[:80]}"
        elif state.human_approval == "rejected":
            answer = "抱歉，该操作已被管理员拒绝。"
        else:
            answer = f"关于 '{query[:20]}' 的回答：这是一个很好的问题。"

        print(f"  [生成回答] {answer[:50]}...")
        state.add_message("ai", answer)
        return NodeResult({"is_complete": True})


class QualityCheckNode(BaseNode):
    """质量检查节点 — 检查回答质量。"""

    def __init__(self):
        super().__init__("quality_check")

    def execute(self, state: GraphState) -> NodeResult:
        """检查生成回答的质量。"""
        last_ai_msg = ""
        for msg in reversed(state.messages):
            if msg["role"] == "ai":
                last_ai_msg = msg["content"]
                break
        # 模拟质量评分
        quality_score = min(len(last_ai_msg) / 50, 1.0)
        is_good = quality_score > 0.3
        print(f"  [质量检查] 评分: {quality_score:.2f}, 通过: {is_good}")
        return NodeResult({"metadata": {**state.metadata, "quality_score": quality_score, "quality_pass": is_good}})


# ============================================================
# 3. Edge — 边与条件路由
# ============================================================

class ConditionalEdge:
    """条件边 — 根据状态动态选择下一个节点。"""

    def __init__(self, condition_fn: Callable[[GraphState], str], routes: dict[str, str]):
        self.condition_fn = condition_fn
        self.routes = routes

    def resolve(self, state: GraphState) -> str:
        """根据条件函数返回下一个节点名称。"""
        result = self.condition_fn(state)
        return self.routes.get(result, "END")


# ============================================================
# 4. StateGraph — 图状态机
# ============================================================

class StateGraph:
    """图状态机 — LangGraph 核心模拟。"""

    def __init__(self, name: str = "workflow"):
        self.name = name
        self.nodes: dict[str, BaseNode] = {}
        self.edges: dict[str, str | ConditionalEdge] = {}
        self.entry_point: str = ""

    def add_node(self, node: BaseNode) -> None:
        """添加节点。"""
        self.nodes[node.name] = node

    def add_edge(self, from_node: str, to_node: str) -> None:
        """添加固定边。"""
        self.edges[from_node] = to_node

    def add_conditional_edge(self, from_node: str, condition_fn: Callable, routes: dict) -> None:
        """添加条件边。"""
        self.edges[from_node] = ConditionalEdge(condition_fn, routes)

    def set_entry_point(self, node_name: str) -> None:
        """设置入口节点。"""
        self.entry_point = node_name

    def compile(self) -> "CompiledGraph":
        """编译图，返回可执行的图。"""
        return CompiledGraph(self)


class CompiledGraph:
    """编译后的图 — 可执行。"""

    def __init__(self, graph: StateGraph):
        self.graph = graph
        self.checkpoints: list[dict] = []  # 持久化检查点

    def invoke(self, initial_state: GraphState, max_steps: int = 20) -> GraphState:
        """执行图。"""
        state = initial_state
        current = self.graph.entry_point
        step = 0

        print(f"\n  === 图 '{self.graph.name}' 开始执行 ===")

        while current != "END" and step < max_steps:
            step += 1
            state.current_node = current

            # 保存检查点
            self.checkpoints.append({"step": step, "node": current, "state": state.to_dict()})

            # 执行当前节点
            node = self.graph.nodes.get(current)
            if not node:
                print(f"  ⚠️ 节点 '{current}' 不存在，终止")
                break

            print(f"\n  [Step {step}] 执行节点: {current}")
            result = node.execute(state)

            # 应用状态更新
            for key, value in result.updates.items():
                if hasattr(state, key):
                    setattr(state, key, value)

            # 确定下一个节点
            edge = self.graph.edges.get(current)
            if edge is None:
                current = "END"
            elif isinstance(edge, str):
                current = edge
            elif isinstance(edge, ConditionalEdge):
                current = edge.resolve(state)
                print(f"  [路由] → {current}")
            else:
                current = "END"

            if state.is_complete:
                print(f"\n  === 图执行完成（{step} 步）===")
                break

        return state


# ============================================================
# 5. Persistence — 持久化模拟
# ============================================================

class MemorySaver:
    """内存持久化 — 开发调试用。"""

    def __init__(self):
        self.storage: dict[str, list[dict]] = {}

    def save(self, thread_id: str, state: dict) -> None:
        """保存状态。"""
        if thread_id not in self.storage:
            self.storage[thread_id] = []
        self.storage[thread_id].append({"timestamp": time.time(), "state": state})

    def load(self, thread_id: str) -> dict | None:
        """加载最新状态。"""
        if thread_id in self.storage and self.storage[thread_id]:
            return self.storage[thread_id][-1]["state"]
        return None

    def list_threads(self) -> list[str]:
        """列出所有线程。"""
        return list(self.storage.keys())


# ============================================================
# 6. 构建客服工作流示例
# ============================================================

def build_customer_service_graph() -> CompiledGraph:
    """构建智能客服工作流图。"""
    graph = StateGraph("customer_service")

    # 添加节点
    graph.add_node(IntentClassifierNode())
    graph.add_node(RAGRetrievalNode())
    graph.add_node(RelevanceCheckNode())
    graph.add_node(QueryRewriteNode())
    graph.add_node(ToolExecutionNode())
    graph.add_node(HumanApprovalNode(auto_approve=True))
    graph.add_node(GenerateResponseNode())
    graph.add_node(QualityCheckNode())

    # 设置入口
    graph.set_entry_point("intent_classifier")

    # 意图路由
    def route_by_intent(state: GraphState) -> str:
        return state.intent

    graph.add_conditional_edge("intent_classifier", route_by_intent, {
        "query": "rag_retrieval",
        "action": "tool_execution",
        "sensitive": "human_approval",
        "chat": "generate_response",
    })

    # RAG 检索 → 相关性检查
    graph.add_edge("rag_retrieval", "relevance_check")

    # 相关性路由
    def route_by_relevance(state: GraphState) -> str:
        is_relevant = state.metadata.get("is_relevant", False)
        if is_relevant:
            return "generate"
        elif state.retry_count < state.max_retries:
            return "rewrite"
        else:
            return "generate"

    graph.add_conditional_edge("relevance_check", route_by_relevance, {
        "generate": "generate_response",
        "rewrite": "query_rewrite",
    })

    # 查询改写 → 重新检索
    graph.add_edge("query_rewrite", "rag_retrieval")

    # 工具执行 → 生成回答
    graph.add_edge("tool_execution", "generate_response")

    # 人工审批路由
    def route_by_approval(state: GraphState) -> str:
        return "approved" if state.human_approval == "approved" else "rejected"

    graph.add_conditional_edge("human_approval", route_by_approval, {
        "approved": "tool_execution",
        "rejected": "generate_response",
    })

    # 生成回答 → 质量检查
    graph.add_edge("generate_response", "quality_check")

    # 质量检查 → 结束
    graph.add_edge("quality_check", "END")

    return graph.compile()


# ============================================================
# 演示函数
# ============================================================

def demo_basic_workflow() -> None:
    """演示基础工作流。"""
    print("\n" + "=" * 60)
    print("1. 基础工作流 — 知识查询")
    print("=" * 60)

    app = build_customer_service_graph()
    state = GraphState()
    state.add_message("human", "什么是 RAG 检索增强生成？")
    result = app.invoke(state)
    print(f"\n  最终回答: {result.messages[-1]['content'][:80]}...")


def demo_tool_workflow() -> None:
    """演示工具调用工作流。"""
    print("\n" + "=" * 60)
    print("2. 工具调用工作流 — 执行操作")
    print("=" * 60)

    app = build_customer_service_graph()
    state = GraphState()
    state.add_message("human", "执行搜索 LangChain 相关文档")
    result = app.invoke(state)
    print(f"\n  最终回答: {result.messages[-1]['content'][:80]}...")


def demo_human_in_the_loop() -> None:
    """演示 Human-in-the-Loop。"""
    print("\n" + "=" * 60)
    print("3. Human-in-the-Loop — 敏感操作审批")
    print("=" * 60)

    app = build_customer_service_graph()
    state = GraphState()
    state.add_message("human", "删除数据库中的用户数据")
    result = app.invoke(state)
    print(f"\n  审批结果: {result.human_approval}")
    print(f"  最终回答: {result.messages[-1]['content'][:80]}...")


def demo_persistence() -> None:
    """演示持久化。"""
    print("\n" + "=" * 60)
    print("4. 持久化 — 状态保存与恢复")
    print("=" * 60)

    saver = MemorySaver()

    # 保存状态
    state = GraphState()
    state.add_message("human", "什么是 LangGraph？")
    state.intent = "query"
    saver.save("thread_001", state.to_dict())
    print(f"  保存线程: thread_001")

    state2 = GraphState()
    state2.add_message("human", "如何使用 Agent？")
    saver.save("thread_002", state2.to_dict())
    print(f"  保存线程: thread_002")

    # 恢复状态
    loaded = saver.load("thread_001")
    print(f"  恢复线程 thread_001: {loaded['messages'][-1]['content'] if loaded else 'N/A'}")
    print(f"  所有线程: {saver.list_threads()}")


def demo_chat_workflow() -> None:
    """演示闲聊工作流。"""
    print("\n" + "=" * 60)
    print("5. 闲聊工作流 — 直接回答")
    print("=" * 60)

    app = build_customer_service_graph()
    state = GraphState()
    state.add_message("human", "你好，今天天气不错")
    result = app.invoke(state)
    print(f"\n  最终回答: {result.messages[-1]['content'][:80]}...")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行所有 LangGraph 工作流演示。"""
    print("LangGraph 工作流 — 图状态机/节点/条件路由/Human-in-the-Loop 模拟")
    print("=" * 60)

    demo_basic_workflow()
    demo_tool_workflow()
    demo_human_in_the_loop()
    demo_persistence()
    demo_chat_workflow()

    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("\n关键要点:")
    print("  1. StateGraph 将工作流建模为有向图，节点执行操作，边定义流转")
    print("  2. 条件路由根据 State 动态选择下一个节点")
    print("  3. Human-in-the-Loop 在关键步骤暂停等待人工审批")
    print("  4. 持久化支持状态保存和恢复，实现长任务续传")
    print("  5. 每个节点单一职责，通过 State 传递数据")
    print("  6. 设置最大步数和重试次数，防止无限循环")


if __name__ == "__main__":
    main()
