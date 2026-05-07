"""
Multi-Agent 协作 — Supervisor 模式模拟

知识点：Multi-Agent 架构设计、Supervisor 模式、Worker Agent、
       任务分解与分配、Agent 间通信、结果汇总、
       Sequential/Hierarchical/Debate 模式

Python 版本：3.11+
依赖：标准库（默认模式）、ollama>=0.1（服务模式）
最后验证：2024-12-01

外部服务（可选）：
  Ollama 本地 LLM 推理服务
  启动命令：docker compose -f docker/docker-compose.yml up -d ollama
  模型下载：docker exec guide-ai-ollama ollama pull qwen2
"""

from __future__ import annotations

import json
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ============================================================
# 1. 基础数据结构
# ============================================================

class AgentRole(Enum):
    """Agent 角色类型。"""
    SUPERVISOR = "supervisor"
    RESEARCHER = "researcher"
    CODER = "coder"
    WRITER = "writer"
    REVIEWER = "reviewer"
    ANALYST = "analyst"


class TaskStatus(Enum):
    """任务状态。"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentMessage:
    """Agent 间通信消息。"""
    sender: str
    receiver: str
    content: str
    message_type: str = "task"  # task, result, feedback
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class Task:
    """任务定义。"""
    id: str
    name: str
    description: str
    assigned_to: str = ""
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    dependencies: list[str] = field(default_factory=list)
    context: dict = field(default_factory=dict)


@dataclass
class AgentProfile:
    """Agent 能力描述。"""
    name: str
    role: AgentRole
    description: str
    capabilities: list[str] = field(default_factory=list)
    model: str = "gpt-4o"  # 使用的 LLM 模型


# ============================================================
# 2. Worker Agent 基类与实现
# ============================================================

class BaseAgent(ABC):
    """Agent 基类。"""

    def __init__(self, profile: AgentProfile):
        self.profile = profile
        self.message_history: list[AgentMessage] = []

    @property
    def name(self) -> str:
        return self.profile.name

    @abstractmethod
    def execute(self, task: Task) -> str:
        """执行任务，返回结果。"""

    def receive_message(self, message: AgentMessage) -> None:
        """接收消息。"""
        self.message_history.append(message)

    def __repr__(self) -> str:
        return f"<Agent: {self.name} [{self.profile.role.value}]>"


class ResearcherAgent(BaseAgent):
    """研究员 Agent — 负责信息搜索和调研。"""

    def __init__(self):
        super().__init__(AgentProfile(
            name="Researcher",
            role=AgentRole.RESEARCHER,
            description="负责搜索和调研技术信息，整理研究报告",
            capabilities=["web_search", "paper_search", "summarize"],
            model="gpt-4o",
        ))

    def execute(self, task: Task) -> str:
        """模拟研究调研。"""
        topic = task.description
        # 模拟搜索和整理
        research_results = {
            "topic": topic,
            "findings": [
                f"关于 {topic} 的核心概念和定义",
                f"{topic} 的主要技术方案和实现方式",
                f"{topic} 的最新发展趋势和业界实践",
                f"{topic} 的优缺点分析和适用场景",
            ],
            "sources": [
                "arXiv 论文 3 篇",
                "技术博客 5 篇",
                "官方文档 2 份",
            ],
            "summary": f"经过调研，{topic} 是当前 AI 领域的重要技术方向。"
                       f"主要应用场景包括知识问答、文档分析和智能客服。"
                       f"核心挑战在于检索质量和生成准确性的平衡。",
        }
        return json.dumps(research_results, ensure_ascii=False, indent=2)


class CoderAgent(BaseAgent):
    """编码 Agent — 负责编写代码示例。"""

    def __init__(self):
        super().__init__(AgentProfile(
            name="Coder",
            role=AgentRole.CODER,
            description="负责编写 Python 代码示例和技术实现",
            capabilities=["python", "code_review", "debugging"],
            model="gpt-4o",
        ))

    def execute(self, task: Task) -> str:
        """模拟代码编写。"""
        topic = task.description
        context = task.context

        code_result = {
            "topic": topic,
            "language": "Python",
            "code": f'''# {topic} 示例代码
class {topic.replace(" ", "")}Demo:
    """基于调研结果的代码实现。"""

    def __init__(self):
        self.config = {{"model": "gpt-4o", "temperature": 0.7}}

    def run(self, query: str) -> str:
        """执行核心逻辑。"""
        # 步骤 1: 预处理
        processed = self._preprocess(query)
        # 步骤 2: 核心处理
        result = self._process(processed)
        # 步骤 3: 后处理
        return self._postprocess(result)

    def _preprocess(self, text: str) -> str:
        return text.strip().lower()

    def _process(self, text: str) -> str:
        return f"处理结果: {{text}}"

    def _postprocess(self, text: str) -> str:
        return text
''',
            "tests": "包含 3 个单元测试用例",
            "dependencies": ["openai", "langchain"],
        }
        return json.dumps(code_result, ensure_ascii=False, indent=2)


class WriterAgent(BaseAgent):
    """写作 Agent — 负责撰写文档和报告。"""

    def __init__(self):
        super().__init__(AgentProfile(
            name="Writer",
            role=AgentRole.WRITER,
            description="负责撰写技术文档、报告和教程",
            capabilities=["technical_writing", "documentation", "tutorial"],
            model="gpt-4o",
        ))

    def execute(self, task: Task) -> str:
        """模拟文档撰写。"""
        topic = task.description
        context = task.context

        # 从上下文中获取研究结果和代码
        research = context.get("research", "无调研数据")
        code = context.get("code", "无代码示例")

        doc_result = {
            "topic": topic,
            "document_type": "技术报告",
            "sections": [
                {"title": "概述", "content": f"{topic} 是当前 AI 领域的热门技术..."},
                {"title": "核心原理", "content": f"基于调研结果，{topic} 的核心原理包括..."},
                {"title": "代码实现", "content": "参见附录中的代码示例..."},
                {"title": "最佳实践", "content": "在生产环境中，建议注意以下几点..."},
                {"title": "总结", "content": f"综上所述，{topic} 在实际应用中具有重要价值。"},
            ],
            "word_count": 2500,
            "references_count": 10,
        }
        return json.dumps(doc_result, ensure_ascii=False, indent=2)


class ReviewerAgent(BaseAgent):
    """审查 Agent — 负责质量审查和反馈。"""

    def __init__(self):
        super().__init__(AgentProfile(
            name="Reviewer",
            role=AgentRole.REVIEWER,
            description="负责审查文档和代码质量，提供改进建议",
            capabilities=["review", "quality_check", "feedback"],
            model="gpt-4o",
        ))

    def execute(self, task: Task) -> str:
        """模拟质量审查。"""
        content = task.context.get("content", "")

        review_result = {
            "overall_score": 8.5,
            "max_score": 10,
            "verdict": "通过",
            "strengths": [
                "内容结构清晰，逻辑连贯",
                "技术细节准确，示例代码可运行",
                "覆盖了核心概念和实践要点",
            ],
            "improvements": [
                "建议增加性能对比数据",
                "可以补充更多实际案例",
            ],
            "approved": True,
        }
        return json.dumps(review_result, ensure_ascii=False, indent=2)


# ============================================================
# 3. Supervisor Agent
# ============================================================

class SupervisorAgent:
    """Supervisor Agent — 负责任务分解、分配和协调。

    核心职责：
    1. 理解用户请求，分解为子任务
    2. 选择合适的 Worker Agent
    3. 传递上下文和中间结果
    4. 审查输出质量
    5. 汇总最终结果
    """

    def __init__(self, workers: list[BaseAgent], max_rounds: int = 5):
        self.workers: dict[str, BaseAgent] = {w.name: w for w in workers}
        self.max_rounds = max_rounds
        self.task_history: list[Task] = []
        self.message_log: list[AgentMessage] = []

    def plan(self, user_request: str) -> list[Task]:
        """将用户请求分解为子任务。"""
        # 模拟任务分解
        tasks = [
            Task(
                id="task_1",
                name="调研",
                description=user_request,
                assigned_to="Researcher",
            ),
            Task(
                id="task_2",
                name="编码",
                description=user_request,
                assigned_to="Coder",
                dependencies=["task_1"],
            ),
            Task(
                id="task_3",
                name="撰写",
                description=user_request,
                assigned_to="Writer",
                dependencies=["task_1", "task_2"],
            ),
            Task(
                id="task_4",
                name="审查",
                description=user_request,
                assigned_to="Reviewer",
                dependencies=["task_3"],
            ),
        ]
        return tasks

    def route(self, task: Task) -> BaseAgent | None:
        """根据任务分配选择 Worker。"""
        return self.workers.get(task.assigned_to)

    def run(self, user_request: str, verbose: bool = True) -> dict:
        """执行完整的 Multi-Agent 工作流。"""
        if verbose:
            print(f"\n  🎯 用户请求: {user_request}")
            print(f"  📋 开始任务分解...")

        # 1. 任务分解
        tasks = self.plan(user_request)
        if verbose:
            print(f"  📋 分解为 {len(tasks)} 个子任务:")
            for t in tasks:
                deps = f" (依赖: {t.dependencies})" if t.dependencies else ""
                print(f"     [{t.id}] {t.name} → {t.assigned_to}{deps}")

        # 2. 按依赖顺序执行
        results: dict[str, str] = {}
        context: dict[str, Any] = {}

        for task in tasks:
            # 检查依赖是否完成
            for dep_id in task.dependencies:
                dep_task = next((t for t in tasks if t.id == dep_id), None)
                if dep_task and dep_task.status != TaskStatus.COMPLETED:
                    if verbose:
                        print(f"  ⚠️ 任务 {task.id} 的依赖 {dep_id} 未完成")
                    continue

            # 传递上下文
            task.context = dict(context)

            # 选择 Worker
            worker = self.route(task)
            if not worker:
                if verbose:
                    print(f"  ❌ 未找到 Worker: {task.assigned_to}")
                task.status = TaskStatus.FAILED
                continue

            # 执行任务
            task.status = TaskStatus.IN_PROGRESS
            if verbose:
                print(f"\n  🔄 [{task.id}] {task.name} → {worker.name} 执行中...")

            # 发送消息
            msg = AgentMessage(
                sender="Supervisor",
                receiver=worker.name,
                content=task.description,
                message_type="task",
            )
            worker.receive_message(msg)
            self.message_log.append(msg)

            # 执行
            result = worker.execute(task)
            task.result = result
            task.status = TaskStatus.COMPLETED
            results[task.id] = result

            # 更新上下文
            context_key = task.name.lower()
            context[context_key] = result

            if verbose:
                result_preview = result[:100].replace("\n", " ")
                print(f"  ✅ [{task.id}] {task.name} 完成: {result_preview}...")

            self.task_history.append(task)

        # 3. 汇总结果
        summary = self._summarize(results, verbose)
        return summary

    def _summarize(self, results: dict[str, str], verbose: bool) -> dict:
        """汇总所有 Worker 的结果。"""
        summary = {
            "status": "completed",
            "total_tasks": len(results),
            "completed_tasks": len(results),
            "results": {},
        }

        for task_id, result in results.items():
            try:
                summary["results"][task_id] = json.loads(result)
            except json.JSONDecodeError:
                summary["results"][task_id] = result

        if verbose:
            print(f"\n  📊 汇总: 共 {len(results)} 个任务全部完成")

        return summary


# ============================================================
# 4. Sequential 模式（顺序执行）
# ============================================================

class SequentialPipeline:
    """顺序执行管道 — Agent 按固定顺序处理。"""

    def __init__(self, agents: list[BaseAgent]):
        self.agents = agents

    def run(self, initial_input: str, verbose: bool = True) -> str:
        """按顺序执行所有 Agent。"""
        if verbose:
            print(f"\n  🔗 顺序管道: {' → '.join(a.name for a in self.agents)}")

        current_input = initial_input
        for i, agent in enumerate(self.agents):
            task = Task(
                id=f"seq_{i}",
                name=f"步骤 {i + 1}",
                description=current_input,
                assigned_to=agent.name,
                context={"previous_output": current_input},
            )

            if verbose:
                print(f"  [{i + 1}/{len(self.agents)}] {agent.name} 处理中...")

            result = agent.execute(task)
            current_input = result

            if verbose:
                preview = result[:80].replace("\n", " ")
                print(f"  ✅ {agent.name} 完成: {preview}...")

        return current_input


# ============================================================
# 5. Debate 模式（协商讨论）
# ============================================================

class DebateSystem:
    """Debate 模式 — 多个 Agent 讨论得出最佳方案。"""

    def __init__(self, agents: list[BaseAgent], rounds: int = 3):
        self.agents = agents
        self.rounds = rounds

    def run(self, topic: str, verbose: bool = True) -> dict:
        """执行多轮讨论。"""
        if verbose:
            print(f"\n  🗣️ 讨论主题: {topic}")
            print(f"  👥 参与者: {', '.join(a.name for a in self.agents)}")
            print(f"  🔄 讨论轮数: {self.rounds}")

        discussion_log: list[dict] = []

        for round_num in range(1, self.rounds + 1):
            if verbose:
                print(f"\n  --- 第 {round_num} 轮 ---")

            round_opinions = []
            for agent in self.agents:
                task = Task(
                    id=f"debate_{round_num}_{agent.name}",
                    name=f"讨论第 {round_num} 轮",
                    description=topic,
                    context={"round": round_num, "previous_opinions": discussion_log},
                )
                opinion = agent.execute(task)
                round_opinions.append({
                    "agent": agent.name,
                    "round": round_num,
                    "opinion": opinion[:200],
                })

                if verbose:
                    print(f"  💬 {agent.name}: {opinion[:60].replace(chr(10), ' ')}...")

            discussion_log.extend(round_opinions)

        # 汇总讨论结果
        result = {
            "topic": topic,
            "rounds": self.rounds,
            "participants": [a.name for a in self.agents],
            "total_opinions": len(discussion_log),
            "consensus": f"经过 {self.rounds} 轮讨论，各方就 {topic} 达成了基本共识。",
        }

        if verbose:
            print(f"\n  📊 讨论结束: {result['consensus']}")

        return result


# ============================================================
# 6. 演示函数
# ============================================================

def demo_supervisor_mode() -> None:
    """演示 Supervisor 模式。"""
    print("\n" + "=" * 60)
    print("1. Supervisor 模式")
    print("=" * 60)

    workers = [
        ResearcherAgent(),
        CoderAgent(),
        WriterAgent(),
        ReviewerAgent(),
    ]

    supervisor = SupervisorAgent(workers)
    result = supervisor.run("RAG 检索增强生成技术")

    print(f"\n  📊 最终结果: {result['status']}, 完成 {result['completed_tasks']} 个任务")


def demo_sequential_mode() -> None:
    """演示 Sequential 模式。"""
    print("\n" + "=" * 60)
    print("2. Sequential 模式（顺序管道）")
    print("=" * 60)

    pipeline = SequentialPipeline([
        ResearcherAgent(),
        WriterAgent(),
        ReviewerAgent(),
    ])

    result = pipeline.run("Agent 记忆机制")
    print(f"\n  📊 管道输出长度: {len(result)} 字符")


def demo_debate_mode() -> None:
    """演示 Debate 模式。"""
    print("\n" + "=" * 60)
    print("3. Debate 模式（协商讨论）")
    print("=" * 60)

    debate = DebateSystem(
        agents=[ResearcherAgent(), CoderAgent(), WriterAgent()],
        rounds=2,
    )

    result = debate.run("LangChain vs LlamaIndex 哪个更适合 RAG 应用？")


def demo_agent_communication() -> None:
    """演示 Agent 间通信。"""
    print("\n" + "=" * 60)
    print("4. Agent 间通信机制")
    print("=" * 60)

    # 创建消息
    messages = [
        AgentMessage("Supervisor", "Researcher", "请调研 RAG 技术", "task"),
        AgentMessage("Researcher", "Supervisor", "调研完成，发现 3 个关键技术点", "result"),
        AgentMessage("Supervisor", "Coder", "基于调研结果编写代码", "task",
                     metadata={"research_data": "..."}),
        AgentMessage("Coder", "Supervisor", "代码编写完成", "result"),
        AgentMessage("Supervisor", "Reviewer", "请审查代码和文档", "task"),
        AgentMessage("Reviewer", "Supervisor", "审查通过，评分 8.5/10", "result"),
    ]

    print(f"\n  📨 通信记录:")
    for msg in messages:
        direction = "→" if msg.message_type == "task" else "←"
        print(f"  {msg.sender} {direction} {msg.receiver}: {msg.content}")


def demo_agent_profiles() -> None:
    """演示 Agent 能力描述。"""
    print("\n" + "=" * 60)
    print("5. Agent 能力描述与路由")
    print("=" * 60)

    agents = [ResearcherAgent(), CoderAgent(), WriterAgent(), ReviewerAgent()]

    print(f"\n  👥 Agent 团队:")
    for agent in agents:
        p = agent.profile
        print(f"  {agent}")
        print(f"     描述: {p.description}")
        print(f"     能力: {', '.join(p.capabilities)}")
        print(f"     模型: {p.model}")

    # 模拟路由决策
    print(f"\n  🔀 路由决策示例:")
    task_types = {
        "搜索 RAG 最新论文": "Researcher",
        "编写 Python 代码示例": "Coder",
        "撰写技术文档": "Writer",
        "审查代码质量": "Reviewer",
    }
    for task_desc, agent_name in task_types.items():
        print(f"  '{task_desc}' → {agent_name}")


# ============================================================
# 服务模式 — 调用 Ollama API
# ============================================================

def demo_ollama_multi_agent() -> None:
    """服务模式：用 Ollama 演示 Multi-Agent 任务分配。"""
    print("\n" + "=" * 60)
    print("服务模式 — 调用 Ollama API 演示 Multi-Agent 任务分配")
    print("=" * 60)

    try:
        import ollama
    except ImportError:
        print("  ⚠️ ollama 未安装: pip install ollama")
        return

    try:
        ollama.list()
    except Exception:
        print("  ❌ 无法连接 Ollama 服务")
        print("  💡 启动: docker compose -f docker/docker-compose.yml up -d ollama")
        return

    model = "qwen2"

    # Supervisor 角色：分解任务并分配给 Worker
    supervisor_prompt = """你是一个 Supervisor Agent，负责将复杂任务分解并分配给专业 Worker。

可用 Worker：
- researcher: 负责信息搜索和调研
- coder: 负责编写和审查代码
- writer: 负责撰写文档和报告

用户任务：帮我调研 RAG 技术，写一段 Python 示例代码，并生成一份技术报告。

请将任务分解并分配给合适的 Worker，以 JSON 格式输出：
{"task_plan": [{"worker": "名称", "task": "具体任务", "priority": 1-3}]}"""

    response = ollama.generate(model=model, prompt=supervisor_prompt, options={"num_predict": 300})
    print(f"\n  🤖 Supervisor 任务分配:")
    print(f"  {response['response'][:400]}...")
    print(f"\n  💡 实际生产中，每个 Worker 会独立执行任务并汇报结果给 Supervisor")


# ============================================================
# 主入口
# ============================================================

def main(server_mode: bool = False) -> None:
    """运行所有 Multi-Agent 演示。"""
    print("🐍 Multi-Agent 协作 — Supervisor 模式模拟")
    print("=" * 60)

    demo_supervisor_mode()
    demo_sequential_mode()
    demo_debate_mode()
    demo_agent_communication()
    demo_agent_profiles()

    if server_mode:
        demo_ollama_multi_agent()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. Supervisor 模式最实用：一个协调者 + 多个专业 Worker")
    print("  2. 任务分解是关键：将复杂任务拆分为原子子任务")
    print("  3. 上下文传递：前序 Agent 的输出作为后续 Agent 的输入")
    print("  4. Sequential 适合流水线，Debate 适合需要多角度分析的场景")
    print("  5. Agent 间通信要结构化，支持消息追踪和审计")
    print("  6. 设置最大轮次和超时，避免 Agent 间无限来回")

    if not server_mode:
        print("\n💡 要测试真实 LLM 调用: python 04_multi_agent.py server")


if __name__ == "__main__":
    is_server = len(sys.argv) > 1 and sys.argv[1].lower() == "server"
    main(server_mode=is_server)
