"""
多 Agent 系统 — LangGraph 风格多 Agent 协作

知识点：基于图状态机的多 Agent 协作系统，包括 Supervisor 模式、
       专业化 Agent（研究/编码/审查）、任务分解、状态管理、结果汇总

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
# 1. 状态定义
# ============================================================

class AgentRole(Enum):
    """Agent 角色。"""
    SUPERVISOR = "supervisor"
    RESEARCHER = "researcher"
    CODER = "coder"
    REVIEWER = "reviewer"
    PLANNER = "planner"


class TaskStatus(Enum):
    """任务状态。"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REVISION = "needs_revision"


@dataclass
class AgentMessage:
    """Agent 间通信消息。"""
    sender: str
    receiver: str
    content: str
    msg_type: str = "task"  # task / result / feedback / question
    timestamp: float = field(default_factory=time.time)


@dataclass
class SubTask:
    """子任务。"""
    id: str
    name: str
    description: str
    assigned_to: AgentRole = AgentRole.RESEARCHER
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    dependencies: list[str] = field(default_factory=list)
    feedback: str = ""


@dataclass
class WorkflowState:
    """工作流状态 — 在 Agent 间传递。"""
    user_request: str = ""
    plan: list[SubTask] = field(default_factory=list)
    current_phase: str = "planning"  # planning / execution / review / complete
    messages: list[AgentMessage] = field(default_factory=list)
    results: dict[str, str] = field(default_factory=dict)
    iteration: int = 0
    max_iterations: int = 3
    final_output: str = ""

    def add_message(self, sender: str, receiver: str, content: str, msg_type: str = "task") -> None:
        self.messages.append(AgentMessage(sender, receiver, content, msg_type))

    def get_completed_tasks(self) -> list[SubTask]:
        return [t for t in self.plan if t.status == TaskStatus.COMPLETED]


# ============================================================
# 2. Agent 基类与实现
# ============================================================

class BaseAgent(ABC):
    """Agent 基类。"""

    def __init__(self, name: str, role: AgentRole):
        self.name = name
        self.role = role

    @abstractmethod
    def process(self, state: WorkflowState, task: SubTask | None = None) -> dict[str, Any]:
        """处理任务，返回状态更新。"""

    def __repr__(self) -> str:
        return f"<{self.name} [{self.role.value}]>"


class PlannerAgent(BaseAgent):
    """规划 Agent — 将用户请求分解为子任务。"""

    def __init__(self):
        super().__init__("Planner", AgentRole.PLANNER)

    def process(self, state: WorkflowState, task: SubTask | None = None) -> dict:
        """分解用户请求为子任务。"""
        request = state.user_request
        print(f"  [{self.name}] 分析请求: '{request[:40]}...'")

        # 模拟任务分解
        plan = [
            SubTask("t1", "需求分析", f"分析 '{request[:20]}' 的需求和技术方案",
                    AgentRole.RESEARCHER),
            SubTask("t2", "技术调研", f"调研 '{request[:20]}' 的最佳实践和技术选型",
                    AgentRole.RESEARCHER, dependencies=["t1"]),
            SubTask("t3", "代码实现", f"编写 '{request[:20]}' 的核心代码",
                    AgentRole.CODER, dependencies=["t2"]),
            SubTask("t4", "代码审查", f"审查代码质量和安全性",
                    AgentRole.REVIEWER, dependencies=["t3"]),
        ]

        print(f"  [{self.name}] 分解为 {len(plan)} 个子任务:")
        for t in plan:
            deps = f" (依赖: {t.dependencies})" if t.dependencies else ""
            print(f"    [{t.id}] {t.name} → {t.assigned_to.value}{deps}")

        return {"plan": plan, "current_phase": "execution"}


class ResearcherAgent(BaseAgent):
    """研究 Agent — 负责调研和分析。"""

    def __init__(self):
        super().__init__("Researcher", AgentRole.RESEARCHER)

    def process(self, state: WorkflowState, task: SubTask | None = None) -> dict:
        if not task:
            return {}
        print(f"  [{self.name}] 执行: {task.name}")

        # 模拟调研
        if "需求" in task.name:
            result = json.dumps({
                "需求分析": {
                    "核心功能": ["数据处理", "API 接口", "错误处理"],
                    "技术栈": ["Python 3.11+", "FastAPI", "Pydantic"],
                    "预估工时": "3-5 天",
                },
            }, ensure_ascii=False, indent=2)
        else:
            result = json.dumps({
                "技术调研": {
                    "推荐方案": "FastAPI + Pydantic + SQLAlchemy",
                    "参考项目": ["project-a", "project-b"],
                    "注意事项": ["异步处理", "输入验证", "错误处理"],
                },
            }, ensure_ascii=False, indent=2)

        task.status = TaskStatus.COMPLETED
        task.result = result
        print(f"  [{self.name}] 完成: {task.name}")
        return {"results": {**state.results, task.id: result}}


class CoderAgent(BaseAgent):
    """编码 Agent — 负责代码实现。"""

    def __init__(self):
        super().__init__("Coder", AgentRole.CODER)

    def process(self, state: WorkflowState, task: SubTask | None = None) -> dict:
        if not task:
            return {}
        print(f"  [{self.name}] 执行: {task.name}")

        # 获取前序任务的上下文
        context = {tid: state.results.get(tid, "") for tid in task.dependencies}

        result = json.dumps({
            "代码实现": {
                "文件": ["main.py", "models.py", "api.py"],
                "代码行数": 350,
                "测试覆盖": "85%",
                "核心功能": "基于调研结果实现了完整的 API 服务",
            },
        }, ensure_ascii=False, indent=2)

        task.status = TaskStatus.COMPLETED
        task.result = result
        print(f"  [{self.name}] 完成: {task.name}")
        return {"results": {**state.results, task.id: result}}


class ReviewerAgent(BaseAgent):
    """审查 Agent — 负责代码审查和质量检查。"""

    def __init__(self):
        super().__init__("Reviewer", AgentRole.REVIEWER)

    def process(self, state: WorkflowState, task: SubTask | None = None) -> dict:
        if not task:
            return {}
        print(f"  [{self.name}] 执行: {task.name}")

        # 模拟代码审查
        review_result = {
            "审查结果": {
                "总体评分": 8.5,
                "通过": True,
                "优点": ["代码结构清晰", "错误处理完善", "类型注解完整"],
                "改进建议": ["增加日志记录", "补充边界测试"],
            },
        }

        result = json.dumps(review_result, ensure_ascii=False, indent=2)
        task.status = TaskStatus.COMPLETED
        task.result = result
        print(f"  [{self.name}] 完成: {task.name} (评分: 8.5/10)")
        return {"results": {**state.results, task.id: result}}


class SupervisorAgent(BaseAgent):
    """Supervisor Agent — 协调所有 Agent。"""

    def __init__(self, agents: dict[AgentRole, BaseAgent]):
        super().__init__("Supervisor", AgentRole.SUPERVISOR)
        self.agents = agents
        self.planner = PlannerAgent()

    def run(self, user_request: str, verbose: bool = True) -> WorkflowState:
        """执行完整的多 Agent 工作流。"""
        state = WorkflowState(user_request=user_request)

        if verbose:
            print(f"\n  🎯 用户请求: {user_request}")

        # 阶段 1: 规划
        print(f"\n  === 阶段 1: 任务规划 ===")
        updates = self.planner.process(state)
        state.plan = updates["plan"]
        state.current_phase = updates["current_phase"]

        # 阶段 2: 执行
        print(f"\n  === 阶段 2: 任务执行 ===")
        for task in state.plan:
            # 检查依赖
            deps_met = all(
                any(t.id == dep and t.status == TaskStatus.COMPLETED for t in state.plan)
                for dep in task.dependencies
            )
            if not deps_met and task.dependencies:
                print(f"  ⚠️ 任务 {task.id} 的依赖未满足，跳过")
                continue

            # 选择 Agent
            agent = self.agents.get(task.assigned_to)
            if not agent:
                print(f"  ❌ 未找到 {task.assigned_to.value} Agent")
                task.status = TaskStatus.FAILED
                continue

            # 执行任务
            task.status = TaskStatus.IN_PROGRESS
            state.add_message(self.name, agent.name, task.description, "task")
            updates = agent.process(state, task)
            state.results.update(updates.get("results", {}))
            state.add_message(agent.name, self.name, task.result[:50], "result")

        # 阶段 3: 汇总
        print(f"\n  === 阶段 3: 结果汇总 ===")
        completed = state.get_completed_tasks()
        state.current_phase = "complete"
        state.final_output = self._summarize(state)

        if verbose:
            print(f"  📊 完成 {len(completed)}/{len(state.plan)} 个任务")
            print(f"  📨 通信消息: {len(state.messages)} 条")
            print(f"  📝 最终输出: {state.final_output[:100]}...")

        return state

    def _summarize(self, state: WorkflowState) -> str:
        """汇总所有结果。"""
        parts = []
        for task in state.plan:
            if task.status == TaskStatus.COMPLETED:
                parts.append(f"[{task.name}] {task.result[:80]}")
        return "\n".join(parts)


# ============================================================
# 3. 工作流构建器
# ============================================================

class MultiAgentWorkflow:
    """多 Agent 工作流构建器。"""

    def __init__(self):
        self.agents: dict[AgentRole, BaseAgent] = {
            AgentRole.RESEARCHER: ResearcherAgent(),
            AgentRole.CODER: CoderAgent(),
            AgentRole.REVIEWER: ReviewerAgent(),
        }
        self.supervisor = SupervisorAgent(self.agents)

    def run(self, request: str) -> WorkflowState:
        """运行工作流。"""
        return self.supervisor.run(request)


# ============================================================
# 演示
# ============================================================

def demo_basic_workflow() -> None:
    """演示基础多 Agent 工作流。"""
    print("\n" + "=" * 60)
    print("1. 基础多 Agent 工作流")
    print("=" * 60)

    workflow = MultiAgentWorkflow()
    state = workflow.run("开发一个 RAG 知识库问答 API 服务")


def demo_communication_log() -> None:
    """演示 Agent 间通信日志。"""
    print("\n" + "=" * 60)
    print("2. Agent 间通信日志")
    print("=" * 60)

    workflow = MultiAgentWorkflow()
    state = workflow.run("实现一个智能客服系统")

    print(f"\n  📨 通信记录:")
    for msg in state.messages:
        arrow = "→" if msg.msg_type == "task" else "←"
        print(f"  {msg.sender} {arrow} {msg.receiver}: {msg.content[:40]}...")


def demo_task_dependencies() -> None:
    """演示任务依赖管理。"""
    print("\n" + "=" * 60)
    print("3. 任务依赖关系")
    print("=" * 60)

    workflow = MultiAgentWorkflow()
    state = workflow.run("构建多模态 AI 应用")

    print(f"\n  📋 任务执行顺序:")
    for task in state.plan:
        status_icon = "✅" if task.status == TaskStatus.COMPLETED else "❌"
        deps = f" ← {task.dependencies}" if task.dependencies else ""
        print(f"  {status_icon} [{task.id}] {task.name} ({task.assigned_to.value}){deps}")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行多 Agent 系统演示。"""
    print("多 Agent 系统 — LangGraph 风格多 Agent 协作")
    print("=" * 60)

    demo_basic_workflow()
    demo_communication_log()
    demo_task_dependencies()

    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("\n关键要点:")
    print("  1. Supervisor 模式：一个协调者分配任务给专业 Worker")
    print("  2. 任务分解是关键：将复杂请求拆分为原子子任务")
    print("  3. 依赖管理确保任务按正确顺序执行")
    print("  4. Agent 间通过结构化消息通信")
    print("  5. 审查 Agent 保证输出质量")
    print("  6. 状态管理支持工作流的暂停和恢复")


if __name__ == "__main__":
    main()
