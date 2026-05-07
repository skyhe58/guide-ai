"""
ReAct 模式 — 推理-行动-观察循环模拟

知识点：ReAct（Reasoning + Acting）模式、Thought-Action-Observation 循环、
       ReAct Prompt 设计、输出解析、停止条件、
       多工具 ReAct Agent、结构化 ReAct

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
import re
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


# ============================================================
# 1. ReAct 数据结构
# ============================================================

class StepType(Enum):
    """ReAct 步骤类型。"""
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL_ANSWER = "final_answer"


@dataclass
class ReActStep:
    """ReAct 单步记录。"""
    step_num: int
    type: StepType
    content: str
    action_name: str | None = None
    action_input: dict | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReActTrace:
    """ReAct 完整执行轨迹。"""
    question: str
    steps: list[ReActStep] = field(default_factory=list)
    final_answer: str = ""
    total_steps: int = 0
    success: bool = False

    def add_step(self, step: ReActStep) -> None:
        """添加步骤。"""
        self.steps.append(step)
        self.total_steps = len(self.steps)

    def summary(self) -> str:
        """输出执行摘要。"""
        lines = [f"问题: {self.question}"]
        lines.append(f"总步数: {self.total_steps}")
        lines.append(f"成功: {self.success}")
        for step in self.steps:
            prefix = {
                StepType.THOUGHT: "💭",
                StepType.ACTION: "⚡",
                StepType.OBSERVATION: "👁️",
                StepType.FINAL_ANSWER: "✅",
            }.get(step.type, "?")
            content_preview = step.content[:80]
            lines.append(f"  {prefix} [{step.step_num}] {step.type.value}: {content_preview}")
        return "\n".join(lines)


# ============================================================
# 2. 模拟工具集
# ============================================================

def tool_search(query: str) -> str:
    """搜索工具（模拟）。"""
    mock_results = {
        "RAG": "RAG（检索增强生成）通过先检索相关文档再生成回答，解决 LLM 知识过时和幻觉问题。核心流程：查询→检索→重排→生成。",
        "LangChain": "LangChain 是最流行的 LLM 应用开发框架，提供 Chain、Agent、Memory、Retriever 等核心抽象。",
        "向量数据库": "主流向量数据库：Chroma（轻量原型）、FAISS（高性能本地）、Milvus（分布式生产）、Pinecone（全托管云服务）。",
        "ReAct": "ReAct 模式结合推理和行动，让 Agent 在每步先思考再行动，通过观察结果决定下一步。",
        "Transformer": "Transformer 通过自注意力机制实现并行计算，是现代 LLM 的基础架构。",
    }
    for key, value in mock_results.items():
        if key.lower() in query.lower():
            return json.dumps({"query": query, "result": value}, ensure_ascii=False)
    return json.dumps({"query": query, "result": f"未找到与 '{query}' 相关的结果"}, ensure_ascii=False)


def tool_calculate(expression: str) -> str:
    """计算工具（模拟）。"""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return json.dumps({"error": "非法表达式"}, ensure_ascii=False)
    try:
        result = eval(expression)  # noqa: S307
        return json.dumps({"expression": expression, "result": result}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def tool_weather(city: str) -> str:
    """天气工具（模拟）。"""
    data = {
        "北京": {"temp": 25, "weather": "晴", "aqi": 75},
        "上海": {"temp": 28, "weather": "多云", "aqi": 60},
        "广州": {"temp": 32, "weather": "雷阵雨", "aqi": 45},
    }
    info = data.get(city)
    if info:
        return json.dumps({"city": city, **info}, ensure_ascii=False)
    return json.dumps({"error": f"未找到城市 '{city}'"}, ensure_ascii=False)


def tool_lookup_person(name: str) -> str:
    """人物查询工具（模拟）。"""
    people = {
        "Yann LeCun": "Yann LeCun 是深度学习先驱，CNN 之父，2018 年图灵奖得主，现任 Meta 首席 AI 科学家。",
        "Andrej Karpathy": "Andrej Karpathy 是 AI 研究者和教育者，前 Tesla AI 总监，前 OpenAI 研究员。",
        "吴恩达": "吴恩达（Andrew Ng）是 AI 领域领军人物，Coursera 联合创始人，DeepLearning.AI 创始人。",
    }
    info = people.get(name)
    if info:
        return json.dumps({"name": name, "info": info}, ensure_ascii=False)
    return json.dumps({"error": f"未找到人物 '{name}'"}, ensure_ascii=False)


# 工具注册表
TOOLS: dict[str, dict] = {
    "search": {
        "description": "搜索 AI 知识库，获取技术概念和框架信息",
        "handler": tool_search,
        "params": ["query"],
    },
    "calculate": {
        "description": "计算数学表达式",
        "handler": tool_calculate,
        "params": ["expression"],
    },
    "weather": {
        "description": "获取城市天气信息",
        "handler": tool_weather,
        "params": ["city"],
    },
    "lookup_person": {
        "description": "查询 AI 领域知名人物信息",
        "handler": tool_lookup_person,
        "params": ["name"],
    },
}


# ============================================================
# 3. ReAct Prompt 模板
# ============================================================

REACT_PROMPT_TEMPLATE = """Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {question}
Thought:"""


def build_react_prompt(question: str, tools: dict[str, dict]) -> str:
    """构建 ReAct Prompt。"""
    tool_descriptions = "\n".join(
        f"  {name}: {info['description']}" for name, info in tools.items()
    )
    tool_names = ", ".join(tools.keys())
    return REACT_PROMPT_TEMPLATE.format(
        tool_descriptions=tool_descriptions,
        tool_names=tool_names,
        question=question,
    )


# ============================================================
# 4. 模拟 LLM 的 ReAct 推理
# ============================================================

class MockReActLLM:
    """模拟 LLM 的 ReAct 推理行为。

    根据问题和当前上下文，模拟生成 Thought/Action/Final Answer。
    """

    def __init__(self, tools: dict[str, dict]):
        self.tools = tools

    def generate(self, prompt: str, observations: list[str]) -> str:
        """模拟 LLM 生成 ReAct 输出。"""
        # 从 prompt 中提取问题
        question = self._extract_question(prompt)

        # 根据已有观察决定下一步
        if not observations:
            return self._first_step(question)
        elif len(observations) == 1:
            return self._second_step(question, observations)
        else:
            return self._final_step(question, observations)

    def _extract_question(self, prompt: str) -> str:
        """提取问题。"""
        match = re.search(r"Question:\s*(.+?)(?:\n|Thought:)", prompt, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _first_step(self, question: str) -> str:
        """第一步推理。"""
        q_lower = question.lower()

        if any(kw in q_lower for kw in ["天气", "weather"]):
            city = "北京"
            for c in ["北京", "上海", "广州"]:
                if c in question:
                    city = c
                    break
            return (f"我需要查询{city}的天气信息。\n"
                    f"Action: weather\n"
                    f"Action Input: {city}")

        if any(kw in q_lower for kw in ["计算", "算", "等于", "多少"]):
            expr = "42 * 2 + 8"
            # 尝试提取数字表达式
            numbers = re.findall(r'\d+', question)
            if len(numbers) >= 2:
                expr = f"{numbers[0]} + {numbers[1]}"
            return (f"用户需要进行数学计算。\n"
                    f"Action: calculate\n"
                    f"Action Input: {expr}")

        if any(name in question for name in ["Yann LeCun", "Karpathy", "吴恩达"]):
            name = "吴恩达"
            for n in ["Yann LeCun", "Andrej Karpathy", "吴恩达"]:
                if n in question:
                    name = n
                    break
            return (f"用户想了解 {name} 的信息，我来查询一下。\n"
                    f"Action: lookup_person\n"
                    f"Action Input: {name}")

        # 默认搜索
        search_query = question[:20]
        return (f"我需要搜索相关信息来回答这个问题。\n"
                f"Action: search\n"
                f"Action Input: {search_query}")

    def _second_step(self, question: str, observations: list[str]) -> str:
        """第二步推理（有一个观察结果后）。"""
        obs = observations[0]
        # 检查是否需要更多信息
        if "error" in obs.lower() or "未找到" in obs:
            return (f"第一次搜索没有找到结果，让我换个关键词试试。\n"
                    f"Action: search\n"
                    f"Action Input: {question[:15]}")

        # 大多数情况下一次工具调用就够了
        return (f"我已经获得了足够的信息来回答问题。\n"
                f"Final Answer: 根据查询结果，{self._summarize(obs)}")

    def _final_step(self, question: str, observations: list[str]) -> str:
        """最终步骤。"""
        combined = " ".join(observations)
        return (f"综合所有信息，我可以给出最终答案。\n"
                f"Final Answer: {self._summarize(combined)}")

    def _summarize(self, text: str) -> str:
        """简单摘要。"""
        try:
            data = json.loads(text)
            if "result" in data:
                return str(data["result"])
            if "info" in data:
                return str(data["info"])
            # 天气数据
            if "temp" in data:
                return f"{data.get('city', '')}天气{data.get('weather', '')}，气温{data.get('temp', '')}°C"
            return json.dumps(data, ensure_ascii=False)[:200]
        except (json.JSONDecodeError, TypeError):
            return text[:200]


# ============================================================
# 5. ReAct Agent 执行引擎
# ============================================================

class ReActAgent:
    """ReAct Agent — 推理-行动-观察循环执行引擎。"""

    def __init__(self, tools: dict[str, dict], max_steps: int = 10,
                 verbose: bool = True):
        self.tools = tools
        self.max_steps = max_steps
        self.verbose = verbose
        self.llm = MockReActLLM(tools)

    def run(self, question: str) -> ReActTrace:
        """执行 ReAct 循环。"""
        trace = ReActTrace(question=question)
        prompt = build_react_prompt(question, self.tools)
        observations: list[str] = []
        step_num = 0

        if self.verbose:
            print(f"\n  🎯 问题: {question}")

        for _ in range(self.max_steps):
            step_num += 1

            # 1. LLM 生成 Thought + Action 或 Final Answer
            llm_output = self.llm.generate(prompt, observations)
            parsed = self._parse_output(llm_output)

            # 2. 记录 Thought
            if parsed.get("thought"):
                thought_step = ReActStep(
                    step_num=step_num,
                    type=StepType.THOUGHT,
                    content=parsed["thought"],
                )
                trace.add_step(thought_step)
                if self.verbose:
                    print(f"  💭 Thought [{step_num}]: {parsed['thought']}")

            # 3. 检查是否是 Final Answer
            if parsed.get("final_answer"):
                answer_step = ReActStep(
                    step_num=step_num,
                    type=StepType.FINAL_ANSWER,
                    content=parsed["final_answer"],
                )
                trace.add_step(answer_step)
                trace.final_answer = parsed["final_answer"]
                trace.success = True
                if self.verbose:
                    print(f"  ✅ Final Answer: {parsed['final_answer']}")
                break

            # 4. 执行 Action
            if parsed.get("action"):
                action_name = parsed["action"]
                action_input = parsed.get("action_input", "")

                action_step = ReActStep(
                    step_num=step_num,
                    type=StepType.ACTION,
                    content=f"{action_name}({action_input})",
                    action_name=action_name,
                    action_input={"input": action_input},
                )
                trace.add_step(action_step)
                if self.verbose:
                    print(f"  ⚡ Action [{step_num}]: {action_name}({action_input})")

                # 5. 执行工具获取 Observation
                observation = self._execute_tool(action_name, action_input)
                observations.append(observation)

                obs_step = ReActStep(
                    step_num=step_num,
                    type=StepType.OBSERVATION,
                    content=observation,
                )
                trace.add_step(obs_step)
                if self.verbose:
                    print(f"  👁️ Observation [{step_num}]: {observation[:100]}...")

                # 更新 prompt
                prompt += f"\n{llm_output}\nObservation: {observation}\nThought:"

        if not trace.success:
            trace.final_answer = "达到最大步数限制，无法完成任务"
            if self.verbose:
                print(f"  ⚠️ 达到最大步数限制 ({self.max_steps})")

        return trace

    def _parse_output(self, output: str) -> dict:
        """解析 LLM 输出，提取 Thought/Action/Final Answer。"""
        result: dict[str, Any] = {}

        # 提取 Final Answer
        final_match = re.search(r"Final Answer:\s*(.+)", output, re.DOTALL)
        if final_match:
            result["final_answer"] = final_match.group(1).strip()
            # 提取 Final Answer 之前的 Thought
            thought_part = output[:final_match.start()].strip()
            if thought_part:
                result["thought"] = thought_part.split("Thought:")[-1].strip() if "Thought:" in thought_part else thought_part
            return result

        # 提取 Action
        action_match = re.search(r"Action:\s*(\w+)", output)
        input_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", output)

        if action_match:
            result["action"] = action_match.group(1).strip()
        if input_match:
            result["action_input"] = input_match.group(1).strip()

        # 提取 Thought（Action 之前的部分）
        if action_match:
            thought_part = output[:action_match.start()].strip()
        else:
            thought_part = output.strip()

        if thought_part:
            result["thought"] = thought_part

        return result

    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """执行工具。"""
        tool_info = self.tools.get(tool_name)
        if not tool_info:
            return json.dumps({"error": f"工具 '{tool_name}' 不存在"}, ensure_ascii=False)

        handler = tool_info["handler"]
        params = tool_info["params"]

        try:
            # 将输入映射到第一个参数
            kwargs = {params[0]: tool_input}
            return handler(**kwargs)
        except Exception as e:
            return json.dumps({"error": f"工具执行失败: {e}"}, ensure_ascii=False)


# ============================================================
# 6. 结构化 ReAct（JSON 格式）
# ============================================================

@dataclass
class StructuredReActOutput:
    """结构化 ReAct 输出。"""
    thought: str
    action: str | None = None
    action_input: dict | None = None
    final_answer: str | None = None

    def to_json(self) -> str:
        """转为 JSON。"""
        data = {"thought": self.thought}
        if self.final_answer:
            data["final_answer"] = self.final_answer
        else:
            data["action"] = self.action or ""
            data["action_input"] = self.action_input or {}
        return json.dumps(data, ensure_ascii=False, indent=2)


def demo_structured_react() -> None:
    """演示结构化 ReAct 格式。"""
    print("\n" + "=" * 60)
    print("4. 结构化 ReAct（JSON 格式）")
    print("=" * 60)

    # 模拟结构化输出
    steps = [
        StructuredReActOutput(
            thought="用户想了解 RAG 技术，我需要搜索相关信息",
            action="search",
            action_input={"query": "RAG 检索增强生成"},
        ),
        StructuredReActOutput(
            thought="已获得 RAG 的基本信息，可以回答用户问题",
            final_answer="RAG 是检索增强生成技术，通过先检索再生成的方式提升 LLM 回答质量",
        ),
    ]

    for i, step in enumerate(steps, 1):
        print(f"\n  步骤 {i}:")
        print(f"  {step.to_json()}")

    print("\n  💡 结构化格式比文本格式更容易解析，推荐在生产环境使用")


# ============================================================
# 7. 演示函数
# ============================================================

def demo_basic_react() -> None:
    """演示基本 ReAct 循环。"""
    print("\n" + "=" * 60)
    print("1. 基本 ReAct 循环")
    print("=" * 60)

    agent = ReActAgent(TOOLS, max_steps=6, verbose=True)
    trace = agent.run("什么是 RAG？它有什么优势？")
    print(f"\n  📊 执行摘要: 共 {trace.total_steps} 步, 成功={trace.success}")


def demo_multi_tool_react() -> None:
    """演示多工具 ReAct。"""
    print("\n" + "=" * 60)
    print("2. 多工具 ReAct")
    print("=" * 60)

    agent = ReActAgent(TOOLS, max_steps=8, verbose=True)

    questions = [
        "北京今天天气怎么样？",
        "吴恩达是谁？他有什么贡献？",
    ]

    for q in questions:
        trace = agent.run(q)
        print(f"  📊 结果: {trace.final_answer[:80]}...")
        print()


def demo_react_trace() -> None:
    """演示 ReAct 执行轨迹。"""
    print("\n" + "=" * 60)
    print("3. ReAct 执行轨迹分析")
    print("=" * 60)

    agent = ReActAgent(TOOLS, max_steps=6, verbose=False)
    trace = agent.run("LangChain 是什么框架？")

    print(f"\n  {trace.summary()}")

    # 统计各类型步骤数
    type_counts: dict[str, int] = {}
    for step in trace.steps:
        key = step.type.value
        type_counts[key] = type_counts.get(key, 0) + 1
    print(f"\n  步骤统计: {type_counts}")


def demo_stop_conditions() -> None:
    """演示停止条件。"""
    print("\n" + "=" * 60)
    print("5. 停止条件设计")
    print("=" * 60)

    conditions = {
        "Final Answer": "LLM 输出 Final Answer 标记",
        "最大步数": "超过 max_iterations（通常 5-10 步）",
        "超时限制": "总执行时间超过上限（如 60 秒）",
        "重复检测": "连续两次相同的 Action + Input",
        "错误累积": "连续 N 次工具调用失败",
    }

    for name, desc in conditions.items():
        print(f"  🛑 {name}: {desc}")

    # 演示最大步数限制
    print("\n  --- 演示最大步数限制 ---")
    agent = ReActAgent(TOOLS, max_steps=2, verbose=True)
    trace = agent.run("这是一个需要很多步骤的复杂问题")
    print(f"  结果: {trace.final_answer}")


# ============================================================
# 服务模式 — 调用 Ollama API
# ============================================================

def demo_ollama_react() -> None:
    """服务模式：用 Ollama 演示 ReAct 推理循环。"""
    print("\n" + "=" * 60)
    print("服务模式 — 调用 Ollama API 演示 ReAct 推理")
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

    react_prompt = """你是一个 ReAct Agent。请按照以下格式逐步推理和行动：

Thought: 分析当前情况，决定下一步行动
Action: 选择工具 [search/calculate/weather]
Action Input: 工具的输入参数
Observation: （等待工具返回结果）
... （重复以上步骤直到得出答案）
Thought: 我已经得到了足够的信息
Final Answer: 最终答案

可用工具：
- search: 搜索知识库
- calculate: 数学计算
- weather: 查询天气

用户问题：北京今天气温多少度？如果气温乘以 2 是多少？

请开始推理："""

    response = ollama.generate(model=model, prompt=react_prompt, options={"num_predict": 400})
    print(f"\n  🤖 Ollama ReAct 推理过程:")
    print(f"  {response['response'][:500]}...")
    print(f"\n  💡 实际生产中，每步 Action 会真正调用工具，Observation 是真实结果")


# ============================================================
# 主入口
# ============================================================

def main(server_mode: bool = False) -> None:
    """运行所有 ReAct 模式演示。"""
    print("🐍 ReAct 模式 — 推理-行动-观察循环模拟")
    print("=" * 60)

    demo_basic_react()
    demo_multi_tool_react()
    demo_react_trace()
    demo_structured_react()
    demo_stop_conditions()

    if server_mode:
        demo_ollama_react()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. ReAct = Thought（推理）+ Action（行动）+ Observation（观察）循环")
    print("  2. Prompt 设计要严格定义格式，方便正则解析")
    print("  3. 必须设置停止条件：Final Answer / 最大步数 / 超时")
    print("  4. 结构化 ReAct（JSON 格式）比文本格式更可靠")
    print("  5. 每步的 Thought 提供可解释性，方便调试")
    print("  6. 生产环境需要流式输出、缓存、监控等优化")

    if not server_mode:
        print("\n💡 要测试真实 LLM 调用: python 03_react_pattern.py server")


if __name__ == "__main__":
    is_server = len(sys.argv) > 1 and sys.argv[1].lower() == "server"
    main(server_mode=is_server)
