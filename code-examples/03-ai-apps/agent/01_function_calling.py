"""
Function Calling — 工具定义、参数解析、多工具调用

知识点：OpenAI Function Calling 格式、JSON Schema 工具定义、
       参数解析与验证、多工具并行调用、工具选择策略、
       错误处理与重试机制

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
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ============================================================
# 1. 工具定义 — JSON Schema 格式
# ============================================================

class ParameterType(Enum):
    """JSON Schema 支持的参数类型。"""
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class ToolParameter:
    """工具参数定义。"""
    name: str
    type: ParameterType
    description: str
    required: bool = True
    enum: list[str] | None = None
    default: Any = None

    def to_schema(self) -> dict:
        """转换为 JSON Schema 格式。"""
        schema: dict[str, Any] = {
            "type": self.type.value,
            "description": self.description,
        }
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class ToolDefinition:
    """工具定义 — 对应 OpenAI Function Calling 的 tool 格式。"""
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    handler: Callable[..., str] | None = None

    def to_openai_tool(self) -> dict:
        """导出为 OpenAI API 的 tools 格式。"""
        properties = {}
        required = []
        for param in self.parameters:
            properties[param.name] = param.to_schema()
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


# ============================================================
# 2. 工具实现 — 模拟真实工具
# ============================================================

def get_weather(city: str, unit: str = "celsius") -> str:
    """获取指定城市的天气信息（模拟）。"""
    # 模拟天气数据
    weather_data = {
        "北京": {"temp": 25, "weather": "晴", "humidity": 40},
        "上海": {"temp": 28, "weather": "多云", "humidity": 65},
        "广州": {"temp": 32, "weather": "雷阵雨", "humidity": 80},
        "深圳": {"temp": 30, "weather": "阴", "humidity": 75},
        "杭州": {"temp": 27, "weather": "晴转多云", "humidity": 55},
    }
    data = weather_data.get(city)
    if not data:
        return json.dumps({"error": f"未找到城市 '{city}' 的天气数据"}, ensure_ascii=False)

    temp = data["temp"]
    if unit == "fahrenheit":
        temp = round(temp * 9 / 5 + 32, 1)

    return json.dumps({
        "city": city,
        "temperature": temp,
        "unit": unit,
        "weather": data["weather"],
        "humidity": data["humidity"],
    }, ensure_ascii=False)


def search_knowledge(query: str, top_k: int = 3) -> str:
    """搜索知识库（模拟）。"""
    # 模拟知识库搜索结果
    knowledge_base = [
        {"title": "RAG 架构概述", "content": "RAG 通过检索增强生成，先检索再生成", "score": 0.95},
        {"title": "向量数据库选型", "content": "Chroma 适合原型，Milvus 适合生产", "score": 0.88},
        {"title": "Embedding 模型对比", "content": "text-embedding-3-small 性价比最高", "score": 0.82},
        {"title": "LangChain 入门", "content": "LangChain 是最流行的 LLM 应用框架", "score": 0.75},
        {"title": "Agent 设计模式", "content": "ReAct 是最常用的 Agent 模式", "score": 0.70},
    ]
    # 简单模拟：根据关键词匹配
    results = []
    for item in knowledge_base:
        if any(kw in item["title"] or kw in item["content"] for kw in query):
            results.append(item)
    # 如果没有匹配，返回前 top_k 个
    if not results:
        results = knowledge_base[:top_k]
    else:
        results = results[:top_k]

    return json.dumps({"query": query, "results": results}, ensure_ascii=False)


def calculate(expression: str) -> str:
    """安全计算数学表达式（模拟）。"""
    # 白名单字符校验，防止代码注入
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in expression):
        return json.dumps({"error": "表达式包含非法字符"}, ensure_ascii=False)
    try:
        # 注意：生产环境应使用 ast.literal_eval 或专用数学库
        result = eval(expression)  # noqa: S307 — 仅用于演示
        return json.dumps({"expression": expression, "result": result}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"计算错误: {e}"}, ensure_ascii=False)


def get_stock_price(symbol: str, market: str = "A股") -> str:
    """获取股票价格（模拟）。"""
    mock_prices = {
        "AAPL": {"price": 178.50, "change": 2.3, "market": "美股"},
        "GOOGL": {"price": 141.20, "change": -0.8, "market": "美股"},
        "600519": {"price": 1680.00, "change": 1.5, "market": "A股"},
        "000001": {"price": 10.25, "change": -0.3, "market": "A股"},
    }
    data = mock_prices.get(symbol)
    if not data:
        return json.dumps({"error": f"未找到股票 '{symbol}'"}, ensure_ascii=False)
    return json.dumps({
        "symbol": symbol,
        "price": data["price"],
        "change_percent": data["change"],
        "market": data["market"],
    }, ensure_ascii=False)


# ============================================================
# 3. 工具注册表
# ============================================================

# 定义所有可用工具
TOOL_DEFINITIONS: list[ToolDefinition] = [
    ToolDefinition(
        name="get_weather",
        description="获取指定城市的当前天气信息，包括温度、天气状况和湿度",
        parameters=[
            ToolParameter("city", ParameterType.STRING, "城市名称，如'北京'、'上海'"),
            ToolParameter("unit", ParameterType.STRING, "温度单位",
                          required=False, enum=["celsius", "fahrenheit"], default="celsius"),
        ],
        handler=get_weather,
    ),
    ToolDefinition(
        name="search_knowledge",
        description="在 AI 知识库中搜索相关文档，返回最相关的结果",
        parameters=[
            ToolParameter("query", ParameterType.STRING, "搜索关键词"),
            ToolParameter("top_k", ParameterType.INTEGER, "返回结果数量",
                          required=False, default=3),
        ],
        handler=search_knowledge,
    ),
    ToolDefinition(
        name="calculate",
        description="计算数学表达式，支持加减乘除和括号",
        parameters=[
            ToolParameter("expression", ParameterType.STRING,
                          "数学表达式，如 '(3 + 5) * 2'"),
        ],
        handler=calculate,
    ),
    ToolDefinition(
        name="get_stock_price",
        description="获取股票的当前价格和涨跌幅",
        parameters=[
            ToolParameter("symbol", ParameterType.STRING, "股票代码，如 'AAPL'、'600519'"),
            ToolParameter("market", ParameterType.STRING, "市场",
                          required=False, enum=["A股", "美股"], default="A股"),
        ],
        handler=get_stock_price,
    ),
]


# ============================================================
# 4. Function Calling 执行引擎
# ============================================================

@dataclass
class ToolCallRequest:
    """LLM 返回的工具调用请求。"""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolCallResult:
    """工具调用结果。"""
    id: str
    name: str
    result: str
    success: bool
    error: str | None = None


class FunctionCallingEngine:
    """Function Calling 执行引擎。

    负责：
    - 管理工具注册表
    - 解析 LLM 返回的 function_call
    - 执行工具并返回结果
    - 处理错误和重试
    """

    def __init__(self, tools: list[ToolDefinition] | None = None):
        self._tools: dict[str, ToolDefinition] = {}
        if tools:
            for tool in tools:
                self.register(tool)

    def register(self, tool: ToolDefinition) -> None:
        """注册工具。"""
        self._tools[tool.name] = tool
        print(f"  ✅ 注册工具: {tool.name} — {tool.description[:40]}...")

    def get_openai_tools(self) -> list[dict]:
        """导出所有工具为 OpenAI API 格式。"""
        return [tool.to_openai_tool() for tool in self._tools.values()]

    def parse_tool_calls(self, raw_response: dict) -> list[ToolCallRequest]:
        """解析 LLM 响应中的工具调用请求。

        兼容 OpenAI 的 tool_calls 格式。
        """
        tool_calls = raw_response.get("tool_calls", [])
        parsed = []
        for tc in tool_calls:
            try:
                arguments = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  ⚠️ 参数解析失败: {e}")
                arguments = {}
            parsed.append(ToolCallRequest(
                id=tc.get("id", str(uuid.uuid4())),
                name=tc["function"]["name"],
                arguments=arguments,
            ))
        return parsed

    def execute(self, tool_call: ToolCallRequest) -> ToolCallResult:
        """执行单个工具调用。"""
        tool = self._tools.get(tool_call.name)
        if not tool:
            return ToolCallResult(
                id=tool_call.id,
                name=tool_call.name,
                result="",
                success=False,
                error=f"工具 '{tool_call.name}' 不存在",
            )
        if not tool.handler:
            return ToolCallResult(
                id=tool_call.id,
                name=tool_call.name,
                result="",
                success=False,
                error=f"工具 '{tool_call.name}' 没有实现 handler",
            )
        try:
            result = tool.handler(**tool_call.arguments)
            return ToolCallResult(
                id=tool_call.id,
                name=tool_call.name,
                result=result,
                success=True,
            )
        except TypeError as e:
            return ToolCallResult(
                id=tool_call.id,
                name=tool_call.name,
                result="",
                success=False,
                error=f"参数错误: {e}",
            )
        except Exception as e:
            return ToolCallResult(
                id=tool_call.id,
                name=tool_call.name,
                result="",
                success=False,
                error=f"执行错误: {e}",
            )

    def execute_batch(self, tool_calls: list[ToolCallRequest]) -> list[ToolCallResult]:
        """批量执行工具调用（模拟并行）。"""
        return [self.execute(tc) for tc in tool_calls]


# ============================================================
# 5. 模拟 LLM 的工具选择行为
# ============================================================

class MockLLM:
    """模拟 LLM 的 Function Calling 行为。

    根据用户输入中的关键词，模拟 LLM 选择工具和生成参数。
    """

    def __init__(self, tools: list[dict]):
        self._tools = tools
        self._tool_names = [t["function"]["name"] for t in tools]

    def generate(self, messages: list[dict]) -> dict:
        """模拟 LLM 生成响应。"""
        user_msg = messages[-1].get("content", "")

        # 根据关键词模拟工具选择
        tool_calls = self._select_tools(user_msg)

        if tool_calls:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls,
            }
        return {
            "role": "assistant",
            "content": f"我理解你的问题是关于：{user_msg}。但我没有找到合适的工具来处理。",
        }

    def _select_tools(self, query: str) -> list[dict]:
        """根据查询内容模拟工具选择。"""
        calls = []

        # 天气相关
        weather_cities = ["北京", "上海", "广州", "深圳", "杭州"]
        matched_cities = [c for c in weather_cities if c in query]
        for city in matched_cities:
            calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city": city}, ensure_ascii=False),
                },
            })

        # 计算相关
        if any(kw in query for kw in ["计算", "算", "等于", "加", "减", "乘", "除"]):
            # 提取简单表达式
            expr = "42 * 2 + 8"  # 模拟提取
            calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": "calculate",
                    "arguments": json.dumps({"expression": expr}),
                },
            })

        # 知识搜索相关
        if any(kw in query for kw in ["搜索", "查找", "什么是", "RAG", "Agent", "LLM"]):
            calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": "search_knowledge",
                    "arguments": json.dumps({"query": query, "top_k": 3}, ensure_ascii=False),
                },
            })

        # 股票相关
        if any(kw in query for kw in ["股票", "股价", "AAPL", "茅台", "600519"]):
            symbol = "600519" if "茅台" in query else "AAPL"
            calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": "get_stock_price",
                    "arguments": json.dumps({"symbol": symbol}),
                },
            })

        return calls


# ============================================================
# 6. 演示函数
# ============================================================

def demo_tool_definitions() -> None:
    """演示工具定义和 JSON Schema 导出。"""
    print("\n" + "=" * 60)
    print("1. 工具定义 — JSON Schema 格式")
    print("=" * 60)

    for tool_def in TOOL_DEFINITIONS:
        schema = tool_def.to_openai_tool()
        print(f"\n  📋 {tool_def.name}:")
        print(f"     描述: {tool_def.description}")
        params = schema["function"]["parameters"]
        print(f"     参数: {list(params['properties'].keys())}")
        print(f"     必填: {params.get('required', [])}")


def demo_single_tool_call() -> None:
    """演示单工具调用流程。"""
    print("\n" + "=" * 60)
    print("2. 单工具调用流程")
    print("=" * 60)

    engine = FunctionCallingEngine(TOOL_DEFINITIONS)
    llm = MockLLM(engine.get_openai_tools())

    # 模拟用户提问
    user_query = "北京今天天气怎么样？"
    print(f"\n  👤 用户: {user_query}")

    messages = [{"role": "user", "content": user_query}]
    response = llm.generate(messages)

    if response.get("tool_calls"):
        tool_calls = engine.parse_tool_calls(response)
        for tc in tool_calls:
            print(f"  🤖 LLM 选择工具: {tc.name}({tc.arguments})")
            result = engine.execute(tc)
            print(f"  🔧 工具结果: {result.result}")
            print(f"  ✅ 执行成功: {result.success}")
    else:
        print(f"  🤖 LLM 直接回答: {response['content']}")


def demo_parallel_tool_calls() -> None:
    """演示多工具并行调用。"""
    print("\n" + "=" * 60)
    print("3. 多工具并行调用")
    print("=" * 60)

    engine = FunctionCallingEngine(TOOL_DEFINITIONS)
    llm = MockLLM(engine.get_openai_tools())

    # 需要多个工具的查询
    user_query = "对比北京和上海的天气，顺便帮我搜索一下 RAG 相关知识"
    print(f"\n  👤 用户: {user_query}")

    messages = [{"role": "user", "content": user_query}]
    response = llm.generate(messages)

    if response.get("tool_calls"):
        tool_calls = engine.parse_tool_calls(response)
        print(f"  🤖 LLM 返回 {len(tool_calls)} 个工具调用:")

        # 批量执行（模拟并行）
        results = engine.execute_batch(tool_calls)
        for result in results:
            print(f"\n  🔧 [{result.name}] 成功={result.success}")
            if result.success:
                data = json.loads(result.result)
                print(f"     结果: {json.dumps(data, ensure_ascii=False, indent=2)[:200]}...")


def demo_error_handling() -> None:
    """演示错误处理。"""
    print("\n" + "=" * 60)
    print("4. 错误处理")
    print("=" * 60)

    engine = FunctionCallingEngine(TOOL_DEFINITIONS)

    # 场景 1：工具不存在
    print("\n  --- 场景 1：调用不存在的工具 ---")
    bad_call = ToolCallRequest(id="err_1", name="nonexistent_tool", arguments={})
    result = engine.execute(bad_call)
    print(f"  ❌ {result.error}")

    # 场景 2：参数错误
    print("\n  --- 场景 2：参数类型错误 ---")
    bad_args = ToolCallRequest(id="err_2", name="get_weather", arguments={"city": 123})
    result = engine.execute(bad_args)
    print(f"  结果: success={result.success}, result={result.result[:100] if result.result else result.error}")

    # 场景 3：计算表达式非法
    print("\n  --- 场景 3：非法计算表达式 ---")
    bad_expr = ToolCallRequest(id="err_3", name="calculate",
                               arguments={"expression": "import os; os.system('rm -rf /')"})
    result = engine.execute(bad_expr)
    print(f"  🛡️ 安全拦截: {result.result}")

    # 场景 4：城市不存在
    print("\n  --- 场景 4：查询不存在的城市 ---")
    unknown_city = ToolCallRequest(id="err_4", name="get_weather",
                                   arguments={"city": "亚特兰蒂斯"})
    result = engine.execute(unknown_city)
    print(f"  📍 {result.result}")


def demo_tool_choice_strategies() -> None:
    """演示工具选择策略。"""
    print("\n" + "=" * 60)
    print("5. 工具选择策略 (tool_choice)")
    print("=" * 60)

    strategies = {
        "auto": "LLM 自行决定是否调用工具（默认）",
        "none": "禁止调用工具，强制文本回答",
        "required": "必须调用至少一个工具",
        '{"type":"function","function":{"name":"get_weather"}}': "强制调用指定工具",
    }

    for choice, desc in strategies.items():
        print(f"\n  📋 tool_choice = {choice}")
        print(f"     行为: {desc}")

    # 模拟不同策略的效果
    print("\n  --- 策略效果对比 ---")
    print("  查询: '你好，今天心情怎么样？'")
    print("  auto   → 🤖 直接文本回答（不需要工具）")
    print("  none   → 🤖 强制文本回答")
    print("  required → 🤖 强制调用工具（可能选择不合适的工具）")


def demo_conversation_with_tools() -> None:
    """演示完整的多轮对话 + 工具调用流程。"""
    print("\n" + "=" * 60)
    print("6. 完整多轮对话 + 工具调用")
    print("=" * 60)

    engine = FunctionCallingEngine(TOOL_DEFINITIONS)
    llm = MockLLM(engine.get_openai_tools())

    # 模拟多轮对话
    conversations = [
        "帮我查一下北京的天气",
        "计算一下 42 乘以 2 加 8 等于多少",
        "搜索一下什么是 RAG",
        "查一下茅台的股价",
    ]

    messages: list[dict] = []

    for i, query in enumerate(conversations, 1):
        print(f"\n  --- 第 {i} 轮 ---")
        print(f"  👤 用户: {query}")

        messages.append({"role": "user", "content": query})
        response = llm.generate(messages)

        if response.get("tool_calls"):
            tool_calls = engine.parse_tool_calls(response)
            for tc in tool_calls:
                result = engine.execute(tc)
                print(f"  🔧 调用 {tc.name}({tc.arguments}) → {result.result[:80]}...")
                # 将工具结果加入对话历史
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result.result,
                })
        else:
            print(f"  🤖 {response['content']}")
            messages.append(response)


# ============================================================
# 7. 服务模式 — 调用 Ollama API
# ============================================================

def demo_ollama_function_calling() -> None:
    """服务模式：用 Ollama 演示 Function Calling 流程。"""
    print("\n" + "=" * 60)
    print("7. 服务模式 — 调用 Ollama API 演示 Function Calling")
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
    engine = FunctionCallingEngine(TOOL_DEFINITIONS)
    tools_schema = engine.get_openai_tools()

    # 使用 Ollama 进行工具选择推理
    user_query = "帮我查一下北京的天气，再搜索一下 RAG 相关知识"
    print(f"\n  👤 用户: {user_query}")

    prompt = f"""你是一个智能助手，可以使用以下工具：

{json.dumps(tools_schema, ensure_ascii=False, indent=2)}

用户问题：{user_query}

请分析用户意图，选择合适的工具。以 JSON 格式输出你的工具调用计划：
{{"tool_calls": [{{"name": "工具名", "arguments": {{...}}}}]}}"""

    response = ollama.generate(model=model, prompt=prompt, options={"num_predict": 300})
    print(f"  🤖 Ollama 响应: {response['response'][:300]}...")
    print(f"\n  💡 实际生产中，OpenAI/Claude 原生支持 Function Calling，无需手动解析")


# ============================================================
# 主入口
# ============================================================

def main(server_mode: bool = False) -> None:
    """运行所有 Function Calling 演示。"""
    print("🐍 Function Calling — 工具定义、参数解析、多工具调用")
    print("=" * 60)

    demo_tool_definitions()
    demo_single_tool_call()
    demo_parallel_tool_calls()
    demo_error_handling()
    demo_tool_choice_strategies()
    demo_conversation_with_tools()

    if server_mode:
        demo_ollama_function_calling()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. 工具定义用 JSON Schema 格式，description 决定 LLM 的选择")
    print("  2. 参数需要类型验证和安全校验，防止注入攻击")
    print("  3. 多工具并行调用提升效率，但需要错误隔离")
    print("  4. tool_choice 控制 LLM 的工具选择行为")
    print("  5. 生产环境需要超时控制、重试机制、日志追踪")
    print("  6. 工具调用结果要加入对话历史，保持上下文连贯")

    if not server_mode:
        print("\n💡 要测试真实 LLM 调用: python 01_function_calling.py server")


if __name__ == "__main__":
    is_server = len(sys.argv) > 1 and sys.argv[1].lower() == "server"
    main(server_mode=is_server)
