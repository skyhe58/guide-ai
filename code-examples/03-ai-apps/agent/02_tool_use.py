"""
Tool Use — 自定义工具开发、工具注册、工具选择策略

知识点：BaseTool 抽象基类、装饰器注册、工具注册中心、
       工具选择策略（全量/语义路由/分类路由）、
       工具执行与错误处理、工具安全性

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
import math
import sys
import time
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable


# ============================================================
# 1. 工具基类与接口定义
# ============================================================

class ToolCategory(Enum):
    """工具分类。"""
    SEARCH = "search"
    CALCULATION = "calculation"
    DATA = "data"
    FILE = "file"
    API = "api"
    UTILITY = "utility"


@dataclass
class ToolResult:
    """工具执行结果。"""
    success: bool
    data: Any = None
    error: str | None = None
    execution_time_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


class BaseTool(ABC):
    """工具抽象基类。

    所有自定义工具必须继承此类，实现 execute 方法。
    """

    name: str = ""
    description: str = ""
    category: ToolCategory = ToolCategory.UTILITY
    version: str = "1.0"
    parameters_schema: dict = {}

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """执行工具逻辑，返回字符串结果。"""

    def validate_params(self, **kwargs: Any) -> tuple[bool, str]:
        """验证输入参数。子类可覆盖实现自定义验证。"""
        return True, ""

    def to_openai_schema(self) -> dict:
        """导出为 OpenAI Function Calling 格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    def __repr__(self) -> str:
        return f"<Tool: {self.name} v{self.version} [{self.category.value}]>"


# ============================================================
# 2. 自定义工具实现
# ============================================================

class WebSearchTool(BaseTool):
    """网页搜索工具（模拟）。"""

    name = "web_search"
    description = "在互联网上搜索信息，返回相关网页摘要"
    category = ToolCategory.SEARCH
    version = "1.2"
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"},
            "num_results": {"type": "integer", "description": "返回结果数量", "default": 5},
        },
        "required": ["query"],
    }

    def execute(self, query: str, num_results: int = 5, **kwargs: Any) -> str:
        """模拟网页搜索。"""
        # 模拟搜索结果
        mock_results = [
            {"title": f"关于 {query} 的详细介绍", "url": f"https://example.com/{i}",
             "snippet": f"这是关于 {query} 的第 {i} 条搜索结果摘要..."}
            for i in range(1, min(num_results, 5) + 1)
        ]
        return json.dumps({"query": query, "results": mock_results}, ensure_ascii=False)


class MathCalculatorTool(BaseTool):
    """数学计算工具。"""

    name = "math_calculator"
    description = "执行数学计算，支持基本运算和常用数学函数（sin, cos, sqrt, log 等）"
    category = ToolCategory.CALCULATION
    version = "1.0"
    parameters_schema = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "数学表达式"},
            "precision": {"type": "integer", "description": "小数精度", "default": 4},
        },
        "required": ["expression"],
    }

    # 安全的数学函数白名单
    SAFE_FUNCTIONS = {
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
        "abs": abs, "round": round, "pow": pow,
        "pi": math.pi, "e": math.e,
    }

    def validate_params(self, **kwargs: Any) -> tuple[bool, str]:
        """验证数学表达式安全性。"""
        expr = kwargs.get("expression", "")
        # 检查危险关键词
        dangerous = ["import", "exec", "eval", "open", "os", "sys", "__"]
        for d in dangerous:
            if d in expr.lower():
                return False, f"表达式包含危险关键词: {d}"
        return True, ""

    def execute(self, expression: str, precision: int = 4, **kwargs: Any) -> str:
        """安全执行数学计算。"""
        valid, msg = self.validate_params(expression=expression)
        if not valid:
            return json.dumps({"error": msg}, ensure_ascii=False)

        try:
            # 使用安全的命名空间执行
            result = eval(expression, {"__builtins__": {}}, self.SAFE_FUNCTIONS)  # noqa: S307
            if isinstance(result, float):
                result = round(result, precision)
            return json.dumps({
                "expression": expression,
                "result": result,
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"计算错误: {e}"}, ensure_ascii=False)


class DateTimeTool(BaseTool):
    """日期时间工具。"""

    name = "datetime_tool"
    description = "获取当前日期时间、计算日期差、格式化日期"
    category = ToolCategory.UTILITY
    version = "1.0"
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["now", "format", "diff"],
                "description": "操作类型：now=当前时间, format=格式化, diff=日期差",
            },
            "format": {"type": "string", "description": "日期格式", "default": "%Y-%m-%d %H:%M:%S"},
        },
        "required": ["action"],
    }

    def execute(self, action: str = "now", format: str = "%Y-%m-%d %H:%M:%S", **kwargs: Any) -> str:
        """执行日期时间操作。"""
        now = datetime.now()
        if action == "now":
            return json.dumps({
                "datetime": now.strftime(format),
                "timestamp": int(now.timestamp()),
                "weekday": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()],
            }, ensure_ascii=False)
        elif action == "format":
            return json.dumps({"formatted": now.strftime(format)}, ensure_ascii=False)
        else:
            return json.dumps({"error": f"不支持的操作: {action}"}, ensure_ascii=False)


class TextAnalysisTool(BaseTool):
    """文本分析工具。"""

    name = "text_analysis"
    description = "分析文本的字数、词频、语言等基本信息"
    category = ToolCategory.UTILITY
    version = "1.1"
    parameters_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "要分析的文本"},
            "analysis_type": {
                "type": "string",
                "enum": ["basic", "keywords", "summary"],
                "description": "分析类型",
                "default": "basic",
            },
        },
        "required": ["text"],
    }

    def execute(self, text: str, analysis_type: str = "basic", **kwargs: Any) -> str:
        """分析文本。"""
        if analysis_type == "basic":
            char_count = len(text)
            word_count = len(text.split())
            line_count = len(text.splitlines()) or 1
            # 简单判断语言
            has_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)
            language = "中文" if has_chinese else "英文"
            return json.dumps({
                "char_count": char_count,
                "word_count": word_count,
                "line_count": line_count,
                "language": language,
                "md5": hashlib.md5(text.encode()).hexdigest()[:8],
            }, ensure_ascii=False)
        elif analysis_type == "keywords":
            # 简单词频统计
            words = text.split()
            freq: dict[str, int] = {}
            for w in words:
                w = w.strip("，。！？、；：""''（）")
                if len(w) > 1:
                    freq[w] = freq.get(w, 0) + 1
            top_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
            return json.dumps({"keywords": [{"word": w, "count": c} for w, c in top_words]},
                              ensure_ascii=False)
        return json.dumps({"error": f"不支持的分析类型: {analysis_type}"}, ensure_ascii=False)


class HashTool(BaseTool):
    """哈希计算工具。"""

    name = "hash_tool"
    description = "计算文本的哈希值，支持 MD5、SHA256 等算法"
    category = ToolCategory.UTILITY
    version = "1.0"
    parameters_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "要计算哈希的文本"},
            "algorithm": {
                "type": "string",
                "enum": ["md5", "sha256", "sha1"],
                "description": "哈希算法",
                "default": "sha256",
            },
        },
        "required": ["text"],
    }

    def execute(self, text: str, algorithm: str = "sha256", **kwargs: Any) -> str:
        """计算哈希值。"""
        hash_funcs = {
            "md5": hashlib.md5,
            "sha256": hashlib.sha256,
            "sha1": hashlib.sha1,
        }
        func = hash_funcs.get(algorithm)
        if not func:
            return json.dumps({"error": f"不支持的算法: {algorithm}"}, ensure_ascii=False)
        digest = func(text.encode()).hexdigest()
        return json.dumps({
            "text_preview": text[:50],
            "algorithm": algorithm,
            "hash": digest,
        }, ensure_ascii=False)


# ============================================================
# 3. 工具注册中心
# ============================================================

class ToolRegistry:
    """工具注册中心 — 统一管理工具的注册、查询、分组。"""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._categories: dict[ToolCategory, list[str]] = {}

    def register(self, tool: BaseTool) -> None:
        """注册工具。"""
        if tool.name in self._tools:
            print(f"  ⚠️ 工具 '{tool.name}' 已存在，将被覆盖")
        self._tools[tool.name] = tool
        # 更新分类索引
        if tool.category not in self._categories:
            self._categories[tool.category] = []
        if tool.name not in self._categories[tool.category]:
            self._categories[tool.category].append(tool.name)

    def unregister(self, name: str) -> bool:
        """注销工具。"""
        tool = self._tools.pop(name, None)
        if tool and tool.category in self._categories:
            self._categories[tool.category] = [
                n for n in self._categories[tool.category] if n != name
            ]
            return True
        return False

    def get(self, name: str) -> BaseTool | None:
        """按名称获取工具。"""
        return self._tools.get(name)

    def list_all(self) -> list[BaseTool]:
        """列出所有工具。"""
        return list(self._tools.values())

    def list_by_category(self, category: ToolCategory) -> list[BaseTool]:
        """按分类获取工具列表。"""
        names = self._categories.get(category, [])
        return [self._tools[n] for n in names if n in self._tools]

    def to_openai_tools(self, category: ToolCategory | None = None) -> list[dict]:
        """导出为 OpenAI Function Calling 格式。"""
        tools = self.list_by_category(category) if category else self.list_all()
        return [t.to_openai_schema() for t in tools]

    def summary(self) -> str:
        """输出注册中心摘要。"""
        lines = [f"工具注册中心: 共 {len(self._tools)} 个工具"]
        for cat, names in self._categories.items():
            lines.append(f"  [{cat.value}] {len(names)} 个: {', '.join(names)}")
        return "\n".join(lines)


# ============================================================
# 4. 装饰器注册
# ============================================================

# 全局注册中心实例
_global_registry = ToolRegistry()


def register_tool(
    name: str,
    description: str,
    category: ToolCategory = ToolCategory.UTILITY,
    parameters_schema: dict | None = None,
) -> Callable:
    """装饰器：将函数注册为工具。"""

    def decorator(func: Callable) -> Callable:
        class DecoratedTool(BaseTool):
            def execute(self, **kwargs: Any) -> str:
                return func(**kwargs)

        tool = DecoratedTool()
        tool.name = name
        tool.description = description
        tool.category = category
        tool.parameters_schema = parameters_schema or {"type": "object", "properties": {}}
        _global_registry.register(tool)
        return func

    return decorator


# 使用装饰器注册工具
@register_tool(
    name="unit_converter",
    description="单位转换工具，支持温度、长度、重量等单位转换",
    category=ToolCategory.CALCULATION,
    parameters_schema={
        "type": "object",
        "properties": {
            "value": {"type": "number", "description": "要转换的数值"},
            "from_unit": {"type": "string", "description": "源单位"},
            "to_unit": {"type": "string", "description": "目标单位"},
        },
        "required": ["value", "from_unit", "to_unit"],
    },
)
def unit_converter(value: float, from_unit: str, to_unit: str, **kwargs: Any) -> str:
    """单位转换。"""
    conversions = {
        ("celsius", "fahrenheit"): lambda v: v * 9 / 5 + 32,
        ("fahrenheit", "celsius"): lambda v: (v - 32) * 5 / 9,
        ("km", "mile"): lambda v: v * 0.621371,
        ("mile", "km"): lambda v: v / 0.621371,
        ("kg", "lb"): lambda v: v * 2.20462,
        ("lb", "kg"): lambda v: v / 2.20462,
    }
    key = (from_unit.lower(), to_unit.lower())
    converter = conversions.get(key)
    if not converter:
        return json.dumps({"error": f"不支持的转换: {from_unit} → {to_unit}"}, ensure_ascii=False)
    result = round(converter(value), 4)
    return json.dumps({
        "original": f"{value} {from_unit}",
        "converted": f"{result} {to_unit}",
    }, ensure_ascii=False)


# ============================================================
# 5. 工具选择策略
# ============================================================

class ToolSelector:
    """工具选择器 — 根据查询选择最相关的工具。"""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def select_all(self) -> list[BaseTool]:
        """全量选择：返回所有工具（工具数 ≤ 10 时推荐）。"""
        return self.registry.list_all()

    def select_by_category(self, category: ToolCategory) -> list[BaseTool]:
        """分类选择：按工具分类筛选。"""
        return self.registry.list_by_category(category)

    def select_by_keywords(self, query: str, top_k: int = 5) -> list[BaseTool]:
        """关键词匹配选择（模拟语义路由）。

        生产环境应使用 embedding 相似度匹配。
        """
        scores: list[tuple[BaseTool, float]] = []
        query_lower = query.lower()

        for tool in self.registry.list_all():
            # 简单的关键词匹配评分
            score = 0.0
            desc_lower = tool.description.lower()
            name_lower = tool.name.lower()

            # 名称匹配
            for word in query_lower.split():
                if word in name_lower:
                    score += 2.0
                if word in desc_lower:
                    score += 1.0

            if score > 0:
                scores.append((tool, score))

        # 按分数排序，返回 Top-K
        scores.sort(key=lambda x: x[1], reverse=True)
        return [tool for tool, _ in scores[:top_k]]


# ============================================================
# 6. 工具执行器
# ============================================================

@dataclass
class ExecutionLog:
    """工具执行日志。"""
    tool_name: str
    arguments: dict
    result: ToolResult
    timestamp: str
    duration_ms: float


class ToolExecutor:
    """工具执行器 — 带日志、超时、重试的工具执行。"""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.logs: list[ExecutionLog] = []

    def execute(self, tool_name: str, arguments: dict,
                max_retries: int = 2, timeout_ms: float = 5000) -> ToolResult:
        """执行工具，带重试和日志。"""
        tool = self.registry.get(tool_name)
        if not tool:
            return ToolResult(success=False, error=f"工具 '{tool_name}' 不存在")

        # 参数验证
        valid, msg = tool.validate_params(**arguments)
        if not valid:
            return ToolResult(success=False, error=f"参数验证失败: {msg}")

        # 重试执行
        last_error = ""
        for attempt in range(max_retries + 1):
            start = time.time()
            try:
                result_str = tool.execute(**arguments)
                duration = (time.time() - start) * 1000
                result = ToolResult(
                    success=True,
                    data=result_str,
                    execution_time_ms=round(duration, 2),
                    metadata={"attempt": attempt + 1},
                )
                self._log(tool_name, arguments, result, duration)
                return result
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    time.sleep(0.1)  # 重试间隔

        result = ToolResult(success=False, error=f"执行失败（重试 {max_retries} 次）: {last_error}")
        self._log(tool_name, arguments, result, 0)
        return result

    def _log(self, tool_name: str, arguments: dict,
             result: ToolResult, duration: float) -> None:
        """记录执行日志。"""
        self.logs.append(ExecutionLog(
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            timestamp=datetime.now().isoformat(),
            duration_ms=round(duration, 2),
        ))

    def get_stats(self) -> dict:
        """获取执行统计。"""
        total = len(self.logs)
        success = sum(1 for log in self.logs if log.result.success)
        avg_duration = (
            sum(log.duration_ms for log in self.logs) / total if total > 0 else 0
        )
        return {
            "total_calls": total,
            "success_count": success,
            "failure_count": total - success,
            "success_rate": f"{success / total * 100:.1f}%" if total > 0 else "N/A",
            "avg_duration_ms": round(avg_duration, 2),
        }


# ============================================================
# 7. 演示函数
# ============================================================

def demo_tool_registration() -> None:
    """演示工具注册和管理。"""
    print("\n" + "=" * 60)
    print("1. 工具注册与管理")
    print("=" * 60)

    registry = ToolRegistry()

    # 注册工具
    tools = [
        WebSearchTool(),
        MathCalculatorTool(),
        DateTimeTool(),
        TextAnalysisTool(),
        HashTool(),
    ]
    for tool in tools:
        registry.register(tool)
        print(f"  ✅ {tool}")

    print(f"\n  {registry.summary()}")


def demo_tool_execution() -> None:
    """演示工具执行。"""
    print("\n" + "=" * 60)
    print("2. 工具执行与错误处理")
    print("=" * 60)

    registry = ToolRegistry()
    for tool in [WebSearchTool(), MathCalculatorTool(), DateTimeTool(),
                 TextAnalysisTool(), HashTool()]:
        registry.register(tool)

    executor = ToolExecutor(registry)

    # 正常执行
    test_cases = [
        ("web_search", {"query": "RAG 架构设计", "num_results": 3}),
        ("math_calculator", {"expression": "sqrt(144) + pi", "precision": 2}),
        ("datetime_tool", {"action": "now"}),
        ("text_analysis", {"text": "RAG 是检索增强生成技术，结合了检索和生成两种能力", "analysis_type": "basic"}),
        ("hash_tool", {"text": "Hello, Agent!", "algorithm": "sha256"}),
    ]

    for tool_name, args in test_cases:
        result = executor.execute(tool_name, args)
        status = "✅" if result.success else "❌"
        data_preview = str(result.data)[:80] if result.data else result.error
        print(f"  {status} {tool_name}: {data_preview}...")

    # 错误场景
    print("\n  --- 错误场景 ---")
    result = executor.execute("nonexistent", {})
    print(f"  ❌ 不存在的工具: {result.error}")

    result = executor.execute("math_calculator", {"expression": "import os"})
    print(f"  🛡️ 安全拦截: {result.error}")

    # 统计
    print(f"\n  📊 执行统计: {executor.get_stats()}")


def demo_tool_selection() -> None:
    """演示工具选择策略。"""
    print("\n" + "=" * 60)
    print("3. 工具选择策略")
    print("=" * 60)

    registry = ToolRegistry()
    for tool in [WebSearchTool(), MathCalculatorTool(), DateTimeTool(),
                 TextAnalysisTool(), HashTool()]:
        registry.register(tool)

    selector = ToolSelector(registry)

    # 全量选择
    all_tools = selector.select_all()
    print(f"\n  全量选择: {len(all_tools)} 个工具")

    # 分类选择
    calc_tools = selector.select_by_category(ToolCategory.CALCULATION)
    print(f"  计算类工具: {[t.name for t in calc_tools]}")

    # 关键词匹配
    queries = ["搜索 RAG 相关论文", "计算 sqrt(144)", "分析这段文本"]
    for query in queries:
        matched = selector.select_by_keywords(query, top_k=2)
        print(f"  查询 '{query}' → 匹配: {[t.name for t in matched]}")


def demo_decorator_registration() -> None:
    """演示装饰器注册。"""
    print("\n" + "=" * 60)
    print("4. 装饰器注册")
    print("=" * 60)

    print(f"  全局注册中心: {_global_registry.summary()}")

    # 使用装饰器注册的工具
    result = unit_converter(value=100, from_unit="celsius", to_unit="fahrenheit")
    print(f"  单位转换: 100°C → {result}")

    result = unit_converter(value=10, from_unit="km", to_unit="mile")
    print(f"  单位转换: 10km → {result}")


def demo_openai_export() -> None:
    """演示导出为 OpenAI 格式。"""
    print("\n" + "=" * 60)
    print("5. 导出为 OpenAI Function Calling 格式")
    print("=" * 60)

    registry = ToolRegistry()
    for tool in [WebSearchTool(), MathCalculatorTool(), DateTimeTool()]:
        registry.register(tool)

    openai_tools = registry.to_openai_tools()
    for tool_schema in openai_tools:
        func = tool_schema["function"]
        print(f"\n  📋 {func['name']}:")
        print(f"     描述: {func['description'][:50]}...")
        print(f"     参数: {list(func['parameters'].get('properties', {}).keys())}")


# ============================================================
# 服务模式 — 调用 Ollama API
# ============================================================

def demo_ollama_tool_selection() -> None:
    """服务模式：用 Ollama 演示工具选择。"""
    print("\n" + "=" * 60)
    print("服务模式 — 调用 Ollama API 演示工具选择")
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

    # 让 LLM 根据用户意图选择工具
    tool_descriptions = [
        "get_weather: 获取城市天气信息",
        "search_knowledge: 搜索 AI 知识库",
        "calculate: 计算数学表达式",
        "translate: 翻译文本",
    ]

    user_query = "帮我查一下北京今天的天气"
    prompt = f"""你是一个工具选择助手。根据用户问题选择最合适的工具。

可用工具：
{chr(10).join(f'- {t}' for t in tool_descriptions)}

用户问题：{user_query}

请输出 JSON 格式：{{"selected_tool": "工具名", "reason": "选择原因", "arguments": {{...}}}}"""

    response = ollama.generate(model=model, prompt=prompt, options={"num_predict": 200})
    print(f"\n  👤 用户: {user_query}")
    print(f"  🤖 Ollama 工具选择: {response['response'][:250]}...")


# ============================================================
# 主入口
# ============================================================

def main(server_mode: bool = False) -> None:
    """运行所有 Tool Use 演示。"""
    print("🐍 Tool Use — 自定义工具开发、工具注册、工具选择策略")
    print("=" * 60)

    demo_tool_registration()
    demo_tool_execution()
    demo_tool_selection()
    demo_decorator_registration()
    demo_openai_export()

    if server_mode:
        demo_ollama_tool_selection()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. BaseTool 抽象基类统一工具接口，支持参数验证和 schema 导出")
    print("  2. 工具注册中心集中管理，支持分类查询和动态注册/注销")
    print("  3. 装饰器注册简化工具开发流程")
    print("  4. 工具选择策略：全量 / 分类路由 / 关键词匹配（语义路由）")
    print("  5. 工具执行器提供重试、日志、统计等生产级功能")
    print("  6. 安全性：参数验证、白名单、危险关键词检测")

    if not server_mode:
        print("\n💡 要测试真实 LLM 调用: python 02_tool_use.py server")


if __name__ == "__main__":
    is_server = len(sys.argv) > 1 and sys.argv[1].lower() == "server"
    main(server_mode=is_server)
