"""
MCP Multi-Agent 系统 — 里程碑项目

知识点：MCP 协议集成、多 Agent 协作、任务编排、工具注册、
       资源共享、安全防护、可观测性

Python 版本：3.11+
依赖：标准库
最后验证：2024-12-01

项目说明：
  本项目实现一个基于 MCP 协议的多 Agent 协作系统。
  包含三个专业 Agent（研究、分析、写作），通过 Orchestrator 统一调度。
  每个 Agent 作为 MCP Server 暴露能力，Orchestrator 作为 MCP Client 调用。

运行方式：
  python main.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional


# ============================================================
# 1. MCP 协议基础设施（简化版）
# ============================================================

class SimpleMCPServer:
    """简化的 MCP Server 基类"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self._tools: dict[str, dict] = {}

    def register_tool(self, name: str, description: str,
                      handler: Callable, schema: dict = None):
        """注册工具"""
        self._tools[name] = {
            "name": name,
            "description": description,
            "handler": handler,
            "inputSchema": schema or {"type": "object", "properties": {}},
        }

    def list_tools(self) -> list[dict]:
        """列出所有工具"""
        return [
            {"name": t["name"], "description": t["description"],
             "inputSchema": t["inputSchema"]}
            for t in self._tools.values()
        ]

    async def call_tool(self, name: str, arguments: dict) -> str:
        """调用工具"""
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"工具 {name} 不存在于 {self.name}")
        handler = tool["handler"]
        if asyncio.iscoroutinefunction(handler):
            return await handler(**arguments)
        return handler(**arguments)


class SimpleMCPClient:
    """简化的 MCP Client"""

    def __init__(self, name: str):
        self.name = name
        self._servers: dict[str, SimpleMCPServer] = {}
        self._tool_map: dict[str, str] = {}  # tool_name -> server_name

    async def connect(self, server: SimpleMCPServer):
        """连接到 Server"""
        self._servers[server.name] = server
        for tool in server.list_tools():
            self._tool_map[tool["name"]] = server.name
        print(f"  [Client] 已连接 {server.name}，"
              f"发现 {len(server.list_tools())} 个工具")

    def get_all_tools(self) -> list[dict]:
        """获取所有可用工具"""
        tools = []
        for server in self._servers.values():
            for tool in server.list_tools():
                tools.append({**tool, "_server": server.name})
        return tools

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """调用工具（自动路由到正确的 Server）"""
        server_name = self._tool_map.get(tool_name)
        if not server_name:
            raise ValueError(f"工具 {tool_name} 未找到")
        server = self._servers[server_name]
        return await server.call_tool(tool_name, arguments)


# ============================================================
# 2. 专业 Agent 定义
# ============================================================

class ResearchAgent(SimpleMCPServer):
    """
    研究 Agent — 负责信息搜索和资料收集

    暴露的 MCP 工具：
    - search_topic: 搜索指定主题的信息
    - get_latest_news: 获取最新新闻
    - summarize_sources: 总结多个信息源
    """

    def __init__(self):
        super().__init__("research-agent", "研究助手 — 信息搜索和资料收集")

        # 模拟知识库
        self._knowledge_base = {
            "AI": {
                "trends": ["大模型小型化", "多模态融合", "Agent 自主化", "AI 安全与对齐"],
                "tools": ["GPT-4o", "Claude 3.5", "Gemini 1.5", "Llama 3"],
                "applications": ["RAG 知识库", "AI 编程助手", "智能客服", "内容生成"],
            },
            "MCP": {
                "trends": ["标准化 Agent 通信", "MCP Server 生态", "IDE 集成"],
                "tools": ["MCP Python SDK", "MCP Inspector", "FastMCP"],
                "applications": ["IDE 工具集成", "Agent 协作", "数据源连接"],
            },
            "安全": {
                "trends": ["Prompt Injection 防御", "红队测试自动化", "AI 对齐"],
                "tools": ["Garak", "ART", "Fairlearn"],
                "applications": ["安全审计", "偏见检测", "内容过滤"],
            },
        }

        # 注册工具
        self.register_tool(
            "search_topic",
            "搜索指定主题的相关信息，返回结构化的搜索结果",
            self._search_topic,
            {"type": "object", "properties": {
                "topic": {"type": "string", "description": "搜索主题"},
                "max_results": {"type": "integer", "description": "最大结果数"},
            }, "required": ["topic"]},
        )

        self.register_tool(
            "get_latest_news",
            "获取指定领域的最新动态和新闻",
            self._get_latest_news,
            {"type": "object", "properties": {
                "domain": {"type": "string", "description": "领域名称"},
            }, "required": ["domain"]},
        )

        self.register_tool(
            "summarize_sources",
            "总结多个信息源的内容，提取关键要点",
            self._summarize_sources,
            {"type": "object", "properties": {
                "sources": {"type": "string", "description": "信息源内容（JSON 格式）"},
            }, "required": ["sources"]},
        )

    def _search_topic(self, topic: str, max_results: int = 5) -> str:
        """搜索主题"""
        results = []
        for key, data in self._knowledge_base.items():
            if key.lower() in topic.lower() or topic.lower() in key.lower():
                results.append({
                    "topic": key,
                    "trends": data["trends"][:max_results],
                    "tools": data["tools"][:max_results],
                    "applications": data["applications"][:max_results],
                })
        if not results:
            results.append({"topic": topic, "message": "未找到相关信息，建议扩大搜索范围"})
        return json.dumps(results, ensure_ascii=False)

    def _get_latest_news(self, domain: str) -> str:
        """获取最新动态"""
        news = {
            "AI": [
                {"title": "GPT-5 即将发布", "date": "2025-07-01", "source": "OpenAI Blog"},
                {"title": "MCP 协议 2.0 规范发布", "date": "2025-06-15", "source": "Anthropic"},
                {"title": "Kiro IDE 正式发布", "date": "2025-07-14", "source": "AWS"},
            ],
            "安全": [
                {"title": "OWASP LLM Top 10 更新", "date": "2025-06-01", "source": "OWASP"},
                {"title": "新型 Prompt Injection 攻击被发现", "date": "2025-05-20", "source": "Security Research"},
            ],
        }
        domain_news = news.get(domain, [{"title": f"{domain} 领域暂无最新动态"}])
        return json.dumps(domain_news, ensure_ascii=False)

    def _summarize_sources(self, sources: str) -> str:
        """总结信息源"""
        try:
            data = json.loads(sources)
            summary_points = []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        for key, value in item.items():
                            if isinstance(value, list):
                                summary_points.extend(value[:2])
                            elif isinstance(value, str) and len(value) > 10:
                                summary_points.append(value)
            return json.dumps({
                "summary": f"共分析 {len(summary_points)} 个要点",
                "key_points": summary_points[:5],
            }, ensure_ascii=False)
        except json.JSONDecodeError:
            return json.dumps({"summary": "无法解析输入数据", "key_points": []})


class AnalysisAgent(SimpleMCPServer):
    """
    分析 Agent — 负责数据分析和洞察提取

    暴露的 MCP 工具：
    - analyze_data: 分析数据并提取洞察
    - compare_items: 对比分析多个项目
    - generate_insights: 生成分析洞察
    """

    def __init__(self):
        super().__init__("analysis-agent", "分析助手 — 数据分析和洞察提取")

        self.register_tool(
            "analyze_data",
            "对输入数据进行分析，提取关键指标和趋势",
            self._analyze_data,
            {"type": "object", "properties": {
                "data": {"type": "string", "description": "待分析的数据（JSON 格式）"},
                "analysis_type": {"type": "string", "description": "分析类型"},
            }, "required": ["data"]},
        )

        self.register_tool(
            "compare_items",
            "对比分析多个项目，生成对比报告",
            self._compare_items,
            {"type": "object", "properties": {
                "items": {"type": "string", "description": "待对比的项目（JSON 格式）"},
            }, "required": ["items"]},
        )

        self.register_tool(
            "generate_insights",
            "基于分析结果生成商业洞察和建议",
            self._generate_insights,
            {"type": "object", "properties": {
                "analysis": {"type": "string", "description": "分析结果"},
                "context": {"type": "string", "description": "业务上下文"},
            }, "required": ["analysis"]},
        )

    def _analyze_data(self, data: str, analysis_type: str = "general") -> str:
        """分析数据"""
        try:
            parsed = json.loads(data)
            # 提取统计信息
            if isinstance(parsed, list):
                count = len(parsed)
                return json.dumps({
                    "analysis_type": analysis_type,
                    "data_points": count,
                    "summary": f"分析了 {count} 条数据",
                    "findings": [
                        f"数据集包含 {count} 个条目",
                        "数据质量良好，无明显异常",
                        "建议进一步深入分析趋势",
                    ],
                }, ensure_ascii=False)
            return json.dumps({"summary": "数据格式不支持", "findings": []})
        except json.JSONDecodeError:
            return json.dumps({"summary": "数据解析失败", "findings": []})

    def _compare_items(self, items: str) -> str:
        """对比分析"""
        try:
            parsed = json.loads(items)
            if isinstance(parsed, list) and len(parsed) >= 2:
                return json.dumps({
                    "comparison": f"对比了 {len(parsed)} 个项目",
                    "similarities": ["都属于 AI 领域", "都在快速发展"],
                    "differences": ["技术路线不同", "适用场景不同", "成熟度不同"],
                    "recommendation": "根据具体需求选择最合适的方案",
                }, ensure_ascii=False)
            return json.dumps({"comparison": "需要至少 2 个项目进行对比"})
        except json.JSONDecodeError:
            return json.dumps({"comparison": "数据解析失败"})

    def _generate_insights(self, analysis: str,
                           context: str = "AI 技术") -> str:
        """生成洞察"""
        return json.dumps({
            "context": context,
            "insights": [
                "AI 技术正在从单一模态向多模态融合演进",
                "MCP 协议有望成为 Agent 通信的标准",
                "AI 安全将成为企业采用 AI 的关键考量",
                "Vibe Coding 正在改变软件开发的方式",
            ],
            "action_items": [
                "关注 MCP 协议生态发展",
                "建立 AI 安全评估流程",
                "探索多模态 AI 应用场景",
            ],
        }, ensure_ascii=False)


class WriterAgent(SimpleMCPServer):
    """
    写作 Agent — 负责内容生成和报告撰写

    暴露的 MCP 工具：
    - write_report: 撰写分析报告
    - format_output: 格式化输出内容
    """

    def __init__(self):
        super().__init__("writer-agent", "写作助手 — 内容生成和报告撰写")

        self.register_tool(
            "write_report",
            "基于分析结果撰写结构化报告",
            self._write_report,
            {"type": "object", "properties": {
                "title": {"type": "string", "description": "报告标题"},
                "content": {"type": "string", "description": "报告内容（JSON 格式）"},
                "format": {"type": "string", "description": "输出格式"},
            }, "required": ["title", "content"]},
        )

        self.register_tool(
            "format_output",
            "将数据格式化为可读的文本输出",
            self._format_output,
            {"type": "object", "properties": {
                "data": {"type": "string", "description": "待格式化的数据"},
                "style": {"type": "string", "description": "格式化风格"},
            }, "required": ["data"]},
        )

    def _write_report(self, title: str, content: str,
                      format: str = "markdown") -> str:
        """撰写报告"""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = {"raw": content}

        report_lines = [
            f"# {title}",
            f"\n> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "\n## 摘要\n",
        ]

        # 提取关键信息
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    report_lines.append(f"\n### {key}\n")
                    for item in value:
                        report_lines.append(f"- {item}")
                elif isinstance(value, str):
                    report_lines.append(f"\n**{key}**: {value}")

        report_lines.append("\n\n## 结论\n")
        report_lines.append("基于以上分析，建议持续关注 AI 前沿技术发展，"
                            "特别是 MCP 协议和 AI 安全领域。")

        return "\n".join(report_lines)

    def _format_output(self, data: str, style: str = "bullet") -> str:
        """格式化输出"""
        try:
            parsed = json.loads(data)
            if isinstance(parsed, dict):
                lines = []
                for key, value in parsed.items():
                    if isinstance(value, list):
                        lines.append(f"\n{key}:")
                        for item in value:
                            lines.append(f"  • {item}")
                    else:
                        lines.append(f"{key}: {value}")
                return "\n".join(lines)
            return str(parsed)
        except json.JSONDecodeError:
            return data


# ============================================================
# 3. Orchestrator — 多 Agent 编排器
# ============================================================

class Orchestrator:
    """
    多 Agent 编排器

    通过 MCP Client 连接多个 Agent，实现任务分解和协调执行。
    """

    def __init__(self):
        self.client = SimpleMCPClient("orchestrator")
        self._execution_log: list[dict] = []

    async def setup(self):
        """初始化：连接所有 Agent"""
        print("\n--- 初始化 Agent 网络 ---\n")

        # 创建并连接 Agent
        research = ResearchAgent()
        analysis = AnalysisAgent()
        writer = WriterAgent()

        await self.client.connect(research)
        await self.client.connect(analysis)
        await self.client.connect(writer)

        # 显示可用工具
        tools = self.client.get_all_tools()
        print(f"\n  总计 {len(tools)} 个可用工具:")
        for tool in tools:
            print(f"    🔧 {tool['name']} ({tool['_server']})")

    async def execute_research_task(self, topic: str) -> str:
        """
        执行研究任务

        流程：
        1. 研究 Agent 搜索信息
        2. 研究 Agent 获取最新动态
        3. 分析 Agent 分析数据
        4. 分析 Agent 生成洞察
        5. 写作 Agent 撰写报告
        """
        print(f"\n--- 执行研究任务: {topic} ---\n")
        start_time = time.time()

        # 步骤 1：搜索信息
        print("  步骤 1: 搜索相关信息...")
        search_result = await self.client.call_tool(
            "search_topic", {"topic": topic, "max_results": 5}
        )
        self._log("search_topic", True)
        print(f"    ✅ 搜索完成")

        # 步骤 2：获取最新动态
        print("  步骤 2: 获取最新动态...")
        news_result = await self.client.call_tool(
            "get_latest_news", {"domain": topic}
        )
        self._log("get_latest_news", True)
        print(f"    ✅ 动态获取完成")

        # 步骤 3：总结信息源
        print("  步骤 3: 总结信息源...")
        summary_result = await self.client.call_tool(
            "summarize_sources", {"sources": search_result}
        )
        self._log("summarize_sources", True)
        print(f"    ✅ 总结完成")

        # 步骤 4：分析数据
        print("  步骤 4: 分析数据...")
        analysis_result = await self.client.call_tool(
            "analyze_data", {"data": search_result, "analysis_type": "trend"}
        )
        self._log("analyze_data", True)
        print(f"    ✅ 分析完成")

        # 步骤 5：生成洞察
        print("  步骤 5: 生成洞察...")
        insights_result = await self.client.call_tool(
            "generate_insights", {"analysis": analysis_result, "context": topic}
        )
        self._log("generate_insights", True)
        print(f"    ✅ 洞察生成完成")

        # 步骤 6：撰写报告
        print("  步骤 6: 撰写报告...")
        report = await self.client.call_tool(
            "write_report", {
                "title": f"{topic} 研究报告",
                "content": insights_result,
                "format": "markdown",
            }
        )
        self._log("write_report", True)
        print(f"    ✅ 报告撰写完成")

        duration = time.time() - start_time
        print(f"\n  总耗时: {duration:.2f}s")
        print(f"  执行步骤: {len(self._execution_log)}")

        return report

    def _log(self, tool_name: str, success: bool):
        """记录执行日志"""
        self._execution_log.append({
            "tool": tool_name,
            "success": success,
            "timestamp": datetime.now().isoformat(),
        })

    def get_execution_summary(self) -> dict:
        """获取执行摘要"""
        total = len(self._execution_log)
        success = sum(1 for log in self._execution_log if log["success"])
        return {
            "total_steps": total,
            "success_count": success,
            "success_rate": f"{success / total * 100:.0f}%" if total > 0 else "N/A",
        }


# ============================================================
# 4. 运行演示
# ============================================================

async def main():
    """运行 MCP Multi-Agent 系统演示"""
    print("=" * 60)
    print("MCP Multi-Agent 系统 — 里程碑项目")
    print("=" * 60)

    # 创建编排器
    orchestrator = Orchestrator()

    # 初始化 Agent 网络
    await orchestrator.setup()

    # 执行研究任务
    report = await orchestrator.execute_research_task("AI")

    # 显示报告
    print("\n" + "=" * 60)
    print("生成的研究报告")
    print("=" * 60)
    print(report)

    # 执行摘要
    print("\n" + "=" * 60)
    print("执行摘要")
    print("=" * 60)
    summary = orchestrator.get_execution_summary()
    print(f"  总步骤: {summary['total_steps']}")
    print(f"  成功: {summary['success_count']}")
    print(f"  成功率: {summary['success_rate']}")

    print("\n" + "=" * 60)
    print("演示完成！")


if __name__ == "__main__":
    asyncio.run(main())
