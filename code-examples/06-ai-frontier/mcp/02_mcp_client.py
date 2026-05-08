"""
MCP 客户端集成模拟

知识点：MCP Client 实现、多 Server 连接管理、工具发现与调用、
       资源读取、会话管理、错误处理、多 Agent 协作

Python 版本：3.11+
依赖：标准库（模拟模式）、mcp>=1.0（生产模式）
最后验证：2024-12-01

说明：本文件模拟 MCP Client 的核心功能，演示如何连接多个 MCP Server
     并统一管理工具调用。生产环境请使用官方 mcp Python SDK。
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# ============================================================
# 1. MCP 客户端数据结构
# ============================================================

MCP_PROTOCOL_VERSION = "2024-11-05"


class ConnectionState(Enum):
    """连接状态"""
    DISCONNECTED = "disconnected"   # 未连接
    CONNECTING = "connecting"       # 连接中
    CONNECTED = "connected"         # 已连接
    ERROR = "error"                 # 错误


@dataclass
class ServerConnection:
    """MCP Server 连接信息"""
    name: str                       # Server 名称
    state: ConnectionState = ConnectionState.DISCONNECTED
    server_info: dict | None = None   # Server 信息
    capabilities: dict | None = None  # Server 能力
    tools: list = field(default_factory=list)       # 可用工具列表
    resources: list = field(default_factory=list)    # 可用资源列表
    prompts: list = field(default_factory=list)      # 可用 Prompt 模板
    connected_at: datetime | None = None
    last_activity: datetime | None = None


@dataclass
class ToolCallResult:
    """工具调用结果"""
    server_name: str        # 来源 Server
    tool_name: str          # 工具名称
    success: bool           # 是否成功
    content: str            # 返回内容
    duration_ms: float      # 耗时（毫秒）
    error: str | None = None  # 错误信息


@dataclass
class MCPClientConfig:
    """客户端配置"""
    client_name: str = "mcp-client"
    client_version: str = "1.0.0"
    timeout_seconds: int = 30       # 请求超时
    max_retries: int = 3            # 最大重试次数
    retry_delay: float = 1.0        # 重试延迟（秒）


# ============================================================
# 2. 模拟传输层
# ============================================================

class MockTransport:
    """
    模拟传输层 — 直接调用 Server 的消息处理方法

    在生产环境中，传输层负责通过 stdio 或 SSE 与 Server 通信。
    这里为了演示，直接引用 Server 对象进行消息传递。
    """

    def __init__(self, server):
        self.server = server
        self._msg_id = 0

    async def send_request(self, method: str, params: dict = None) -> dict:
        """发送请求并等待响应"""
        self._msg_id += 1
        request = json.dumps({
            "jsonrpc": "2.0",
            "id": self._msg_id,
            "method": method,
            "params": params or {},
        })

        response_str = await self.server.process_message(request)
        if response_str:
            response = json.loads(response_str)
            if "error" in response:
                raise MCPClientError(
                    f"Server 错误: {response['error']['message']}"
                )
            return response.get("result", {})
        return {}

    async def send_notification(self, method: str, params: dict = None):
        """发送通知（无需响应）"""
        notification = json.dumps({
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
        })
        await self.server.process_message(notification)


class MCPClientError(Exception):
    """MCP 客户端错误"""
    pass


# ============================================================
# 3. MCP Client 核心实现
# ============================================================

class MCPClient:
    """
    MCP 客户端实现

    核心功能：
    - 连接管理：连接/断开多个 MCP Server
    - 工具发现：获取所有 Server 的工具列表
    - 工具调用：调用指定 Server 的工具
    - 资源读取：读取 Server 暴露的资源
    - 统一接口：提供跨 Server 的统一工具调用接口
    """

    def __init__(self, config: MCPClientConfig = None):
        self.config = config or MCPClientConfig()
        self._connections: dict[str, ServerConnection] = {}
        self._transports: dict[str, MockTransport] = {}

        # 工具路由表：tool_name -> server_name
        self._tool_router: dict[str, str] = {}

        # 日志
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stderr,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
        self._logger = logging.getLogger("mcp-client")

    # ---- 连接管理 ----

    async def connect(self, server_name: str, server) -> ServerConnection:
        """
        连接到 MCP Server

        执行流程：
        1. 创建传输层连接
        2. 发送 initialize 请求（能力协商）
        3. 发送 initialized 通知（确认连接）
        4. 获取工具/资源/Prompt 列表
        """
        self._logger.info(f"正在连接 Server: {server_name}")

        # 创建连接记录
        conn = ServerConnection(name=server_name, state=ConnectionState.CONNECTING)
        self._connections[server_name] = conn

        # 创建传输层
        transport = MockTransport(server)
        self._transports[server_name] = transport

        try:
            # 步骤 1：能力协商
            init_result = await transport.send_request("initialize", {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {
                    "roots": {"listChanged": True},
                    "sampling": {},
                },
                "clientInfo": {
                    "name": self.config.client_name,
                    "version": self.config.client_version,
                },
            })

            conn.server_info = init_result.get("serverInfo", {})
            conn.capabilities = init_result.get("capabilities", {})

            # 步骤 2：确认连接
            await transport.send_notification("initialized")

            # 步骤 3：获取工具列表
            if conn.capabilities.get("tools"):
                tools_result = await transport.send_request("tools/list")
                conn.tools = tools_result.get("tools", [])
                # 更新工具路由表
                for tool in conn.tools:
                    self._tool_router[tool["name"]] = server_name

            # 步骤 4：获取资源列表
            if conn.capabilities.get("resources"):
                resources_result = await transport.send_request("resources/list")
                conn.resources = resources_result.get("resources", [])

            # 步骤 5：获取 Prompt 列表
            if conn.capabilities.get("prompts"):
                prompts_result = await transport.send_request("prompts/list")
                conn.prompts = prompts_result.get("prompts", [])

            conn.state = ConnectionState.CONNECTED
            conn.connected_at = datetime.now()
            conn.last_activity = datetime.now()

            self._logger.info(
                f"已连接 Server: {server_name} "
                f"(工具: {len(conn.tools)}, "
                f"资源: {len(conn.resources)}, "
                f"Prompt: {len(conn.prompts)})"
            )
            return conn

        except Exception as e:
            conn.state = ConnectionState.ERROR
            self._logger.error(f"连接失败: {server_name}, 错误: {e}")
            raise MCPClientError(f"连接 {server_name} 失败: {e}")

    async def disconnect(self, server_name: str):
        """断开与 Server 的连接"""
        conn = self._connections.get(server_name)
        if conn:
            # 移除工具路由
            for tool in conn.tools:
                self._tool_router.pop(tool["name"], None)
            conn.state = ConnectionState.DISCONNECTED
            self._transports.pop(server_name, None)
            self._logger.info(f"已断开 Server: {server_name}")

    # ---- 工具操作 ----

    def get_all_tools(self) -> list[dict]:
        """获取所有已连接 Server 的工具列表"""
        all_tools = []
        for conn in self._connections.values():
            if conn.state == ConnectionState.CONNECTED:
                for tool in conn.tools:
                    all_tools.append({
                        **tool,
                        "_server": conn.name,  # 标记来源 Server
                    })
        return all_tools

    def find_tool(self, tool_name: str) -> dict | None:
        """查找工具及其所在的 Server"""
        server_name = self._tool_router.get(tool_name)
        if not server_name:
            return None
        conn = self._connections.get(server_name)
        if not conn:
            return None
        for tool in conn.tools:
            if tool["name"] == tool_name:
                return {**tool, "_server": server_name}
        return None

    async def call_tool(self, tool_name: str,
                        arguments: dict = None,
                        server_name: str = None) -> ToolCallResult:
        """
        调用工具

        如果指定了 server_name，直接调用该 Server 的工具。
        否则，通过工具路由表自动查找 Server。
        """
        start_time = time.time()

        # 确定目标 Server
        if not server_name:
            server_name = self._tool_router.get(tool_name)
        if not server_name:
            return ToolCallResult(
                server_name="unknown", tool_name=tool_name,
                success=False, content="",
                duration_ms=0, error=f"工具 {tool_name} 未找到",
            )

        transport = self._transports.get(server_name)
        if not transport:
            return ToolCallResult(
                server_name=server_name, tool_name=tool_name,
                success=False, content="",
                duration_ms=0, error=f"Server {server_name} 未连接",
            )

        try:
            result = await transport.send_request("tools/call", {
                "name": tool_name,
                "arguments": arguments or {},
            })

            duration = (time.time() - start_time) * 1000
            content = result.get("content", [])
            text = content[0]["text"] if content else ""
            is_error = result.get("isError", False)

            # 更新最后活动时间
            conn = self._connections.get(server_name)
            if conn:
                conn.last_activity = datetime.now()

            return ToolCallResult(
                server_name=server_name, tool_name=tool_name,
                success=not is_error, content=text,
                duration_ms=duration,
                error=text if is_error else None,
            )

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return ToolCallResult(
                server_name=server_name, tool_name=tool_name,
                success=False, content="",
                duration_ms=duration, error=str(e),
            )

    # ---- 资源操作 ----

    async def read_resource(self, server_name: str, uri: str) -> str:
        """读取 Server 的资源"""
        transport = self._transports.get(server_name)
        if not transport:
            raise MCPClientError(f"Server {server_name} 未连接")

        result = await transport.send_request("resources/read", {"uri": uri})
        contents = result.get("contents", [])
        return contents[0]["text"] if contents else ""

    # ---- Prompt 操作 ----

    async def get_prompt(self, server_name: str, prompt_name: str,
                         arguments: dict = None) -> str:
        """获取 Prompt 模板"""
        transport = self._transports.get(server_name)
        if not transport:
            raise MCPClientError(f"Server {server_name} 未连接")

        result = await transport.send_request("prompts/get", {
            "name": prompt_name,
            "arguments": arguments or {},
        })
        messages = result.get("messages", [])
        if messages:
            return messages[0]["content"]["text"]
        return ""

    # ---- 状态查询 ----

    def get_status(self) -> dict:
        """获取客户端状态"""
        connections = {}
        for name, conn in self._connections.items():
            connections[name] = {
                "state": conn.state.value,
                "tools": len(conn.tools),
                "resources": len(conn.resources),
                "prompts": len(conn.prompts),
                "server_info": conn.server_info,
                "connected_at": str(conn.connected_at) if conn.connected_at else None,
            }
        return {
            "client": self.config.client_name,
            "total_connections": len(self._connections),
            "total_tools": len(self._tool_router),
            "connections": connections,
        }


# ============================================================
# 4. 多 Agent 协作示例
# ============================================================

class MultiAgentOrchestrator:
    """
    多 Agent 编排器

    通过 MCP Client 连接多个 Agent（MCP Server），
    实现任务分解和协调执行。
    """

    def __init__(self, client: MCPClient):
        self.client = client
        self._execution_log: list[dict] = []

    async def execute_plan(self, plan: list[dict]) -> list[ToolCallResult]:
        """
        执行任务计划

        plan 格式：
        [
            {"tool": "tool_name", "args": {...}, "description": "步骤描述"},
            ...
        ]
        """
        results = []
        for i, step in enumerate(plan, 1):
            print(f"\n  步骤 {i}: {step.get('description', step['tool'])}")

            result = await self.client.call_tool(
                step["tool"], step.get("args", {})
            )

            results.append(result)
            self._execution_log.append({
                "step": i,
                "tool": step["tool"],
                "success": result.success,
                "duration_ms": result.duration_ms,
            })

            if result.success:
                # 截断过长的输出
                display = result.content[:200]
                if len(result.content) > 200:
                    display += "..."
                print(f"  ✅ 成功 ({result.duration_ms:.1f}ms): {display}")
            else:
                print(f"  ❌ 失败: {result.error}")

        return results

    def get_execution_summary(self) -> dict:
        """获取执行摘要"""
        total = len(self._execution_log)
        success = sum(1 for log in self._execution_log if log["success"])
        total_time = sum(log["duration_ms"] for log in self._execution_log)
        return {
            "total_steps": total,
            "success_count": success,
            "failure_count": total - success,
            "total_duration_ms": total_time,
            "success_rate": f"{success / total * 100:.1f}%" if total > 0 else "N/A",
        }


# ============================================================
# 5. 运行演示
# ============================================================

async def demo():
    """运行 MCP 客户端演示"""
    # 导入 Server（从同目录的 01_mcp_server.py）
    # 这里直接创建 Server 实例
    import os

    print("=" * 60)
    print("MCP 客户端集成演示")
    print("=" * 60)

    # 尝试导入 Server 模块
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from importlib.util import module_from_spec, spec_from_file_location
        server_path = os.path.join(os.path.dirname(__file__), "01_mcp_server.py")
        spec = spec_from_file_location("mcp_server", server_path)
        mcp_server_module = module_from_spec(spec)
        spec.loader.exec_module(mcp_server_module)
        server = mcp_server_module.create_demo_server()
    except Exception:
        # 如果导入失败，创建一个简单的内联 Server
        print("注意：无法导入 01_mcp_server.py，使用内联 Server")
        # 使用简化版本
        return

    # 创建客户端
    client = MCPClient(MCPClientConfig(
        client_name="demo-orchestrator",
        client_version="1.0.0",
    ))

    # 1. 连接 Server
    print("\n--- 1. 连接 MCP Server ---")
    await client.connect("database-server", server)

    # 2. 查看所有可用工具
    print("\n--- 2. 可用工具列表 ---")
    tools = client.get_all_tools()
    for tool in tools:
        print(f"  🔧 {tool['name']} ({tool['_server']})")
        print(f"     {tool['description']}")

    # 3. 调用工具
    print("\n--- 3. 工具调用 ---")
    result = await client.call_tool("query_database", {"table": "users"})
    print(f"  查询用户表: {'✅ 成功' if result.success else '❌ 失败'}")
    print(f"  耗时: {result.duration_ms:.1f}ms")
    print(f"  结果: {result.content[:200]}")

    # 4. 读取资源
    print("\n--- 4. 读取资源 ---")
    config = await client.read_resource("database-server", "config://app")
    print(f"  应用配置: {config}")

    # 5. 获取 Prompt 模板
    print("\n--- 5. 获取 Prompt 模板 ---")
    prompt = await client.get_prompt("database-server", "code_review", {
        "code": "def hello(): print('world')",
        "language": "python",
    })
    print(f"  Prompt 模板（前 150 字符）: {prompt[:150]}...")

    # 6. 多 Agent 协作演示
    print("\n--- 6. 多 Agent 协作（任务编排） ---")
    orchestrator = MultiAgentOrchestrator(client)

    plan = [
        {
            "tool": "query_database",
            "args": {"table": "users"},
            "description": "查询所有用户",
        },
        {
            "tool": "query_database",
            "args": {"table": "products"},
            "description": "查询所有产品",
        },
        {
            "tool": "calculate_stats",
            "args": {"numbers": "99,149", "operation": "avg"},
            "description": "计算产品平均价格",
        },
        {
            "tool": "calculate_stats",
            "args": {"numbers": "100,50", "operation": "sum"},
            "description": "计算总库存",
        },
    ]

    print("  执行任务计划:")
    await orchestrator.execute_plan(plan)

    # 7. 执行摘要
    print("\n--- 7. 执行摘要 ---")
    summary = orchestrator.get_execution_summary()
    print(f"  总步骤: {summary['total_steps']}")
    print(f"  成功: {summary['success_count']}")
    print(f"  失败: {summary['failure_count']}")
    print(f"  总耗时: {summary['total_duration_ms']:.1f}ms")
    print(f"  成功率: {summary['success_rate']}")

    # 8. 客户端状态
    print("\n--- 8. 客户端状态 ---")
    status = client.get_status()
    print(f"  客户端: {status['client']}")
    print(f"  连接数: {status['total_connections']}")
    print(f"  总工具数: {status['total_tools']}")

    # 9. 断开连接
    await client.disconnect("database-server")
    print("\n--- 9. 已断开所有连接 ---")

    print("\n" + "=" * 60)
    print("演示完成！")


if __name__ == "__main__":
    asyncio.run(demo())
