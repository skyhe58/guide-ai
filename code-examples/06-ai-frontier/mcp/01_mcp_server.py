"""
MCP Server 实现模拟

知识点：MCP 协议规范、JSON-RPC 2.0 消息处理、工具注册与调用、
       资源暴露、Prompt 模板、能力协商、传输层模拟

Python 版本：3.11+
依赖：标准库（模拟模式）、mcp>=1.0（生产模式）
最后验证：2024-12-01

说明：本文件模拟 MCP Server 的核心功能，无需安装 mcp 包即可运行。
     生产环境请使用官方 mcp Python SDK。
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional


# ============================================================
# 1. MCP 协议常量和数据结构
# ============================================================

# MCP 协议版本
MCP_PROTOCOL_VERSION = "2024-11-05"

# JSON-RPC 版本
JSONRPC_VERSION = "2.0"


class MCPErrorCode(Enum):
    """MCP 错误码定义"""
    PARSE_ERROR = -32700       # JSON 解析错误
    INVALID_REQUEST = -32600   # 无效请求
    METHOD_NOT_FOUND = -32601  # 方法不存在
    INVALID_PARAMS = -32602    # 无效参数
    INTERNAL_ERROR = -32603    # 内部错误


@dataclass
class ToolDefinition:
    """工具定义 — 描述一个可被 LLM 调用的工具"""
    name: str                  # 工具名称（snake_case）
    description: str           # 工具描述（LLM 依赖此描述决定何时调用）
    input_schema: dict         # 输入参数的 JSON Schema
    handler: Callable          # 工具处理函数


@dataclass
class ResourceDefinition:
    """资源定义 — 描述一个可被读取的数据源"""
    uri_template: str          # 资源 URI 模板（如 "config://{name}"）
    name: str                  # 资源名称
    description: str           # 资源描述
    mime_type: str = "text/plain"  # MIME 类型
    handler: Callable = None   # 资源读取处理函数


@dataclass
class PromptDefinition:
    """Prompt 模板定义 — 预定义的交互模板"""
    name: str                  # 模板名称
    description: str           # 模板描述
    arguments: list = field(default_factory=list)  # 模板参数
    handler: Callable = None   # 模板生成函数


@dataclass
class ServerCapabilities:
    """服务端能力声明"""
    tools: bool = True         # 是否支持工具
    resources: bool = True     # 是否支持资源
    prompts: bool = True       # 是否支持 Prompt 模板
    tools_list_changed: bool = True  # 工具列表变更通知
    resources_subscribe: bool = False  # 资源订阅


@dataclass
class MCPMessage:
    """MCP 消息封装"""
    jsonrpc: str = JSONRPC_VERSION
    id: Optional[int] = None
    method: Optional[str] = None
    params: Optional[dict] = None
    result: Optional[Any] = None
    error: Optional[dict] = None


# ============================================================
# 2. JSON-RPC 消息处理器
# ============================================================

class JSONRPCHandler:
    """JSON-RPC 2.0 消息处理器"""

    def __init__(self):
        # 方法路由表：method -> handler
        self._routes: dict[str, Callable] = {}

    def register_method(self, method: str, handler: Callable):
        """注册 JSON-RPC 方法处理器"""
        self._routes[method] = handler
        logging.debug(f"注册方法: {method}")

    async def handle_message(self, raw_message: str) -> Optional[str]:
        """处理原始 JSON-RPC 消息"""
        try:
            # 解析 JSON
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            return self._error_response(
                None, MCPErrorCode.PARSE_ERROR, "JSON 解析错误"
            )

        # 验证 JSON-RPC 版本
        if message.get("jsonrpc") != JSONRPC_VERSION:
            return self._error_response(
                message.get("id"), MCPErrorCode.INVALID_REQUEST,
                "不支持的 JSON-RPC 版本"
            )

        method = message.get("method")
        msg_id = message.get("id")
        params = message.get("params", {})

        # 通知消息（无 id）不需要响应
        is_notification = msg_id is None

        # 查找方法处理器
        handler = self._routes.get(method)
        if not handler:
            if is_notification:
                return None
            return self._error_response(
                msg_id, MCPErrorCode.METHOD_NOT_FOUND,
                f"方法不存在: {method}"
            )

        try:
            # 执行处理器
            result = await handler(params)
            if is_notification:
                return None
            return self._success_response(msg_id, result)
        except Exception as e:
            if is_notification:
                return None
            return self._error_response(
                msg_id, MCPErrorCode.INTERNAL_ERROR, str(e)
            )

    def _success_response(self, msg_id: int, result: Any) -> str:
        """构建成功响应"""
        response = {
            "jsonrpc": JSONRPC_VERSION,
            "id": msg_id,
            "result": result,
        }
        return json.dumps(response, ensure_ascii=False)

    def _error_response(self, msg_id: Optional[int],
                        code: MCPErrorCode, message: str) -> str:
        """构建错误响应"""
        response = {
            "jsonrpc": JSONRPC_VERSION,
            "id": msg_id,
            "error": {
                "code": code.value,
                "message": message,
            },
        }
        return json.dumps(response, ensure_ascii=False)


# ============================================================
# 3. MCP Server 核心实现
# ============================================================

class MCPServer:
    """
    MCP Server 模拟实现

    模拟 MCP 协议的核心功能：
    - 能力协商（initialize/initialized）
    - 工具注册与调用（tools/list, tools/call）
    - 资源暴露与读取（resources/list, resources/read）
    - Prompt 模板管理（prompts/list, prompts/get）
    """

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.capabilities = ServerCapabilities()

        # 注册表
        self._tools: dict[str, ToolDefinition] = {}
        self._resources: dict[str, ResourceDefinition] = {}
        self._prompts: dict[str, PromptDefinition] = {}

        # JSON-RPC 处理器
        self._rpc = JSONRPCHandler()
        self._setup_routes()

        # 会话状态
        self._initialized = False
        self._client_info: Optional[dict] = None

        # 日志配置（输出到 stderr，避免干扰 stdio 传输）
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stderr,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
        self._logger = logging.getLogger(f"mcp-server-{name}")

    def _setup_routes(self):
        """注册 MCP 协议方法路由"""
        self._rpc.register_method("initialize", self._handle_initialize)
        self._rpc.register_method("initialized", self._handle_initialized)
        self._rpc.register_method("tools/list", self._handle_tools_list)
        self._rpc.register_method("tools/call", self._handle_tools_call)
        self._rpc.register_method("resources/list", self._handle_resources_list)
        self._rpc.register_method("resources/read", self._handle_resources_read)
        self._rpc.register_method("prompts/list", self._handle_prompts_list)
        self._rpc.register_method("prompts/get", self._handle_prompts_get)
        self._rpc.register_method("ping", self._handle_ping)

    # ---- 装饰器 API ----

    def tool(self, name: str = None, description: str = None,
             schema: dict = None):
        """工具注册装饰器"""
        def decorator(func):
            tool_name = name or func.__name__
            tool_desc = description or func.__doc__ or "无描述"
            tool_schema = schema or self._infer_schema(func)
            self._tools[tool_name] = ToolDefinition(
                name=tool_name,
                description=tool_desc.strip(),
                input_schema=tool_schema,
                handler=func,
            )
            self._logger.info(f"注册工具: {tool_name}")
            return func
        return decorator

    def resource(self, uri_template: str, name: str = None,
                 description: str = None, mime_type: str = "text/plain"):
        """资源注册装饰器"""
        def decorator(func):
            res_name = name or func.__name__
            res_desc = description or func.__doc__ or "无描述"
            self._resources[uri_template] = ResourceDefinition(
                uri_template=uri_template,
                name=res_name,
                description=res_desc.strip(),
                mime_type=mime_type,
                handler=func,
            )
            self._logger.info(f"注册资源: {uri_template}")
            return func
        return decorator

    def prompt(self, name: str = None, description: str = None,
               arguments: list = None):
        """Prompt 模板注册装饰器"""
        def decorator(func):
            prompt_name = name or func.__name__
            prompt_desc = description or func.__doc__ or "无描述"
            self._prompts[prompt_name] = PromptDefinition(
                name=prompt_name,
                description=prompt_desc.strip(),
                arguments=arguments or [],
                handler=func,
            )
            self._logger.info(f"注册 Prompt 模板: {prompt_name}")
            return func
        return decorator

    # ---- 协议处理方法 ----

    async def _handle_initialize(self, params: dict) -> dict:
        """处理 initialize 请求 — 能力协商"""
        client_version = params.get("protocolVersion", "unknown")
        self._client_info = params.get("clientInfo", {})

        self._logger.info(
            f"客户端连接: {self._client_info.get('name', 'unknown')} "
            f"(协议版本: {client_version})"
        )

        # 返回服务端能力声明
        return {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {
                "tools": {"listChanged": self.capabilities.tools_list_changed}
                if self.capabilities.tools else None,
                "resources": {"subscribe": self.capabilities.resources_subscribe}
                if self.capabilities.resources else None,
                "prompts": {}
                if self.capabilities.prompts else None,
            },
            "serverInfo": {
                "name": self.name,
                "version": self.version,
            },
        }

    async def _handle_initialized(self, params: dict) -> None:
        """处理 initialized 通知 — 确认连接建立"""
        self._initialized = True
        self._logger.info("连接已建立")

    async def _handle_tools_list(self, params: dict) -> dict:
        """处理 tools/list 请求 — 返回工具列表"""
        tools = []
        for tool in self._tools.values():
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
            })
        return {"tools": tools}

    async def _handle_tools_call(self, params: dict) -> dict:
        """处理 tools/call 请求 — 调用工具"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        tool = self._tools.get(tool_name)
        if not tool:
            raise ValueError(f"工具不存在: {tool_name}")

        self._logger.info(f"调用工具: {tool_name}, 参数: {arguments}")

        try:
            # 执行工具处理函数
            if asyncio.iscoroutinefunction(tool.handler):
                result = await tool.handler(**arguments)
            else:
                result = tool.handler(**arguments)

            return {
                "content": [
                    {"type": "text", "text": str(result)}
                ],
            }
        except Exception as e:
            self._logger.error(f"工具执行失败: {tool_name}, 错误: {e}")
            return {
                "content": [
                    {"type": "text", "text": f"错误: {str(e)}"}
                ],
                "isError": True,
            }

    async def _handle_resources_list(self, params: dict) -> dict:
        """处理 resources/list 请求 — 返回资源列表"""
        resources = []
        for res in self._resources.values():
            resources.append({
                "uri": res.uri_template,
                "name": res.name,
                "description": res.description,
                "mimeType": res.mime_type,
            })
        return {"resources": resources}

    async def _handle_resources_read(self, params: dict) -> dict:
        """处理 resources/read 请求 — 读取资源"""
        uri = params.get("uri", "")

        # 简单的 URI 匹配（生产环境应使用正则匹配）
        for template, res in self._resources.items():
            if self._match_uri(template, uri):
                uri_params = self._extract_uri_params(template, uri)
                if asyncio.iscoroutinefunction(res.handler):
                    content = await res.handler(**uri_params)
                else:
                    content = res.handler(**uri_params)
                return {
                    "contents": [{
                        "uri": uri,
                        "mimeType": res.mime_type,
                        "text": str(content),
                    }],
                }

        raise ValueError(f"资源不存在: {uri}")

    async def _handle_prompts_list(self, params: dict) -> dict:
        """处理 prompts/list 请求 — 返回 Prompt 模板列表"""
        prompts = []
        for p in self._prompts.values():
            prompts.append({
                "name": p.name,
                "description": p.description,
                "arguments": p.arguments,
            })
        return {"prompts": prompts}

    async def _handle_prompts_get(self, params: dict) -> dict:
        """处理 prompts/get 请求 — 获取 Prompt 模板"""
        prompt_name = params.get("name")
        arguments = params.get("arguments", {})

        prompt = self._prompts.get(prompt_name)
        if not prompt:
            raise ValueError(f"Prompt 模板不存在: {prompt_name}")

        if asyncio.iscoroutinefunction(prompt.handler):
            content = await prompt.handler(**arguments)
        else:
            content = prompt.handler(**arguments)

        return {
            "description": prompt.description,
            "messages": [
                {"role": "user", "content": {"type": "text", "text": content}}
            ],
        }

    async def _handle_ping(self, params: dict) -> dict:
        """处理 ping 请求 — 健康检查"""
        return {}

    # ---- 辅助方法 ----

    def _infer_schema(self, func: Callable) -> dict:
        """从函数签名推断 JSON Schema"""
        import inspect
        sig = inspect.signature(func)
        properties = {}
        required = []

        type_map = {
            str: "string", int: "integer", float: "number",
            bool: "boolean", list: "array", dict: "object",
        }

        for name, param in sig.parameters.items():
            if name == "self":
                continue
            annotation = param.annotation
            prop_type = type_map.get(annotation, "string")
            properties[name] = {"type": prop_type}
            if param.default is inspect.Parameter.empty:
                required.append(name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def _match_uri(self, template: str, uri: str) -> bool:
        """简单的 URI 模板匹配"""
        import re
        pattern = re.sub(r"\{(\w+)\}", r"(?P<\1>[^/]+)", template)
        return bool(re.fullmatch(pattern, uri))

    def _extract_uri_params(self, template: str, uri: str) -> dict:
        """从 URI 中提取参数"""
        import re
        pattern = re.sub(r"\{(\w+)\}", r"(?P<\1>[^/]+)", template)
        match = re.fullmatch(pattern, uri)
        return match.groupdict() if match else {}

    # ---- 运行方法 ----

    async def process_message(self, raw_message: str) -> Optional[str]:
        """处理单条消息（用于测试和模拟）"""
        return await self._rpc.handle_message(raw_message)

    def get_stats(self) -> dict:
        """获取 Server 统计信息"""
        return {
            "name": self.name,
            "version": self.version,
            "initialized": self._initialized,
            "tools_count": len(self._tools),
            "resources_count": len(self._resources),
            "prompts_count": len(self._prompts),
            "client_info": self._client_info,
        }


# ============================================================
# 4. 示例：创建一个数据库查询 MCP Server
# ============================================================

def create_demo_server() -> MCPServer:
    """创建演示用的 MCP Server"""
    server = MCPServer("demo-database-server", "0.1.0")

    # 模拟数据库
    mock_db = {
        "users": [
            {"id": 1, "name": "张三", "email": "zhangsan@example.com", "role": "admin"},
            {"id": 2, "name": "李四", "email": "lisi@example.com", "role": "user"},
            {"id": 3, "name": "王五", "email": "wangwu@example.com", "role": "user"},
        ],
        "products": [
            {"id": 1, "name": "AI 入门课程", "price": 99.0, "stock": 100},
            {"id": 2, "name": "Python 实战", "price": 149.0, "stock": 50},
        ],
    }

    # 注册工具：查询数据库
    @server.tool(
        name="query_database",
        description="查询模拟数据库中的数据。支持查询 users 和 products 表。",
        schema={
            "type": "object",
            "properties": {
                "table": {
                    "type": "string",
                    "description": "要查询的表名（users 或 products）",
                    "enum": ["users", "products"],
                },
                "filter_field": {
                    "type": "string",
                    "description": "过滤字段名（可选）",
                },
                "filter_value": {
                    "type": "string",
                    "description": "过滤字段值（可选）",
                },
            },
            "required": ["table"],
        },
    )
    def query_database(table: str, filter_field: str = None,
                       filter_value: str = None) -> str:
        """查询数据库"""
        data = mock_db.get(table, [])
        if filter_field and filter_value:
            data = [row for row in data
                    if str(row.get(filter_field, "")) == filter_value]
        return json.dumps(data, ensure_ascii=False, indent=2)

    # 注册工具：计算统计
    @server.tool(
        name="calculate_stats",
        description="对数据进行简单的统计计算（求和、平均值、计数）。",
        schema={
            "type": "object",
            "properties": {
                "numbers": {
                    "type": "string",
                    "description": "逗号分隔的数字列表",
                },
                "operation": {
                    "type": "string",
                    "description": "统计操作",
                    "enum": ["sum", "avg", "count", "max", "min"],
                },
            },
            "required": ["numbers", "operation"],
        },
    )
    def calculate_stats(numbers: str, operation: str) -> str:
        """计算统计值"""
        nums = [float(n.strip()) for n in numbers.split(",")]
        ops = {
            "sum": sum(nums),
            "avg": sum(nums) / len(nums),
            "count": len(nums),
            "max": max(nums),
            "min": min(nums),
        }
        result = ops.get(operation, "不支持的操作")
        return f"{operation}({numbers}) = {result}"

    # 注册资源：应用配置
    @server.resource(
        uri_template="config://{name}",
        name="app_config",
        description="获取应用配置信息",
        mime_type="application/json",
    )
    def get_config(name: str) -> str:
        """获取配置"""
        configs = {
            "database": json.dumps({
                "host": "localhost", "port": 5432,
                "database": "demo", "pool_size": 10,
            }),
            "redis": json.dumps({
                "host": "localhost", "port": 6379,
                "db": 0, "max_connections": 20,
            }),
            "app": json.dumps({
                "name": "AI Assistant", "version": "1.0.0",
                "debug": False, "log_level": "INFO",
            }),
        }
        return configs.get(name, json.dumps({"error": f"配置 {name} 不存在"}))

    # 注册 Prompt 模板：代码审查
    @server.prompt(
        name="code_review",
        description="代码审查模板 — 对代码进行全面审查",
        arguments=[
            {"name": "code", "description": "要审查的代码", "required": True},
            {"name": "language", "description": "编程语言", "required": False},
        ],
    )
    def code_review_prompt(code: str, language: str = "python") -> str:
        """生成代码审查 Prompt"""
        return f"""请对以下 {language} 代码进行全面审查，关注以下方面：

1. **代码质量**：可读性、命名规范、代码结构
2. **潜在 Bug**：逻辑错误、边界条件、异常处理
3. **安全问题**：输入验证、SQL 注入、XSS 等
4. **性能优化**：算法效率、资源使用、缓存策略
5. **最佳实践**：设计模式、SOLID 原则、测试覆盖

代码：
```{language}
{code}
```

请按严重程度（高/中/低）分类列出发现的问题，并给出改进建议。"""

    return server


# ============================================================
# 5. 运行演示
# ============================================================

async def demo():
    """运行 MCP Server 演示"""
    print("=" * 60)
    print("MCP Server 模拟演示")
    print("=" * 60)

    # 创建 Server
    server = create_demo_server()

    # 1. 模拟 initialize 请求
    print("\n--- 1. 能力协商 (initialize) ---")
    init_request = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {"roots": {"listChanged": True}},
            "clientInfo": {"name": "demo-client", "version": "1.0.0"},
        },
    })
    response = await server.process_message(init_request)
    print(f"响应: {json.dumps(json.loads(response), indent=2, ensure_ascii=False)}")

    # 2. 模拟 initialized 通知
    init_notification = json.dumps({
        "jsonrpc": "2.0",
        "method": "initialized",
        "params": {},
    })
    await server.process_message(init_notification)
    print("\n--- 2. 连接确认 (initialized) ---")
    print("连接已建立")

    # 3. 获取工具列表
    print("\n--- 3. 获取工具列表 (tools/list) ---")
    tools_request = json.dumps({
        "jsonrpc": "2.0", "id": 2,
        "method": "tools/list", "params": {},
    })
    response = await server.process_message(tools_request)
    result = json.loads(response)
    for tool in result["result"]["tools"]:
        print(f"  工具: {tool['name']} — {tool['description']}")

    # 4. 调用工具：查询数据库
    print("\n--- 4. 调用工具 (tools/call) ---")
    call_request = json.dumps({
        "jsonrpc": "2.0", "id": 3,
        "method": "tools/call",
        "params": {
            "name": "query_database",
            "arguments": {"table": "users"},
        },
    })
    response = await server.process_message(call_request)
    result = json.loads(response)
    print(f"查询结果:\n{result['result']['content'][0]['text']}")

    # 5. 调用工具：统计计算
    print("\n--- 5. 统计计算 ---")
    calc_request = json.dumps({
        "jsonrpc": "2.0", "id": 4,
        "method": "tools/call",
        "params": {
            "name": "calculate_stats",
            "arguments": {"numbers": "10,20,30,40,50", "operation": "avg"},
        },
    })
    response = await server.process_message(calc_request)
    result = json.loads(response)
    print(f"计算结果: {result['result']['content'][0]['text']}")

    # 6. 读取资源
    print("\n--- 6. 读取资源 (resources/read) ---")
    resource_request = json.dumps({
        "jsonrpc": "2.0", "id": 5,
        "method": "resources/read",
        "params": {"uri": "config://database"},
    })
    response = await server.process_message(resource_request)
    result = json.loads(response)
    print(f"配置内容: {result['result']['contents'][0]['text']}")

    # 7. 获取 Prompt 模板
    print("\n--- 7. 获取 Prompt 模板 (prompts/get) ---")
    prompt_request = json.dumps({
        "jsonrpc": "2.0", "id": 6,
        "method": "prompts/get",
        "params": {
            "name": "code_review",
            "arguments": {
                "code": "def add(a, b): return a + b",
                "language": "python",
            },
        },
    })
    response = await server.process_message(prompt_request)
    result = json.loads(response)
    prompt_text = result["result"]["messages"][0]["content"]["text"]
    print(f"Prompt 模板:\n{prompt_text[:200]}...")

    # 8. 服务器统计
    print("\n--- 8. 服务器统计 ---")
    stats = server.get_stats()
    print(f"服务器名称: {stats['name']}")
    print(f"工具数量: {stats['tools_count']}")
    print(f"资源数量: {stats['resources_count']}")
    print(f"Prompt 模板数量: {stats['prompts_count']}")
    print(f"已初始化: {stats['initialized']}")

    print("\n" + "=" * 60)
    print("演示完成！")


if __name__ == "__main__":
    asyncio.run(demo())
