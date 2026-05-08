"""
异步上下文管理器 — async with、资源管理、超时控制

知识点：__aenter__/__aexit__ 协议、自定义异步上下文管理器、
       contextlib.asynccontextmanager 装饰器、
       异步资源管理最佳实践、asyncio.timeout 超时控制

Python 版本：3.11+
依赖：asyncio（标准库）、contextlib（标准库）
最后验证：2024-12-01
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

# ============================================================
# 1. 异步上下文管理器基础 — __aenter__ / __aexit__
# ============================================================

class AsyncDatabasePool:
    """模拟异步数据库连接池。

    在 AI 应用中，数据库连接池是常见的异步资源：
    - 向量数据库连接（Chroma、Pinecone）
    - 关系型数据库连接（asyncpg、aiomysql）
    - 缓存连接（aioredis）

    通过 async with 语法确保连接正确获取和释放，
    即使发生异常也不会泄漏连接。
    """

    def __init__(self, dsn: str, pool_size: int = 5):
        self.dsn = dsn
        self.pool_size = pool_size
        self._connections: list[str] = []
        self._initialized = False

    async def __aenter__(self) -> AsyncDatabasePool:
        """异步初始化连接池。

        __aenter__ 在 async with 块开始时调用，
        返回值绑定到 as 后面的变量。
        """
        print(f"  🔌 正在初始化连接池: {self.dsn} (大小: {self.pool_size})")
        # 模拟异步建立多个数据库连接
        await asyncio.sleep(0.2)
        self._connections = [
            f"conn_{i}@{self.dsn}" for i in range(self.pool_size)
        ]
        self._initialized = True
        print(f"  ✅ 连接池就绪，{len(self._connections)} 个连接可用")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        """异步释放连接池资源。

        __aexit__ 在 async with 块结束时调用（无论是否发生异常）。
        参数包含异常信息，返回 True 表示抑制异常，False 表示继续传播。
        """
        if exc_type is not None:
            print(f"  ⚠️ 检测到异常: {exc_type.__name__}: {exc_val}")

        print(f"  🔒 正在关闭连接池: 释放 {len(self._connections)} 个连接...")
        await asyncio.sleep(0.1)  # 模拟异步关闭连接
        self._connections.clear()
        self._initialized = False
        print("  ✅ 连接池已关闭")

        # 返回 False — 不抑制异常，让调用方处理
        return False

    async def execute(self, query: str) -> dict:
        """模拟执行异步数据库查询。"""
        if not self._initialized:
            raise RuntimeError("连接池未初始化，请使用 async with 语句")
        conn = self._connections[0]  # 简化：总是使用第一个连接
        print(f"  📝 [{conn}] 执行查询: {query}")
        await asyncio.sleep(0.1)
        return {"query": query, "rows": 42, "connection": conn}


async def demo_async_context_basics() -> None:
    """演示基本的异步上下文管理器用法。"""
    print("\n" + "=" * 60)
    print("1. 异步上下文管理器基础 — __aenter__ / __aexit__")
    print("=" * 60)

    # async with 确保资源正确获取和释放
    async with AsyncDatabasePool("postgresql://localhost/ai_app") as pool:
        result = await pool.execute("SELECT * FROM embeddings LIMIT 10")
        print(f"  查询结果: {result}")

        result2 = await pool.execute("SELECT COUNT(*) FROM documents")
        print(f"  查询结果: {result2}")

    print("  （连接池已自动关闭，即使中间发生异常也会正确释放）")


# ============================================================
# 2. 异常安全性演示
# ============================================================

async def demo_exception_safety() -> None:
    """演示异步上下文管理器的异常安全性。

    关键点：即使 async with 块内抛出异常，
    __aexit__ 仍然会被调用，确保资源释放。
    """
    print("\n" + "=" * 60)
    print("2. 异常安全性 — __aexit__ 始终执行")
    print("=" * 60)

    try:
        async with AsyncDatabasePool("postgresql://localhost/ai_app") as pool:
            await pool.execute("SELECT * FROM models")
            # 模拟业务逻辑异常
            raise ValueError("模型版本不兼容")
    except ValueError as e:
        print(f"  🔄 异常已被外层捕获: {e}")
        print("  💡 注意：连接池在异常发生后仍然被正确关闭了")


# ============================================================
# 3. contextlib.asynccontextmanager — 装饰器方式
# ============================================================

@contextlib.asynccontextmanager
async def async_llm_session(
    model: str = "qwen2",
    base_url: str = "http://localhost:11434",
):
    """使用装饰器创建异步上下文管理器。

    相比类方式，装饰器方式更简洁，适合简单的资源管理场景。
    yield 之前的代码相当于 __aenter__，yield 之后相当于 __aexit__。

    实际场景：管理 LLM 推理会话的生命周期。
    """
    # --- 进入上下文（相当于 __aenter__）---
    session_id = f"session_{id(model) % 10000}"
    print(f"  🤖 创建 LLM 会话: model={model}, url={base_url}")
    print(f"  📋 会话 ID: {session_id}")
    await asyncio.sleep(0.1)  # 模拟连接建立

    session_info = {
        "session_id": session_id,
        "model": model,
        "base_url": base_url,
        "created_at": time.time(),
    }

    try:
        # yield 的值会绑定到 as 后面的变量
        yield session_info
    finally:
        # --- 退出上下文（相当于 __aexit__）---
        # finally 确保无论是否发生异常都会执行清理
        duration = time.time() - session_info["created_at"]
        print(f"  🔒 关闭 LLM 会话: {session_id} (持续 {duration:.2f}s)")
        await asyncio.sleep(0.05)  # 模拟连接关闭


@contextlib.asynccontextmanager
async def async_vector_store(collection: str = "default"):
    """模拟异步向量数据库连接管理。"""
    print(f"  📦 连接向量数据库，集合: {collection}")
    await asyncio.sleep(0.1)

    store = {
        "collection": collection,
        "status": "connected",
        "vectors_count": 10000,
    }

    try:
        yield store
    finally:
        print(f"  📦 断开向量数据库连接: {collection}")
        await asyncio.sleep(0.05)


async def demo_asynccontextmanager() -> None:
    """演示 asynccontextmanager 装饰器。"""
    print("\n" + "=" * 60)
    print("3. contextlib.asynccontextmanager — 装饰器方式")
    print("=" * 60)

    # 使用装饰器创建的异步上下文管理器
    async with async_llm_session(model="qwen2-7b") as session:
        print(f"  💬 使用会话 {session['session_id']} 进行推理...")
        await asyncio.sleep(0.2)  # 模拟推理
        print("  💬 推理完成")

    # 嵌套使用多个异步上下文管理器
    print("\n  --- 嵌套使用多个异步上下文管理器 ---")
    async with async_llm_session(model="qwen2") as llm:
        async with async_vector_store(collection="knowledge_base") as store:
            print(f"  🔗 LLM ({llm['model']}) + 向量库 ({store['collection']}) 联合查询")
            await asyncio.sleep(0.1)
            print("  🔗 RAG 查询完成")


# ============================================================
# 4. 异步资源管理最佳实践 — 组合多个资源
# ============================================================

class AsyncMLPipeline:
    """模拟异步 ML 推理流水线，管理多个异步资源。

    实际场景：一个 RAG 服务需要同时管理：
    - LLM 推理客户端
    - 向量数据库连接
    - 缓存连接
    - 监控指标上报器
    """

    def __init__(self, config: dict[str, str]):
        self.config = config
        self._resources: dict[str, Any] = {}

    async def __aenter__(self) -> AsyncMLPipeline:
        """按依赖顺序初始化所有资源。"""
        print("  🏗️ 初始化 ML 流水线资源...")

        # 并发初始化无依赖关系的资源
        init_tasks = [
            self._init_resource("vector_db", "向量数据库", 0.2),
            self._init_resource("cache", "Redis 缓存", 0.1),
            self._init_resource("metrics", "Prometheus 指标", 0.05),
        ]
        await asyncio.gather(*init_tasks)

        # LLM 客户端可能依赖其他资源的配置，串行初始化
        await self._init_resource("llm", "LLM 推理客户端", 0.15)

        print(f"  ✅ 所有资源就绪: {list(self._resources.keys())}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """按逆序释放所有资源。"""
        print("  🔒 释放 ML 流水线资源...")
        # 逆序释放，确保依赖关系正确
        for name in reversed(list(self._resources.keys())):
            print(f"    关闭: {name}")
            await asyncio.sleep(0.05)
        self._resources.clear()
        print("  ✅ 所有资源已释放")
        return False

    async def _init_resource(self, name: str, desc: str, delay: float) -> None:
        """初始化单个资源。"""
        await asyncio.sleep(delay)
        self._resources[name] = {"name": desc, "status": "ready"}
        print(f"    ✅ {desc} 就绪")

    async def predict(self, query: str) -> str:
        """模拟推理请求。"""
        print(f"  🔮 处理查询: {query}")
        await asyncio.sleep(0.2)
        return f"基于 RAG 的回答: {query} 的答案是..."


async def demo_resource_management() -> None:
    """演示异步资源管理最佳实践。"""
    print("\n" + "=" * 60)
    print("4. 异步资源管理最佳实践 — 组合多个资源")
    print("=" * 60)

    config = {
        "llm_url": "http://localhost:11434",
        "vector_db_url": "http://localhost:8000",
        "redis_url": "redis://localhost:6379",
    }

    async with AsyncMLPipeline(config) as pipeline:
        result = await pipeline.predict("什么是 Transformer 的自注意力机制？")
        print(f"  结果: {result}")


# ============================================================
# 5. asyncio.timeout — 超时控制（Python 3.11+）
# ============================================================

async def slow_llm_call(prompt: str, delay: float = 3.0) -> str:
    """模拟一个可能很慢的 LLM 调用。"""
    print(f"  🤖 LLM 开始处理: {prompt[:30]}...")
    await asyncio.sleep(delay)
    return f"LLM 回答: {prompt} 的结果"


async def demo_timeout_control() -> None:
    """演示 asyncio.timeout 超时控制（Python 3.11+）。

    在 AI 应用中，超时控制至关重要：
    - LLM 推理可能因为模型过大或 GPU 负载高而超时
    - 向量数据库检索在数据量大时可能变慢
    - 外部 API 调用可能因网络问题超时
    """
    print("\n" + "=" * 60)
    print("5. asyncio.timeout — 超时控制（Python 3.11+）")
    print("=" * 60)

    # --- 正常完成的情况 ---
    print("\n  --- 场景 1: 在超时时间内完成 ---")
    try:
        async with asyncio.timeout(2.0):
            result = await slow_llm_call("简单问题", delay=0.5)
            print(f"  ✅ {result}")
    except TimeoutError:
        print("  ❌ 超时！")

    # --- 超时的情况 ---
    print("\n  --- 场景 2: 超过超时时间 ---")
    try:
        async with asyncio.timeout(1.0):
            result = await slow_llm_call("复杂推理问题", delay=3.0)
            print(f"  ✅ {result}")
    except TimeoutError:
        print("  ❌ LLM 调用超时（1.0s），启动降级策略...")
        print("  🔄 降级: 使用更小的模型或返回缓存结果")

    # --- 超时 + 重试模式 ---
    print("\n  --- 场景 3: 超时重试模式 ---")
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            # 每次重试增加超时时间
            timeout_seconds = 0.5 * attempt
            print(f"  🔄 第 {attempt} 次尝试 (超时: {timeout_seconds}s)")
            async with asyncio.timeout(timeout_seconds):
                # 模拟：第三次尝试成功（延迟 1.2s）
                delay = 1.2 if attempt < 3 else 0.3
                result = await slow_llm_call("重试测试", delay=delay)
                print(f"  ✅ 第 {attempt} 次成功: {result}")
                break
        except TimeoutError:
            if attempt == max_retries:
                print(f"  ❌ 已达最大重试次数 ({max_retries})，放弃")
            else:
                print(f"  ⏰ 第 {attempt} 次超时，准备重试...")

    # --- asyncio.timeout_at 绝对时间超时 ---
    print("\n  --- 场景 4: asyncio.timeout_at 绝对时间超时 ---")
    loop = asyncio.get_event_loop()
    # 设置一个绝对截止时间（当前时间 + 1秒）
    deadline = loop.time() + 1.0
    try:
        async with asyncio.timeout_at(deadline):
            result = await slow_llm_call("绝对时间测试", delay=0.3)
            print(f"  ✅ {result}")
    except TimeoutError:
        print("  ❌ 超过绝对截止时间")


# ============================================================
# 6. 综合实战 — 带超时和资源管理的 RAG 服务
# ============================================================

@contextlib.asynccontextmanager
async def managed_rag_service(timeout: float = 5.0):
    """一个带完整资源管理和超时控制的 RAG 服务上下文。

    展示如何组合多种异步模式：
    - 异步上下文管理器管理资源生命周期
    - asyncio.timeout 控制整体超时
    - asyncio.gather 并发执行子任务
    """
    print("  🚀 启动 RAG 服务...")
    resources = {
        "llm": "qwen2-7b",
        "vector_db": "chroma://localhost:8000",
        "started_at": time.time(),
    }
    await asyncio.sleep(0.1)  # 模拟初始化
    print("  ✅ RAG 服务就绪")

    try:
        yield resources
    finally:
        duration = time.time() - resources["started_at"]
        print(f"  🔒 RAG 服务关闭 (运行 {duration:.2f}s)")


async def demo_comprehensive() -> None:
    """综合演示：带超时和资源管理的 RAG 查询。"""
    print("\n" + "=" * 60)
    print("6. 综合实战 — 带超时和资源管理的 RAG 服务")
    print("=" * 60)

    async with managed_rag_service() as service:
        print(f"  使用模型: {service['llm']}")

        # 带超时的 RAG 查询
        try:
            async with asyncio.timeout(3.0):
                # 并发执行检索和 Embedding
                retrieval, embedding = await asyncio.gather(
                    slow_llm_call("向量检索", delay=0.3),
                    slow_llm_call("Embedding 编码", delay=0.2),
                )
                print(f"  📚 检索完成: {retrieval}")
                print(f"  🔢 编码完成: {embedding}")

                # 串行执行 LLM 生成
                answer = await slow_llm_call("生成最终回答", delay=0.5)
                print(f"  💬 回答: {answer}")

        except TimeoutError:
            print("  ❌ RAG 查询超时，返回降级结果")

    print("  （所有资源已自动释放）")


# ============================================================
# 主入口
# ============================================================

async def main() -> None:
    """运行所有演示。"""
    print("🐍 异步上下文管理器 — async with、资源管理、超时控制")
    print("=" * 60)

    await demo_async_context_basics()
    await demo_exception_safety()
    await demo_asynccontextmanager()
    await demo_resource_management()
    await demo_timeout_control()
    await demo_comprehensive()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. async with 确保异步资源正确获取和释放")
    print("  2. __aenter__/__aexit__ 是异步上下文管理器的核心协议")
    print("  3. @asynccontextmanager 装饰器适合简单场景")
    print("  4. asyncio.timeout (3.11+) 提供优雅的超时控制")
    print("  5. 组合使用上下文管理器 + 超时 + 并发是 AI 服务的标准模式")


if __name__ == "__main__":
    asyncio.run(main())
