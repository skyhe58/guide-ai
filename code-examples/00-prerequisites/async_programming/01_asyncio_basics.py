"""
asyncio 基础 — 事件循环、async/await、并发执行

知识点：asyncio 事件循环机制、协程定义与调用、
       asyncio.gather 并发、asyncio.create_task 任务创建、
       串行 vs 并发性能对比、Python 3.11+ TaskGroup

Python 版本：3.11+
依赖：asyncio（标准库）
最后验证：2024-12-01
"""

from __future__ import annotations

import asyncio
import time


# ============================================================
# 1. async/await 基础 — 协程定义与调用
# ============================================================

async def fetch_data(source: str, delay: float = 1.0) -> dict:
    """模拟从远程数据源异步获取数据。

    在 AI 应用中，这类操作对应：
    - 调用 LLM API（如 OpenAI、Ollama）
    - 查询向量数据库（如 Chroma、Pinecone）
    - 读取远程文档用于 RAG 流水线

    Args:
        source: 数据源名称
        delay: 模拟网络延迟（秒）

    Returns:
        包含数据源名称和状态的字典
    """
    print(f"  ⏳ 开始请求: {source}")
    # asyncio.sleep 是非阻塞的，事件循环可以在等待期间执行其他协程
    # 注意：绝对不要在 async 函数中使用 time.sleep()，那会阻塞整个事件循环
    await asyncio.sleep(delay)
    print(f"  ✅ 请求完成: {source} (耗时 {delay:.1f}s)")
    return {"source": source, "status": "ok", "latency": delay}


async def demo_basic_coroutine() -> None:
    """演示基本的协程调用。"""
    print("\n" + "=" * 60)
    print("1. async/await 基础 — 协程定义与调用")
    print("=" * 60)

    # 直接 await 一个协程 — 等待它完成后才继续
    result = await fetch_data("用户服务", delay=0.5)
    print(f"  结果: {result}")

    # 协程对象 vs 协程调用的区别
    # coro = fetch_data("test")  # 这只是创建了协程对象，并未执行
    # result = await coro        # await 才会真正执行协程


# ============================================================
# 2. asyncio.gather — 并发执行多个协程
# ============================================================

async def demo_gather() -> None:
    """演示 asyncio.gather 并发执行。

    gather 是 AI 应用中最常用的并发原语：
    - RAG 流水线中并发检索多个数据源
    - 同时调用多个 LLM 进行对比评测
    - 批量处理多个用户请求
    """
    print("\n" + "=" * 60)
    print("2. asyncio.gather — 并发执行多个协程")
    print("=" * 60)

    start = time.perf_counter()

    # gather 同时启动所有协程，总耗时 ≈ 最慢的那个
    results = await asyncio.gather(
        fetch_data("LLM 推理服务", delay=1.5),
        fetch_data("向量数据库检索", delay=0.8),
        fetch_data("用户画像服务", delay=1.0),
    )

    elapsed = time.perf_counter() - start
    print(f"\n  📊 并发执行 3 个任务，总耗时: {elapsed:.2f}s（串行需 3.3s）")
    print(f"  返回结果数: {len(results)}")

    # gather 的 return_exceptions 参数 — 收集异常而非立即抛出
    # 在生产环境中，部分服务失败不应影响其他服务的结果
    print("\n  --- return_exceptions=True 演示 ---")

    async def failing_service() -> str:
        """模拟一个会失败的服务调用。"""
        await asyncio.sleep(0.3)
        raise ConnectionError("向量数据库连接超时")

    results_with_errors = await asyncio.gather(
        fetch_data("LLM 服务", delay=0.5),
        failing_service(),
        fetch_data("缓存服务", delay=0.3),
        return_exceptions=True,  # 异常作为结果返回，不中断其他任务
    )

    for i, r in enumerate(results_with_errors):
        if isinstance(r, Exception):
            print(f"  任务 {i}: ❌ 失败 — {type(r).__name__}: {r}")
        else:
            print(f"  任务 {i}: ✅ 成功 — {r}")


# ============================================================
# 3. asyncio.create_task — 创建后台任务
# ============================================================

async def background_logger(interval: float = 0.5, count: int = 3) -> None:
    """模拟后台日志/监控任务。

    在 AI 服务中，后台任务常用于：
    - 定期上报推理指标到 Prometheus
    - 异步写入审计日志
    - 心跳检测下游服务健康状态
    """
    for i in range(count):
        await asyncio.sleep(interval)
        print(f"  📝 [后台日志] 第 {i + 1}/{count} 次心跳检测 — 服务正常")


async def demo_create_task() -> None:
    """演示 asyncio.create_task 创建并发任务。"""
    print("\n" + "=" * 60)
    print("3. asyncio.create_task — 创建后台任务")
    print("=" * 60)

    # create_task 立即将协程调度到事件循环，不会阻塞当前协程
    log_task = asyncio.create_task(background_logger(interval=0.4, count=3))

    # 主任务继续执行，与后台日志任务并发运行
    print("  🚀 主任务: 开始处理用户请求...")
    result = await fetch_data("主业务逻辑", delay=1.0)
    print(f"  🚀 主任务完成: {result}")

    # 等待后台任务完成（如果还没完成的话）
    await log_task
    print("  后台日志任务也已完成")


# ============================================================
# 4. 串行 vs 并发 — 性能对比
# ============================================================

async def demo_serial_vs_concurrent() -> None:
    """对比串行和并发执行的耗时差异。

    这是异步编程最直观的价值体现：
    在 AI 应用中，一个请求往往需要调用多个下游服务，
    串行调用的总耗时是所有调用之和，而并发调用接近最慢的那个。
    """
    print("\n" + "=" * 60)
    print("4. 串行 vs 并发 — 性能对比")
    print("=" * 60)

    services = [
        ("Embedding 模型", 0.8),
        ("向量检索", 0.6),
        ("LLM 生成", 1.5),
        ("Rerank 重排序", 0.4),
    ]

    # --- 串行执行 ---
    print("\n  📌 串行执行:")
    start = time.perf_counter()
    serial_results = []
    for name, delay in services:
        r = await fetch_data(name, delay)
        serial_results.append(r)
    serial_time = time.perf_counter() - start
    print(f"  串行总耗时: {serial_time:.2f}s")

    # --- 并发执行 ---
    print("\n  📌 并发执行:")
    start = time.perf_counter()
    concurrent_results = await asyncio.gather(
        *(fetch_data(name, delay) for name, delay in services)
    )
    concurrent_time = time.perf_counter() - start
    print(f"  并发总耗时: {concurrent_time:.2f}s")

    # --- 对比 ---
    speedup = serial_time / concurrent_time if concurrent_time > 0 else float("inf")
    print(f"\n  🚀 加速比: {speedup:.1f}x（串行 {serial_time:.2f}s → 并发 {concurrent_time:.2f}s）")


# ============================================================
# 5. Python 3.11+ TaskGroup — 结构化并发
# ============================================================

async def demo_task_group() -> None:
    """演示 Python 3.11+ 的 TaskGroup 结构化并发。

    TaskGroup 相比 gather 的优势：
    - 任一任务失败时自动取消所有其他任务（更安全）
    - 异常通过 ExceptionGroup 统一收集
    - 更清晰的作用域管理（async with 块结束时所有任务必须完成）
    """
    print("\n" + "=" * 60)
    print("5. Python 3.11+ TaskGroup — 结构化并发")
    print("=" * 60)

    results: list[dict] = []

    async def collect_result(source: str, delay: float) -> None:
        """获取数据并收集结果。"""
        data = await fetch_data(source, delay)
        results.append(data)

    start = time.perf_counter()

    # TaskGroup 确保所有任务在 async with 块结束前完成
    async with asyncio.TaskGroup() as tg:
        tg.create_task(collect_result("文档加载器", 0.6))
        tg.create_task(collect_result("文本切分器", 0.4))
        tg.create_task(collect_result("Embedding 编码", 0.8))
        tg.create_task(collect_result("向量存储", 0.5))

    elapsed = time.perf_counter() - start
    print(f"\n  TaskGroup 完成，共 {len(results)} 个任务，耗时: {elapsed:.2f}s")

    # --- TaskGroup 异常处理演示 ---
    print("\n  --- TaskGroup 异常处理 ---")

    async def risky_task(name: str, should_fail: bool = False) -> str:
        """可能失败的任务。"""
        await asyncio.sleep(0.3)
        if should_fail:
            raise ValueError(f"{name} 处理失败: 输入数据格式错误")
        return f"{name} 完成"

    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(risky_task("任务A"))
            tg.create_task(risky_task("任务B", should_fail=True))
            tg.create_task(risky_task("任务C"))
    except* ValueError as eg:
        # Python 3.11+ except* 语法处理 ExceptionGroup
        print(f"  捕获到 {len(eg.exceptions)} 个 ValueError:")
        for exc in eg.exceptions:
            print(f"    ❌ {exc}")
        print("  💡 TaskGroup 中任一任务失败，其他任务会被自动取消")


# ============================================================
# 6. 实战模式 — 模拟 RAG 流水线的异步编排
# ============================================================

async def rag_pipeline_async(query: str) -> dict:
    """模拟一个异步 RAG（检索增强生成）流水线。

    实际 AI 应用中的典型异步编排：
    1. 并发执行查询改写 + Embedding 编码
    2. 并发检索多个数据源
    3. Rerank 重排序（依赖检索结果，串行）
    4. LLM 生成最终回答（依赖重排序结果，串行）
    """
    print("\n" + "=" * 60)
    print("6. 实战模式 — 模拟 RAG 流水线异步编排")
    print("=" * 60)
    print(f"  用户查询: {query}")

    start = time.perf_counter()

    # 阶段 1：并发执行查询改写和 Embedding 编码
    print("\n  --- 阶段 1: 查询预处理（并发） ---")
    query_rewrite, embedding = await asyncio.gather(
        fetch_data("查询改写 (HyDE)", delay=0.3),
        fetch_data("Embedding 编码 (BGE-M3)", delay=0.2),
    )

    # 阶段 2：并发检索多个数据源
    print("\n  --- 阶段 2: 多源检索（并发） ---")
    search_results = await asyncio.gather(
        fetch_data("Chroma 向量检索", delay=0.4),
        fetch_data("BM25 关键词检索", delay=0.3),
        fetch_data("知识图谱检索", delay=0.5),
    )

    # 阶段 3：Rerank 重排序（依赖检索结果）
    print("\n  --- 阶段 3: Rerank 重排序（串行） ---")
    reranked = await fetch_data("BGE-Reranker 重排序", delay=0.3)

    # 阶段 4：LLM 生成回答（依赖重排序结果）
    print("\n  --- 阶段 4: LLM 生成（串行） ---")
    answer = await fetch_data("Qwen2 生成回答", delay=0.8)

    elapsed = time.perf_counter() - start
    print(f"\n  🎯 RAG 流水线完成，总耗时: {elapsed:.2f}s")
    print(f"  （如果全部串行，预计耗时: {0.3+0.2+0.4+0.3+0.5+0.3+0.8:.1f}s）")

    return {
        "query": query,
        "sources": len(search_results),
        "answer": answer,
        "latency": round(elapsed, 2),
    }


# ============================================================
# 主入口
# ============================================================

async def main() -> None:
    """运行所有演示。"""
    print("🐍 asyncio 基础 — 事件循环、async/await、并发执行")
    print("=" * 60)

    await demo_basic_coroutine()
    await demo_gather()
    await demo_create_task()
    await demo_serial_vs_concurrent()
    await demo_task_group()
    await rag_pipeline_async("如何使用 LangGraph 构建多 Agent 系统？")

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. async def 定义协程，await 等待协程完成")
    print("  2. asyncio.gather 并发执行多个协程，总耗时 ≈ 最慢的那个")
    print("  3. asyncio.create_task 创建后台任务，不阻塞当前协程")
    print("  4. TaskGroup (3.11+) 提供更安全的结构化并发")
    print("  5. 在 AI 应用中，合理编排串行/并发可以显著降低延迟")


if __name__ == "__main__":
    # asyncio.run() 是 Python 3.7+ 推荐的启动方式
    # 它会创建事件循环、运行协程、然后关闭事件循环
    asyncio.run(main())
