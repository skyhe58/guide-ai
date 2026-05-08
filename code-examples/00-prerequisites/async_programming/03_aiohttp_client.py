"""
aiohttp 异步 HTTP 客户端 — 请求、并发、超时、重试

知识点：aiohttp.ClientSession 使用、异步 GET/POST 请求、
       并发请求多个 URL、ClientTimeout 超时控制、
       错误处理与重试机制、Ollama API 调用示例

Python 版本：3.11+
依赖：aiohttp>=3.9
最后验证：2024-12-01

外部服务（可选）：
  Ollama 本地 LLM 推理服务
  启动命令：docker compose -f docker/docker-compose.yml up -d ollama
  API 地址：http://localhost:11434
  默认模型：qwen2

运行方式：
  默认模式（内存模拟，无需外部服务）：
    python 03_aiohttp_client.py

  服务模式（调用本地 Ollama API）：
    python 03_aiohttp_client.py server
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import Any

import aiohttp

# ============================================================
# 1. aiohttp.ClientSession 基础 — GET/POST 请求
# ============================================================

async def demo_basic_requests() -> None:
    """演示 aiohttp 基本的 GET 和 POST 请求。

    aiohttp.ClientSession 是异步 HTTP 客户端的核心：
    - 内部维护连接池，复用 TCP 连接
    - 支持 Cookie 管理、请求头设置
    - 必须在 async with 中使用，确保连接正确释放
    """
    print("\n" + "=" * 60)
    print("1. aiohttp.ClientSession 基础 — GET/POST 请求")
    print("=" * 60)

    # ClientSession 应该在整个应用生命周期中复用
    # 不要为每个请求创建新的 session
    async with aiohttp.ClientSession() as session:

        # --- GET 请求 ---
        print("\n  --- GET 请求 ---")
        try:
            async with session.get(
                "https://httpbin.org/get",
                params={"query": "asyncio tutorial", "lang": "zh"},
                headers={"User-Agent": "guide-ai/1.0"},
            ) as response:
                print(f"  状态码: {response.status}")
                print(f"  Content-Type: {response.content_type}")
                data = await response.json()
                print(f"  请求参数: {data.get('args', {})}")
        except aiohttp.ClientError as e:
            print(f"  ⚠️ GET 请求失败（可能无网络）: {e}")
            print("  💡 使用模拟数据继续演示...")
            _demo_mock_get()

        # --- POST 请求 ---
        print("\n  --- POST 请求 (JSON) ---")
        try:
            payload = {
                "model": "qwen2",
                "prompt": "什么是 RAG？",
                "max_tokens": 100,
            }
            async with session.post(
                "https://httpbin.org/post",
                json=payload,
            ) as response:
                print(f"  状态码: {response.status}")
                data = await response.json()
                # httpbin 会回显我们发送的数据
                sent_data = json.loads(data.get("data", "{}"))
                print(f"  发送的数据: {sent_data}")
        except aiohttp.ClientError as e:
            print(f"  ⚠️ POST 请求失败（可能无网络）: {e}")
            print("  💡 使用模拟数据继续演示...")
            _demo_mock_post()


def _demo_mock_get() -> None:
    """无网络时的 GET 模拟输出。"""
    print("  [模拟] 状态码: 200")
    print("  [模拟] 请求参数: {'query': 'asyncio tutorial', 'lang': 'zh'}")


def _demo_mock_post() -> None:
    """无网络时的 POST 模拟输出。"""
    print("  [模拟] 状态码: 200")
    print("  [模拟] 发送的数据: {'model': 'qwen2', 'prompt': '什么是 RAG？'}")


# ============================================================
# 2. 并发请求多个 URL
# ============================================================

async def fetch_url(
    session: aiohttp.ClientSession,
    url: str,
    label: str = "",
) -> dict[str, Any]:
    """异步获取单个 URL 的内容。

    Args:
        session: 复用的 HTTP 会话
        url: 目标 URL
        label: 请求标签（用于日志）

    Returns:
        包含状态码、耗时等信息的字典
    """
    start = time.perf_counter()
    try:
        async with session.get(url) as response:
            # 读取响应体（这里只读取前 200 字节以节省内存）
            body = await response.text()
            elapsed = time.perf_counter() - start
            return {
                "label": label or url,
                "status": response.status,
                "size": len(body),
                "latency": round(elapsed, 3),
                "success": True,
            }
    except aiohttp.ClientError as e:
        elapsed = time.perf_counter() - start
        return {
            "label": label or url,
            "status": 0,
            "error": str(e),
            "latency": round(elapsed, 3),
            "success": False,
        }


async def demo_concurrent_requests() -> None:
    """演示并发请求多个 URL。

    在 AI 应用中的典型场景：
    - 并发调用多个 LLM API 进行对比评测
    - RAG 中并发检索多个数据源
    - 批量获取文档用于知识库构建
    """
    print("\n" + "=" * 60)
    print("2. 并发请求多个 URL")
    print("=" * 60)

    # 模拟并发请求多个服务（使用 httpbin 的延迟接口）
    urls = [
        ("https://httpbin.org/delay/1", "服务 A (1s 延迟)"),
        ("https://httpbin.org/delay/1", "服务 B (1s 延迟)"),
        ("https://httpbin.org/delay/1", "服务 C (1s 延迟)"),
        ("https://httpbin.org/get", "服务 D (快速)"),
    ]

    start = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        # 使用 gather 并发发起所有请求
        tasks = [
            fetch_url(session, url, label)
            for url, label in urls
        ]
        results = await asyncio.gather(*tasks)

    total_time = time.perf_counter() - start

    print(f"\n  📊 并发请求 {len(urls)} 个 URL 的结果:")
    for r in results:
        status = "✅" if r["success"] else "❌"
        print(f"    {status} {r['label']}: "
              f"状态={r.get('status', 'N/A')}, "
              f"耗时={r['latency']}s")

    print(f"\n  总耗时: {total_time:.2f}s（串行预计: ~4s）")

    if not results[0]["success"]:
        print("  💡 提示: 请求失败可能是因为无网络连接，这不影响学习")


# ============================================================
# 3. ClientTimeout — 超时控制
# ============================================================

async def demo_timeout() -> None:
    """演示 aiohttp 的超时控制机制。

    ClientTimeout 提供细粒度的超时配置：
    - total: 整个请求的总超时（从发起到读取完响应）
    - connect: TCP 连接建立超时
    - sock_connect: socket 连接超时
    - sock_read: 读取响应数据超时

    在 AI 应用中，合理设置超时非常重要：
    - LLM API 调用通常需要较长的 total 超时（30-120s）
    - 向量数据库查询通常较快（5-10s）
    - 健康检查需要短超时（1-3s）
    """
    print("\n" + "=" * 60)
    print("3. ClientTimeout — 超时控制")
    print("=" * 60)

    # --- 不同场景的超时配置 ---
    timeout_configs = {
        "LLM 推理": aiohttp.ClientTimeout(total=60, connect=5),
        "向量检索": aiohttp.ClientTimeout(total=10, connect=3),
        "健康检查": aiohttp.ClientTimeout(total=3, connect=1),
    }

    for name, timeout in timeout_configs.items():
        print(f"\n  📋 {name} 超时配置:")
        print(f"    total={timeout.total}s, connect={timeout.connect}s")

    # --- 超时演示 ---
    print("\n  --- 超时演示 ---")

    # 设置一个很短的超时来触发超时异常
    short_timeout = aiohttp.ClientTimeout(total=0.5)

    try:
        async with aiohttp.ClientSession(timeout=short_timeout) as session:
            async with session.get("https://httpbin.org/delay/3") as response:
                data = await response.text()
                print(f"  ✅ 请求成功: {len(data)} bytes")
    except TimeoutError:
        print("  ⏰ 请求超时（0.5s），符合预期")
        print("  💡 在生产环境中，超时后应启动降级或重试策略")
    except aiohttp.ClientError as e:
        print(f"  ⚠️ 请求失败: {e}")
        print("  💡 无网络环境下跳过超时演示")

    # --- 为不同请求设置不同超时 ---
    print("\n  --- 请求级别超时覆盖 ---")
    default_timeout = aiohttp.ClientTimeout(total=30)

    try:
        async with aiohttp.ClientSession(timeout=default_timeout) as session:
            # 单个请求可以覆盖 session 级别的超时
            async with session.get(
                "https://httpbin.org/get",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:
                print(f"  ✅ 使用请求级别超时 (5s): 状态码={response.status}")
    except (TimeoutError, aiohttp.ClientError) as e:
        print(f"  ⚠️ 请求失败: {e}")


# ============================================================
# 4. 错误处理与重试机制
# ============================================================

async def fetch_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """带指数退避重试的异步 HTTP 请求。

    重试策略在 AI 应用中非常重要：
    - LLM API 可能因为限流返回 429
    - 推理服务可能因为 GPU 负载高暂时不可用
    - 网络波动可能导致偶发性失败

    指数退避（Exponential Backoff）避免在服务恢复时造成请求风暴。

    Args:
        session: HTTP 会话
        url: 请求 URL
        max_retries: 最大重试次数
        base_delay: 基础重试延迟（秒），每次重试翻倍
        timeout: 单次请求超时（秒）

    Returns:
        请求结果字典
    """
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            req_timeout = aiohttp.ClientTimeout(total=timeout)
            async with session.get(url, timeout=req_timeout) as response:
                if response.status == 429:
                    # 限流：从响应头获取建议的等待时间
                    retry_after = float(
                        response.headers.get("Retry-After", base_delay * attempt)
                    )
                    print(f"    ⚠️ 第 {attempt} 次: 被限流 (429)，等待 {retry_after}s")
                    await asyncio.sleep(retry_after)
                    continue

                if response.status >= 500:
                    # 服务端错误：重试
                    print(f"    ⚠️ 第 {attempt} 次: 服务端错误 ({response.status})")
                    await asyncio.sleep(base_delay * attempt)
                    continue

                # 成功
                body = await response.text()
                return {
                    "status": response.status,
                    "body_size": len(body),
                    "attempts": attempt,
                    "success": True,
                }

        except TimeoutError:
            last_error = TimeoutError(f"请求超时 ({timeout}s)")
            print(f"    ⏰ 第 {attempt} 次: 超时")
        except aiohttp.ClientConnectorError as e:
            last_error = e
            print(f"    🔌 第 {attempt} 次: 连接失败 — {e}")
        except aiohttp.ClientError as e:
            last_error = e
            print(f"    ❌ 第 {attempt} 次: 请求错误 — {e}")

        # 指数退避：1s → 2s → 4s
        if attempt < max_retries:
            delay = base_delay * (2 ** (attempt - 1))
            print(f"    🔄 等待 {delay}s 后重试...")
            await asyncio.sleep(delay)

    # 所有重试都失败
    return {
        "status": 0,
        "error": str(last_error),
        "attempts": max_retries,
        "success": False,
    }


async def demo_retry() -> None:
    """演示错误处理和重试机制。"""
    print("\n" + "=" * 60)
    print("4. 错误处理与重试机制")
    print("=" * 60)

    async with aiohttp.ClientSession() as session:
        # 场景 1：请求一个正常的 URL
        print("\n  --- 场景 1: 正常请求 ---")
        result = await fetch_with_retry(
            session,
            "https://httpbin.org/get",
            max_retries=3,
            timeout=5.0,
        )
        if result["success"]:
            print(f"  ✅ 成功 (第 {result['attempts']} 次尝试)")
        else:
            print(f"  ❌ 失败: {result.get('error', '未知错误')}")

        # 场景 2：请求一个不存在的服务（模拟连接失败）
        print("\n  --- 场景 2: 连接失败 + 重试 ---")
        result = await fetch_with_retry(
            session,
            "http://localhost:19999/nonexistent",
            max_retries=2,
            base_delay=0.3,  # 缩短延迟以加快演示
            timeout=1.0,
        )
        print(f"  最终结果: {'✅ 成功' if result['success'] else '❌ 失败'}")
        print(f"  尝试次数: {result['attempts']}")


# ============================================================
# 5. 实战模式 — 模拟并发调用多个 LLM API
# ============================================================

async def mock_llm_api(
    session: aiohttp.ClientSession,
    model: str,
    prompt: str,
    delay: float = 0.5,
) -> dict[str, Any]:
    """模拟调用 LLM API（内存模式，无需外部服务）。

    模拟不同模型的响应特征：
    - 不同模型有不同的延迟
    - 返回模拟的推理结果
    """
    print(f"  🤖 [{model}] 开始推理...")
    await asyncio.sleep(delay)

    # 模拟不同模型的响应
    responses = {
        "qwen2-7b": "RAG 是检索增强生成技术，结合了信息检索和文本生成...",
        "llama3-8b": "RAG (Retrieval-Augmented Generation) combines retrieval...",
        "deepseek-v2": "RAG 通过检索外部知识库来增强 LLM 的生成能力...",
    }

    return {
        "model": model,
        "prompt": prompt,
        "response": responses.get(model, f"{model} 的回答..."),
        "latency": delay,
        "tokens": len(prompt) * 3,  # 模拟 token 计数
    }


async def demo_concurrent_llm_calls() -> None:
    """演示并发调用多个 LLM 进行对比评测。

    这是 AI 应用开发中的常见场景：
    - 多模型对比评测
    - A/B 测试不同的 Prompt
    - 多模型投票（ensemble）提高准确率
    """
    print("\n" + "=" * 60)
    print("5. 实战模式 — 并发调用多个 LLM API（内存模拟）")
    print("=" * 60)

    prompt = "请解释什么是 RAG（检索增强生成）？"
    models = [
        ("qwen2-7b", 0.8),
        ("llama3-8b", 1.2),
        ("deepseek-v2", 0.6),
    ]

    start = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        # 并发调用所有模型
        tasks = [
            mock_llm_api(session, model, prompt, delay)
            for model, delay in models
        ]
        results = await asyncio.gather(*tasks)

    total_time = time.perf_counter() - start

    print(f"\n  📊 多模型对比结果 (总耗时: {total_time:.2f}s):")
    print(f"  {'模型':<15} {'延迟':>6} {'Token 数':>8}  回答摘要")
    print("  " + "-" * 70)
    for r in results:
        print(f"  {r['model']:<15} {r['latency']:>5.1f}s {r['tokens']:>8}  "
              f"{r['response'][:35]}...")


# ============================================================
# 6. 服务模式 — 调用本地 Ollama API
# ============================================================

async def call_ollama_api(
    session: aiohttp.ClientSession,
    prompt: str,
    model: str = "qwen2",
    base_url: str = "http://localhost:11434",
) -> dict[str, Any]:
    """调用本地 Ollama API 进行 LLM 推理。

    Ollama 提供 OpenAI 兼容的 API 接口：
    - /api/generate — 文本生成
    - /api/chat — 对话
    - /api/embeddings — 向量编码

    Args:
        session: HTTP 会话
        prompt: 用户提示词
        model: 模型名称（需要先 ollama pull 下载）
        base_url: Ollama 服务地址

    Returns:
        包含模型回答的字典
    """
    url = f"{base_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,  # 非流式，等待完整响应
        "options": {
            "temperature": 0.7,
            "num_predict": 200,  # 限制生成长度
        },
    }

    timeout = aiohttp.ClientTimeout(total=60)  # LLM 推理可能较慢

    start = time.perf_counter()
    async with session.post(url, json=payload, timeout=timeout) as response:
        if response.status != 200:
            error_text = await response.text()
            raise aiohttp.ClientResponseError(
                response.request_info,
                response.history,
                status=response.status,
                message=f"Ollama API 错误: {error_text}",
            )

        data = await response.json()
        elapsed = time.perf_counter() - start

        return {
            "model": data.get("model", model),
            "response": data.get("response", ""),
            "total_duration_ms": data.get("total_duration", 0) / 1e6,
            "eval_count": data.get("eval_count", 0),
            "request_latency": round(elapsed, 2),
        }


async def demo_ollama_server_mode() -> None:
    """服务模式：调用本地 Ollama API。

    前置条件：
    1. 启动 Ollama: docker compose -f docker/docker-compose.yml up -d ollama
    2. 下载模型: docker exec ollama ollama pull qwen2
    """
    print("\n" + "=" * 60)
    print("6. 服务模式 — 调用本地 Ollama API")
    print("=" * 60)

    base_url = "http://localhost:11434"

    async with aiohttp.ClientSession() as session:
        # 检查 Ollama 服务是否可用
        print(f"\n  🔍 检查 Ollama 服务: {base_url}")
        try:
            health_timeout = aiohttp.ClientTimeout(total=3)
            async with session.get(base_url, timeout=health_timeout) as resp:
                if resp.status == 200:
                    print("  ✅ Ollama 服务正常运行")
                else:
                    print(f"  ⚠️ Ollama 返回异常状态: {resp.status}")
                    return
        except (TimeoutError, aiohttp.ClientError):
            print("  ❌ 无法连接 Ollama 服务")
            print("  💡 请先启动 Ollama:")
            print("     docker compose -f docker/docker-compose.yml up -d ollama")
            return

        # 单次调用
        print("\n  --- 单次 LLM 调用 ---")
        try:
            result = await call_ollama_api(
                session,
                prompt="用一句话解释什么是向量数据库？",
                model="qwen2",
            )
            print(f"  模型: {result['model']}")
            print(f"  回答: {result['response'][:200]}")
            print(f"  延迟: {result['request_latency']}s")
            print(f"  Token 数: {result['eval_count']}")
        except aiohttp.ClientError as e:
            print(f"  ❌ 调用失败: {e}")
            return

        # 并发调用 — 多个 Prompt 同时推理
        print("\n  --- 并发多 Prompt 推理 ---")
        prompts = [
            "什么是 Transformer 的自注意力机制？",
            "LoRA 微调的核心原理是什么？",
            "RAG 和 Fine-tuning 的区别是什么？",
        ]

        start = time.perf_counter()
        tasks = [
            call_ollama_api(session, prompt=p, model="qwen2")
            for p in prompts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.perf_counter() - start

        print(f"\n  📊 并发推理结果 (总耗时: {total_time:.2f}s):")
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                print(f"    ❌ Prompt {i+1}: {r}")
            else:
                print(f"    ✅ Prompt {i+1}: {r['response'][:60]}... "
                      f"({r['request_latency']}s)")


# ============================================================
# 主入口
# ============================================================

async def main(server_mode: bool = False) -> None:
    """运行所有演示。

    Args:
        server_mode: 是否启用服务模式（调用本地 Ollama）
    """
    print("🐍 aiohttp 异步 HTTP 客户端 — 请求、并发、超时、重试")
    print("=" * 60)

    if server_mode:
        print("📡 运行模式: 服务模式（调用本地 Ollama API）")
    else:
        print("💾 运行模式: 默认模式（内存模拟，无需外部服务）")

    # 默认模式的演示
    await demo_basic_requests()
    await demo_concurrent_requests()
    await demo_timeout()
    await demo_retry()
    await demo_concurrent_llm_calls()

    # 服务模式：调用本地 Ollama
    if server_mode:
        await demo_ollama_server_mode()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. ClientSession 应在应用生命周期中复用，不要每次请求都创建")
    print("  2. 使用 asyncio.gather 并发请求多个 URL，大幅降低总耗时")
    print("  3. ClientTimeout 提供细粒度超时控制（total/connect/read）")
    print("  4. 生产环境必须实现重试机制（指数退避 + 限流处理）")
    print("  5. 调用 LLM API 时注意设置较长的超时（30-120s）")

    if not server_mode:
        print("\n💡 要测试真实 LLM 调用，请运行:")
        print("   python 03_aiohttp_client.py server")


if __name__ == "__main__":
    # 检查命令行参数决定运行模式
    is_server_mode = len(sys.argv) > 1 and sys.argv[1].lower() == "server"
    asyncio.run(main(server_mode=is_server_mode))
