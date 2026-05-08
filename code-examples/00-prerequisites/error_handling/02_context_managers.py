"""
上下文管理器 — with 语句、资源管理、contextlib 工具

知识点：__enter__/__exit__ 协议、with 语句资源管理、
       contextlib.contextmanager 装饰器、
       contextlib.suppress/closing/ExitStack、
       AI 应用中的资源管理模式

Python 版本：3.11+
依赖：标准库
最后验证：2024-12-01
"""

from __future__ import annotations

import contextlib
import time
from typing import Any

# ============================================================
# 1. __enter__ / __exit__ 协议
# ============================================================

class GPUMemoryManager:
    """模拟 GPU 显存管理器。

    在 AI 应用中，GPU 显存是稀缺资源：
    - 模型加载需要分配显存
    - 推理完成后需要释放显存
    - 异常情况下也必须确保显存释放

    上下文管理器是管理这类资源的最佳模式。
    """

    def __init__(self, model_name: str, memory_gb: float = 4.0):
        self.model_name = model_name
        self.memory_gb = memory_gb
        self._allocated = False

    def __enter__(self) -> GPUMemoryManager:
        """分配 GPU 显存。"""
        print(f"  🔋 分配 {self.memory_gb}GB GPU 显存: {self.model_name}")
        self._allocated = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        """释放 GPU 显存。

        参数说明：
        - exc_type: 异常类型（无异常时为 None）
        - exc_val: 异常实例
        - exc_tb: 异常 traceback
        - 返回 True 抑制异常，False 继续传播
        """
        if exc_type is not None:
            print(f"  ⚠️ 检测到异常: {exc_type.__name__}: {exc_val}")
        print(f"  🔌 释放 {self.memory_gb}GB GPU 显存: {self.model_name}")
        self._allocated = False
        return False  # 不抑制异常

    def predict(self, prompt: str) -> str:
        """模拟模型推理。"""
        if not self._allocated:
            raise RuntimeError("GPU 显存未分配，请使用 with 语句")
        print(f"  🤖 [{self.model_name}] 推理: {prompt[:40]}...")
        return f"回答: {prompt} 的结果"


def demo_enter_exit() -> None:
    """演示 __enter__/__exit__ 协议。"""
    print("\n" + "=" * 60)
    print("1. __enter__ / __exit__ 协议")
    print("=" * 60)

    # 正常使用
    print("\n  --- 正常使用 ---")
    with GPUMemoryManager("qwen2-7b", memory_gb=8.0) as gpu:
        result = gpu.predict("什么是 RAG？")
        print(f"  结果: {result}")
    print("  （显存已自动释放）")

    # 异常情况 — 显存仍然会被释放
    print("\n  --- 异常情况 ---")
    try:
        with GPUMemoryManager("llama3-70b", memory_gb=40.0) as gpu:
            raise MemoryError("GPU 显存不足")
    except MemoryError as e:
        print(f"  外层捕获: {e}")
        print("  （显存仍然被正确释放了）")


# ============================================================
# 2. contextlib.contextmanager 装饰器
# ============================================================

@contextlib.contextmanager
def timer(label: str = "操作"):
    """计时上下文管理器 — 测量代码块执行时间。

    在 AI 应用中常用于：
    - 测量推理延迟
    - 测量 RAG 各阶段耗时
    - 性能基准测试
    """
    start = time.perf_counter()
    print(f"  ⏱️ [{label}] 开始计时...")
    try:
        yield  # yield 之前 = __enter__，yield 之后 = __exit__
    finally:
        elapsed = time.perf_counter() - start
        print(f"  ⏱️ [{label}] 耗时: {elapsed:.3f}s")


@contextlib.contextmanager
def temporary_config(config: dict, overrides: dict):
    """临时配置覆盖 — 测试不同参数时使用。

    在 AI 应用中常用于：
    - A/B 测试不同的 temperature/top_p 参数
    - 临时切换模型
    - 测试不同的检索策略
    """
    original = {k: config.get(k) for k in overrides}
    config.update(overrides)
    print(f"  📝 临时配置: {overrides}")
    try:
        yield config
    finally:
        # 恢复原始配置
        for k, v in original.items():
            if v is None:
                config.pop(k, None)
            else:
                config[k] = v
        print(f"  📝 配置已恢复")


def demo_contextmanager_decorator() -> None:
    """演示 contextlib.contextmanager 装饰器。"""
    print("\n" + "=" * 60)
    print("2. contextlib.contextmanager 装饰器")
    print("=" * 60)

    # 计时器
    print("\n  --- 计时器 ---")
    with timer("模拟推理"):
        time.sleep(0.1)  # 模拟耗时操作

    # 临时配置
    print("\n  --- 临时配置覆盖 ---")
    llm_config = {"model": "qwen2", "temperature": 0.7, "max_tokens": 200}
    print(f"  原始配置: {llm_config}")

    with temporary_config(llm_config, {"temperature": 0.1, "max_tokens": 50}):
        print(f"  临时配置: {llm_config}")

    print(f"  恢复后: {llm_config}")


# ============================================================
# 3. contextlib 实用工具
# ============================================================

def demo_contextlib_tools() -> None:
    """演示 contextlib 模块的实用工具。"""
    print("\n" + "=" * 60)
    print("3. contextlib 实用工具")
    print("=" * 60)

    # --- suppress：静默忽略特定异常 ---
    print("\n  --- contextlib.suppress ---")
    # 等价于 try/except + pass，但更简洁
    with contextlib.suppress(FileNotFoundError):
        # 尝试删除缓存文件，不存在也没关系
        open("/tmp/nonexistent_cache.json")  # noqa: SIM115
    print("  FileNotFoundError 被静默忽略")

    # --- ExitStack：动态管理多个上下文管理器 ---
    print("\n  --- contextlib.ExitStack ---")

    models = ["embedding-model", "reranker-model", "llm-model"]

    with contextlib.ExitStack() as stack:
        # 动态注册多个上下文管理器
        loaded = []
        for name in models:
            gpu = stack.enter_context(GPUMemoryManager(name, memory_gb=2.0))
            loaded.append(gpu)
        print(f"  已加载 {len(loaded)} 个模型")
        # ExitStack 退出时，按注册的逆序释放所有资源
    print("  所有模型资源已释放")


# ============================================================
# 4. 实战模式 — RAG 服务资源管理
# ============================================================

@contextlib.contextmanager
def rag_service_context(config: dict):
    """RAG 服务完整资源管理。

    管理 RAG 服务需要的所有资源：
    1. Embedding 模型
    2. 向量数据库连接
    3. LLM 推理客户端
    """
    resources = {}
    print("  🚀 启动 RAG 服务...")

    try:
        # 按依赖顺序初始化
        resources["embedding"] = f"embedding@{config.get('embedding', 'bge-m3')}"
        print(f"    ✅ Embedding: {resources['embedding']}")

        resources["vector_db"] = f"chroma@{config.get('db_host', 'localhost:8000')}"
        print(f"    ✅ 向量数据库: {resources['vector_db']}")

        resources["llm"] = f"llm@{config.get('model', 'qwen2')}"
        print(f"    ✅ LLM: {resources['llm']}")

        print("  ✅ RAG 服务就绪")
        yield resources

    except Exception as e:
        print(f"  ❌ RAG 服务异常: {e}")
        raise

    finally:
        # 逆序释放资源
        print("  🔒 关闭 RAG 服务...")
        for name in reversed(list(resources.keys())):
            print(f"    关闭: {name}")
        resources.clear()
        print("  ✅ 所有资源已释放")


def demo_rag_resource_management() -> None:
    """演示 RAG 服务的资源管理。"""
    print("\n" + "=" * 60)
    print("4. 实战模式 — RAG 服务资源管理")
    print("=" * 60)

    config = {
        "embedding": "bge-m3",
        "db_host": "localhost:8000",
        "model": "qwen2-7b",
    }

    with rag_service_context(config) as svc:
        print(f"\n  使用资源: {list(svc.keys())}")
        print("  执行 RAG 查询...")
        print("  查询完成\n")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行所有演示。"""
    print("🐍 上下文管理器 — with 语句、资源管理、contextlib 工具")
    print("=" * 60)

    demo_enter_exit()
    demo_contextmanager_decorator()
    demo_contextlib_tools()
    demo_rag_resource_management()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. with 语句确保资源正确释放，即使发生异常")
    print("  2. __enter__/__exit__ 是上下文管理器的核心协议")
    print("  3. @contextmanager 装饰器适合简单场景（yield 前后分别对应进入/退出）")
    print("  4. ExitStack 适合动态管理数量不确定的资源")
    print("  5. AI 应用中，模型/连接/显存等资源都应该用上下文管理器管理")


if __name__ == "__main__":
    main()
