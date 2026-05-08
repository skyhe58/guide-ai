"""
自定义异常 — 异常层次设计、异常链、优雅降级

知识点：自定义异常类设计、异常链（raise ... from ...）、
       异常上下文信息、AI 应用异常处理模式、优雅降级策略

Python 版本：3.11+
依赖：标准库
最后验证：2024-12-01
"""

from __future__ import annotations

# ============================================================
# 1. 自定义异常层次 — AI 应用场景
# ============================================================

class AIServiceError(Exception):
    """AI 服务基础异常类。

    所有 AI 相关的自定义异常都继承此类，
    便于在最外层统一捕获 AI 服务异常。
    """

    def __init__(
        self,
        message: str,
        service: str = "",
        error_code: str = "",
        retry_after: float = 0,
    ):
        super().__init__(message)
        self.service = service        # 出错的服务名
        self.error_code = error_code  # 错误码
        self.retry_after = retry_after  # 建议重试等待时间（秒）

    def to_dict(self) -> dict:
        """序列化为字典，方便日志记录和 API 响应。"""
        return {
            "error_type": type(self).__name__,
            "message": str(self),
            "service": self.service,
            "error_code": self.error_code,
            "retry_after": self.retry_after,
        }


class ModelLoadError(AIServiceError):
    """模型加载失败（显存不足、文件损坏、版本不兼容）"""

    def __init__(self, model_name: str, reason: str, **kwargs):
        super().__init__(
            f"模型 '{model_name}' 加载失败: {reason}",
            service="model_loader",
            **kwargs,
        )
        self.model_name = model_name
        self.reason = reason


class InferenceTimeoutError(AIServiceError):
    """推理超时"""

    def __init__(self, model: str, timeout: float, **kwargs):
        super().__init__(
            f"模型 '{model}' 推理超时 ({timeout}s)",
            service="inference",
            retry_after=timeout * 0.5,  # 建议等待一半超时时间后重试
            **kwargs,
        )
        self.model = model
        self.timeout = timeout


class RateLimitError(AIServiceError):
    """API 限流"""

    def __init__(self, api: str, retry_after: float = 60, **kwargs):
        super().__init__(
            f"API '{api}' 限流，请在 {retry_after}s 后重试",
            service=api,
            error_code="RATE_LIMITED",
            retry_after=retry_after,
            **kwargs,
        )


class VectorDBError(AIServiceError):
    """向量数据库操作失败"""

    def __init__(self, operation: str, reason: str, **kwargs):
        super().__init__(
            f"向量数据库 {operation} 失败: {reason}",
            service="vector_db",
            **kwargs,
        )
        self.operation = operation


class DocumentParseError(AIServiceError):
    """文档解析失败"""

    def __init__(self, file_path: str, reason: str, **kwargs):
        super().__init__(
            f"文档解析失败 '{file_path}': {reason}",
            service="document_parser",
            **kwargs,
        )
        self.file_path = file_path


# ============================================================
# 2. 异常链演示 — raise ... from ...
# ============================================================

def demo_exception_chaining() -> None:
    """演示异常链的三种模式。"""
    print("\n" + "=" * 60)
    print("2. 异常链 — raise ... from ...")
    print("=" * 60)

    # --- 显式链接：raise X from Y ---
    print("\n  --- 显式链接 (raise X from Y) ---")
    try:
        try:
            # 模拟底层连接错误
            raise ConnectionRefusedError("Connection refused: localhost:8000")
        except ConnectionRefusedError as e:
            # 转换为业务异常，保留原始异常链
            raise VectorDBError("connect", "Chroma 服务未启动") from e
    except VectorDBError as e:
        print(f"  捕获: {e}")
        print(f"  原始异常: {e.__cause__}")
        print(f"  类型: {type(e.__cause__).__name__}")

    # --- 隐式链接：raise X（不带 from）---
    print("\n  --- 隐式链接 (raise X) ---")
    try:
        try:
            raise KeyError("model_name")
        except KeyError:
            # 不带 from，Python 自动设置 __context__
            raise ModelLoadError("qwen2", "配置文件缺少 model_name 字段")
    except ModelLoadError as e:
        print(f"  捕获: {e}")
        print(f"  隐式上下文: {e.__context__}")

    # --- 断开链接：raise X from None ---
    print("\n  --- 断开链接 (raise X from None) ---")
    try:
        try:
            raise ValueError("内部实现细节错误")
        except ValueError:
            # from None 隐藏原始异常（不想暴露内部实现）
            raise AIServiceError("服务暂时不可用，请稍后重试") from None
    except AIServiceError as e:
        print(f"  捕获: {e}")
        print(f"  原始异常已隐藏: __cause__ = {e.__cause__}")


# ============================================================
# 3. try/except/else/finally 完整语法
# ============================================================

def demo_try_except_else_finally() -> None:
    """演示 try/except/else/finally 的执行顺序。"""
    print("\n" + "=" * 60)
    print("3. try/except/else/finally 完整语法")
    print("=" * 60)

    def process_prompt(prompt: str) -> str:
        """模拟处理用户 Prompt。"""
        if not prompt.strip():
            raise ValueError("Prompt 不能为空")
        if len(prompt) > 1000:
            raise ValueError(f"Prompt 过长: {len(prompt)} 字符（最大 1000）")
        return f"处理结果: {prompt[:50]}..."

    # --- 场景 1：正常执行 ---
    print("\n  --- 场景 1: 正常执行 ---")
    try:
        result = process_prompt("什么是 RAG？")
    except ValueError as e:
        print(f"  except: 参数错误 — {e}")
    else:
        # 仅在 try 没有异常时执行
        print(f"  else: 处理成功 — {result}")
    finally:
        print("  finally: 清理资源（无论是否异常都执行）")

    # --- 场景 2：有异常 ---
    print("\n  --- 场景 2: 有异常 ---")
    try:
        result = process_prompt("")
    except ValueError as e:
        print(f"  except: 参数错误 — {e}")
    else:
        print(f"  else: 不会执行")
    finally:
        print("  finally: 清理资源（仍然执行）")

    # --- 场景 3：多层 except 匹配 ---
    print("\n  --- 场景 3: 多层 except 匹配顺序 ---")
    errors = [
        RateLimitError("openai", retry_after=30),
        ModelLoadError("llama3", "显存不足"),
        AIServiceError("未知 AI 服务错误"),
        RuntimeError("非 AI 相关错误"),
    ]

    for err in errors:
        try:
            raise err
        except RateLimitError as e:
            # 最具体的异常放在最前面
            print(f"  限流处理: {e.retry_after}s 后重试")
        except ModelLoadError as e:
            print(f"  模型加载失败: 尝试加载量化版本 {e.model_name}-Q4")
        except AIServiceError as e:
            # 父类异常兜底
            print(f"  AI 服务异常: {e}")
        except Exception as e:
            # 最后兜底
            print(f"  未预期异常: {type(e).__name__}: {e}")


# ============================================================
# 4. 优雅降级模式
# ============================================================

def demo_graceful_degradation() -> None:
    """演示 AI 应用中的优雅降级策略。"""
    print("\n" + "=" * 60)
    print("4. 优雅降级模式")
    print("=" * 60)

    # 模拟模型列表（按优先级排序）
    model_chain = [
        ("qwen2-72b", "高精度模型"),
        ("qwen2-7b", "中等模型"),
        ("qwen2-1.5b", "轻量模型"),
    ]

    def try_load_model(name: str) -> dict:
        """模拟模型加载，前两个会失败。"""
        if name == "qwen2-72b":
            raise ModelLoadError(name, "GPU 显存不足 (需要 40GB)")
        if name == "qwen2-7b":
            raise ModelLoadError(name, "模型文件损坏")
        return {"model": name, "status": "loaded"}

    # 级联降级：依次尝试加载，直到成功
    print("\n  --- 级联降级策略 ---")
    model = None
    for name, desc in model_chain:
        try:
            model = try_load_model(name)
            print(f"  ✅ 成功加载: {name} ({desc})")
            break
        except ModelLoadError as e:
            print(f"  ⚠️ {name} 失败: {e.reason}")
            print(f"     降级到下一个模型...")

    if model is None:
        print("  ❌ 所有模型加载失败，返回缓存结果")
    else:
        print(f"  🎯 最终使用: {model['model']}")


# ============================================================
# 5. 异常信息序列化 — 用于日志和 API 响应
# ============================================================

def demo_exception_serialization() -> None:
    """演示异常信息的结构化序列化。"""
    print("\n" + "=" * 60)
    print("5. 异常信息序列化")
    print("=" * 60)

    errors = [
        RateLimitError("openai", retry_after=60),
        ModelLoadError("qwen2-72b", "显存不足"),
        InferenceTimeoutError("llama3-8b", timeout=30.0),
        VectorDBError("search", "索引未构建"),
        DocumentParseError("/data/report.pdf", "PDF 加密无法解析"),
    ]

    print("\n  结构化异常信息:")
    for err in errors:
        info = err.to_dict()
        print(f"\n  [{info['error_type']}]")
        print(f"    消息: {info['message']}")
        print(f"    服务: {info['service']}")
        if info['error_code']:
            print(f"    错误码: {info['error_code']}")
        if info['retry_after']:
            print(f"    重试等待: {info['retry_after']}s")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行所有演示。"""
    print("🐍 自定义异常 — 异常层次设计、异常链、优雅降级")
    print("=" * 60)

    # 1. 展示异常层次
    print("\n" + "=" * 60)
    print("1. 自定义异常层次")
    print("=" * 60)
    print("\n  AIServiceError（基础）")
    print("  ├── ModelLoadError（模型加载失败）")
    print("  ├── InferenceTimeoutError（推理超时）")
    print("  ├── RateLimitError（API 限流）")
    print("  ├── VectorDBError（向量数据库错误）")
    print("  └── DocumentParseError（文档解析失败）")

    demo_exception_chaining()
    demo_try_except_else_finally()
    demo_graceful_degradation()
    demo_exception_serialization()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. 按业务域设计异常层次，携带上下文信息")
    print("  2. 用 raise X from Y 保留异常链，方便排查根因")
    print("  3. except 从具体到通用排列，避免裸 except")
    print("  4. 实现级联降级策略，避免单点故障导致服务崩溃")
    print("  5. 异常信息结构化序列化，方便日志分析和监控告警")


if __name__ == "__main__":
    main()
