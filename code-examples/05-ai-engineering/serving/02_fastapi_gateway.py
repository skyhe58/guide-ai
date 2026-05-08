"""
FastAPI API 网关模拟

知识点：API 网关设计、认证鉴权、速率限制、请求路由、
       缓存层、日志中间件、健康检查、优雅降级

Python 版本：3.11+
依赖：标准库（默认模式）、fastapi>=0.100, uvicorn>=0.20（服务模式）
最后验证：2024-12-01

外部服务（可选）：
  启动命令：uvicorn 02_fastapi_gateway:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field

# ============================================================
# 1. 请求/响应数据结构
# ============================================================

@dataclass
class ChatMessage:
    """聊天消息"""
    role: str       # system / user / assistant
    content: str


@dataclass
class ChatRequest:
    """聊天请求"""
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = False
    user: str | None = None

    def validate(self) -> list[str]:
        """验证请求"""
        errors = []
        if not self.messages:
            errors.append("messages 不能为空")
        if not 0 <= self.temperature <= 2:
            errors.append(f"temperature 应在 [0, 2]，当前: {self.temperature}")
        if self.max_tokens < 1 or self.max_tokens > 4096:
            errors.append(f"max_tokens 应在 [1, 4096]，当前: {self.max_tokens}")
        for msg in self.messages:
            if msg.role not in ("system", "user", "assistant"):
                errors.append(f"无效的角色: {msg.role}")
            if len(msg.content) > 10000:
                errors.append("消息内容过长（最大 10000 字符）")
        return errors


@dataclass
class ChatResponse:
    """聊天响应"""
    id: str
    model: str
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cached: bool = False


@dataclass
class ErrorResponse:
    """错误响应"""
    error: str
    code: int
    message: str


# ============================================================
# 2. API Key 认证
# ============================================================

@dataclass
class APIKeyInfo:
    """API Key 信息"""
    key: str
    user_id: str
    tier: str           # free / pro / enterprise
    rate_limit_rpm: int  # 每分钟请求数限制
    rate_limit_tpm: int  # 每分钟 Token 数限制
    allowed_models: list[str] = field(default_factory=list)


class APIKeyManager:
    """API Key 管理器"""

    def __init__(self):
        # 预设 API Key（生产环境应使用数据库）
        self.keys: dict[str, APIKeyInfo] = {
            "sk-free-001": APIKeyInfo(
                key="sk-free-001", user_id="user_free",
                tier="free", rate_limit_rpm=10, rate_limit_tpm=10000,
                allowed_models=["qwen2-1.5b"],
            ),
            "sk-pro-001": APIKeyInfo(
                key="sk-pro-001", user_id="user_pro",
                tier="pro", rate_limit_rpm=60, rate_limit_tpm=100000,
                allowed_models=["qwen2-7b", "qwen2-1.5b"],
            ),
            "sk-enterprise-001": APIKeyInfo(
                key="sk-enterprise-001", user_id="user_enterprise",
                tier="enterprise", rate_limit_rpm=600, rate_limit_tpm=1000000,
                allowed_models=["qwen2-72b", "qwen2-7b", "qwen2-1.5b"],
            ),
        }

    def verify(self, api_key: str) -> APIKeyInfo | None:
        """验证 API Key"""
        # 去除 Bearer 前缀
        key = api_key.replace("Bearer ", "").strip()
        return self.keys.get(key)


# ============================================================
# 3. 速率限制器
# ============================================================

class RateLimiter:
    """
    令牌桶速率限制器

    支持按用户限制 RPM（每分钟请求数）和 TPM（每分钟 Token 数）。
    """

    def __init__(self):
        # 请求计数：user_id -> [(timestamp, count)]
        self.request_counts: dict[str, list[float]] = defaultdict(list)
        # Token 计数：user_id -> [(timestamp, tokens)]
        self.token_counts: dict[str, list[tuple[float, int]]] = defaultdict(list)

    def check_rpm(self, user_id: str, limit: int) -> tuple[bool, dict]:
        """检查 RPM 限制"""
        now = time.time()
        window = 60  # 1 分钟窗口

        # 清理过期记录
        self.request_counts[user_id] = [
            ts for ts in self.request_counts[user_id]
            if now - ts < window
        ]

        current_count = len(self.request_counts[user_id])
        remaining = max(0, limit - current_count)

        if current_count >= limit:
            return False, {
                "limit": limit,
                "remaining": 0,
                "reset_after_seconds": round(
                    window - (now - self.request_counts[user_id][0]), 1
                ),
            }

        self.request_counts[user_id].append(now)
        return True, {"limit": limit, "remaining": remaining - 1}

    def check_tpm(self, user_id: str, tokens: int, limit: int) -> tuple[bool, dict]:
        """检查 TPM 限制"""
        now = time.time()
        window = 60

        # 清理过期记录
        self.token_counts[user_id] = [
            (ts, t) for ts, t in self.token_counts[user_id]
            if now - ts < window
        ]

        current_tokens = sum(t for _, t in self.token_counts[user_id])
        remaining = max(0, limit - current_tokens)

        if current_tokens + tokens > limit:
            return False, {"limit": limit, "used": current_tokens, "remaining": remaining}

        self.token_counts[user_id].append((now, tokens))
        return True, {"limit": limit, "remaining": remaining - tokens}


# ============================================================
# 4. 缓存层
# ============================================================

class ResponseCache:
    """响应缓存（精确匹配）"""

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: dict[str, tuple[float, ChatResponse]] = {}
        self.hits = 0
        self.misses = 0

    def _make_key(self, request: ChatRequest) -> str:
        """生成缓存键"""
        # 只缓存确定性请求（temperature=0）
        if request.temperature > 0:
            return ""
        content = json.dumps({
            "model": request.model,
            "messages": [(m.role, m.content) for m in request.messages],
            "max_tokens": request.max_tokens,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, request: ChatRequest) -> ChatResponse | None:
        """查询缓存"""
        key = self._make_key(request)
        if not key:
            self.misses += 1
            return None

        if key in self.cache:
            timestamp, response = self.cache[key]
            if time.time() - timestamp < self.ttl:
                self.hits += 1
                response.cached = True
                return response
            else:
                del self.cache[key]

        self.misses += 1
        return None

    def set(self, request: ChatRequest, response: ChatResponse) -> None:
        """设置缓存"""
        key = self._make_key(request)
        if not key:
            return

        # LRU 淘汰
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache, key=lambda k: self.cache[k][0])
            del self.cache[oldest_key]

        self.cache[key] = (time.time(), response)

    def get_stats(self) -> dict:
        """获取缓存统计"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 4),
        }


# ============================================================
# 5. 模型路由器
# ============================================================

@dataclass
class BackendConfig:
    """后端配置"""
    name: str
    url: str
    model: str
    weight: int = 1
    healthy: bool = True
    max_concurrent: int = 100
    current_concurrent: int = 0


class ModelRouter:
    """模型路由器"""

    def __init__(self):
        self.backends: dict[str, list[BackendConfig]] = {
            "qwen2-72b": [
                BackendConfig("vllm-72b-1", "http://vllm-72b-1:8000", "qwen2-72b", weight=1),
            ],
            "qwen2-7b": [
                BackendConfig("vllm-7b-1", "http://vllm-7b-1:8000", "qwen2-7b", weight=2),
                BackendConfig("vllm-7b-2", "http://vllm-7b-2:8000", "qwen2-7b", weight=1),
            ],
            "qwen2-1.5b": [
                BackendConfig("vllm-1.5b-1", "http://vllm-1.5b-1:8000", "qwen2-1.5b", weight=1),
            ],
        }

    def route(self, model: str) -> BackendConfig | None:
        """路由到后端"""
        backends = self.backends.get(model, [])
        healthy_backends = [b for b in backends if b.healthy]

        if not healthy_backends:
            return None

        # 加权最少连接
        best = min(
            healthy_backends,
            key=lambda b: b.current_concurrent / b.weight,
        )
        return best

    def get_available_models(self) -> list[str]:
        """获取可用模型列表"""
        return [
            model for model, backends in self.backends.items()
            if any(b.healthy for b in backends)
        ]


# ============================================================
# 6. API 网关
# ============================================================

class APIGateway:
    """
    LLM API 网关

    整合认证、限流、缓存、路由等功能，
    提供统一的 API 入口。
    """

    def __init__(self):
        self.key_manager = APIKeyManager()
        self.rate_limiter = RateLimiter()
        self.cache = ResponseCache()
        self.router = ModelRouter()
        self.request_log: list[dict] = []

        print("[Gateway] API 网关初始化完成")
        print(f"[Gateway] 可用模型: {self.router.get_available_models()}")

    def handle_request(
        self,
        api_key: str,
        request: ChatRequest,
    ) -> ChatResponse | ErrorResponse:
        """处理请求"""
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        # 1. 认证
        key_info = self.key_manager.verify(api_key)
        if key_info is None:
            return ErrorResponse("unauthorized", 401, "无效的 API Key")

        # 2. 模型权限检查
        if request.model not in key_info.allowed_models:
            return ErrorResponse(
                "forbidden", 403,
                f"当前套餐不支持模型 {request.model}，"
                f"可用模型: {key_info.allowed_models}",
            )

        # 3. 请求验证
        errors = request.validate()
        if errors:
            return ErrorResponse("bad_request", 400, "; ".join(errors))

        # 4. 速率限制
        rpm_ok, rpm_info = self.rate_limiter.check_rpm(
            key_info.user_id, key_info.rate_limit_rpm
        )
        if not rpm_ok:
            return ErrorResponse(
                "rate_limit_exceeded", 429,
                f"请求频率超限 (RPM: {key_info.rate_limit_rpm})",
            )

        # 5. 缓存查询
        cached_response = self.cache.get(request)
        if cached_response:
            self._log_request(request_id, key_info, request, cached_response, True)
            return cached_response

        # 6. 路由到后端
        backend = self.router.route(request.model)
        if backend is None:
            return ErrorResponse("service_unavailable", 503, f"模型 {request.model} 不可用")

        # 7. 模拟推理
        response = self._simulate_inference(request_id, request, backend)

        # 8. 缓存响应
        self.cache.set(request, response)

        # 9. 记录日志
        latency = (time.time() - start_time) * 1000
        response.latency_ms = round(latency, 1)
        self._log_request(request_id, key_info, request, response, False)

        return response

    def _simulate_inference(
        self,
        request_id: str,
        request: ChatRequest,
        backend: BackendConfig,
    ) -> ChatResponse:
        """模拟推理调用"""
        # 模拟 token 计数
        prompt_tokens = sum(len(m.content) // 2 for m in request.messages)
        completion_tokens = min(request.max_tokens, 50 + len(request.messages) * 20)

        content = f"这是来自 {backend.name} 的模拟回答。"

        return ChatResponse(
            id=f"chatcmpl-{request_id}",
            model=request.model,
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=0,
        )

    def _log_request(
        self,
        request_id: str,
        key_info: APIKeyInfo,
        request: ChatRequest,
        response: ChatResponse,
        cached: bool,
    ) -> None:
        """记录请求日志"""
        self.request_log.append({
            "request_id": request_id,
            "user_id": key_info.user_id,
            "tier": key_info.tier,
            "model": request.model,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "latency_ms": response.latency_ms,
            "cached": cached,
            "timestamp": time.time(),
        })

    def get_stats(self) -> dict:
        """获取网关统计"""
        total_requests = len(self.request_log)
        cached_requests = sum(1 for r in self.request_log if r["cached"])
        total_tokens = sum(r["total_tokens"] for r in self.request_log if "total_tokens" in r)

        return {
            "total_requests": total_requests,
            "cached_requests": cached_requests,
            "cache_stats": self.cache.get_stats(),
            "total_tokens": total_tokens,
            "available_models": self.router.get_available_models(),
        }


# ============================================================
# 7. 主程序入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FastAPI API 网关模拟演示")
    print("=" * 60)

    gateway = APIGateway()

    # --- 演示 1: 正常请求 ---
    print("\n--- 正常请求 ---")
    request = ChatRequest(
        model="qwen2-7b",
        messages=[
            ChatMessage("system", "你是一个有帮助的助手"),
            ChatMessage("user", "什么是 RAG？"),
        ],
        temperature=0,
    )
    response = gateway.handle_request("sk-pro-001", request)
    if isinstance(response, ChatResponse):
        print(f"  响应: {response.content}")
        print(f"  Token: {response.total_tokens}, 延迟: {response.latency_ms}ms")
    else:
        print(f"  错误: {response.message}")

    # --- 演示 2: 缓存命中 ---
    print("\n--- 缓存命中 ---")
    response2 = gateway.handle_request("sk-pro-001", request)
    if isinstance(response2, ChatResponse):
        print(f"  缓存命中: {response2.cached}")

    # --- 演示 3: 认证失败 ---
    print("\n--- 认证失败 ---")
    response3 = gateway.handle_request("invalid-key", request)
    if isinstance(response3, ErrorResponse):
        print(f"  错误 {response3.code}: {response3.message}")

    # --- 演示 4: 权限不足 ---
    print("\n--- 权限不足 ---")
    request_72b = ChatRequest(
        model="qwen2-72b",
        messages=[ChatMessage("user", "你好")],
    )
    response4 = gateway.handle_request("sk-free-001", request_72b)
    if isinstance(response4, ErrorResponse):
        print(f"  错误 {response4.code}: {response4.message}")

    # --- 演示 5: 速率限制 ---
    print("\n--- 速率限制测试 ---")
    for i in range(12):
        req = ChatRequest(
            model="qwen2-1.5b",
            messages=[ChatMessage("user", f"测试 {i}")],
            temperature=0.7,
        )
        resp = gateway.handle_request("sk-free-001", req)
        status = "✅" if isinstance(resp, ChatResponse) else f"❌ {resp.message}"
        print(f"  请求 {i+1}: {status}")

    # --- 统计信息 ---
    print(f"\n--- 网关统计 ---")
    stats = gateway.get_stats()
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    print("\n✅ FastAPI API 网关模拟演示完成！")
