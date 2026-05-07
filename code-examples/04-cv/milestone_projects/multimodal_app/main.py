"""
多模态图文理解应用 — 里程碑项目

项目说明：LLaVA/Qwen-VL 图文理解 + API 服务
功能：图像描述、视觉问答、OCR、多轮对话、多模型切换

知识点：多模态模型服务化、图文对话管理、
       多模型后端切换、流式输出、会话管理

Python 版本：3.11+
依赖：numpy>=1.24, pydantic>=2.0（模拟模式）
真实环境依赖：
  pip install fastapi uvicorn transformers torch Pillow
  pip install ollama  # Ollama 后端
最后验证：2024-12-01

运行方式（模拟）：
  python main.py
真实运行：
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import json
import time
import uuid
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Generator
from enum import Enum


# ============================================================
# 1. 数据模型
# ============================================================

class ModelBackend(Enum):
    """模型后端。"""
    OLLAMA = "ollama"
    TRANSFORMERS = "transformers"
    API = "api"


@dataclass
class ChatMessage:
    """对话消息。"""
    role: str           # user / assistant / system
    content: str
    image: np.ndarray | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "has_image": self.image is not None,
            "timestamp": self.timestamp,
        }


@dataclass
class ChatSession:
    """对话会话。"""
    session_id: str
    model: str
    messages: list[ChatMessage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def add_message(self, message: ChatMessage) -> None:
        self.messages.append(message)

    @property
    def turn_count(self) -> int:
        return len([m for m in self.messages if m.role == "user"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "model": self.model,
            "turn_count": self.turn_count,
            "message_count": len(self.messages),
            "created_at": self.created_at,
        }


@dataclass
class ChatRequest:
    """对话请求。"""
    message: str
    image: np.ndarray | None = None
    session_id: str | None = None
    model: str = "llava"
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False


@dataclass
class ChatResponse:
    """对话响应。"""
    session_id: str
    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    elapsed_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "content": self.content,
            "model": self.model,
            "usage": self.usage,
            "elapsed_time_ms": round(self.elapsed_time * 1000, 2),
        }


# ============================================================
# 2. 模拟多模态模型后端
# ============================================================

class MockMultimodalBackend:
    """模拟多模态模型后端。"""

    # 模拟回复库
    RESPONSES = {
        "describe": {
            "llava": "这张图片展示了一个充满活力的场景。画面中央有一个主要物体，"
                     "周围环绕着丰富的背景元素。整体色调温暖，光线柔和。",
            "qwen-vl": "图片内容丰富，主体清晰可辨。从构图来看，采用了经典的三分法则，"
                       "前景和背景层次分明。色彩搭配和谐自然。",
        },
        "ocr": {
            "llava": "图片中的文字内容为：'Hello World 2024'，使用白色字体，"
                     "位于图片中央偏上位置。",
            "qwen-vl": "识别到以下文字：'Hello World 2024'。字体为无衬线体，"
                       "白色，居中显示。文字清晰可读。",
        },
        "qa": {
            "llava": "根据图片内容分析，这个问题的答案是：图片中展示了一个典型的场景，"
                     "包含了多个视觉元素。",
            "qwen-vl": "基于对图片的理解，我认为：画面中的主要元素表明这是一个特定的场景，"
                       "具有明确的主题和意义。",
        },
        "default": {
            "llava": "我已经仔细观察了这张图片。这是一个视觉内容丰富的场景，"
                     "包含了多个值得注意的元素。",
            "qwen-vl": "经过分析，这张图片展示了一个有趣的视觉场景。"
                       "画面中的各个元素相互配合，形成了和谐的整体。",
        },
    }

    AVAILABLE_MODELS = {
        "llava": {
            "name": "LLaVA 1.6",
            "backend": "ollama",
            "description": "开源多模态模型，本地部署",
            "speed": "中等",
        },
        "qwen-vl": {
            "name": "Qwen-VL-Max",
            "backend": "api",
            "description": "通义千问视觉模型，中文最强",
            "speed": "快",
        },
        "gpt-4o": {
            "name": "GPT-4o",
            "backend": "api",
            "description": "OpenAI 旗舰多模态模型",
            "speed": "中等",
        },
    }

    def __init__(self) -> None:
        self.loaded_models: set[str] = set()

    def load_model(self, model_name: str) -> bool:
        """加载模型。"""
        if model_name in self.AVAILABLE_MODELS:
            self.loaded_models.add(model_name)
            print(f"  ✅ 模型加载: {model_name}")
            return True
        print(f"  ❌ 未知模型: {model_name}")
        return False

    def generate(self, model: str, prompt: str,
                 image: np.ndarray | None = None,
                 max_tokens: int = 512) -> tuple[str, dict[str, int]]:
        """生成回复。"""
        # 选择回复类型
        p_lower = prompt.lower()
        if any(kw in p_lower for kw in ["描述", "describe", "看到"]):
            category = "describe"
        elif any(kw in p_lower for kw in ["文字", "ocr", "识别"]):
            category = "ocr"
        elif "?" in prompt or "？" in prompt:
            category = "qa"
        else:
            category = "default"

        model_key = model if model in ("llava", "qwen-vl") else "llava"
        response = self.RESPONSES.get(category, self.RESPONSES["default"])
        text = response.get(model_key, response["llava"])

        # 模拟 token 统计
        prompt_tokens = len(prompt) * 2
        image_tokens = 576 if image is not None else 0
        completion_tokens = len(text)
        usage = {
            "prompt_tokens": prompt_tokens,
            "image_tokens": image_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + image_tokens + completion_tokens,
        }

        return text, usage

    def stream_generate(self, model: str, prompt: str,
                        image: np.ndarray | None = None) -> Generator[str, None, None]:
        """流式生成回复。"""
        full_text, _ = self.generate(model, prompt, image)
        # 模拟逐字输出
        for char in full_text:
            yield char
            time.sleep(0.01)


# ============================================================
# 3. 会话管理器
# ============================================================

class SessionManager:
    """对话会话管理器。"""

    def __init__(self, max_sessions: int = 100) -> None:
        self.sessions: dict[str, ChatSession] = {}
        self.max_sessions = max_sessions

    def create_session(self, model: str = "llava") -> ChatSession:
        """创建新会话。"""
        session_id = str(uuid.uuid4())[:8]
        session = ChatSession(session_id=session_id, model=model)
        self.sessions[session_id] = session

        # 清理过期会话
        if len(self.sessions) > self.max_sessions:
            self._cleanup()

        print(f"  📝 创建会话: {session_id} (model={model})")
        return session

    def get_session(self, session_id: str) -> ChatSession | None:
        """获取会话。"""
        return self.sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """删除会话。"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def list_sessions(self) -> list[dict[str, Any]]:
        """列出所有会话。"""
        return [s.to_dict() for s in self.sessions.values()]

    def _cleanup(self) -> None:
        """清理最旧的会话。"""
        sorted_sessions = sorted(
            self.sessions.items(),
            key=lambda x: x[1].created_at,
        )
        # 删除最旧的 20%
        to_remove = len(sorted_sessions) // 5
        for sid, _ in sorted_sessions[:to_remove]:
            del self.sessions[sid]


# ============================================================
# 4. 多模态应用服务
# ============================================================

class MultimodalApp:
    """多模态图文理解应用。

    真实 FastAPI 实现：
        @app.post("/chat")
        async def chat(request: ChatRequest):
            session = session_manager.get_or_create(request.session_id)
            response = model.generate(request.message, request.image)
            session.add_message(response)
            return response
    """

    def __init__(self) -> None:
        self.backend = MockMultimodalBackend()
        self.session_manager = SessionManager()
        self.start_time = time.time()
        self.request_count = 0
        print(f"\n  🚀 多模态应用初始化")

    def startup(self) -> None:
        """启动服务。"""
        self.backend.load_model("llava")
        self.backend.load_model("qwen-vl")
        print(f"  ✅ 服务就绪")

    def chat(self, request: ChatRequest) -> ChatResponse:
        """处理对话请求。"""
        start = time.time()
        self.request_count += 1

        # 获取或创建会话
        if request.session_id:
            session = self.session_manager.get_session(request.session_id)
            if not session:
                session = self.session_manager.create_session(request.model)
        else:
            session = self.session_manager.create_session(request.model)

        # 添加用户消息
        user_msg = ChatMessage(
            role="user",
            content=request.message,
            image=request.image,
        )
        session.add_message(user_msg)

        # 生成回复
        response_text, usage = self.backend.generate(
            model=request.model,
            prompt=request.message,
            image=request.image,
            max_tokens=request.max_tokens,
        )

        # 添加助手消息
        assistant_msg = ChatMessage(role="assistant", content=response_text)
        session.add_message(assistant_msg)

        elapsed = time.time() - start + np.random.uniform(0.5, 2.0)

        return ChatResponse(
            session_id=session.session_id,
            content=response_text,
            model=request.model,
            usage=usage,
            elapsed_time=elapsed,
        )

    def describe_image(self, image: np.ndarray,
                       model: str = "llava") -> ChatResponse:
        """图像描述快捷接口。"""
        request = ChatRequest(
            message="请详细描述这张图片的内容",
            image=image,
            model=model,
        )
        return self.chat(request)

    def visual_qa(self, image: np.ndarray, question: str,
                  model: str = "llava") -> ChatResponse:
        """视觉问答快捷接口。"""
        request = ChatRequest(
            message=question,
            image=image,
            model=model,
        )
        return self.chat(request)

    def ocr(self, image: np.ndarray,
             model: str = "qwen-vl") -> ChatResponse:
        """OCR 文字识别快捷接口。"""
        request = ChatRequest(
            message="请识别并提取图片中的所有文字内容",
            image=image,
            model=model,
        )
        return self.chat(request)

    def list_models(self) -> dict[str, Any]:
        """列出可用模型。"""
        return self.backend.AVAILABLE_MODELS

    def get_stats(self) -> dict[str, Any]:
        """获取服务统计。"""
        return {
            "uptime_seconds": round(time.time() - self.start_time, 1),
            "total_requests": self.request_count,
            "active_sessions": len(self.session_manager.sessions),
            "loaded_models": list(self.backend.loaded_models),
        }


# ============================================================
# 5. API 路由文档
# ============================================================

API_ROUTES = [
    ("POST", "/chat", "图文对话", "发送文本+图像，获取模型回复"),
    ("POST", "/describe", "图像描述", "上传图像，获取详细描述"),
    ("POST", "/vqa", "视觉问答", "上传图像+问题，获取答案"),
    ("POST", "/ocr", "文字识别", "上传图像，提取文字内容"),
    ("GET", "/models", "模型列表", "获取可用模型列表"),
    ("GET", "/sessions", "会话列表", "获取活跃会话列表"),
    ("DELETE", "/sessions/{id}", "删除会话", "删除指定会话"),
    ("GET", "/stats", "服务统计", "获取服务运行统计"),
    ("GET", "/health", "健康检查", "服务健康状态"),
]


# ============================================================
# 6. 演示函数
# ============================================================

def demo_startup() -> MultimodalApp:
    """演示服务启动。"""
    print("\n" + "=" * 60)
    print("1. 服务启动")
    print("=" * 60)

    app = MultimodalApp()
    app.startup()
    return app


def demo_image_description(app: MultimodalApp) -> None:
    """演示图像描述。"""
    print("\n" + "=" * 60)
    print("2. 图像描述")
    print("=" * 60)

    image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

    for model in ["llava", "qwen-vl"]:
        resp = app.describe_image(image, model=model)
        print(f"\n  📷 {model} 描述:")
        print(f"     {resp.content[:80]}...")
        print(f"     {resp.to_dict()['elapsed_time_ms']:.0f}ms, "
              f"tokens={resp.usage.get('total_tokens', 0)}")


def demo_visual_qa(app: MultimodalApp) -> None:
    """演示视觉问答。"""
    print("\n" + "=" * 60)
    print("3. 视觉问答（VQA）")
    print("=" * 60)

    image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

    questions = [
        "图片中有什么物体？",
        "这张图片的主题是什么？",
        "图片中的颜色搭配如何？",
    ]

    for q in questions:
        resp = app.visual_qa(image, q)
        print(f"\n  ❓ {q}")
        print(f"  💬 {resp.content[:60]}...")


def demo_ocr(app: MultimodalApp) -> None:
    """演示 OCR。"""
    print("\n" + "=" * 60)
    print("4. OCR 文字识别")
    print("=" * 60)

    image = np.random.randint(0, 256, (256, 512, 3), dtype=np.uint8)
    resp = app.ocr(image)
    print(f"\n  📝 OCR 结果: {resp.content[:80]}...")


def demo_multi_turn(app: MultimodalApp) -> None:
    """演示多轮对话。"""
    print("\n" + "=" * 60)
    print("5. 多轮图文对话")
    print("=" * 60)

    image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

    # 第一轮：带图像
    req1 = ChatRequest(message="描述这张图片", image=image, model="llava")
    resp1 = app.chat(req1)
    session_id = resp1.session_id
    print(f"\n  [轮次 1] 用户: 描述这张图片")
    print(f"  [轮次 1] 助手: {resp1.content[:60]}...")

    # 第二轮：基于上下文
    req2 = ChatRequest(message="图片中有什么特别的地方？",
                       session_id=session_id, model="llava")
    resp2 = app.chat(req2)
    print(f"\n  [轮次 2] 用户: 图片中有什么特别的地方？")
    print(f"  [轮次 2] 助手: {resp2.content[:60]}...")

    # 查看会话信息
    session = app.session_manager.get_session(session_id)
    if session:
        print(f"\n  会话信息: {session.to_dict()}")


def demo_model_comparison(app: MultimodalApp) -> None:
    """演示多模型对比。"""
    print("\n" + "=" * 60)
    print("6. 多模型对比")
    print("=" * 60)

    models = app.list_models()
    print(f"\n  可用模型:")
    for name, info in models.items():
        print(f"    {name}: {info['name']} ({info['backend']}) — {info['description']}")

    # 同一图像不同模型对比
    image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    print(f"\n  同一图像多模型对比:")
    for model_name in ["llava", "qwen-vl"]:
        resp = app.describe_image(image, model=model_name)
        print(f"    {model_name}: {resp.content[:50]}... ({resp.elapsed_time*1000:.0f}ms)")


def demo_stats(app: MultimodalApp) -> None:
    """演示服务统计。"""
    print("\n" + "=" * 60)
    print("7. 服务统计")
    print("=" * 60)

    stats = app.get_stats()
    print(f"\n  服务统计:")
    for key, value in stats.items():
        print(f"    {key}: {value}")

    # API 路由
    print(f"\n  API 路由:")
    for method, path, name, desc in API_ROUTES:
        print(f"    {method:<7} {path:<25} — {name}: {desc}")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行多模态应用演示。"""
    print("🐍 多模态图文理解应用 — 里程碑项目")
    print("=" * 60)

    app = demo_startup()
    demo_image_description(app)
    demo_visual_qa(app)
    demo_ocr(app)
    demo_multi_turn(app)
    demo_model_comparison(app)
    demo_stats(app)

    print("\n" + "=" * 60)
    print("✅ 里程碑项目演示完成！")
    print("\n💡 项目要点:")
    print("  1. 统一接口支持多模型后端（Ollama/API/Transformers）")
    print("  2. 会话管理支持多轮图文对话")
    print("  3. 快捷接口: 图像描述/VQA/OCR")
    print("  4. 多模型对比帮助选择最佳模型")
    print("  5. 流式输出提升用户体验")
    print("  6. 生产部署: FastAPI + Redis(会话) + Nginx")


if __name__ == "__main__":
    main()
