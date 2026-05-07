"""
智能客服系统 — FastAPI 服务

知识点：将智能客服引擎封装为 RESTful API 服务，包括对话接口、
       会话管理、健康检查、CORS 配置

Python 版本：3.11+
依赖：fastapi>=0.100.0, uvicorn>=0.23.0
最后验证：2024-12-01

启动命令：
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

# ============================================================
# 模拟 FastAPI（无需安装依赖即可运行演示）
# ============================================================

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

# 导入客服引擎
from main import CustomerServiceEngine


# ============================================================
# 请求/响应模型
# ============================================================

if HAS_FASTAPI:
    class ChatRequest(BaseModel):
        """对话请求。"""
        message: str = Field(..., min_length=1, max_length=4000, description="用户消息")
        session_id: str = Field(default="", description="会话 ID（为空则创建新会话）")
        user_id: str = Field(default="anonymous", description="用户 ID")

    class ChatResponse(BaseModel):
        """对话响应。"""
        session_id: str = Field(description="会话 ID")
        answer: str = Field(description="AI 回答")
        intent: str = Field(description="识别的意图")
        sources: list[dict[str, Any]] = Field(default_factory=list, description="参考来源")
        confidence: float = Field(description="置信度 0-1")
        latency_ms: float = Field(description="响应延迟（毫秒）")

    class HealthResponse(BaseModel):
        """健康检查响应。"""
        status: str = "healthy"
        version: str = "1.0.0"
        uptime_seconds: float = 0.0

    class SessionResponse(BaseModel):
        """会话信息响应。"""
        session_id: str
        message_count: int
        created_at: float


# ============================================================
# FastAPI 应用
# ============================================================

if HAS_FASTAPI:
    # 创建 FastAPI 应用
    app = FastAPI(
        title="智能客服 API",
        description="基于 RAG + Agent 的智能客服系统 API",
        version="1.0.0",
    )

    # CORS 配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 全局变量
    engine = CustomerServiceEngine()
    start_time = time.time()

    # ---- API 路由 ----

    @app.get("/health", response_model=HealthResponse, tags=["系统"])
    async def health_check():
        """健康检查接口。"""
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            uptime_seconds=round(time.time() - start_time, 2),
        )

    @app.post("/api/chat", response_model=ChatResponse, tags=["对话"])
    async def chat(request: ChatRequest):
        """对话接口 — 发送用户消息，返回 AI 回答。"""
        try:
            result = engine.chat(
                message=request.message,
                session_id=request.session_id,
                user_id=request.user_id,
            )
            return ChatResponse(**result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

    @app.get("/api/sessions/{session_id}", response_model=SessionResponse, tags=["会话"])
    async def get_session(session_id: str):
        """获取会话信息。"""
        session = engine.conversation_mgr.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")
        return SessionResponse(
            session_id=session.session_id,
            message_count=len(session.messages),
            created_at=session.created_at,
        )

    @app.get("/api/sessions/{session_id}/history", tags=["会话"])
    async def get_history(session_id: str, max_turns: int = 10):
        """获取对话历史。"""
        session = engine.conversation_mgr.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")
        return {"session_id": session_id, "history": session.get_history(max_turns)}


# ============================================================
# 模拟 API 测试（无 FastAPI 时）
# ============================================================

def demo_api_simulation() -> None:
    """模拟 API 调用演示。"""
    print("智能客服 API 服务 — 模拟演示")
    print("=" * 60)

    engine = CustomerServiceEngine()

    # 模拟 API 请求
    requests = [
        {"message": "你好", "session_id": "", "user_id": "user_001"},
        {"message": "如何重置密码？", "session_id": "", "user_id": "user_001"},
        {"message": "帮我创建工单", "session_id": "", "user_id": "user_002"},
    ]

    print("\n  模拟 API 调用:")
    for req in requests:
        result = engine.chat(**req)
        print(f"\n  POST /api/chat")
        print(f"  请求: {{'message': '{req['message']}', 'user_id': '{req['user_id']}'}}")
        print(f"  响应: {{")
        print(f"    'session_id': '{result['session_id']}',")
        print(f"    'answer': '{result['answer'][:60]}...',")
        print(f"    'intent': '{result['intent']}',")
        print(f"    'confidence': {result['confidence']},")
        print(f"    'latency_ms': {result['latency_ms']}")
        print(f"  }}")

    print(f"\n  GET /health")
    print(f"  响应: {{'status': 'healthy', 'version': '1.0.0'}}")

    if HAS_FASTAPI:
        print(f"\n  💡 启动真实 API 服务:")
        print(f"     uvicorn api:app --host 0.0.0.0 --port 8000 --reload")
        print(f"     访问 http://localhost:8000/docs 查看 API 文档")
    else:
        print(f"\n  💡 安装 FastAPI 启动真实服务:")
        print(f"     pip install fastapi uvicorn")
        print(f"     uvicorn api:app --host 0.0.0.0 --port 8000 --reload")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    if HAS_FASTAPI:
        import uvicorn
        print("启动智能客服 API 服务...")
        print("API 文档: http://localhost:8000/docs")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        demo_api_simulation()
