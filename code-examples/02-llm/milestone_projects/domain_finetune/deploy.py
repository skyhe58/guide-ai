"""
部署为 API 服务 — 微调模型部署

知识点：FastAPI 服务、Ollama 后端、流式输出、
       健康检查、错误处理

Python 版本：3.11+
依赖：无额外依赖（演示模式）
可选依赖：fastapi>=0.104, uvicorn>=0.24, requests>=2.31
最后验证：2024-12-01

⚠️ 部署前提：
  1. 微调完成，导出 GGUF 模型
  2. 导入 Ollama: ollama create my-model -f Modelfile
  3. 启动 Ollama: ollama serve

⚠️ 运行方式：
  pip install fastapi uvicorn requests
  python deploy.py
  # 或: uvicorn deploy:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

from dataclasses import dataclass

# ============================================================
# 1. 服务配置
# ============================================================

@dataclass
class ServiceConfig:
    """API 服务配置。"""
    ollama_url: str = "http://localhost:11434"
    model_name: str = "qwen2:7b"  # 替换为你的微调模型
    max_tokens: int = 1024
    temperature: float = 0.7
    host: str = "0.0.0.0"
    port: int = 8080


# ============================================================
# 2. FastAPI 服务（伪代码 + 可运行框架）
# ============================================================

def create_app_code() -> str:
    """生成 FastAPI 服务代码。"""
    return '''
# ===== FastAPI 服务完整代码 =====
# 文件: api_server.py
# 运行: uvicorn api_server:app --host 0.0.0.0 --port 8080

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import requests
import json

app = FastAPI(title="微调模型 API", version="1.0.0")

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "qwen2:7b"  # 替换为你的微调模型名


class ChatRequest(BaseModel):
    """聊天请求。"""
    message: str = Field(min_length=1, max_length=4096)
    system_prompt: str = Field(default="你是一个专业的 Python 编程助手。")
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    stream: bool = Field(default=False)


class ChatResponse(BaseModel):
    """聊天回复。"""
    response: str
    model: str
    latency_ms: float


@app.get("/health")
async def health_check():
    """健康检查。"""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        return {"status": "healthy", "models": models}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama 不可用: {e}")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """聊天接口。"""
    import time
    start = time.perf_counter()

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": req.system_prompt},
            {"role": "user", "content": req.message},
        ],
        "stream": False,
        "options": {
            "temperature": req.temperature,
            "num_predict": req.max_tokens,
        },
    }

    try:
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        latency = (time.perf_counter() - start) * 1000

        return ChatResponse(
            response=data["message"]["content"],
            model=MODEL_NAME,
            latency_ms=round(latency, 1),
        )
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Ollama 服务未启动")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """流式聊天接口。"""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": req.system_prompt},
            {"role": "user", "content": req.message},
        ],
        "stream": True,
    }

    def generate():
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=120)
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                content = data.get("message", {}).get("content", "")
                if content:
                    yield f"data: {json.dumps({'content': content})}\\n\\n"
        yield "data: [DONE]\\n\\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
'''


# ============================================================
# 3. 部署方案对比
# ============================================================

def show_deployment_options() -> None:
    """展示部署方案对比。"""
    print("\n" + "=" * 60)
    print("2. 部署方案对比")
    print("=" * 60)

    print("""
    ┌──────────────┬──────────────┬──────────────┬──────────────┐
    │ 方案         │ 适用场景     │ 性能         │ 复杂度       │
    ├──────────────┼──────────────┼──────────────┼──────────────┤
    │ Ollama+FastAPI│ 个人/小团队 │ 中等         │ ⭐           │
    │ vLLM         │ 生产环境     │ 高           │ ⭐⭐         │
    │ TGI          │ 生产环境     │ 高           │ ⭐⭐         │
    │ llama.cpp    │ 边缘设备     │ 中等         │ ⭐           │
    │ Triton       │ 大规模服务   │ 最高         │ ⭐⭐⭐       │
    └──────────────┴──────────────┴──────────────┴──────────────┘

    推荐路径：
    1. 开发测试: Ollama + FastAPI（本脚本）
    2. 小规模生产: vLLM + Docker
    3. 大规模生产: vLLM + Kubernetes + 负载均衡
    """)


# ============================================================
# 4. Docker 部署配置
# ============================================================

def show_docker_config() -> None:
    """展示 Docker 部署配置。"""
    print("\n" + "=" * 60)
    print("3. Docker 部署配置")
    print("=" * 60)

    dockerfile = """
    # ===== Dockerfile =====
    FROM python:3.11-slim

    WORKDIR /app
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt

    COPY api_server.py .

    EXPOSE 8080
    CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8080"]


    # ===== docker-compose.yml =====
    version: "3.8"
    services:
      ollama:
        image: ollama/ollama
        ports:
          - "11434:11434"
        volumes:
          - ollama_data:/root/.ollama
        deploy:
          resources:
            reservations:
              devices:
                - capabilities: [gpu]

      api:
        build: .
        ports:
          - "8080:8080"
        depends_on:
          - ollama
        environment:
          - OLLAMA_URL=http://ollama:11434

    volumes:
      ollama_data:
    """
    print(dockerfile)


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行部署演示。"""
    print("🚀 微调模型部署为 API 服务")
    print("=" * 60)

    # 1. 展示 FastAPI 代码
    print("\n1. FastAPI 服务代码")
    print("=" * 60)
    print(create_app_code())

    # 2. 部署方案对比
    show_deployment_options()

    # 3. Docker 配置
    show_docker_config()

    print("\n" + "=" * 60)
    print("✅ 演示完成！")
    print("\n💡 快速开始:")
    print("  1. 安装: pip install fastapi uvicorn requests")
    print("  2. 启动 Ollama: ollama serve")
    print("  3. 运行服务: uvicorn api_server:app --port 8080")
    print("  4. 测试: curl -X POST http://localhost:8080/chat \\")
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"message": "什么是 LoRA？"}\'')


if __name__ == "__main__":
    main()
