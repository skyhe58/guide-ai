"""
Ollama API 调用 — 本地 LLM 部署与使用

知识点：Ollama 安装部署、REST API 调用、模型管理、
       Modelfile 自定义、流式输出

Python 版本：3.11+
依赖：requests>=2.31
最后验证：2024-12-01

⚠️ Docker 启动命令：
  # CPU 模式
  docker run -d -v ollama:/root/.ollama -p 11434:11434 \\
    --name ollama ollama/ollama

  # GPU 模式（NVIDIA）
  docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 \\
    --name ollama ollama/ollama

  # 拉取模型
  docker exec -it ollama ollama pull qwen2:7b

⚠️ 本地安装（macOS/Linux/Windows）：
  curl -fsSL https://ollama.com/install.sh | sh
  ollama pull qwen2:7b
  ollama serve  # 启动服务
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class OllamaConfig:
    """Ollama 服务配置。"""
    base_url: str = "http://localhost:11434"
    model: str = "qwen2:7b"


# ============================================================
# 1. Ollama REST API 调用
# ============================================================

def ollama_chat(
    messages: list[dict[str, str]],
    config: OllamaConfig | None = None,
    stream: bool = False,
) -> str | None:
    """调用 Ollama Chat API。

    Args:
        messages: 对话消息列表
        config: Ollama 配置
        stream: 是否流式输出

    Returns:
        模型回复文本
    """
    try:
        import requests
    except ImportError:
        print("  ⚠️ 需要安装 requests: pip install requests")
        return None

    if config is None:
        config = OllamaConfig()

    url = f"{config.base_url}/api/chat"
    payload = {
        "model": config.model,
        "messages": messages,
        "stream": stream,
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]
    except Exception as e:
        print(f"  ⚠️ Ollama 服务不可用: {e}")
        print("  请先启动 Ollama（见文件头安装命令）")
        return None


def ollama_generate(
    prompt: str,
    config: OllamaConfig | None = None,
    system: str = "",
) -> str | None:
    """调用 Ollama Generate API（单轮生成）。"""
    try:
        import requests
    except ImportError:
        return None

    if config is None:
        config = OllamaConfig()

    url = f"{config.base_url}/api/generate"
    payload = {
        "model": config.model,
        "prompt": prompt,
        "system": system,
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["response"]
    except Exception:
        return None


def ollama_embeddings(
    text: str,
    config: OllamaConfig | None = None,
    model: str = "nomic-embed-text",
) -> list[float] | None:
    """调用 Ollama Embeddings API。"""
    try:
        import requests
    except ImportError:
        return None

    if config is None:
        config = OllamaConfig()

    url = f"{config.base_url}/api/embeddings"
    payload = {"model": model, "prompt": text}

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["embedding"]
    except Exception:
        return None


# ============================================================
# 2. 模型管理
# ============================================================

def list_models(config: OllamaConfig | None = None) -> list[dict] | None:
    """列出已下载的模型。"""
    try:
        import requests
    except ImportError:
        return None

    if config is None:
        config = OllamaConfig()

    try:
        response = requests.get(f"{config.base_url}/api/tags", timeout=10)
        response.raise_for_status()
        return response.json().get("models", [])
    except Exception:
        return None


# ============================================================
# 3. Modelfile 自定义
# ============================================================

def demo_modelfile() -> None:
    """展示 Modelfile 自定义。"""
    print("\n" + "=" * 60)
    print("3. Modelfile 自定义模型")
    print("=" * 60)

    modelfile = '''
    # 文件名: Modelfile
    # 基于 Qwen2 创建自定义模型

    FROM qwen2:7b

    # 系统提示词
    SYSTEM """你是一个专业的 Python 编程助手。
    请用中文回答，代码注释也用中文。
    回答要简洁、准确、实用。"""

    # 参数配置
    PARAMETER temperature 0.7
    PARAMETER top_p 0.9
    PARAMETER num_ctx 4096
    PARAMETER stop "<|im_end|>"

    # 创建命令:
    # ollama create my-python-helper -f Modelfile
    # 使用:
    # ollama run my-python-helper "如何实现单例模式？"
    '''
    print(modelfile)


# ============================================================
# 4. 常用命令速查
# ============================================================

def demo_commands() -> None:
    """展示 Ollama 常用命令。"""
    print("\n" + "=" * 60)
    print("4. Ollama 常用命令速查")
    print("=" * 60)

    commands = """
    # 模型管理
    ollama pull qwen2:7b          # 下载模型
    ollama list                    # 列出已下载模型
    ollama show qwen2:7b           # 查看模型信息
    ollama rm qwen2:7b             # 删除模型
    ollama cp qwen2:7b my-model    # 复制模型

    # 运行模型
    ollama run qwen2:7b            # 交互式对话
    ollama run qwen2:7b "你好"     # 单次提问

    # 服务管理
    ollama serve                   # 启动服务（默认 11434 端口）
    OLLAMA_HOST=0.0.0.0 ollama serve  # 允许远程访问

    # 推荐模型（按大小）
    ollama pull qwen2:0.5b         # 极小模型，测试用
    ollama pull qwen2:7b           # 推荐，效果好
    ollama pull llama3.1:8b        # Meta 开源
    ollama pull deepseek-coder-v2  # 代码生成
    """
    print(commands)


# ============================================================
# 演示
# ============================================================

def demo_api_call() -> None:
    """演示 API 调用。"""
    print("\n" + "=" * 60)
    print("1. Ollama Chat API 调用")
    print("=" * 60)

    messages = [
        {"role": "system", "content": "你是一个专业的 AI 助手，请用中文回答。"},
        {"role": "user", "content": "什么是 KV Cache？用一句话解释。"},
    ]

    print(f"  请求: {json.dumps(messages, ensure_ascii=False)}")

    result = ollama_chat(messages)
    if result:
        print(f"  回复: {result}")
    else:
        print("  （Ollama 服务未启动，显示模拟回复）")
        print("  模拟回复: KV Cache 是在自回归生成时缓存已计算的 Key 和 Value，")
        print("  避免每步重复计算，将推理复杂度从 O(n²) 降到 O(n)。")

    # 列出模型
    print("\n  --- 已下载模型 ---")
    models = list_models()
    if models:
        for m in models:
            size_gb = m.get("size", 0) / 1e9
            print(f"    {m['name']}: {size_gb:.1f} GB")
    else:
        print("    （Ollama 服务未启动）")


def demo_openai_compat() -> None:
    """展示 Ollama 的 OpenAI 兼容 API。"""
    print("\n" + "=" * 60)
    print("2. Ollama OpenAI 兼容 API")
    print("=" * 60)

    code = '''
    # Ollama 支持 OpenAI SDK 直接调用
    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # 任意值
    )

    response = client.chat.completions.create(
        model="qwen2:7b",
        messages=[
            {"role": "user", "content": "什么是 Transformer？"}
        ],
    )
    print(response.choices[0].message.content)

    # 💡 好处: 开发时用 Ollama，生产时切换到 vLLM/OpenAI，代码不变
    '''
    print(code)


def main() -> None:
    """运行所有 Ollama 演示。"""
    print("🦙 Ollama API 调用 — 本地 LLM 部署")
    print("=" * 60)

    demo_api_call()
    demo_openai_compat()
    demo_modelfile()
    demo_commands()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. Ollama: 最简单的本地 LLM 部署方案")
    print("  2. 支持 REST API 和 OpenAI 兼容 API")
    print("  3. Modelfile 自定义系统提示和参数")
    print("  4. 支持 CPU 和 GPU 推理")
    print("  5. 推荐 qwen2:7b 作为入门模型")


if __name__ == "__main__":
    main()
