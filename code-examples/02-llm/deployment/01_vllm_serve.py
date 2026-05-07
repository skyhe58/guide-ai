"""
vLLM 推理服务 — 高性能 LLM 推理引擎

知识点：vLLM 安装部署、OpenAI 兼容 API、PagedAttention、
       Continuous Batching、Tensor Parallelism

Python 版本：3.11+
依赖：requests>=2.31（API 调用）
可选依赖：vllm>=0.4（需 NVIDIA GPU）
最后验证：2024-12-01

⚠️ Docker 启动命令（推荐）：
  docker run --runtime nvidia --gpus all \\
    -v ~/.cache/huggingface:/root/.cache/huggingface \\
    -p 8000:8000 \\
    --ipc=host \\
    vllm/vllm-openai:latest \\
    --model Qwen/Qwen2-7B-Instruct \\
    --max-model-len 4096

⚠️ pip 安装（需 CUDA）：
  pip install vllm
  python -m vllm.entrypoints.openai.api_server \\
    --model Qwen/Qwen2-7B-Instruct \\
    --port 8000

⚠️ 免费替代方案：
  - Ollama（本地部署，见 02_ollama_api.py）
  - Hugging Face Inference Endpoints（免费额度）
"""

from __future__ import annotations

import json
from dataclasses import dataclass


# ============================================================
# 1. vLLM OpenAI 兼容 API 调用
# ============================================================

@dataclass
class VLLMConfig:
    """vLLM 服务配置。"""
    base_url: str = "http://localhost:8000"
    model: str = "Qwen/Qwen2-7B-Instruct"
    api_key: str = "EMPTY"  # vLLM 默认不需要 API Key


def call_vllm_chat(
    messages: list[dict[str, str]],
    config: VLLMConfig | None = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
    stream: bool = False,
) -> str | None:
    """调用 vLLM 的 OpenAI 兼容 Chat API。

    Args:
        messages: 对话消息列表
        config: vLLM 配置
        temperature: 温度参数
        max_tokens: 最大生成 Token 数
        stream: 是否流式输出

    Returns:
        模型回复文本，服务不可用时返回 None
    """
    try:
        import requests
    except ImportError:
        print("  ⚠️ 需要安装 requests: pip install requests")
        return None

    if config is None:
        config = VLLMConfig()

    url = f"{config.base_url}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
    }
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.ConnectionError:
        print(f"  ⚠️ 无法连接 vLLM 服务 ({config.base_url})")
        print("  请先启动 vLLM 服务（见文件头 Docker 命令）")
        return None
    except Exception as e:
        print(f"  ⚠️ 请求失败: {e}")
        return None


def call_vllm_completion(
    prompt: str,
    config: VLLMConfig | None = None,
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> str | None:
    """调用 vLLM 的 Completions API。"""
    try:
        import requests
    except ImportError:
        return None

    if config is None:
        config = VLLMConfig()

    url = f"{config.base_url}/v1/completions"
    payload = {
        "model": config.model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["text"]
    except Exception:
        return None


# ============================================================
# 2. vLLM 离线推理（Python API）
# ============================================================

def demo_offline_inference() -> None:
    """展示 vLLM 离线推理代码（伪代码）。"""
    print("\n" + "=" * 60)
    print("2. vLLM 离线推理（Python API，伪代码）")
    print("=" * 60)

    code = '''
    from vllm import LLM, SamplingParams

    # 加载模型
    llm = LLM(
        model="Qwen/Qwen2-7B-Instruct",
        tensor_parallel_size=1,    # GPU 数量
        max_model_len=4096,
        gpu_memory_utilization=0.9,
    )

    # 采样参数
    params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
    )

    # 批量推理（vLLM 自动优化 batching）
    prompts = [
        "什么是机器学习？",
        "解释 Transformer 架构",
        "Python 异步编程的优势",
    ]
    outputs = llm.generate(prompts, params)

    for output in outputs:
        print(f"Prompt: {output.prompt}")
        print(f"Output: {output.outputs[0].text}")
    '''
    print(code)


# ============================================================
# 3. vLLM 部署配置指南
# ============================================================

def demo_deployment_configs() -> None:
    """展示 vLLM 部署配置。"""
    print("\n" + "=" * 60)
    print("3. vLLM 部署配置指南")
    print("=" * 60)

    configs = {
        "单卡部署（RTX 4090 / A100）": {
            "命令": "python -m vllm.entrypoints.openai.api_server "
                   "--model Qwen/Qwen2-7B-Instruct --max-model-len 4096",
            "适用": "7B-13B 模型",
        },
        "多卡部署（Tensor Parallelism）": {
            "命令": "python -m vllm.entrypoints.openai.api_server "
                   "--model Qwen/Qwen2-72B-Instruct --tensor-parallel-size 4",
            "适用": "70B+ 模型",
        },
        "量化部署（AWQ/GPTQ）": {
            "命令": "python -m vllm.entrypoints.openai.api_server "
                   "--model Qwen/Qwen2-7B-Instruct-AWQ --quantization awq",
            "适用": "显存有限场景",
        },
    }

    for name, cfg in configs.items():
        print(f"\n  [{name}]")
        print(f"    命令: {cfg['命令']}")
        print(f"    适用: {cfg['适用']}")

    print("\n  💡 vLLM 核心优势:")
    print("    - PagedAttention: 显存利用率提升 2-4x")
    print("    - Continuous Batching: 吞吐量提升 10-24x")
    print("    - OpenAI 兼容 API: 无缝替换 OpenAI SDK")
    print("    - Tensor Parallelism: 多卡并行推理")


# ============================================================
# 演示
# ============================================================

def demo_api_call() -> None:
    """演示 API 调用。"""
    print("\n" + "=" * 60)
    print("1. vLLM OpenAI 兼容 API 调用")
    print("=" * 60)

    messages = [
        {"role": "system", "content": "你是一个专业的 AI 助手。"},
        {"role": "user", "content": "什么是 PagedAttention？"},
    ]

    print(f"  请求消息: {json.dumps(messages, ensure_ascii=False, indent=4)}")

    result = call_vllm_chat(messages)
    if result:
        print(f"  回复: {result}")
    else:
        print("  （vLLM 服务未启动，显示模拟回复）")
        print("  模拟回复: PagedAttention 是 vLLM 的核心技术，借鉴操作系统的虚拟内存分页机制，")
        print("  将 KV Cache 分成固定大小的块（pages），按需分配，避免显存碎片化。")


def main() -> None:
    """运行所有 vLLM 演示。"""
    print("🚀 vLLM 推理服务 — 高性能 LLM 推理引擎")
    print("=" * 60)

    demo_api_call()
    demo_offline_inference()
    demo_deployment_configs()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. vLLM: 当前最快的开源 LLM 推理引擎")
    print("  2. PagedAttention: 显存利用率提升 2-4x")
    print("  3. OpenAI 兼容 API: 一行代码切换")
    print("  4. 支持 AWQ/GPTQ 量化部署")
    print("  5. Docker 部署最简单（见文件头命令）")


if __name__ == "__main__":
    main()
