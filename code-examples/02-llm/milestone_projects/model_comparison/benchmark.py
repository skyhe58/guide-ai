"""
多模型对比评测脚本 — 本地/API 模型性能对比

知识点：模型评测指标、延迟/吞吐量测试、质量评估、
       多模型对比框架

Python 版本：3.11+
依赖：requests>=2.31
最后验证：2024-12-01

⚠️ 运行前提：
  - 本地模型: 启动 Ollama（ollama serve）
  - 拉取模型: ollama pull qwen2:7b
  - 可选: ollama pull llama3.1:8b
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field


# ============================================================
# 1. 评测配置
# ============================================================

@dataclass
class ModelConfig:
    """模型配置。"""
    name: str
    provider: str  # "ollama" | "vllm" | "openai"
    base_url: str
    model_id: str
    api_key: str = ""


@dataclass
class BenchmarkResult:
    """评测结果。"""
    model_name: str
    task: str
    response: str = ""
    latency_ms: float = 0.0
    tokens_generated: int = 0
    tokens_per_second: float = 0.0
    success: bool = True
    error: str = ""


# 默认模型配置
DEFAULT_MODELS = [
    ModelConfig("Qwen2-7B", "ollama", "http://localhost:11434", "qwen2:7b"),
    ModelConfig("LLaMA3.1-8B", "ollama", "http://localhost:11434", "llama3.1:8b"),
]

# 评测任务
BENCHMARK_TASKS = [
    {
        "name": "中文问答",
        "prompt": "请用 3 句话解释什么是机器学习。",
        "max_tokens": 200,
    },
    {
        "name": "代码生成",
        "prompt": "用 Python 写一个二分查找函数，包含类型注解和 docstring。",
        "max_tokens": 300,
    },
    {
        "name": "逻辑推理",
        "prompt": "小明比小红大 3 岁，小红比小刚小 2 岁，小刚 10 岁。请问小明几岁？请一步步推理。",
        "max_tokens": 200,
    },
    {
        "name": "摘要生成",
        "prompt": "请用一句话总结以下内容：Transformer 是一种基于注意力机制的神经网络架构，"
                 "由 Google 在 2017 年提出。它完全抛弃了 RNN 的循环结构，通过自注意力机制"
                 "实现并行计算，大幅提升了训练速度。现代所有大语言模型都基于 Transformer 架构。",
        "max_tokens": 100,
    },
]


# ============================================================
# 2. 评测引擎
# ============================================================

def call_ollama(
    config: ModelConfig,
    prompt: str,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """调用 Ollama API。

    Returns:
        (回复文本, 延迟毫秒)
    """
    try:
        import requests
    except ImportError:
        return "", 0.0

    url = f"{config.base_url}/api/generate"
    payload = {
        "model": config.model_id,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens},
    }

    start = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        elapsed = (time.perf_counter() - start) * 1000
        return resp.json().get("response", ""), elapsed
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return f"[ERROR: {e}]", elapsed


def run_benchmark(
    models: list[ModelConfig] | None = None,
    tasks: list[dict] | None = None,
) -> list[BenchmarkResult]:
    """运行评测。"""
    if models is None:
        models = DEFAULT_MODELS
    if tasks is None:
        tasks = BENCHMARK_TASKS

    results: list[BenchmarkResult] = []

    for model in models:
        print(f"\n  === 评测模型: {model.name} ===")
        for task in tasks:
            print(f"    任务: {task['name']}...", end=" ")

            response, latency = call_ollama(model, task["prompt"], task["max_tokens"])

            # 估算 Token 数（粗略：中文 1 字 ≈ 1 Token）
            tokens = len(response)
            tps = tokens / (latency / 1000) if latency > 0 else 0

            result = BenchmarkResult(
                model_name=model.name,
                task=task["name"],
                response=response[:200],  # 截断显示
                latency_ms=latency,
                tokens_generated=tokens,
                tokens_per_second=tps,
                success="ERROR" not in response,
            )
            results.append(result)

            status = "✅" if result.success else "❌"
            print(f"{status} {latency:.0f}ms, ~{tps:.1f} tok/s")

    return results


# ============================================================
# 3. 结果展示
# ============================================================

def print_results(results: list[BenchmarkResult]) -> None:
    """打印评测结果对比。"""
    print("\n" + "=" * 60)
    print("评测结果汇总")
    print("=" * 60)

    # 按模型分组
    models = sorted(set(r.model_name for r in results))
    tasks = sorted(set(r.task for r in results))

    # 延迟对比
    print(f"\n  {'任务':<12}", end="")
    for model in models:
        print(f"  {model:<16}", end="")
    print()
    print("  " + "-" * (12 + 18 * len(models)))

    for task in tasks:
        print(f"  {task:<12}", end="")
        for model in models:
            r = next((r for r in results if r.model_name == model and r.task == task), None)
            if r and r.success:
                print(f"  {r.latency_ms:>6.0f}ms {r.tokens_per_second:>5.1f}t/s", end="")
            else:
                print(f"  {'N/A':>16}", end="")
        print()

    # 平均性能
    print(f"\n  {'平均':<12}", end="")
    for model in models:
        model_results = [r for r in results if r.model_name == model and r.success]
        if model_results:
            avg_latency = sum(r.latency_ms for r in model_results) / len(model_results)
            avg_tps = sum(r.tokens_per_second for r in model_results) / len(model_results)
            print(f"  {avg_latency:>6.0f}ms {avg_tps:>5.1f}t/s", end="")
        else:
            print(f"  {'N/A':>16}", end="")
    print()


def print_responses(results: list[BenchmarkResult]) -> None:
    """打印模型回复对比。"""
    print("\n" + "=" * 60)
    print("模型回复对比")
    print("=" * 60)

    tasks = sorted(set(r.task for r in results))
    for task in tasks:
        print(f"\n  --- {task} ---")
        task_results = [r for r in results if r.task == task]
        for r in task_results:
            response_preview = r.response[:150].replace("\n", " ")
            print(f"  [{r.model_name}]: {response_preview}...")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行多模型对比评测。"""
    print("📊 多模型对比评测")
    print("=" * 60)
    print("  评测模型:", ", ".join(m.name for m in DEFAULT_MODELS))
    print("  评测任务:", ", ".join(t["name"] for t in BENCHMARK_TASKS))

    # 检查 Ollama 服务
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        print(f"  可用模型: {models}")
    except Exception:
        print("\n  ⚠️ Ollama 服务未启动，显示模拟结果")
        print("  请先运行: ollama serve && ollama pull qwen2:7b")

        # 模拟结果
        print("\n  --- 模拟评测结果 ---")
        print("  Qwen2-7B:    平均延迟 ~2000ms, ~30 tok/s (T4 GPU)")
        print("  LLaMA3.1-8B: 平均延迟 ~2500ms, ~25 tok/s (T4 GPU)")
        print("\n  💡 实际性能取决于硬件配置（CPU/GPU/内存）")
        return

    results = run_benchmark()
    print_results(results)
    print_responses(results)

    print("\n" + "=" * 60)
    print("✅ 评测完成！")
    print("\n💡 评测建议:")
    print("  1. 多次运行取平均值（减少波动）")
    print("  2. 关注任务相关性（选择适合的模型）")
    print("  3. 考虑部署成本（显存/延迟/吞吐量）")


if __name__ == "__main__":
    main()
