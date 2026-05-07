"""
vLLM + 监控面板 — 里程碑项目

功能：vLLM 推理配置 + Prometheus 指标采集 + Grafana 看板配置 +
     健康检查 + 告警规则 + 性能基准测试

Python 版本：3.11+
依赖：标准库（默认模式）
最后验证：2024-12-01

运行说明：
  python main.py
  模拟 vLLM 推理服务 + 完整监控体系。
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


# ============================================================
# 1. vLLM 服务配置
# ============================================================

@dataclass
class VLLMServiceConfig:
    """vLLM 服务配置"""
    model: str = "Qwen/Qwen2-7B-Instruct"
    tensor_parallel_size: int = 1
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.90
    port: int = 8000
    max_concurrent: int = 128

    def to_docker_command(self) -> str:
        return (f"docker run --gpus all -p {self.port}:8000 "
                f"vllm/vllm-openai:latest --model {self.model} "
                f"--tensor-parallel-size {self.tensor_parallel_size} "
                f"--max-model-len {self.max_model_len} "
                f"--gpu-memory-utilization {self.gpu_memory_utilization}")


# ============================================================
# 2. 指标采集器
# ============================================================

@dataclass
class MetricPoint:
    """指标数据点"""
    name: str
    value: float
    labels: dict[str, str]
    timestamp: float


class MetricsCollector:
    """指标采集器"""

    def __init__(self):
        self.metrics: list[MetricPoint] = []
        self.counters: dict[str, float] = defaultdict(float)
        self.histograms: dict[str, list[float]] = defaultdict(list)
        self.gauges: dict[str, float] = {}

    def inc_counter(self, name: str, labels: dict[str, str], value: float = 1) -> None:
        key = f"{name}:{json.dumps(labels, sort_keys=True)}"
        self.counters[key] += value
        self.metrics.append(MetricPoint(name, self.counters[key], labels, time.time()))

    def observe_histogram(self, name: str, labels: dict[str, str], value: float) -> None:
        key = f"{name}:{json.dumps(labels, sort_keys=True)}"
        self.histograms[key].append(value)
        self.metrics.append(MetricPoint(name, value, labels, time.time()))

    def set_gauge(self, name: str, labels: dict[str, str], value: float) -> None:
        key = f"{name}:{json.dumps(labels, sort_keys=True)}"
        self.gauges[key] = value
        self.metrics.append(MetricPoint(name, value, labels, time.time()))

    def get_percentile(self, name: str, labels: dict[str, str], p: float) -> float:
        key = f"{name}:{json.dumps(labels, sort_keys=True)}"
        values = sorted(self.histograms.get(key, []))
        if not values:
            return 0
        idx = min(int(len(values) * p / 100), len(values) - 1)
        return values[idx]

    def get_rate(self, name: str, labels: dict[str, str], window: float = 60) -> float:
        key = f"{name}:{json.dumps(labels, sort_keys=True)}"
        return self.counters.get(key, 0) / window


# ============================================================
# 3. 模拟推理服务
# ============================================================

class VLLMSimulator:
    """vLLM 推理服务模拟"""

    def __init__(self, config: VLLMServiceConfig, collector: MetricsCollector):
        self.config = config
        self.collector = collector
        self.active_requests = 0
        self.total_requests = 0

    def handle_request(self, prompt: str, max_tokens: int = 256) -> dict:
        """处理推理请求"""
        self.active_requests += 1
        self.total_requests += 1
        self.collector.set_gauge("vllm_active_requests", {"model": self.config.model}, self.active_requests)

        # 模拟推理
        input_tokens = len(prompt) // 2
        output_tokens = min(max_tokens, random.randint(30, 200))
        latency = 0.1 + output_tokens * random.uniform(0.005, 0.015)
        success = random.random() > 0.01

        # 记录指标
        status = "success" if success else "error"
        self.collector.inc_counter("vllm_requests_total", {"model": self.config.model, "status": status})
        self.collector.observe_histogram("vllm_request_duration_seconds", {"model": self.config.model}, latency)
        self.collector.inc_counter("vllm_tokens_total", {"model": self.config.model, "type": "input"}, input_tokens)
        self.collector.inc_counter("vllm_tokens_total", {"model": self.config.model, "type": "output"}, output_tokens)

        # 模拟 GPU 指标
        gpu_util = 0.4 + self.active_requests * 0.02 + random.uniform(-0.05, 0.05)
        gpu_mem = 0.6 + self.active_requests * 0.01
        self.collector.set_gauge("gpu_utilization", {"gpu_id": "0"}, min(1.0, gpu_util))
        self.collector.set_gauge("gpu_memory_used_ratio", {"gpu_id": "0"}, min(1.0, gpu_mem))

        self.active_requests -= 1
        return {
            "success": success, "latency": round(latency, 3),
            "input_tokens": input_tokens, "output_tokens": output_tokens,
        }

    def health_check(self) -> dict:
        return {"status": "healthy", "model": self.config.model, "active_requests": self.active_requests}


# ============================================================
# 4. Grafana 看板配置生成
# ============================================================

class GrafanaDashboardGenerator:
    """Grafana 看板配置生成器"""

    @staticmethod
    def generate(service_name: str) -> dict:
        """生成 Grafana 看板 JSON"""
        return {
            "dashboard": {
                "title": f"{service_name} 监控看板",
                "panels": [
                    {"title": "QPS", "type": "graph",
                     "targets": [{"expr": 'rate(vllm_requests_total[5m])'}]},
                    {"title": "P99 延迟", "type": "graph",
                     "targets": [{"expr": 'histogram_quantile(0.99, rate(vllm_request_duration_seconds_bucket[5m]))'}]},
                    {"title": "错误率", "type": "stat",
                     "targets": [{"expr": 'rate(vllm_requests_total{status="error"}[5m]) / rate(vllm_requests_total[5m])'}]},
                    {"title": "GPU 利用率", "type": "gauge",
                     "targets": [{"expr": "gpu_utilization"}]},
                    {"title": "GPU 显存", "type": "gauge",
                     "targets": [{"expr": "gpu_memory_used_ratio"}]},
                    {"title": "Token 消耗", "type": "graph",
                     "targets": [{"expr": "rate(vllm_tokens_total[5m])"}]},
                    {"title": "活跃请求", "type": "stat",
                     "targets": [{"expr": "vllm_active_requests"}]},
                ],
                "refresh": "10s",
                "time": {"from": "now-1h", "to": "now"},
            }
        }


# ============================================================
# 5. 告警规则生成
# ============================================================

class AlertRuleGenerator:
    """告警规则生成器"""

    @staticmethod
    def generate() -> list[dict]:
        return [
            {"alert": "VLLMServiceDown", "expr": "up{job='vllm'} == 0",
             "for": "1m", "severity": "critical",
             "summary": "vLLM 服务不可用"},
            {"alert": "VLLMHighLatency",
             "expr": "histogram_quantile(0.99, rate(vllm_request_duration_seconds_bucket[5m])) > 10",
             "for": "5m", "severity": "warning",
             "summary": "vLLM P99 延迟超过 10 秒"},
            {"alert": "VLLMHighErrorRate",
             "expr": 'rate(vllm_requests_total{status="error"}[5m]) / rate(vllm_requests_total[5m]) > 0.01',
             "for": "5m", "severity": "warning",
             "summary": "vLLM 错误率超过 1%"},
            {"alert": "GPUMemoryHigh",
             "expr": "gpu_memory_used_ratio > 0.9",
             "for": "10m", "severity": "warning",
             "summary": "GPU 显存使用率超过 90%"},
        ]


# ============================================================
# 6. 主程序入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("vLLM + 监控面板 — 里程碑项目")
    print("=" * 60)

    # 初始化
    config = VLLMServiceConfig()
    collector = MetricsCollector()
    simulator = VLLMSimulator(config, collector)

    print(f"\n--- vLLM 服务配置 ---")
    print(f"  模型: {config.model}")
    print(f"  Docker: {config.to_docker_command()}")

    # 模拟流量
    print(f"\n--- 模拟 100 个请求 ---")
    prompts = ["什么是 RAG？", "解释 Transformer", "如何微调 LLM？", "什么是向量数据库？"]
    for i in range(100):
        prompt = random.choice(prompts)
        result = simulator.handle_request(prompt)

    # 查看指标
    print(f"\n--- 监控指标 ---")
    model = config.model
    p50 = collector.get_percentile("vllm_request_duration_seconds", {"model": model}, 50)
    p99 = collector.get_percentile("vllm_request_duration_seconds", {"model": model}, 99)
    qps = collector.get_rate("vllm_requests_total", {"model": model, "status": "success"})
    print(f"  QPS: {qps:.1f}")
    print(f"  P50 延迟: {p50:.3f}s")
    print(f"  P99 延迟: {p99:.3f}s")
    print(f"  健康检查: {simulator.health_check()}")

    # Grafana 看板
    print(f"\n--- Grafana 看板配置 ---")
    dashboard = GrafanaDashboardGenerator.generate("vLLM")
    print(f"  面板数: {len(dashboard['dashboard']['panels'])}")
    for panel in dashboard["dashboard"]["panels"]:
        print(f"    📊 {panel['title']} ({panel['type']})")

    # 告警规则
    print(f"\n--- 告警规则 ---")
    rules = AlertRuleGenerator.generate()
    for rule in rules:
        print(f"  🚨 {rule['alert']}: {rule['summary']} (severity: {rule['severity']})")

    print("\n✅ vLLM + 监控面板里程碑项目完成！")
