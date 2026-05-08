"""
Prometheus 指标暴露模拟

知识点：Counter/Histogram/Gauge 指标类型、指标采集、
       PromQL 查询模拟、看板数据生成、指标命名规范

Python 版本：3.11+
依赖：标准库（默认模式）、prometheus-client>=0.17（服务模式）
最后验证：2024-12-01

外部服务（可选）：
  Prometheus + Grafana
  启动命令：docker compose -f docker-compose.monitoring.yml up -d
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

# ============================================================
# 1. Prometheus 指标类型模拟
# ============================================================

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricSample:
    """指标样本"""
    name: str
    labels: dict[str, str]
    value: float
    timestamp: float


class Counter:
    """计数器 — 只增不减"""

    def __init__(self, name: str, description: str, label_names: list[str] | None = None):
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self._values: dict[tuple, float] = defaultdict(float)

    def labels(self, **kwargs: str) -> Counter:
        """设置标签"""
        self._current_labels = tuple(sorted(kwargs.items()))
        return self

    def inc(self, amount: float = 1.0) -> None:
        """递增"""
        labels = getattr(self, "_current_labels", ())
        self._values[labels] += amount

    def get(self, **kwargs: str) -> float:
        """获取值"""
        labels = tuple(sorted(kwargs.items()))
        return self._values.get(labels, 0)

    def collect(self) -> list[MetricSample]:
        """采集所有样本"""
        samples = []
        for labels, value in self._values.items():
            samples.append(MetricSample(
                name=self.name,
                labels=dict(labels),
                value=value,
                timestamp=time.time(),
            ))
        return samples


class Gauge:
    """仪表盘 — 可增可减"""

    def __init__(self, name: str, description: str, label_names: list[str] | None = None):
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self._values: dict[tuple, float] = defaultdict(float)

    def labels(self, **kwargs: str) -> Gauge:
        self._current_labels = tuple(sorted(kwargs.items()))
        return self

    def set(self, value: float) -> None:
        labels = getattr(self, "_current_labels", ())
        self._values[labels] = value

    def inc(self, amount: float = 1.0) -> None:
        labels = getattr(self, "_current_labels", ())
        self._values[labels] += amount

    def dec(self, amount: float = 1.0) -> None:
        labels = getattr(self, "_current_labels", ())
        self._values[labels] -= amount

    def collect(self) -> list[MetricSample]:
        samples = []
        for labels, value in self._values.items():
            samples.append(MetricSample(self.name, dict(labels), value, time.time()))
        return samples


class Histogram:
    """直方图 — 分布统计"""

    def __init__(
        self,
        name: str,
        description: str,
        label_names: list[str] | None = None,
        buckets: list[float] | None = None,
    ):
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self.buckets = buckets or [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 30, 60]
        self._observations: dict[tuple, list[float]] = defaultdict(list)

    def labels(self, **kwargs: str) -> Histogram:
        self._current_labels = tuple(sorted(kwargs.items()))
        return self

    def observe(self, value: float) -> None:
        labels = getattr(self, "_current_labels", ())
        self._observations[labels].append(value)

    def get_percentile(self, percentile: float, **kwargs: str) -> float:
        """获取分位数"""
        labels = tuple(sorted(kwargs.items()))
        values = sorted(self._observations.get(labels, []))
        if not values:
            return 0
        idx = int(len(values) * percentile / 100)
        return values[min(idx, len(values) - 1)]

    def collect(self) -> list[MetricSample]:
        samples = []
        for labels, values in self._observations.items():
            # _count
            samples.append(MetricSample(f"{self.name}_count", dict(labels), len(values), time.time()))
            # _sum
            samples.append(MetricSample(f"{self.name}_sum", dict(labels), sum(values), time.time()))
            # _bucket
            for bucket in self.buckets:
                count = sum(1 for v in values if v <= bucket)
                bucket_labels = {**dict(labels), "le": str(bucket)}
                samples.append(MetricSample(f"{self.name}_bucket", bucket_labels, count, time.time()))
        return samples


# ============================================================
# 2. LLM 推理服务指标
# ============================================================

class LLMMetrics:
    """LLM 推理服务指标集合"""

    def __init__(self):
        self.request_count = Counter(
            "llm_requests_total", "LLM 请求总数", ["model", "status"]
        )
        self.request_latency = Histogram(
            "llm_request_duration_seconds", "LLM 请求延迟",
            ["model"], buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60],
        )
        self.active_requests = Gauge(
            "llm_active_requests", "当前活跃请求数", ["model"]
        )
        self.gpu_utilization = Gauge(
            "gpu_utilization_percent", "GPU 利用率", ["gpu_id"]
        )
        self.gpu_memory = Gauge(
            "gpu_memory_used_bytes", "GPU 显存使用", ["gpu_id"]
        )
        self.token_count = Counter(
            "llm_tokens_total", "Token 总数", ["model", "type"]
        )

    def record_request(self, model: str, latency: float, success: bool,
                       input_tokens: int, output_tokens: int) -> None:
        """记录一次请求"""
        status = "success" if success else "error"
        self.request_count.labels(model=model, status=status).inc()
        self.request_latency.labels(model=model).observe(latency)
        self.token_count.labels(model=model, type="input").inc(input_tokens)
        self.token_count.labels(model=model, type="output").inc(output_tokens)

    def update_gpu_metrics(self, gpu_id: str, utilization: float, memory_bytes: float) -> None:
        """更新 GPU 指标"""
        self.gpu_utilization.labels(gpu_id=gpu_id).set(utilization)
        self.gpu_memory.labels(gpu_id=gpu_id).set(memory_bytes)

    def to_prometheus_format(self) -> str:
        """输出 Prometheus 文本格式"""
        lines = []
        for metric in [self.request_count, self.request_latency,
                       self.active_requests, self.gpu_utilization,
                       self.gpu_memory, self.token_count]:
            lines.append(f"# HELP {metric.name} {metric.description}")
            lines.append(f"# TYPE {metric.name} {type(metric).__name__.lower()}")
            for sample in metric.collect():
                label_str = ",".join(f'{k}="{v}"' for k, v in sample.labels.items())
                if label_str:
                    lines.append(f"{sample.name}{{{label_str}}} {sample.value}")
                else:
                    lines.append(f"{sample.name} {sample.value}")
        return "\n".join(lines)


# ============================================================
# 3. 模拟请求流量
# ============================================================

def simulate_traffic(metrics: LLMMetrics, num_requests: int = 100) -> None:
    """模拟请求流量"""
    models = ["qwen2-7b", "qwen2-72b", "qwen2-1.5b"]
    model_weights = [0.6, 0.1, 0.3]  # 请求分布

    for i in range(num_requests):
        model = random.choices(models, weights=model_weights)[0]
        # 模拟延迟
        base_latency = {"qwen2-7b": 1.0, "qwen2-72b": 5.0, "qwen2-1.5b": 0.3}
        latency = base_latency[model] * random.uniform(0.5, 2.0)
        success = random.random() > 0.02
        input_tokens = random.randint(50, 500)
        output_tokens = random.randint(50, 300)

        metrics.record_request(model, latency, success, input_tokens, output_tokens)

    # 更新 GPU 指标
    for gpu_id in ["0", "1"]:
        metrics.update_gpu_metrics(
            gpu_id,
            utilization=random.uniform(0.5, 0.95),
            memory_bytes=random.uniform(20e9, 70e9),
        )


# ============================================================
# 4. PromQL 查询模拟
# ============================================================

class PromQLSimulator:
    """PromQL 查询模拟器"""

    def __init__(self, metrics: LLMMetrics):
        self.metrics = metrics

    def query_qps(self, model: str | None = None) -> float:
        """查询 QPS"""
        total = 0
        for sample in self.metrics.request_count.collect():
            if model and sample.labels.get("model") != model:
                continue
            total += sample.value
        return round(total / 60, 2)  # 假设 60 秒窗口

    def query_error_rate(self, model: str | None = None) -> float:
        """查询错误率"""
        success = error = 0
        for sample in self.metrics.request_count.collect():
            if model and sample.labels.get("model") != model:
                continue
            if sample.labels.get("status") == "success":
                success += sample.value
            elif sample.labels.get("status") == "error":
                error += sample.value
        total = success + error
        return round(error / max(total, 1), 4)

    def query_latency_percentile(self, percentile: float, model: str) -> float:
        """查询延迟分位数"""
        return round(self.metrics.request_latency.get_percentile(percentile, model=model), 3)


# ============================================================
# 5. 主程序入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Prometheus 指标暴露模拟演示")
    print("=" * 60)

    # 初始化指标
    metrics = LLMMetrics()

    # 模拟流量
    print("\n--- 模拟 200 个请求 ---")
    simulate_traffic(metrics, 200)

    # PromQL 查询
    print("\n--- PromQL 查询 ---")
    pql = PromQLSimulator(metrics)
    print(f"  总 QPS: {pql.query_qps()}")
    print(f"  qwen2-7b QPS: {pql.query_qps('qwen2-7b')}")
    print(f"  总错误率: {pql.query_error_rate():.2%}")
    print(f"  qwen2-7b P50 延迟: {pql.query_latency_percentile(50, 'qwen2-7b')}s")
    print(f"  qwen2-7b P99 延迟: {pql.query_latency_percentile(99, 'qwen2-7b')}s")
    print(f"  qwen2-72b P99 延迟: {pql.query_latency_percentile(99, 'qwen2-72b')}s")

    # Prometheus 格式输出
    print("\n--- Prometheus 格式（前 20 行）---")
    prom_output = metrics.to_prometheus_format()
    for line in prom_output.split("\n")[:20]:
        print(f"  {line}")

    print("\n✅ Prometheus 指标暴露模拟演示完成！")
