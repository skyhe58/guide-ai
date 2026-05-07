"""
负载均衡模拟

知识点：负载均衡算法（轮询/加权轮询/最少连接）、健康检查、
       GPU 感知路由、故障转移、自动扩缩容、请求队列管理

Python 版本：3.11+
依赖：标准库（默认模式）
最后验证：2024-12-01
"""

from __future__ import annotations

import hashlib
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ============================================================
# 1. 后端实例数据结构
# ============================================================

class InstanceStatus(Enum):
    """实例状态"""
    HEALTHY = "healthy"         # 健康
    UNHEALTHY = "unhealthy"     # 不健康
    DRAINING = "draining"       # 排空中（不接受新请求）
    STARTING = "starting"       # 启动中（模型加载）


@dataclass
class GPUMetrics:
    """GPU 指标"""
    utilization: float = 0.0      # GPU 利用率 (0-1)
    memory_used: float = 0.0      # 显存使用率 (0-1)
    temperature: float = 50.0     # 温度 (°C)
    power_usage: float = 100.0    # 功耗 (W)


@dataclass
class BackendInstance:
    """后端推理实例"""
    instance_id: str
    host: str
    port: int
    model: str
    weight: int = 1                # 权重（用于加权路由）
    status: InstanceStatus = InstanceStatus.HEALTHY
    gpu_metrics: GPUMetrics = field(default_factory=GPUMetrics)
    active_connections: int = 0    # 当前活跃连接数
    max_connections: int = 100     # 最大连接数
    total_requests: int = 0        # 总请求数
    total_errors: int = 0          # 总错误数
    avg_latency_ms: float = 0.0    # 平均延迟
    last_health_check: float = 0.0 # 上次健康检查时间

    @property
    def error_rate(self) -> float:
        """错误率"""
        return self.total_errors / max(self.total_requests, 1)

    @property
    def load_score(self) -> float:
        """负载评分（越低越好）"""
        return (
            self.gpu_metrics.utilization * 0.4
            + self.gpu_metrics.memory_used * 0.3
            + (self.active_connections / self.max_connections) * 0.3
        )


# ============================================================
# 2. 负载均衡算法
# ============================================================

class BalancingAlgorithm(Enum):
    """负载均衡算法"""
    ROUND_ROBIN = "round_robin"               # 轮询
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"  # 加权轮询
    LEAST_CONNECTIONS = "least_connections"     # 最少连接
    WEIGHTED_LEAST_CONN = "weighted_least_conn" # 加权最少连接
    GPU_AWARE = "gpu_aware"                    # GPU 感知路由


class LoadBalancer:
    """
    负载均衡器

    支持多种负载均衡算法，包括 GPU 感知路由。
    """

    def __init__(self, algorithm: BalancingAlgorithm = BalancingAlgorithm.WEIGHTED_LEAST_CONN):
        self.algorithm = algorithm
        self.instances: dict[str, BackendInstance] = {}
        self._rr_index = 0  # 轮询索引
        self._wrr_state: dict[str, int] = {}  # 加权轮询状态
        print(f"[LB] 负载均衡器初始化: 算法={algorithm.value}")

    def add_instance(self, instance: BackendInstance) -> None:
        """添加后端实例"""
        self.instances[instance.instance_id] = instance
        print(f"[LB] 添加实例: {instance.instance_id} ({instance.host}:{instance.port})")

    def remove_instance(self, instance_id: str) -> None:
        """移除后端实例"""
        if instance_id in self.instances:
            del self.instances[instance_id]
            print(f"[LB] 移除实例: {instance_id}")

    def get_healthy_instances(self) -> list[BackendInstance]:
        """获取健康实例"""
        return [
            inst for inst in self.instances.values()
            if inst.status == InstanceStatus.HEALTHY
        ]

    def select_instance(self) -> BackendInstance | None:
        """选择后端实例"""
        healthy = self.get_healthy_instances()
        if not healthy:
            return None

        if self.algorithm == BalancingAlgorithm.ROUND_ROBIN:
            return self._round_robin(healthy)
        elif self.algorithm == BalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin(healthy)
        elif self.algorithm == BalancingAlgorithm.LEAST_CONNECTIONS:
            return self._least_connections(healthy)
        elif self.algorithm == BalancingAlgorithm.WEIGHTED_LEAST_CONN:
            return self._weighted_least_connections(healthy)
        elif self.algorithm == BalancingAlgorithm.GPU_AWARE:
            return self._gpu_aware(healthy)
        return healthy[0]

    def _round_robin(self, instances: list[BackendInstance]) -> BackendInstance:
        """轮询算法"""
        instance = instances[self._rr_index % len(instances)]
        self._rr_index += 1
        return instance

    def _weighted_round_robin(self, instances: list[BackendInstance]) -> BackendInstance:
        """加权轮询算法"""
        # 构建加权列表
        weighted_list = []
        for inst in instances:
            weighted_list.extend([inst] * inst.weight)
        instance = weighted_list[self._rr_index % len(weighted_list)]
        self._rr_index += 1
        return instance

    def _least_connections(self, instances: list[BackendInstance]) -> BackendInstance:
        """最少连接算法"""
        return min(instances, key=lambda i: i.active_connections)

    def _weighted_least_connections(self, instances: list[BackendInstance]) -> BackendInstance:
        """加权最少连接算法"""
        return min(instances, key=lambda i: i.active_connections / max(i.weight, 1))

    def _gpu_aware(self, instances: list[BackendInstance]) -> BackendInstance:
        """GPU 感知路由"""
        return min(instances, key=lambda i: i.load_score)


# ============================================================
# 3. 健康检查器
# ============================================================

class HealthChecker:
    """健康检查器"""

    def __init__(
        self,
        check_interval: float = 10.0,
        unhealthy_threshold: int = 3,
        healthy_threshold: int = 2,
    ):
        self.check_interval = check_interval
        self.unhealthy_threshold = unhealthy_threshold
        self.healthy_threshold = healthy_threshold
        self.failure_counts: dict[str, int] = defaultdict(int)
        self.success_counts: dict[str, int] = defaultdict(int)

    def check(self, instance: BackendInstance) -> bool:
        """执行健康检查"""
        # 模拟健康检查（实际应发送 HTTP 请求）
        is_healthy = random.random() > 0.1  # 90% 概率健康

        if is_healthy:
            self.failure_counts[instance.instance_id] = 0
            self.success_counts[instance.instance_id] += 1

            if (instance.status == InstanceStatus.UNHEALTHY
                    and self.success_counts[instance.instance_id] >= self.healthy_threshold):
                instance.status = InstanceStatus.HEALTHY
                print(f"[Health] {instance.instance_id}: 恢复健康 ✅")
        else:
            self.success_counts[instance.instance_id] = 0
            self.failure_counts[instance.instance_id] += 1

            if self.failure_counts[instance.instance_id] >= self.unhealthy_threshold:
                instance.status = InstanceStatus.UNHEALTHY
                print(f"[Health] {instance.instance_id}: 标记不健康 ❌")

        instance.last_health_check = time.time()
        return is_healthy

    def run_checks(self, instances: dict[str, BackendInstance]) -> dict[str, bool]:
        """批量健康检查"""
        results = {}
        for inst_id, instance in instances.items():
            results[inst_id] = self.check(instance)
        return results


# ============================================================
# 4. 自动扩缩容
# ============================================================

@dataclass
class ScalingPolicy:
    """扩缩容策略"""
    min_instances: int = 1
    max_instances: int = 10
    scale_up_threshold: float = 0.8    # GPU 利用率超过 80% 扩容
    scale_down_threshold: float = 0.3  # GPU 利用率低于 30% 缩容
    cooldown_seconds: float = 300      # 冷却期 5 分钟


class AutoScaler:
    """自动扩缩容器"""

    def __init__(self, lb: LoadBalancer, policy: ScalingPolicy):
        self.lb = lb
        self.policy = policy
        self.last_scale_time = 0.0
        self.instance_counter = 0

    def evaluate(self) -> str:
        """评估是否需要扩缩容"""
        now = time.time()
        if now - self.last_scale_time < self.policy.cooldown_seconds:
            return "cooldown"

        healthy = self.lb.get_healthy_instances()
        if not healthy:
            return "no_instances"

        avg_gpu_util = sum(i.gpu_metrics.utilization for i in healthy) / len(healthy)
        current_count = len(healthy)

        if avg_gpu_util > self.policy.scale_up_threshold and current_count < self.policy.max_instances:
            self._scale_up()
            self.last_scale_time = now
            return "scale_up"
        elif avg_gpu_util < self.policy.scale_down_threshold and current_count > self.policy.min_instances:
            self._scale_down()
            self.last_scale_time = now
            return "scale_down"

        return "no_change"

    def _scale_up(self) -> None:
        """扩容"""
        self.instance_counter += 1
        new_instance = BackendInstance(
            instance_id=f"auto-{self.instance_counter}",
            host=f"10.0.0.{100 + self.instance_counter}",
            port=8000,
            model="qwen2-7b",
            status=InstanceStatus.STARTING,
        )
        self.lb.add_instance(new_instance)
        # 模拟模型加载完成
        new_instance.status = InstanceStatus.HEALTHY
        print(f"[AutoScale] 扩容: 新增实例 {new_instance.instance_id}")

    def _scale_down(self) -> None:
        """缩容"""
        healthy = self.lb.get_healthy_instances()
        if len(healthy) <= self.policy.min_instances:
            return
        # 移除负载最低的实例
        target = min(healthy, key=lambda i: i.active_connections)
        target.status = InstanceStatus.DRAINING
        print(f"[AutoScale] 缩容: 排空实例 {target.instance_id}")


# ============================================================
# 5. 请求处理模拟
# ============================================================

@dataclass
class Request:
    """模拟请求"""
    request_id: str
    model: str
    prompt_tokens: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class Response:
    """模拟响应"""
    request_id: str
    instance_id: str
    latency_ms: float
    success: bool
    error: str | None = None


def simulate_request_processing(
    lb: LoadBalancer,
    request: Request,
) -> Response:
    """模拟请求处理"""
    instance = lb.select_instance()
    if instance is None:
        return Response(request.request_id, "none", 0, False, "无可用实例")

    instance.active_connections += 1
    instance.total_requests += 1

    # 模拟处理延迟
    base_latency = 100 + request.prompt_tokens * 0.5
    latency = base_latency * (1 + instance.gpu_metrics.utilization)
    latency += random.gauss(0, 20)
    latency = max(50, latency)

    # 模拟错误
    success = random.random() > 0.02  # 2% 错误率
    if not success:
        instance.total_errors += 1

    instance.active_connections -= 1

    # 更新 GPU 指标（模拟）
    instance.gpu_metrics.utilization = min(1.0, 0.3 + instance.active_connections * 0.05)
    instance.gpu_metrics.memory_used = min(1.0, 0.5 + instance.active_connections * 0.03)

    return Response(
        request_id=request.request_id,
        instance_id=instance.instance_id,
        latency_ms=round(latency, 1),
        success=success,
        error=None if success else "推理超时",
    )


# ============================================================
# 6. 主程序入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("负载均衡模拟演示")
    print("=" * 60)

    # 创建负载均衡器
    lb = LoadBalancer(BalancingAlgorithm.GPU_AWARE)

    # 添加后端实例
    instances = [
        BackendInstance("vllm-1", "10.0.0.1", 8000, "qwen2-7b", weight=3,
                       gpu_metrics=GPUMetrics(utilization=0.6, memory_used=0.7)),
        BackendInstance("vllm-2", "10.0.0.2", 8000, "qwen2-7b", weight=2,
                       gpu_metrics=GPUMetrics(utilization=0.3, memory_used=0.5)),
        BackendInstance("vllm-3", "10.0.0.3", 8000, "qwen2-7b", weight=1,
                       gpu_metrics=GPUMetrics(utilization=0.8, memory_used=0.9)),
    ]
    for inst in instances:
        lb.add_instance(inst)

    # --- 演示 1: 请求路由 ---
    print("\n--- 请求路由测试 ---")
    route_counts = defaultdict(int)
    for i in range(20):
        req = Request(f"req-{i}", "qwen2-7b", random.randint(50, 500))
        resp = simulate_request_processing(lb, req)
        route_counts[resp.instance_id] += 1

    print("路由分布:")
    for inst_id, count in sorted(route_counts.items()):
        print(f"  {inst_id}: {count} 请求")

    # --- 演示 2: 健康检查 ---
    print("\n--- 健康检查 ---")
    checker = HealthChecker(unhealthy_threshold=2)
    for _ in range(5):
        results = checker.run_checks(lb.instances)
        healthy_count = sum(1 for v in results.values() if v)
        print(f"  健康实例: {healthy_count}/{len(results)}")

    # --- 演示 3: 自动扩缩容 ---
    print("\n--- 自动扩缩容 ---")
    scaler = AutoScaler(lb, ScalingPolicy(min_instances=2, max_instances=5, cooldown_seconds=0))

    # 模拟高负载
    for inst in lb.instances.values():
        inst.gpu_metrics.utilization = 0.9
    action = scaler.evaluate()
    print(f"  高负载决策: {action}")

    # 模拟低负载
    for inst in lb.instances.values():
        inst.gpu_metrics.utilization = 0.2
    action = scaler.evaluate()
    print(f"  低负载决策: {action}")

    # --- 统计信息 ---
    print("\n--- 实例状态 ---")
    for inst in lb.instances.values():
        print(f"  {inst.instance_id}: 状态={inst.status.value}, "
              f"GPU={inst.gpu_metrics.utilization:.0%}, "
              f"请求={inst.total_requests}, 错误率={inst.error_rate:.1%}")

    print("\n✅ 负载均衡模拟演示完成！")
