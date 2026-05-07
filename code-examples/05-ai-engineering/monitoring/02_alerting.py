"""
告警策略模拟

知识点：告警规则定义、严重级别（P0-P3）、通知渠道、
       告警抑制、告警聚合、Runbook 集成、告警质量评估

Python 版本：3.11+
依赖：标准库（默认模式）
最后验证：2024-12-01
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable


# ============================================================
# 1. 告警数据结构
# ============================================================

class Severity(Enum):
    """告警严重级别"""
    P0_CRITICAL = "P0"   # 紧急：服务不可用，5 分钟内响应
    P1_HIGH = "P1"       # 严重：性能严重下降，1 小时内响应
    P2_MEDIUM = "P2"     # 警告：潜在风险，工作日内处理
    P3_LOW = "P3"        # 信息：需要关注，下次迭代处理


class AlertState(Enum):
    """告警状态"""
    FIRING = "firing"       # 触发中
    RESOLVED = "resolved"   # 已恢复
    SILENCED = "silenced"   # 已静默
    ACKNOWLEDGED = "acknowledged"  # 已确认


@dataclass
class AlertRule:
    """告警规则"""
    name: str
    description: str
    severity: Severity
    condition: Callable[[dict[str, float]], bool]  # 触发条件
    for_duration: float = 0  # 持续时间（秒）才触发
    labels: dict[str, str] = field(default_factory=dict)
    runbook_url: str = ""
    annotations: dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """告警实例"""
    alert_id: str
    rule_name: str
    severity: Severity
    state: AlertState
    message: str
    fired_at: float
    resolved_at: float | None = None
    acknowledged_at: float | None = None
    acknowledged_by: str | None = None
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)
    notification_sent: bool = False

    @property
    def duration_seconds(self) -> float:
        end = self.resolved_at or time.time()
        return end - self.fired_at


# ============================================================
# 2. 通知渠道
# ============================================================

class NotificationChannel(Enum):
    PHONE = "phone"
    SMS = "sms"
    SLACK = "slack"
    EMAIL = "email"
    WEBHOOK = "webhook"


@dataclass
class NotificationConfig:
    """通知配置"""
    channel: NotificationChannel
    target: str  # 电话号码/邮箱/Slack 频道/Webhook URL
    enabled: bool = True


class NotificationRouter:
    """通知路由器"""

    def __init__(self):
        self.routes: dict[Severity, list[NotificationConfig]] = {
            Severity.P0_CRITICAL: [
                NotificationConfig(NotificationChannel.PHONE, "+86-138xxxx"),
                NotificationConfig(NotificationChannel.SMS, "+86-138xxxx"),
                NotificationConfig(NotificationChannel.SLACK, "#incidents"),
            ],
            Severity.P1_HIGH: [
                NotificationConfig(NotificationChannel.SLACK, "#alerts"),
                NotificationConfig(NotificationChannel.EMAIL, "oncall@team.com"),
            ],
            Severity.P2_MEDIUM: [
                NotificationConfig(NotificationChannel.EMAIL, "team@team.com"),
            ],
            Severity.P3_LOW: [
                NotificationConfig(NotificationChannel.SLACK, "#info"),
            ],
        }

    def send(self, alert: Alert) -> list[str]:
        """发送通知"""
        channels = self.routes.get(alert.severity, [])
        sent = []
        for config in channels:
            if config.enabled:
                print(f"  [通知] {config.channel.value} → {config.target}: "
                      f"[{alert.severity.value}] {alert.message}")
                sent.append(config.channel.value)
        return sent


# ============================================================
# 3. 告警管理器
# ============================================================

class AlertManager:
    """告警管理器"""

    def __init__(self):
        self.rules: list[AlertRule] = []
        self.active_alerts: dict[str, Alert] = {}
        self.alert_history: list[Alert] = []
        self.notifier = NotificationRouter()
        self.silence_rules: list[dict] = []
        self._alert_counter = 0

        # 统计
        self.stats = {
            "total_fired": 0,
            "total_resolved": 0,
            "total_notifications": 0,
        }

    def add_rule(self, rule: AlertRule) -> None:
        """添加告警规则"""
        self.rules.append(rule)

    def add_silence(self, rule_name: str, duration_seconds: float, reason: str) -> None:
        """添加静默规则"""
        self.silence_rules.append({
            "rule_name": rule_name,
            "until": time.time() + duration_seconds,
            "reason": reason,
        })
        print(f"[AlertManager] 静默规则: {rule_name} ({duration_seconds}s) — {reason}")

    def _is_silenced(self, rule_name: str) -> bool:
        """检查是否被静默"""
        now = time.time()
        for silence in self.silence_rules:
            if silence["rule_name"] == rule_name and silence["until"] > now:
                return True
        return False

    def evaluate(self, metrics: dict[str, float]) -> list[Alert]:
        """评估所有规则"""
        new_alerts = []

        for rule in self.rules:
            is_firing = rule.condition(metrics)
            alert_key = rule.name

            if is_firing and alert_key not in self.active_alerts:
                # 新告警
                if self._is_silenced(rule.name):
                    continue

                self._alert_counter += 1
                alert = Alert(
                    alert_id=f"alert-{self._alert_counter:04d}",
                    rule_name=rule.name,
                    severity=rule.severity,
                    state=AlertState.FIRING,
                    message=rule.description,
                    fired_at=time.time(),
                    labels=rule.labels,
                    annotations={**rule.annotations, "runbook": rule.runbook_url},
                )
                self.active_alerts[alert_key] = alert
                self.stats["total_fired"] += 1

                # 发送通知
                sent = self.notifier.send(alert)
                alert.notification_sent = True
                self.stats["total_notifications"] += len(sent)
                new_alerts.append(alert)

            elif not is_firing and alert_key in self.active_alerts:
                # 告警恢复
                alert = self.active_alerts.pop(alert_key)
                alert.state = AlertState.RESOLVED
                alert.resolved_at = time.time()
                self.alert_history.append(alert)
                self.stats["total_resolved"] += 1
                print(f"  [恢复] {alert.rule_name} (持续: {alert.duration_seconds:.0f}s)")

        return new_alerts

    def acknowledge(self, alert_id: str, by: str) -> bool:
        """确认告警"""
        for alert in self.active_alerts.values():
            if alert.alert_id == alert_id:
                alert.state = AlertState.ACKNOWLEDGED
                alert.acknowledged_at = time.time()
                alert.acknowledged_by = by
                return True
        return False

    def get_summary(self) -> dict:
        """获取告警摘要"""
        severity_counts = defaultdict(int)
        for alert in self.active_alerts.values():
            severity_counts[alert.severity.value] += 1

        return {
            "active_alerts": len(self.active_alerts),
            "by_severity": dict(severity_counts),
            "total_fired": self.stats["total_fired"],
            "total_resolved": self.stats["total_resolved"],
            "total_notifications": self.stats["total_notifications"],
        }


# ============================================================
# 4. 预定义告警规则
# ============================================================

def create_llm_alert_rules() -> list[AlertRule]:
    """创建 LLM 推理服务告警规则"""
    return [
        AlertRule(
            name="LLMServiceDown",
            description="LLM 推理服务不可用",
            severity=Severity.P0_CRITICAL,
            condition=lambda m: m.get("service_up", 1) == 0,
            runbook_url="https://wiki/runbook/llm-service-down",
        ),
        AlertRule(
            name="LLMHighLatency",
            description="LLM P99 延迟超过 10 秒",
            severity=Severity.P1_HIGH,
            condition=lambda m: m.get("latency_p99", 0) > 10,
        ),
        AlertRule(
            name="LLMHighErrorRate",
            description="LLM 错误率超过 1%",
            severity=Severity.P1_HIGH,
            condition=lambda m: m.get("error_rate", 0) > 0.01,
        ),
        AlertRule(
            name="GPUMemoryHigh",
            description="GPU 显存使用率超过 90%",
            severity=Severity.P2_MEDIUM,
            condition=lambda m: m.get("gpu_memory_percent", 0) > 90,
        ),
        AlertRule(
            name="CostOverBudget",
            description="日成本超出预算",
            severity=Severity.P2_MEDIUM,
            condition=lambda m: m.get("daily_cost", 0) > m.get("daily_budget", 100),
        ),
        AlertRule(
            name="QueueLengthHigh",
            description="请求队列过长",
            severity=Severity.P2_MEDIUM,
            condition=lambda m: m.get("queue_length", 0) > 100,
        ),
    ]


# ============================================================
# 5. 主程序入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("告警策略模拟演示")
    print("=" * 60)

    # 初始化告警管理器
    manager = AlertManager()
    for rule in create_llm_alert_rules():
        manager.add_rule(rule)
    print(f"[AlertManager] 加载 {len(manager.rules)} 条告警规则")

    # --- 模拟正常状态 ---
    print("\n--- 正常状态 ---")
    normal_metrics = {
        "service_up": 1, "latency_p99": 2.5, "error_rate": 0.005,
        "gpu_memory_percent": 70, "daily_cost": 50, "daily_budget": 100,
        "queue_length": 10,
    }
    alerts = manager.evaluate(normal_metrics)
    print(f"  触发告警: {len(alerts)}")

    # --- 模拟异常状态 ---
    print("\n--- 异常状态（延迟飙升 + GPU 显存高）---")
    abnormal_metrics = {
        "service_up": 1, "latency_p99": 15.0, "error_rate": 0.005,
        "gpu_memory_percent": 95, "daily_cost": 50, "daily_budget": 100,
        "queue_length": 150,
    }
    alerts = manager.evaluate(abnormal_metrics)
    print(f"  触发告警: {len(alerts)}")

    # --- 模拟服务宕机 ---
    print("\n--- 服务宕机 ---")
    down_metrics = {
        "service_up": 0, "latency_p99": 0, "error_rate": 1.0,
        "gpu_memory_percent": 0, "daily_cost": 0, "daily_budget": 100,
        "queue_length": 0,
    }
    alerts = manager.evaluate(down_metrics)

    # --- 确认告警 ---
    print("\n--- 确认告警 ---")
    for alert in manager.active_alerts.values():
        manager.acknowledge(alert.alert_id, "oncall-engineer")
        print(f"  确认: {alert.rule_name} by oncall-engineer")
        break

    # --- 恢复 ---
    print("\n--- 恢复正常 ---")
    manager.evaluate(normal_metrics)

    # --- 告警摘要 ---
    print(f"\n--- 告警摘要 ---")
    summary = manager.get_summary()
    print(f"  活跃告警: {summary['active_alerts']}")
    print(f"  按级别: {summary['by_severity']}")
    print(f"  总触发: {summary['total_fired']}")
    print(f"  总恢复: {summary['total_resolved']}")
    print(f"  总通知: {summary['total_notifications']}")

    print("\n✅ 告警策略模拟演示完成！")
