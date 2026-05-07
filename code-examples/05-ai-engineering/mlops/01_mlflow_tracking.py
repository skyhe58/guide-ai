"""
MLflow 实验追踪模拟

知识点：MLflow Tracking API、实验管理、参数记录、指标记录、
       产物管理、模型日志、实验对比、超参数搜索集成

Python 版本：3.11+
依赖：标准库（默认模式）、mlflow>=2.0（服务模式）
最后验证：2024-12-01

外部服务（可选）：
  MLflow Tracking Server
  启动命令：docker run -p 5000:5000 ghcr.io/mlflow/mlflow:latest mlflow server --host 0.0.0.0
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


# ============================================================
# 1. MLflow 核心数据结构模拟
# ============================================================

class RunStatus(Enum):
    """实验运行状态"""
    RUNNING = "RUNNING"       # 运行中
    FINISHED = "FINISHED"     # 已完成
    FAILED = "FAILED"         # 失败
    KILLED = "KILLED"         # 被终止


@dataclass
class Param:
    """实验参数"""
    key: str          # 参数名
    value: str        # 参数值（MLflow 中参数值都是字符串）


@dataclass
class Metric:
    """实验指标"""
    key: str          # 指标名
    value: float      # 指标值
    timestamp: float  # 记录时间戳
    step: int         # 步骤编号


@dataclass
class Artifact:
    """实验产物"""
    path: str         # 产物路径
    file_size: int    # 文件大小（字节）
    content: Any      # 产物内容


@dataclass
class RunInfo:
    """运行信息"""
    run_id: str                    # 运行 ID
    experiment_id: str             # 实验 ID
    run_name: str                  # 运行名称
    status: RunStatus              # 运行状态
    start_time: float              # 开始时间
    end_time: float | None = None  # 结束时间
    tags: dict[str, str] = field(default_factory=dict)  # 标签


@dataclass
class Run:
    """一次实验运行"""
    info: RunInfo
    params: dict[str, Param] = field(default_factory=dict)
    metrics: dict[str, list[Metric]] = field(default_factory=dict)
    artifacts: dict[str, Artifact] = field(default_factory=dict)


# ============================================================
# 2. MLflow Tracking 模拟实现
# ============================================================

class MLflowTracker:
    """
    MLflow 实验追踪模拟器

    模拟 MLflow 的核心功能：
    - 实验管理（创建、查询）
    - 运行管理（开始、结束、记录）
    - 参数记录
    - 指标记录（支持按步骤记录）
    - 产物管理
    - 实验对比
    """

    def __init__(self, tracking_uri: str = "local"):
        # 追踪服务地址
        self.tracking_uri = tracking_uri
        # 实验存储：experiment_id -> experiment_name
        self.experiments: dict[str, str] = {}
        # 运行存储：run_id -> Run
        self.runs: dict[str, Run] = {}
        # 当前活跃的运行
        self._active_run: Run | None = None
        # 当前实验 ID
        self._current_experiment_id: str | None = None

        print(f"[MLflow] 追踪服务初始化: {tracking_uri}")

    def create_experiment(self, name: str) -> str:
        """创建实验"""
        # 检查是否已存在
        for exp_id, exp_name in self.experiments.items():
            if exp_name == name:
                print(f"[MLflow] 实验已存在: {name} (ID: {exp_id})")
                return exp_id

        # 生成实验 ID
        exp_id = hashlib.md5(name.encode()).hexdigest()[:8]
        self.experiments[exp_id] = name
        print(f"[MLflow] 创建实验: {name} (ID: {exp_id})")
        return exp_id

    def set_experiment(self, name: str) -> str:
        """设置当前实验"""
        exp_id = self.create_experiment(name)
        self._current_experiment_id = exp_id
        return exp_id

    def start_run(self, run_name: str | None = None) -> Run:
        """开始一次实验运行"""
        if self._active_run is not None:
            raise RuntimeError("已有活跃的运行，请先结束当前运行")

        # 生成运行 ID
        run_id = hashlib.md5(
            f"{time.time()}{random.random()}".encode()
        ).hexdigest()[:12]

        # 创建运行信息
        run_info = RunInfo(
            run_id=run_id,
            experiment_id=self._current_experiment_id or "default",
            run_name=run_name or f"run-{run_id[:6]}",
            status=RunStatus.RUNNING,
            start_time=time.time(),
        )

        run = Run(info=run_info)
        self.runs[run_id] = run
        self._active_run = run

        print(f"[MLflow] 开始运行: {run_info.run_name} (ID: {run_id})")
        return run

    def end_run(self, status: RunStatus = RunStatus.FINISHED) -> None:
        """结束当前运行"""
        if self._active_run is None:
            raise RuntimeError("没有活跃的运行")

        self._active_run.info.status = status
        self._active_run.info.end_time = time.time()
        duration = self._active_run.info.end_time - self._active_run.info.start_time

        print(
            f"[MLflow] 运行结束: {self._active_run.info.run_name} "
            f"(状态: {status.value}, 耗时: {duration:.1f}s)"
        )
        self._active_run = None

    def log_param(self, key: str, value: Any) -> None:
        """记录单个参数"""
        self._check_active_run()
        param = Param(key=key, value=str(value))
        self._active_run.params[key] = param

    def log_params(self, params: dict[str, Any]) -> None:
        """批量记录参数"""
        for key, value in params.items():
            self.log_param(key, value)
        print(f"[MLflow] 记录 {len(params)} 个参数")

    def log_metric(self, key: str, value: float, step: int = 0) -> None:
        """记录单个指标"""
        self._check_active_run()
        metric = Metric(
            key=key,
            value=value,
            timestamp=time.time(),
            step=step,
        )
        if key not in self._active_run.metrics:
            self._active_run.metrics[key] = []
        self._active_run.metrics[key].append(metric)

    def log_metrics(self, metrics: dict[str, float], step: int = 0) -> None:
        """批量记录指标"""
        for key, value in metrics.items():
            self.log_metric(key, value, step)

    def log_artifact(self, path: str, content: Any = None) -> None:
        """记录产物"""
        self._check_active_run()
        artifact = Artifact(
            path=path,
            file_size=len(str(content)) if content else 0,
            content=content,
        )
        self._active_run.artifacts[path] = artifact
        print(f"[MLflow] 记录产物: {path}")

    def set_tag(self, key: str, value: str) -> None:
        """设置标签"""
        self._check_active_run()
        self._active_run.info.tags[key] = value

    def set_tags(self, tags: dict[str, str]) -> None:
        """批量设置标签"""
        for key, value in tags.items():
            self.set_tag(key, value)

    def _check_active_run(self) -> None:
        """检查是否有活跃的运行"""
        if self._active_run is None:
            raise RuntimeError("没有活跃的运行，请先调用 start_run()")

    # --------------------------------------------------------
    # 查询和对比功能
    # --------------------------------------------------------

    def search_runs(
        self,
        experiment_id: str | None = None,
        filter_string: str | None = None,
        order_by: str | None = None,
    ) -> list[Run]:
        """搜索实验运行"""
        results = list(self.runs.values())

        # 按实验 ID 过滤
        if experiment_id:
            results = [r for r in results if r.info.experiment_id == experiment_id]

        # 按指标排序
        if order_by:
            metric_name = order_by.replace("metrics.", "").replace(" DESC", "").strip()
            desc = "DESC" in order_by

            def get_metric_value(run: Run) -> float:
                if metric_name in run.metrics:
                    return run.metrics[metric_name][-1].value
                return float("-inf")

            results.sort(key=get_metric_value, reverse=desc)

        return results

    def compare_runs(self, run_ids: list[str]) -> dict:
        """对比多个运行"""
        comparison = {"runs": []}

        for run_id in run_ids:
            run = self.runs.get(run_id)
            if run is None:
                continue

            run_summary = {
                "run_id": run_id,
                "run_name": run.info.run_name,
                "status": run.info.status.value,
                "params": {k: v.value for k, v in run.params.items()},
                "final_metrics": {
                    k: v[-1].value for k, v in run.metrics.items()
                },
            }
            comparison["runs"].append(run_summary)

        return comparison

    def get_best_run(self, experiment_id: str, metric: str) -> Run | None:
        """获取最佳运行"""
        runs = self.search_runs(
            experiment_id=experiment_id,
            order_by=f"metrics.{metric} DESC",
        )
        return runs[0] if runs else None


# ============================================================
# 3. 模拟训练过程
# ============================================================

class SimpleModel:
    """简单模型模拟（用于演示实验追踪）"""

    def __init__(self, learning_rate: float, hidden_size: int, dropout: float):
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.weights = [random.gauss(0, 0.1) for _ in range(hidden_size)]

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """模拟一个训练 epoch"""
        # 模拟训练损失递减（加入随机波动）
        base_loss = 2.0 * math.exp(-0.3 * epoch) + 0.1
        noise = random.gauss(0, 0.05)
        train_loss = max(0.01, base_loss + noise)

        # 模拟验证准确率递增
        base_acc = 1.0 - math.exp(-0.2 * epoch) * 0.5
        acc_noise = random.gauss(0, 0.02)
        val_accuracy = min(0.99, max(0.5, base_acc + acc_noise))

        # 模拟 F1 分数
        val_f1 = val_accuracy * random.uniform(0.95, 1.0)

        return {
            "train_loss": round(train_loss, 4),
            "val_accuracy": round(val_accuracy, 4),
            "val_f1": round(val_f1, 4),
        }

    def get_model_size(self) -> int:
        """获取模型大小（字节）"""
        return self.hidden_size * 4  # 每个权重 4 字节


def run_training_experiment(
    tracker: MLflowTracker,
    model_name: str,
    config: dict[str, Any],
) -> str:
    """运行一次训练实验"""
    # 开始运行
    run = tracker.start_run(run_name=f"{model_name}-{config.get('lr', 0.001)}")

    try:
        # 记录参数
        tracker.log_params({
            "model_name": model_name,
            "learning_rate": config["lr"],
            "hidden_size": config["hidden_size"],
            "dropout": config["dropout"],
            "epochs": config["epochs"],
            "batch_size": config.get("batch_size", 32),
            "optimizer": config.get("optimizer", "adam"),
        })

        # 记录标签
        tracker.set_tags({
            "author": "demo",
            "dataset": "text-classification-v1",
            "framework": "pytorch",
        })

        # 创建模型
        model = SimpleModel(
            learning_rate=config["lr"],
            hidden_size=config["hidden_size"],
            dropout=config["dropout"],
        )

        # 训练循环
        best_accuracy = 0.0
        for epoch in range(config["epochs"]):
            metrics = model.train_epoch(epoch)
            tracker.log_metrics(metrics, step=epoch)

            if metrics["val_accuracy"] > best_accuracy:
                best_accuracy = metrics["val_accuracy"]

            if epoch % 3 == 0:
                print(
                    f"  Epoch {epoch}: loss={metrics['train_loss']:.4f}, "
                    f"acc={metrics['val_accuracy']:.4f}"
                )

        # 记录最终指标
        tracker.log_metric("best_accuracy", best_accuracy)
        tracker.log_metric("model_size_bytes", model.get_model_size())

        # 记录产物
        tracker.log_artifact(
            "model.pt",
            content={"weights_count": len(model.weights)},
        )
        tracker.log_artifact(
            "config.json",
            content=config,
        )

        # 结束运行
        tracker.end_run(RunStatus.FINISHED)
        return run.info.run_id

    except Exception as e:
        print(f"[错误] 训练失败: {e}")
        tracker.end_run(RunStatus.FAILED)
        raise


# ============================================================
# 4. 超参数搜索集成
# ============================================================

class HyperparameterSearch:
    """
    超参数搜索模拟

    模拟 Optuna 风格的超参数搜索，
    与 MLflow 实验追踪集成。
    """

    def __init__(self, tracker: MLflowTracker, n_trials: int = 10):
        self.tracker = tracker
        self.n_trials = n_trials
        self.results: list[dict] = []

    def suggest_params(self) -> dict[str, Any]:
        """随机采样超参数"""
        return {
            "lr": 10 ** random.uniform(-5, -2),           # 1e-5 到 1e-2
            "hidden_size": random.choice([64, 128, 256, 512]),
            "dropout": random.uniform(0.1, 0.5),
            "epochs": random.choice([5, 10, 15]),
            "batch_size": random.choice([16, 32, 64]),
            "optimizer": random.choice(["adam", "sgd", "adamw"]),
        }

    def run(self, experiment_name: str) -> dict:
        """运行超参数搜索"""
        print(f"\n{'='*60}")
        print(f"开始超参数搜索: {self.n_trials} 次试验")
        print(f"{'='*60}")

        self.tracker.set_experiment(experiment_name)

        for trial in range(self.n_trials):
            params = self.suggest_params()
            print(f"\n--- 试验 {trial + 1}/{self.n_trials} ---")
            print(f"  参数: lr={params['lr']:.6f}, hidden={params['hidden_size']}, "
                  f"dropout={params['dropout']:.2f}")

            run_id = run_training_experiment(
                self.tracker,
                model_name=f"trial-{trial}",
                config=params,
            )

            # 获取最终指标
            run = self.tracker.runs[run_id]
            best_acc = run.metrics.get("best_accuracy", [Metric("", 0, 0, 0)])[-1].value

            self.results.append({
                "trial": trial,
                "run_id": run_id,
                "params": params,
                "best_accuracy": best_acc,
            })

        # 找到最佳试验
        best_trial = max(self.results, key=lambda x: x["best_accuracy"])
        print(f"\n{'='*60}")
        print(f"最佳试验: #{best_trial['trial']}")
        print(f"  最佳准确率: {best_trial['best_accuracy']:.4f}")
        print(f"  最佳参数: lr={best_trial['params']['lr']:.6f}, "
              f"hidden={best_trial['params']['hidden_size']}")
        print(f"{'='*60}")

        return best_trial


# ============================================================
# 5. 实验对比和分析
# ============================================================

def analyze_experiments(tracker: MLflowTracker) -> None:
    """分析和对比实验结果"""
    print(f"\n{'='*60}")
    print("实验分析报告")
    print(f"{'='*60}")

    # 统计信息
    total_runs = len(tracker.runs)
    finished_runs = sum(
        1 for r in tracker.runs.values()
        if r.info.status == RunStatus.FINISHED
    )
    failed_runs = total_runs - finished_runs

    print(f"\n总运行数: {total_runs}")
    print(f"成功: {finished_runs}, 失败: {failed_runs}")

    # 按准确率排序
    sorted_runs = sorted(
        tracker.runs.values(),
        key=lambda r: (
            r.metrics.get("best_accuracy", [Metric("", 0, 0, 0)])[-1].value
        ),
        reverse=True,
    )

    print(f"\n{'排名':<4} {'运行名称':<25} {'准确率':<10} {'学习率':<12} {'隐藏层':<8}")
    print("-" * 60)

    for i, run in enumerate(sorted_runs[:5]):
        acc = run.metrics.get("best_accuracy", [Metric("", 0, 0, 0)])[-1].value
        lr = run.params.get("learning_rate", Param("", "N/A")).value
        hidden = run.params.get("hidden_size", Param("", "N/A")).value
        print(f"  {i+1:<4} {run.info.run_name:<25} {acc:<10.4f} {lr:<12} {hidden:<8}")

    # 对比前两名
    if len(sorted_runs) >= 2:
        top_ids = [sorted_runs[0].info.run_id, sorted_runs[1].info.run_id]
        comparison = tracker.compare_runs(top_ids)
        print(f"\n前两名对比:")
        for run_data in comparison["runs"]:
            print(f"  {run_data['run_name']}:")
            print(f"    参数: {json.dumps(run_data['params'], indent=6)}")
            print(f"    指标: {run_data['final_metrics']}")


# ============================================================
# 6. 主程序入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MLflow 实验追踪模拟演示")
    print("=" * 60)

    # 初始化追踪器
    tracker = MLflowTracker(tracking_uri="local")

    # --- 演示 1: 单次实验 ---
    print("\n" + "=" * 60)
    print("演示 1: 单次训练实验")
    print("=" * 60)

    tracker.set_experiment("text-classification")

    config_1 = {
        "lr": 0.001,
        "hidden_size": 128,
        "dropout": 0.3,
        "epochs": 10,
        "batch_size": 32,
    }
    run_id_1 = run_training_experiment(tracker, "bert-base", config_1)

    config_2 = {
        "lr": 0.0005,
        "hidden_size": 256,
        "dropout": 0.2,
        "epochs": 10,
        "batch_size": 64,
    }
    run_id_2 = run_training_experiment(tracker, "bert-large", config_2)

    # 对比两次实验
    comparison = tracker.compare_runs([run_id_1, run_id_2])
    print("\n实验对比:")
    for run_data in comparison["runs"]:
        print(f"  {run_data['run_name']}: {run_data['final_metrics']}")

    # --- 演示 2: 超参数搜索 ---
    print("\n" + "=" * 60)
    print("演示 2: 超参数搜索")
    print("=" * 60)

    search = HyperparameterSearch(tracker, n_trials=5)
    best = search.run("hyperparam-search")

    # --- 演示 3: 实验分析 ---
    analyze_experiments(tracker)

    print("\n✅ MLflow 实验追踪模拟演示完成！")
