"""
Weights & Biases 实验管理模拟

知识点：W&B 实验追踪、项目管理、运行配置、指标可视化、
       表格日志、Sweep 超参数搜索、团队协作、报告生成

Python 版本：3.11+
依赖：标准库（默认模式）、wandb>=0.15（服务模式）
最后验证：2024-12-01

外部服务（可选）：
  Weights & Biases 云服务
  注册地址：https://wandb.ai/
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# ============================================================
# 1. W&B 核心数据结构模拟
# ============================================================

class RunState(Enum):
    """运行状态"""
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    CRASHED = "crashed"


@dataclass
class WandbConfig:
    """W&B 运行配置"""
    data: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def update(self, d: dict[str, Any]) -> None:
        self.data.update(d)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)


@dataclass
class WandbTable:
    """W&B 表格（用于记录结构化数据）"""
    columns: list[str]
    data: list[list[Any]] = field(default_factory=list)

    def add_data(self, *args: Any) -> None:
        """添加一行数据"""
        if len(args) != len(self.columns):
            raise ValueError(f"期望 {len(self.columns)} 列，实际 {len(args)} 列")
        self.data.append(list(args))

    def __len__(self) -> int:
        return len(self.data)

    def to_dict(self) -> dict:
        return {"columns": self.columns, "data": self.data}


@dataclass
class WandbArtifact:
    """W&B 产物"""
    name: str
    artifact_type: str
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    files: list[str] = field(default_factory=list)

    def add_file(self, path: str) -> None:
        """添加文件到产物"""
        self.files.append(path)
        print(f"  [Artifact] 添加文件: {path}")


@dataclass
class WandbSummary:
    """W&B 运行摘要（最终指标）"""
    data: dict[str, Any] = field(default_factory=dict)

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)


# ============================================================
# 2. W&B Run 模拟实现
# ============================================================

class WandbRun:
    """
    W&B 运行模拟

    模拟 wandb.init() 返回的 Run 对象，
    支持配置、日志、表格、产物等功能。
    """

    def __init__(
        self,
        project: str,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
    ):
        # 运行基本信息
        self.id = hashlib.md5(f"{time.time()}{random.random()}".encode()).hexdigest()[:8]
        self.project = project
        self.name = name or f"run-{self.id}"
        self.state = RunState.RUNNING
        self.start_time = time.time()
        self.tags = tags or []
        self.notes = notes or ""

        # 配置
        self.config = WandbConfig(data=config or {})

        # 指标历史
        self.history: list[dict[str, Any]] = []
        self._step = 0

        # 摘要（最终指标）
        self.summary = WandbSummary()

        # 表格
        self.tables: dict[str, WandbTable] = {}

        # 产物
        self.artifacts: list[WandbArtifact] = []

        # 系统指标
        self.system_metrics: list[dict[str, float]] = []

        print(f"[W&B] 初始化运行: {self.name} (项目: {self.project})")
        print(f"[W&B] 运行 ID: {self.id}")
        if config:
            print(f"[W&B] 配置: {json.dumps(config, indent=2, default=str)}")

    def log(self, data: dict[str, Any], step: int | None = None, commit: bool = True) -> None:
        """记录指标"""
        if step is not None:
            self._step = step
        else:
            self._step += 1

        log_entry = {"_step": self._step, **data}
        self.history.append(log_entry)

        # 更新摘要（保留最新值）
        for key, value in data.items():
            if isinstance(value, (int, float)):
                self.summary[key] = value

    def log_table(self, key: str, table: WandbTable) -> None:
        """记录表格"""
        self.tables[key] = table
        print(f"[W&B] 记录表格: {key} ({len(table)} 行)")

    def log_artifact(self, artifact: WandbArtifact) -> None:
        """记录产物"""
        self.artifacts.append(artifact)
        print(f"[W&B] 记录产物: {artifact.name} (类型: {artifact.artifact_type})")

    def finish(self, exit_code: int = 0) -> None:
        """结束运行"""
        self.state = RunState.FINISHED if exit_code == 0 else RunState.FAILED
        duration = time.time() - self.start_time

        print(f"\n[W&B] 运行结束: {self.name}")
        print(f"[W&B] 状态: {self.state.value}")
        print(f"[W&B] 耗时: {duration:.1f}s")
        print(f"[W&B] 记录步数: {self._step}")
        print(f"[W&B] 摘要: {json.dumps(self.summary.to_dict(), indent=2, default=str)}")

    def get_history(self) -> list[dict[str, Any]]:
        """获取指标历史"""
        return self.history

    def alert(self, title: str, text: str, level: str = "INFO") -> None:
        """发送告警"""
        print(f"[W&B Alert] [{level}] {title}: {text}")


# ============================================================
# 3. W&B 项目管理模拟
# ============================================================

class WandbProject:
    """
    W&B 项目管理模拟

    管理一个项目下的所有运行，
    支持运行搜索、对比、报告生成。
    """

    def __init__(self, name: str):
        self.name = name
        self.runs: list[WandbRun] = []
        print(f"[W&B] 项目: {name}")

    def init_run(self, **kwargs: Any) -> WandbRun:
        """初始化一个新运行"""
        run = WandbRun(project=self.name, **kwargs)
        self.runs.append(run)
        return run

    def get_runs(
        self,
        filters: dict[str, Any] | None = None,
        order: str = "-summary.best_accuracy",
    ) -> list[WandbRun]:
        """获取运行列表"""
        runs = list(self.runs)

        # 过滤
        if filters:
            for key, value in filters.items():
                if key.startswith("config."):
                    config_key = key.replace("config.", "")
                    runs = [
                        r for r in runs
                        if r.config.data.get(config_key) == value
                    ]
                elif key == "state":
                    runs = [r for r in runs if r.state.value == value]

        # 排序
        desc = order.startswith("-")
        sort_key = order.lstrip("-").replace("summary.", "")

        def get_sort_value(run: WandbRun) -> float:
            return run.summary.data.get(sort_key, 0)

        runs.sort(key=get_sort_value, reverse=desc)
        return runs

    def generate_report(self) -> str:
        """生成项目报告"""
        report_lines = [
            f"# 项目报告: {self.name}",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"总运行数: {len(self.runs)}",
            "",
            "## 运行摘要",
            "",
        ]

        # 按准确率排序
        sorted_runs = self.get_runs(order="-best_accuracy")

        for i, run in enumerate(sorted_runs):
            acc = run.summary.data.get("best_accuracy", "N/A")
            loss = run.summary.data.get("train_loss", "N/A")
            report_lines.append(
                f"{i+1}. **{run.name}** — "
                f"准确率: {acc}, 损失: {loss}, "
                f"配置: {json.dumps(run.config.to_dict(), default=str)}"
            )

        # 统计分析
        accuracies = [
            r.summary.data.get("best_accuracy", 0)
            for r in self.runs
            if "best_accuracy" in r.summary.data
        ]

        if accuracies:
            report_lines.extend([
                "",
                "## 统计分析",
                f"- 最高准确率: {max(accuracies):.4f}",
                f"- 最低准确率: {min(accuracies):.4f}",
                f"- 平均准确率: {statistics.mean(accuracies):.4f}",
                f"- 标准差: {statistics.stdev(accuracies):.4f}" if len(accuracies) > 1 else "",
            ])

        return "\n".join(report_lines)


# ============================================================
# 4. W&B Sweep 超参数搜索模拟
# ============================================================

class SweepMethod(Enum):
    """搜索方法"""
    GRID = "grid"       # 网格搜索
    RANDOM = "random"   # 随机搜索
    BAYES = "bayes"     # 贝叶斯优化


@dataclass
class SweepConfig:
    """Sweep 配置"""
    method: SweepMethod
    metric: dict[str, str]  # {"name": "val_accuracy", "goal": "maximize"}
    parameters: dict[str, dict[str, Any]]  # 参数空间


class WandbSweep:
    """
    W&B Sweep 超参数搜索模拟

    模拟 W&B 的 Sweep 功能，
    支持随机搜索和简单的贝叶斯优化。
    """

    def __init__(self, project: WandbProject, config: SweepConfig):
        self.project = project
        self.config = config
        self.sweep_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        self.trials: list[dict] = []

        print(f"[W&B Sweep] 创建 Sweep: {self.sweep_id}")
        print(f"[W&B Sweep] 方法: {config.method.value}")
        print(f"[W&B Sweep] 优化目标: {config.metric}")

    def _sample_params(self) -> dict[str, Any]:
        """采样超参数"""
        params = {}
        for name, spec in self.config.parameters.items():
            if "values" in spec:
                # 离散值
                params[name] = random.choice(spec["values"])
            elif "min" in spec and "max" in spec:
                if spec.get("distribution") == "log_uniform":
                    # 对数均匀分布
                    log_min = math.log(spec["min"])
                    log_max = math.log(spec["max"])
                    params[name] = math.exp(random.uniform(log_min, log_max))
                else:
                    # 均匀分布
                    params[name] = random.uniform(spec["min"], spec["max"])
        return params

    def run_agent(self, train_fn: Any, count: int = 10) -> dict:
        """运行 Sweep Agent"""
        print(f"\n[W&B Sweep] 开始搜索: {count} 次试验")

        for trial_idx in range(count):
            # 采样参数
            params = self._sample_params()
            print(f"\n--- Sweep 试验 {trial_idx + 1}/{count} ---")

            # 初始化运行
            run = self.project.init_run(
                name=f"sweep-{self.sweep_id}-{trial_idx}",
                config=params,
                tags=["sweep", self.sweep_id],
            )

            # 执行训练
            try:
                result = train_fn(run, params)
                run.finish(exit_code=0)
            except Exception as e:
                print(f"[错误] 试验失败: {e}")
                run.finish(exit_code=1)
                result = {}

            self.trials.append({
                "trial": trial_idx,
                "params": params,
                "result": result,
                "run_id": run.id,
            })

        # 找到最佳试验
        metric_name = self.config.metric["name"]
        goal = self.config.metric["goal"]

        best_trial = max(
            self.trials,
            key=lambda t: t["result"].get(metric_name, 0)
            if goal == "maximize"
            else -t["result"].get(metric_name, float("inf")),
        )

        print(f"\n[W&B Sweep] 搜索完成！")
        print(f"[W&B Sweep] 最佳试验: #{best_trial['trial']}")
        print(f"[W&B Sweep] 最佳 {metric_name}: {best_trial['result'].get(metric_name, 'N/A')}")
        print(f"[W&B Sweep] 最佳参数: {json.dumps(best_trial['params'], indent=2, default=str)}")

        return best_trial


# ============================================================
# 5. 模拟训练函数
# ============================================================

def simulated_train(run: WandbRun, params: dict[str, Any]) -> dict[str, float]:
    """模拟训练过程"""
    epochs = params.get("epochs", 10)
    lr = params.get("learning_rate", 0.001)
    hidden = params.get("hidden_size", 128)

    best_accuracy = 0.0

    for epoch in range(epochs):
        # 模拟训练指标
        base_loss = 2.0 * math.exp(-0.3 * epoch * (lr * 1000)) + 0.1
        train_loss = max(0.01, base_loss + random.gauss(0, 0.05))

        base_acc = 1.0 - math.exp(-0.15 * epoch * math.log(hidden + 1)) * 0.5
        val_accuracy = min(0.99, max(0.5, base_acc + random.gauss(0, 0.02)))

        # 记录指标
        run.log({
            "train_loss": round(train_loss, 4),
            "val_accuracy": round(val_accuracy, 4),
            "epoch": epoch,
            "learning_rate": lr,
        })

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy

    # 更新摘要
    run.summary["best_accuracy"] = round(best_accuracy, 4)
    run.summary["final_loss"] = round(train_loss, 4)

    # 记录预测表格
    table = WandbTable(columns=["text", "prediction", "label", "correct"])
    sample_texts = ["好评", "差评", "一般", "推荐", "不推荐"]
    for text in sample_texts:
        pred = random.choice(["正面", "负面", "中性"])
        label = random.choice(["正面", "负面", "中性"])
        table.add_data(text, pred, label, pred == label)
    run.log_table("predictions", table)

    return {"best_accuracy": best_accuracy, "final_loss": train_loss}


# ============================================================
# 6. 主程序入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Weights & Biases 实验管理模拟演示")
    print("=" * 60)

    # 创建项目
    project = WandbProject("text-classification")

    # --- 演示 1: 基础实验追踪 ---
    print("\n" + "=" * 60)
    print("演示 1: 基础实验追踪")
    print("=" * 60)

    run1 = project.init_run(
        name="bert-base-v1",
        config={
            "model": "bert-base-chinese",
            "learning_rate": 3e-5,
            "hidden_size": 128,
            "epochs": 8,
        },
        tags=["baseline", "bert"],
        notes="BERT-base 基线实验",
    )
    result1 = simulated_train(run1, run1.config.to_dict())
    run1.finish()

    run2 = project.init_run(
        name="bert-large-v1",
        config={
            "model": "bert-large-chinese",
            "learning_rate": 2e-5,
            "hidden_size": 256,
            "epochs": 8,
        },
        tags=["experiment", "bert"],
    )
    result2 = simulated_train(run2, run2.config.to_dict())
    run2.finish()

    # --- 演示 2: Sweep 超参数搜索 ---
    print("\n" + "=" * 60)
    print("演示 2: Sweep 超参数搜索")
    print("=" * 60)

    sweep_config = SweepConfig(
        method=SweepMethod.RANDOM,
        metric={"name": "best_accuracy", "goal": "maximize"},
        parameters={
            "learning_rate": {"min": 1e-5, "max": 1e-2, "distribution": "log_uniform"},
            "hidden_size": {"values": [64, 128, 256, 512]},
            "epochs": {"values": [5, 8, 10]},
        },
    )

    sweep = WandbSweep(project, sweep_config)
    best = sweep.run_agent(simulated_train, count=5)

    # --- 演示 3: 项目报告 ---
    print("\n" + "=" * 60)
    print("演示 3: 项目报告")
    print("=" * 60)

    report = project.generate_report()
    print(report)

    print("\n✅ W&B 实验管理模拟演示完成！")
