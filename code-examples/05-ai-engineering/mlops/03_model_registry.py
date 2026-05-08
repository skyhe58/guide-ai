"""
模型版本管理模拟

知识点：Model Registry、模型版本化、阶段管理（Staging/Production/Archived）、
       模型血缘追踪、模型打包、版本回滚、访问控制

Python 版本：3.11+
依赖：标准库（默认模式）、mlflow>=2.0（服务模式）
最后验证：2024-12-01

外部服务（可选）：
  MLflow Model Registry
  启动命令：docker run -p 5000:5000 ghcr.io/mlflow/mlflow:latest mlflow server --host 0.0.0.0
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# ============================================================
# 1. 模型版本管理核心数据结构
# ============================================================

class ModelStage(Enum):
    """模型阶段"""
    NONE = "None"              # 未分配阶段
    DEVELOPMENT = "Development"  # 开发中
    STAGING = "Staging"        # 预发布
    PRODUCTION = "Production"  # 生产环境
    ARCHIVED = "Archived"      # 已归档


@dataclass
class ModelLineage:
    """模型血缘信息"""
    training_run_id: str          # 训练运行 ID
    dataset_version: str          # 数据集版本
    code_commit: str              # 代码 commit hash
    framework: str                # 训练框架
    hyperparameters: dict[str, Any] = field(default_factory=dict)  # 超参数
    environment: dict[str, str] = field(default_factory=dict)      # 环境信息


@dataclass
class ModelMetrics:
    """模型评估指标"""
    accuracy: float = 0.0
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p99_ms: float = 0.0
    model_size_mb: float = 0.0


@dataclass
class ModelVersion:
    """模型版本"""
    name: str                     # 模型名称
    version: int                  # 版本号
    stage: ModelStage             # 当前阶段
    description: str              # 版本描述
    created_at: float             # 创建时间
    updated_at: float             # 更新时间
    lineage: ModelLineage         # 血缘信息
    metrics: ModelMetrics         # 评估指标
    tags: dict[str, str] = field(default_factory=dict)  # 标签
    artifacts: dict[str, str] = field(default_factory=dict)  # 产物路径
    stage_history: list[dict] = field(default_factory=list)  # 阶段变更历史


@dataclass
class RegisteredModel:
    """注册模型"""
    name: str                     # 模型名称
    description: str              # 模型描述
    created_at: float             # 创建时间
    versions: dict[int, ModelVersion] = field(default_factory=dict)  # 版本列表
    latest_version: int = 0       # 最新版本号
    tags: dict[str, str] = field(default_factory=dict)  # 标签


# ============================================================
# 2. Model Registry 模拟实现
# ============================================================

class ModelRegistry:
    """
    模型注册表模拟

    模拟 MLflow Model Registry 的核心功能：
    - 模型注册和版本管理
    - 阶段管理（Development → Staging → Production → Archived）
    - 血缘追踪
    - 版本回滚
    - 访问控制
    """

    def __init__(self):
        # 注册模型存储
        self.models: dict[str, RegisteredModel] = {}
        # 操作审计日志
        self.audit_log: list[dict] = []
        print("[Registry] 模型注册表初始化完成")

    def _log_audit(self, action: str, model_name: str, details: dict) -> None:
        """记录审计日志"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "model_name": model_name,
            "details": details,
        }
        self.audit_log.append(entry)

    # --------------------------------------------------------
    # 模型注册
    # --------------------------------------------------------

    def create_registered_model(
        self,
        name: str,
        description: str = "",
        tags: dict[str, str] | None = None,
    ) -> RegisteredModel:
        """创建注册模型"""
        if name in self.models:
            print(f"[Registry] 模型已存在: {name}")
            return self.models[name]

        model = RegisteredModel(
            name=name,
            description=description,
            created_at=time.time(),
            tags=tags or {},
        )
        self.models[name] = model
        self._log_audit("create_model", name, {"description": description})
        print(f"[Registry] 创建模型: {name}")
        return model

    def register_model_version(
        self,
        name: str,
        description: str,
        lineage: ModelLineage,
        metrics: ModelMetrics,
        artifacts: dict[str, str] | None = None,
        tags: dict[str, str] | None = None,
    ) -> ModelVersion:
        """注册新的模型版本"""
        # 确保模型已注册
        if name not in self.models:
            self.create_registered_model(name)

        model = self.models[name]
        model.latest_version += 1
        version_num = model.latest_version

        # 创建版本
        version = ModelVersion(
            name=name,
            version=version_num,
            stage=ModelStage.NONE,
            description=description,
            created_at=time.time(),
            updated_at=time.time(),
            lineage=lineage,
            metrics=metrics,
            tags=tags or {},
            artifacts=artifacts or {},
        )

        model.versions[version_num] = version
        self._log_audit("register_version", name, {
            "version": version_num,
            "description": description,
        })

        print(f"[Registry] 注册版本: {name} v{version_num}")
        print(f"  描述: {description}")
        print(f"  指标: accuracy={metrics.accuracy:.4f}, f1={metrics.f1_score:.4f}")
        return version

    # --------------------------------------------------------
    # 阶段管理
    # --------------------------------------------------------

    def transition_model_version_stage(
        self,
        name: str,
        version: int,
        stage: ModelStage,
        comment: str = "",
    ) -> ModelVersion:
        """推进模型版本阶段"""
        model = self._get_model(name)
        model_version = self._get_version(model, version)

        old_stage = model_version.stage

        # 验证阶段转换合法性
        valid_transitions = {
            ModelStage.NONE: [ModelStage.DEVELOPMENT, ModelStage.STAGING],
            ModelStage.DEVELOPMENT: [ModelStage.STAGING, ModelStage.ARCHIVED],
            ModelStage.STAGING: [ModelStage.PRODUCTION, ModelStage.DEVELOPMENT, ModelStage.ARCHIVED],
            ModelStage.PRODUCTION: [ModelStage.ARCHIVED, ModelStage.STAGING],
            ModelStage.ARCHIVED: [ModelStage.STAGING],  # 允许从归档恢复
        }

        if stage not in valid_transitions.get(old_stage, []):
            raise ValueError(
                f"无效的阶段转换: {old_stage.value} → {stage.value}。"
                f"允许的转换: {[s.value for s in valid_transitions.get(old_stage, [])]}"
            )

        # 如果推进到 Production，归档当前 Production 版本
        if stage == ModelStage.PRODUCTION:
            current_prod = self.get_latest_versions(name, stages=[ModelStage.PRODUCTION])
            for prod_version in current_prod:
                if prod_version.version != version:
                    prod_version.stage = ModelStage.ARCHIVED
                    prod_version.stage_history.append({
                        "from": ModelStage.PRODUCTION.value,
                        "to": ModelStage.ARCHIVED.value,
                        "timestamp": datetime.now().isoformat(),
                        "comment": f"被 v{version} 替换",
                    })
                    print(f"  [自动归档] v{prod_version.version}: Production → Archived")

        # 更新阶段
        model_version.stage = stage
        model_version.updated_at = time.time()
        model_version.stage_history.append({
            "from": old_stage.value,
            "to": stage.value,
            "timestamp": datetime.now().isoformat(),
            "comment": comment,
        })

        self._log_audit("transition_stage", name, {
            "version": version,
            "from_stage": old_stage.value,
            "to_stage": stage.value,
            "comment": comment,
        })

        print(f"[Registry] 阶段推进: {name} v{version}: {old_stage.value} → {stage.value}")
        return model_version

    # --------------------------------------------------------
    # 查询功能
    # --------------------------------------------------------

    def get_latest_versions(
        self,
        name: str,
        stages: list[ModelStage] | None = None,
    ) -> list[ModelVersion]:
        """获取指定阶段的最新版本"""
        model = self._get_model(name)
        versions = list(model.versions.values())

        if stages:
            versions = [v for v in versions if v.stage in stages]

        return sorted(versions, key=lambda v: v.version, reverse=True)

    def get_model_version(self, name: str, version: int) -> ModelVersion:
        """获取指定版本"""
        model = self._get_model(name)
        return self._get_version(model, version)

    def search_model_versions(
        self,
        name: str,
        min_accuracy: float | None = None,
        stage: ModelStage | None = None,
    ) -> list[ModelVersion]:
        """搜索模型版本"""
        model = self._get_model(name)
        results = list(model.versions.values())

        if min_accuracy is not None:
            results = [v for v in results if v.metrics.accuracy >= min_accuracy]
        if stage is not None:
            results = [v for v in results if v.stage == stage]

        return sorted(results, key=lambda v: v.metrics.accuracy, reverse=True)

    # --------------------------------------------------------
    # 版本回滚
    # --------------------------------------------------------

    def rollback(self, name: str, target_version: int, reason: str = "") -> ModelVersion:
        """回滚到指定版本"""
        print(f"\n[Registry] 开始回滚: {name} → v{target_version}")
        print(f"  原因: {reason}")

        # 获取当前 Production 版本
        current_prod = self.get_latest_versions(name, stages=[ModelStage.PRODUCTION])

        if current_prod:
            current_v = current_prod[0].version
            print(f"  当前 Production: v{current_v}")
            # 归档当前版本
            self.transition_model_version_stage(
                name, current_v, ModelStage.ARCHIVED,
                comment=f"回滚: 被 v{target_version} 替换",
            )

        # 将目标版本推进到 Production
        target = self.get_model_version(name, target_version)
        if target.stage == ModelStage.ARCHIVED:
            # 先恢复到 Staging
            self.transition_model_version_stage(
                name, target_version, ModelStage.STAGING,
                comment="回滚: 从归档恢复",
            )

        self.transition_model_version_stage(
            name, target_version, ModelStage.PRODUCTION,
            comment=f"回滚: {reason}",
        )

        print(f"[Registry] 回滚完成: {name} v{target_version} → Production")
        return target

    # --------------------------------------------------------
    # 辅助方法
    # --------------------------------------------------------

    def _get_model(self, name: str) -> RegisteredModel:
        if name not in self.models:
            raise ValueError(f"模型不存在: {name}")
        return self.models[name]

    def _get_version(self, model: RegisteredModel, version: int) -> ModelVersion:
        if version not in model.versions:
            raise ValueError(f"版本不存在: {model.name} v{version}")
        return model.versions[version]

    def print_model_status(self, name: str) -> None:
        """打印模型状态"""
        model = self._get_model(name)
        print(f"\n{'='*60}")
        print(f"模型: {name}")
        print(f"描述: {model.description}")
        print(f"版本数: {len(model.versions)}")
        print(f"{'='*60}")

        print(f"\n{'版本':<6} {'阶段':<15} {'准确率':<10} {'F1':<10} {'描述'}")
        print("-" * 70)
        for v in sorted(model.versions.values(), key=lambda x: x.version):
            print(
                f"  v{v.version:<4} {v.stage.value:<15} "
                f"{v.metrics.accuracy:<10.4f} {v.metrics.f1_score:<10.4f} "
                f"{v.description[:30]}"
            )

    def print_audit_log(self, last_n: int = 10) -> None:
        """打印审计日志"""
        print(f"\n{'='*60}")
        print(f"审计日志（最近 {last_n} 条）")
        print(f"{'='*60}")
        for entry in self.audit_log[-last_n:]:
            print(f"  [{entry['timestamp']}] {entry['action']}: "
                  f"{entry['model_name']} — {entry['details']}")


# ============================================================
# 3. 质量门禁
# ============================================================

class QualityGate:
    """模型质量门禁"""

    def __init__(self, thresholds: dict[str, float]):
        self.thresholds = thresholds

    def check(self, metrics: ModelMetrics) -> tuple[bool, dict]:
        """检查模型是否通过质量门禁"""
        results = {}
        metrics_dict = {
            "accuracy": metrics.accuracy,
            "f1_score": metrics.f1_score,
            "latency_p99_ms": metrics.latency_p99_ms,
        }

        for metric_name, threshold in self.thresholds.items():
            actual = metrics_dict.get(metric_name, 0)
            # 延迟指标是越小越好
            if "latency" in metric_name:
                passed = actual <= threshold
            else:
                passed = actual >= threshold

            results[metric_name] = {
                "actual": actual,
                "threshold": threshold,
                "passed": passed,
            }

        all_passed = all(r["passed"] for r in results.values())
        return all_passed, results


# ============================================================
# 4. 主程序入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("模型版本管理模拟演示")
    print("=" * 60)

    # 初始化注册表
    registry = ModelRegistry()

    # 创建注册模型
    registry.create_registered_model(
        name="text-classifier",
        description="中文文本分类模型",
        tags={"team": "nlp", "task": "classification"},
    )

    # --- 注册多个版本 ---
    print("\n--- 注册模型版本 ---")

    versions_data = [
        ("BERT-base 基线模型", 0.92, 0.90, 45, "v1.0"),
        ("BERT-base + 数据增强", 0.94, 0.92, 48, "v1.1"),
        ("BERT-large 模型", 0.96, 0.95, 120, "v2.0"),
        ("DistilBERT 轻量模型", 0.91, 0.89, 20, "v3.0"),
    ]

    for desc, acc, f1, latency, dataset_v in versions_data:
        lineage = ModelLineage(
            training_run_id=hashlib.md5(desc.encode()).hexdigest()[:12],
            dataset_version=dataset_v,
            code_commit=hashlib.md5(desc.encode()).hexdigest()[:7],
            framework="pytorch",
            hyperparameters={"lr": 3e-5, "epochs": 10},
            environment={"python": "3.11", "torch": "2.1.0"},
        )
        metrics = ModelMetrics(
            accuracy=acc, f1_score=f1,
            precision=acc * 0.98, recall=f1 * 0.97,
            latency_p50_ms=latency * 0.6, latency_p99_ms=latency,
            model_size_mb=400 if "large" in desc.lower() else 200,
        )
        registry.register_model_version(
            "text-classifier", desc, lineage, metrics,
        )

    # --- 阶段推进 ---
    print("\n--- 阶段推进 ---")

    # v1 → Development → Staging → Production
    registry.transition_model_version_stage(
        "text-classifier", 1, ModelStage.STAGING, "基线模型评估"
    )
    registry.transition_model_version_stage(
        "text-classifier", 1, ModelStage.PRODUCTION, "基线模型上线"
    )

    # v3 → Staging → Production（自动归档 v1）
    registry.transition_model_version_stage(
        "text-classifier", 3, ModelStage.STAGING, "BERT-large 评估"
    )

    # 质量门禁检查
    print("\n--- 质量门禁检查 ---")
    gate = QualityGate(thresholds={
        "accuracy": 0.95,
        "f1_score": 0.93,
        "latency_p99_ms": 100,
    })

    v3 = registry.get_model_version("text-classifier", 3)
    passed, details = gate.check(v3.metrics)
    print(f"v3 质量门禁: {'通过 ✅' if passed else '未通过 ❌'}")
    for metric, result in details.items():
        status = "✅" if result["passed"] else "❌"
        print(f"  {metric}: {result['actual']:.2f} (阈值: {result['threshold']}) {status}")

    if passed:
        registry.transition_model_version_stage(
            "text-classifier", 3, ModelStage.PRODUCTION, "质量门禁通过"
        )

    # --- 模型状态 ---
    registry.print_model_status("text-classifier")

    # --- 版本回滚 ---
    print("\n--- 版本回滚演示 ---")
    registry.rollback("text-classifier", 1, "v3 延迟过高，回滚到 v1")

    registry.print_model_status("text-classifier")

    # --- 审计日志 ---
    registry.print_audit_log()

    print("\n✅ 模型版本管理模拟演示完成！")
