"""
数据标注模拟

知识点：标注任务管理、标注规范设计、质量控制（Cohen's Kappa）、
       主动学习、LLM 辅助标注、标注者间一致性、标注效率分析

Python 版本：3.11+
依赖：标准库（默认模式）
最后验证：2024-12-01
"""

from __future__ import annotations

import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ============================================================
# 1. 标注任务数据结构
# ============================================================

class TaskType(Enum):
    """标注任务类型"""
    CLASSIFICATION = "classification"    # 文本分类
    NER = "ner"                         # 命名实体识别
    SENTIMENT = "sentiment"             # 情感分析
    QA_PAIR = "qa_pair"                 # 问答对生成
    PREFERENCE = "preference"           # 偏好排序（RLHF）


class AnnotationStatus(Enum):
    """标注状态"""
    PENDING = "pending"       # 待标注
    IN_PROGRESS = "in_progress"  # 标注中
    COMPLETED = "completed"   # 已完成
    REVIEW = "review"         # 待审核
    REJECTED = "rejected"     # 被拒绝


@dataclass
class LabelingGuideline:
    """标注规范"""
    task_type: TaskType
    labels: list[str]
    rules: list[str]
    examples: list[dict[str, str]]
    edge_cases: list[dict[str, str]] = field(default_factory=list)


@dataclass
class AnnotationSample:
    """标注样本"""
    sample_id: str
    text: str
    task_type: TaskType
    status: AnnotationStatus = AnnotationStatus.PENDING
    annotations: dict[str, str] = field(default_factory=dict)  # annotator_id -> label
    llm_suggestion: str | None = None  # LLM 预标注建议
    final_label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Annotator:
    """标注者"""
    annotator_id: str
    name: str
    accuracy: float = 0.9  # 模拟准确率
    speed: float = 1.0     # 标注速度倍率
    total_annotations: int = 0
    correct_annotations: int = 0


# ============================================================
# 2. 标注质量控制
# ============================================================

class QualityController:
    """标注质量控制器"""

    @staticmethod
    def cohens_kappa(annotations_a: list[str], annotations_b: list[str]) -> float:
        """计算 Cohen's Kappa 一致性系数"""
        if len(annotations_a) != len(annotations_b):
            raise ValueError("标注数量不一致")

        n = len(annotations_a)
        if n == 0:
            return 0.0

        # 计算观察一致率
        agreements = sum(1 for a, b in zip(annotations_a, annotations_b) if a == b)
        po = agreements / n

        # 计算期望一致率
        labels = set(annotations_a + annotations_b)
        pe = 0.0
        for label in labels:
            p_a = annotations_a.count(label) / n
            p_b = annotations_b.count(label) / n
            pe += p_a * p_b

        # Cohen's Kappa
        if pe == 1.0:
            return 1.0
        kappa = (po - pe) / (1 - pe)
        return round(kappa, 4)

    @staticmethod
    def fleiss_kappa_simplified(all_annotations: list[list[str]]) -> float:
        """简化版 Fleiss' Kappa（多标注者一致性）"""
        if len(all_annotations) < 2:
            return 1.0

        # 两两计算 Cohen's Kappa 取平均
        kappas = []
        for i in range(len(all_annotations)):
            for j in range(i + 1, len(all_annotations)):
                k = QualityController.cohens_kappa(all_annotations[i], all_annotations[j])
                kappas.append(k)

        return round(sum(kappas) / len(kappas), 4) if kappas else 0.0

    @staticmethod
    def interpret_kappa(kappa: float) -> str:
        """解释 Kappa 值"""
        if kappa < 0:
            return "低于随机一致 ❌"
        elif kappa < 0.2:
            return "极低一致性 ❌"
        elif kappa < 0.4:
            return "低一致性 ⚠️"
        elif kappa < 0.6:
            return "中等一致性 ⚠️"
        elif kappa < 0.8:
            return "较高一致性 ✅"
        else:
            return "高一致性 ✅"


# ============================================================
# 3. 标注任务管理器
# ============================================================

class LabelingManager:
    """标注任务管理器"""

    def __init__(self, guideline: LabelingGuideline):
        self.guideline = guideline
        self.samples: dict[str, AnnotationSample] = {}
        self.annotators: dict[str, Annotator] = {}
        self.quality_controller = QualityController()

        print(f"[标注] 任务管理器初始化: {guideline.task_type.value}")
        print(f"  标签: {guideline.labels}")
        print(f"  规则数: {len(guideline.rules)}")

    def add_samples(self, texts: list[str]) -> None:
        """添加待标注样本"""
        for i, text in enumerate(texts):
            sample_id = f"sample_{len(self.samples) + i}"
            self.samples[sample_id] = AnnotationSample(
                sample_id=sample_id,
                text=text,
                task_type=self.guideline.task_type,
            )
        print(f"[标注] 添加 {len(texts)} 个样本，总计: {len(self.samples)}")

    def add_annotator(self, annotator: Annotator) -> None:
        """添加标注者"""
        self.annotators[annotator.annotator_id] = annotator
        print(f"[标注] 添加标注者: {annotator.name} (准确率: {annotator.accuracy:.0%})")

    def llm_pre_annotate(self, confidence_threshold: float = 0.8) -> int:
        """LLM 预标注"""
        pre_annotated = 0
        for sample in self.samples.values():
            if sample.status == AnnotationStatus.PENDING:
                # 模拟 LLM 预标注
                label = random.choice(self.guideline.labels)
                confidence = random.uniform(0.5, 1.0)
                sample.llm_suggestion = label
                sample.metadata["llm_confidence"] = round(confidence, 2)
                pre_annotated += 1
        print(f"[标注] LLM 预标注完成: {pre_annotated} 个样本")
        return pre_annotated

    def annotate_batch(
        self,
        annotator_id: str,
        num_samples: int | None = None,
    ) -> list[dict]:
        """批量标注"""
        annotator = self.annotators.get(annotator_id)
        if not annotator:
            raise ValueError(f"标注者不存在: {annotator_id}")

        pending = [s for s in self.samples.values() if s.status == AnnotationStatus.PENDING]
        if num_samples:
            pending = pending[:num_samples]

        results = []
        for sample in pending:
            # 模拟标注（基于标注者准确率）
            if random.random() < annotator.accuracy:
                # 正确标注（假设有一个"正确"标签）
                label = sample.llm_suggestion or random.choice(self.guideline.labels)
            else:
                # 错误标注
                wrong_labels = [l for l in self.guideline.labels if l != sample.llm_suggestion]
                label = random.choice(wrong_labels) if wrong_labels else self.guideline.labels[0]

            sample.annotations[annotator_id] = label
            sample.status = AnnotationStatus.COMPLETED
            annotator.total_annotations += 1

            results.append({
                "sample_id": sample.sample_id,
                "annotator": annotator_id,
                "label": label,
                "had_llm_suggestion": sample.llm_suggestion is not None,
            })

        print(f"[标注] {annotator.name} 完成 {len(results)} 个标注")
        return results

    def compute_agreement(self) -> dict:
        """计算标注者间一致性"""
        # 收集有多个标注的样本
        multi_annotated = [
            s for s in self.samples.values()
            if len(s.annotations) >= 2
        ]

        if not multi_annotated:
            return {"kappa": 0, "message": "没有多标注样本"}

        annotator_ids = list(self.annotators.keys())
        if len(annotator_ids) < 2:
            return {"kappa": 0, "message": "标注者不足"}

        # 收集两个标注者的标注
        a_labels = []
        b_labels = []
        for sample in multi_annotated:
            if annotator_ids[0] in sample.annotations and annotator_ids[1] in sample.annotations:
                a_labels.append(sample.annotations[annotator_ids[0]])
                b_labels.append(sample.annotations[annotator_ids[1]])

        if not a_labels:
            return {"kappa": 0, "message": "没有重叠标注"}

        kappa = self.quality_controller.cohens_kappa(a_labels, b_labels)
        interpretation = self.quality_controller.interpret_kappa(kappa)

        return {
            "kappa": kappa,
            "interpretation": interpretation,
            "num_samples": len(a_labels),
            "agreement_rate": sum(1 for a, b in zip(a_labels, b_labels) if a == b) / len(a_labels),
        }

    def resolve_labels(self, strategy: str = "majority") -> int:
        """解决标注分歧"""
        resolved = 0
        for sample in self.samples.values():
            if len(sample.annotations) == 0:
                continue

            if strategy == "majority":
                # 多数投票
                counter = Counter(sample.annotations.values())
                sample.final_label = counter.most_common(1)[0][0]
            elif strategy == "expert":
                # 取第一个标注者的结果（假设是专家）
                sample.final_label = list(sample.annotations.values())[0]

            resolved += 1

        print(f"[标注] 标签解决: {resolved} 个样本 (策略: {strategy})")
        return resolved

    def get_statistics(self) -> dict:
        """获取标注统计"""
        total = len(self.samples)
        completed = sum(1 for s in self.samples.values() if s.status == AnnotationStatus.COMPLETED)
        with_final = sum(1 for s in self.samples.values() if s.final_label is not None)

        label_dist = Counter(
            s.final_label for s in self.samples.values() if s.final_label
        )

        return {
            "total_samples": total,
            "completed": completed,
            "with_final_label": with_final,
            "completion_rate": round(completed / max(total, 1), 2),
            "label_distribution": dict(label_dist),
        }


# ============================================================
# 4. 主程序入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("数据标注模拟演示")
    print("=" * 60)

    # 创建标注规范
    guideline = LabelingGuideline(
        task_type=TaskType.SENTIMENT,
        labels=["正面", "负面", "中性"],
        rules=[
            "只根据文本内容判断情感",
            "讽刺表达标注为'负面'",
            "纯事实陈述标注为'中性'",
        ],
        examples=[
            {"text": "这个产品太好用了！", "label": "正面"},
            {"text": "服务态度很差", "label": "负面"},
            {"text": "今天天气不错", "label": "中性"},
        ],
    )

    # 初始化管理器
    manager = LabelingManager(guideline)

    # 添加样本
    texts = [
        "这款手机拍照效果非常好", "物流太慢了，等了一周",
        "产品质量一般般", "客服态度很好，问题解决了",
        "价格偏贵但质量不错", "包装破损，差评",
        "性价比很高，推荐购买", "用了一个月就坏了",
        "颜色和图片一样", "发货速度快，好评",
        "不值这个价格", "功能齐全，满足需求",
    ]
    manager.add_samples(texts)

    # LLM 预标注
    manager.llm_pre_annotate()

    # 添加标注者
    manager.add_annotator(Annotator("a1", "标注员A", accuracy=0.9))
    manager.add_annotator(Annotator("a2", "标注员B", accuracy=0.85))

    # 批量标注
    manager.annotate_batch("a1")
    manager.annotate_batch("a2")

    # 计算一致性
    agreement = manager.compute_agreement()
    print(f"\n标注者间一致性:")
    print(f"  Kappa: {agreement['kappa']}")
    print(f"  解释: {agreement['interpretation']}")
    print(f"  一致率: {agreement.get('agreement_rate', 0):.0%}")

    # 解决分歧
    manager.resolve_labels("majority")

    # 统计信息
    stats = manager.get_statistics()
    print(f"\n标注统计: {stats}")

    print("\n✅ 数据标注模拟演示完成！")
