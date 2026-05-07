"""
逻辑回归 — 二分类与多分类

知识点：逻辑回归原理（Sigmoid/Softmax）、交叉熵损失、
       scikit-learn 实现、分类评估指标（准确率/精确率/召回率/F1）

Python 版本：3.11+
依赖：scikit-learn>=1.4, numpy>=1.26
最后验证：2024-12-01
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def demo_binary_classification() -> None:
    """二分类逻辑回归。"""
    print("\n" + "=" * 60)
    print("1. 二分类逻辑回归")
    print("=" * 60)

    # 生成二分类数据（模拟：垃圾邮件检测）
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=5,
        n_classes=2, random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # 评估
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  准确率: {accuracy:.4f}")
    print(f"\n  分类报告:\n{classification_report(y_test, y_pred, target_names=['正常', '垃圾'])}")

    # Sigmoid 概率输出
    y_proba = model.predict_proba(X_test)[:5]
    print(f"  前 5 个样本的概率:\n{y_proba.round(3)}")


def demo_multiclass() -> None:
    """多分类逻辑回归。"""
    print("\n" + "=" * 60)
    print("2. 多分类逻辑回归")
    print("=" * 60)

    # 生成多分类数据（模拟：情感分析 正面/中性/负面）
    X, y = make_classification(
        n_samples=600, n_features=10, n_informative=6,
        n_classes=3, n_clusters_per_class=1, random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练（多分类自动使用 Softmax）
    model = LogisticRegression(max_iter=200, multi_class="multinomial")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"  准确率: {accuracy_score(y_test, y_pred):.4f}")
    print(f"\n  分类报告:\n{classification_report(y_test, y_pred, target_names=['负面', '中性', '正面'])}")


def main() -> None:
    print("🐍 逻辑回归 — 二分类与多分类")
    print("=" * 60)
    demo_binary_classification()
    demo_multiclass()
    print("\n✅ 完成！逻辑回归虽然名字有'回归'，但它是分类算法。")


if __name__ == "__main__":
    main()
