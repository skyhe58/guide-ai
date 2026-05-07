"""
随机森林 — 集成学习（Bagging）

知识点：随机森林原理（Bagging + 特征随机）、
       与决策树对比、超参数调优、特征重要性

Python 版本：3.11+
依赖：scikit-learn>=1.4, numpy>=1.26
最后验证：2024-12-01
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score


def demo_random_forest() -> None:
    """随机森林 vs 决策树对比。"""
    print("\n" + "=" * 60)
    print("随机森林 vs 决策树")
    print("=" * 60)

    # 生成数据
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=6,
        n_classes=2, random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 决策树
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    dt_train = accuracy_score(y_train, dt.predict(X_train))
    dt_test = accuracy_score(y_test, dt.predict(X_test))

    # 随机森林
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_train = accuracy_score(y_train, rf.predict(X_train))
    rf_test = accuracy_score(y_test, rf.predict(X_test))

    print(f"  {'模型':<12} {'训练集':>8} {'测试集':>8} {'过拟合差距':>10}")
    print(f"  {'-'*42}")
    print(f"  {'决策树':<12} {dt_train:>8.4f} {dt_test:>8.4f} {dt_train-dt_test:>10.4f}")
    print(f"  {'随机森林':<12} {rf_train:>8.4f} {rf_test:>8.4f} {rf_train-rf_test:>10.4f}")
    print(f"\n  💡 随机森林通过集成多棵树，显著降低过拟合")

    # 交叉验证
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring="accuracy")
    print(f"\n  5 折交叉验证: {cv_scores.round(4)}")
    print(f"  平均准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 特征重要性
    feature_names = [f"feat_{i}" for i in range(10)]
    importances = rf.feature_importances_
    top_5 = np.argsort(importances)[-5:][::-1]
    print(f"\n  Top-5 重要特征:")
    for idx in top_5:
        bar = "█" * int(importances[idx] * 40)
        print(f"    {feature_names[idx]}: {importances[idx]:.3f} {bar}")


def main() -> None:
    print("🐍 随机森林 — 集成学习（Bagging）")
    print("=" * 60)
    demo_random_forest()
    print("\n✅ 完成！随机森林 = 多棵决策树投票，降低方差，减少过拟合。")


if __name__ == "__main__":
    main()
