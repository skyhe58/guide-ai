"""
决策树 — 分类与回归树

知识点：决策树原理（信息增益/基尼系数）、树的构建与剪枝、
       scikit-learn 实现、特征重要性、可视化

Python 版本：3.11+
依赖：scikit-learn>=1.4, numpy>=1.26
最后验证：2024-12-01
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def demo_decision_tree() -> None:
    """决策树分类。"""
    print("\n" + "=" * 60)
    print("决策树分类")
    print("=" * 60)

    # 生成数据
    X, y = make_classification(
        n_samples=500, n_features=6, n_informative=4,
        n_classes=2, random_state=42,
    )
    feature_names = [f"特征_{i}" for i in range(6)]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练（限制深度防止过拟合）
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    # 评估
    y_pred = model.predict(X_test)
    print(f"  准确率: {accuracy_score(y_test, y_pred):.4f}")

    # 特征重要性（决策树的优势：可解释性）
    importances = model.feature_importances_
    print(f"\n  特征重要性:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 30)
        print(f"    {name}: {imp:.3f} {bar}")

    # 树结构文本可视化
    tree_text = export_text(model, feature_names=feature_names, max_depth=2)
    print(f"\n  决策树结构（前 2 层）:\n{tree_text}")

    # 过拟合演示：不限制深度
    model_overfit = DecisionTreeClassifier(random_state=42)  # 无深度限制
    model_overfit.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model_overfit.predict(X_train))
    test_acc = accuracy_score(y_test, model_overfit.predict(X_test))
    print(f"  过拟合演示（无深度限制）:")
    print(f"    训练集准确率: {train_acc:.4f}")
    print(f"    测试集准确率: {test_acc:.4f}")
    print(f"    差距: {train_acc - test_acc:.4f}（差距大 = 过拟合）")


def main() -> None:
    print("🐍 决策树 — 分类与可解释性")
    print("=" * 60)
    demo_decision_tree()
    print("\n✅ 完成！决策树的核心优势是可解释性，但容易过拟合。")
    print("💡 解决方案：限制深度、剪枝，或使用随机森林（集成多棵树）。")


if __name__ == "__main__":
    main()
