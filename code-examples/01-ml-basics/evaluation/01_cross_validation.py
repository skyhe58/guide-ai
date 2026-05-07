"""
交叉验证 — K-Fold、分层采样、学习曲线

Python 版本：3.11+
依赖：scikit-learn>=1.4, numpy>=1.26
最后验证：2024-12-01
"""
from __future__ import annotations
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def demo_cross_validation() -> None:
    """交叉验证演示。"""
    print("\n" + "=" * 60)
    print("交叉验证")
    print("=" * 60)
    X, y = make_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)

    models = {
        "逻辑回归": LogisticRegression(max_iter=200),
        "随机森林": RandomForestClassifier(n_estimators=50, random_state=42),
    }
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="f1")
        print(f"  {name}: F1 = {scores.mean():.4f} ± {scores.std():.4f}  {scores.round(4)}")

def demo_learning_curve() -> None:
    """学习曲线 — 诊断过拟合/欠拟合。"""
    print("\n" + "=" * 60)
    print("学习曲线")
    print("=" * 60)
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    train_sizes, train_scores, val_scores = learning_curve(model, X, y, cv=5, train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0])
    print(f"  {'训练集比例':>10} {'训练分数':>10} {'验证分数':>10} {'差距':>8}")
    print(f"  {'-'*40}")
    for size, tr, va in zip(train_sizes, train_scores.mean(axis=1), val_scores.mean(axis=1)):
        print(f"  {size/len(X):>10.0%} {tr:>10.4f} {va:>10.4f} {tr-va:>8.4f}")

def main() -> None:
    print("🐍 交叉验证 — K-Fold、学习曲线")
    demo_cross_validation()
    demo_learning_curve()
    print("\n✅ 完成！交叉验证是模型评估的标准方法。")

if __name__ == "__main__":
    main()
