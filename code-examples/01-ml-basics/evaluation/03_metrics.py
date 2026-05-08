"""
评估指标 — 分类指标、回归指标、混淆矩阵

Python 版本：3.11+
依赖：scikit-learn>=1.4, numpy>=1.26
最后验证：2024-12-01
"""
from __future__ import annotations

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def demo_classification_metrics() -> None:
    """分类评估指标。"""
    print("\n" + "=" * 60)
    print("分类评估指标")
    print("=" * 60)
    X, y = make_classification(n_samples=500, n_features=10, n_classes=2, weights=[0.7, 0.3], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"  准确率 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
    print(f"  精确率 (Precision): {precision_score(y_test, y_pred):.4f}")
    print(f"  召回率 (Recall): {recall_score(y_test, y_pred):.4f}")
    print(f"  F1 分数: {f1_score(y_test, y_pred):.4f}")
    print(f"  AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  混淆矩阵:")
    print(f"              预测正  预测负")
    print(f"    实际正    {cm[1][1]:>5}   {cm[1][0]:>5}  (TP={cm[1][1]}, FN={cm[1][0]})")
    print(f"    实际负    {cm[0][1]:>5}   {cm[0][0]:>5}  (FP={cm[0][1]}, TN={cm[0][0]})")

    # 完整分类报告
    print(f"\n  分类报告:\n{classification_report(y_test, y_pred, target_names=['负类', '正类'])}")

def demo_threshold_tuning() -> None:
    """阈值调优 — 调整精确率/召回率平衡。"""
    print("\n" + "=" * 60)
    print("阈值调优")
    print("=" * 60)
    X, y = make_classification(n_samples=500, n_features=10, weights=[0.7, 0.3], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"  {'阈值':>6} {'精确率':>8} {'召回率':>8} {'F1':>8}")
    print(f"  {'-'*32}")
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred = (y_proba >= threshold).astype(int)
        p = precision_score(y_test, y_pred, zero_division=0)
        r = recall_score(y_test, y_pred, zero_division=0)
        f = f1_score(y_test, y_pred, zero_division=0)
        print(f"  {threshold:>6.1f} {p:>8.4f} {r:>8.4f} {f:>8.4f}")
    print(f"\n  💡 降低阈值 → 召回率↑精确率↓，升高阈值 → 精确率↑召回率↓")

def main() -> None:
    print("🐍 评估指标 — 分类指标、混淆矩阵、阈值调优")
    demo_classification_metrics()
    demo_threshold_tuning()
    print("\n✅ 完成！选对评估指标比调参更重要。")

if __name__ == "__main__":
    main()
