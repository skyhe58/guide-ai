"""
SVM 与 KNN — 支持向量机和 K 近邻

知识点：SVM 原理（最大间隔/核函数）、KNN 原理（距离度量/K 值选择）、
       scikit-learn 实现、两者对比

Python 版本：3.11+
依赖：scikit-learn>=1.4, numpy>=1.26
最后验证：2024-12-01
"""

from __future__ import annotations

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def demo_svm_knn_comparison() -> None:
    """SVM 与 KNN 对比。"""
    print("\n" + "=" * 60)
    print("SVM vs KNN 对比")
    print("=" * 60)

    # 生成数据
    X, y = make_classification(
        n_samples=500, n_features=8, n_informative=5,
        n_classes=2, random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 标准化（SVM 和 KNN 都对特征尺度敏感）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SVM（RBF 核）
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    svm.fit(X_train_scaled, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test_scaled))

    # SVM（线性核）
    svm_linear = SVC(kernel="linear", C=1.0, random_state=42)
    svm_linear.fit(X_train_scaled, y_train)
    svm_linear_acc = accuracy_score(y_test, svm_linear.predict(X_test_scaled))

    # KNN（K=5）
    knn5 = KNeighborsClassifier(n_neighbors=5)
    knn5.fit(X_train_scaled, y_train)
    knn5_acc = accuracy_score(y_test, knn5.predict(X_test_scaled))

    # KNN（K=3）
    knn3 = KNeighborsClassifier(n_neighbors=3)
    knn3.fit(X_train_scaled, y_train)
    knn3_acc = accuracy_score(y_test, knn3.predict(X_test_scaled))

    print(f"  {'模型':<16} {'准确率':>8}")
    print(f"  {'-'*26}")
    print(f"  {'SVM (RBF 核)':<16} {svm_acc:>8.4f}")
    print(f"  {'SVM (线性核)':<16} {svm_linear_acc:>8.4f}")
    print(f"  {'KNN (K=5)':<16} {knn5_acc:>8.4f}")
    print(f"  {'KNN (K=3)':<16} {knn3_acc:>8.4f}")

    # 对比总结
    print(f"\n  📊 对比总结:")
    print(f"  {'维度':<12} {'SVM':<20} {'KNN':<20}")
    print(f"  {'-'*52}")
    print(f"  {'训练速度':<12} {'中等':<20} {'无需训练（懒学习）':<20}")
    print(f"  {'预测速度':<12} {'快':<20} {'慢（需计算距离）':<20}")
    print(f"  {'可解释性':<12} {'低':<20} {'高（看邻居）':<20}")
    print(f"  {'特征缩放':<12} {'必须':<20} {'必须':<20}")
    print(f"  {'适用场景':<12} {'中小数据集':<20} {'小数据集':<20}")


def main() -> None:
    print("🐍 SVM 与 KNN — 支持向量机和 K 近邻")
    print("=" * 60)
    demo_svm_knn_comparison()
    print("\n✅ 完成！")
    print("💡 SVM 适合中小数据集的高维分类，KNN 适合小数据集的快速原型。")
    print("💡 两者都必须做特征标准化（StandardScaler）。")


if __name__ == "__main__":
    main()
