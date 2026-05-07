"""
PCA 降维 — 主成分分析、方差解释、Embedding 可视化

知识点：PCA 原理（特征值分解/方差最大化）、scikit-learn 实现、
       方差解释比例、Embedding 降维可视化、与 t-SNE 对比

Python 版本：3.11+
依赖：scikit-learn>=1.4, numpy>=1.26
最后验证：2024-12-01
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


def demo_pca_basic() -> None:
    """PCA 基础用法。"""
    print("\n" + "=" * 60)
    print("1. PCA 基础")
    print("=" * 60)

    # 生成高维数据
    X, y = make_blobs(n_samples=200, n_features=10, centers=3, random_state=42)
    X = StandardScaler().fit_transform(X)
    print(f"  原始数据: {X.shape}")

    # PCA 降到 2 维
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    print(f"  降维后: {X_2d.shape}")

    # 方差解释比例
    print(f"  各主成分方差解释比例: {pca.explained_variance_ratio_.round(4)}")
    print(f"  累计方差解释: {pca.explained_variance_ratio_.sum():.4f}")


def demo_variance_analysis() -> None:
    """方差解释分析 — 选择合适的维度数。"""
    print("\n" + "=" * 60)
    print("2. 方差解释分析")
    print("=" * 60)

    # 生成 50 维数据
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 50))
    X = StandardScaler().fit_transform(X)

    # 完整 PCA
    pca_full = PCA()
    pca_full.fit(X)

    # 累计方差解释
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    print(f"  原始维度: {X.shape[1]}")
    print(f"\n  累计方差解释:")
    thresholds = [0.80, 0.90, 0.95, 0.99]
    for t in thresholds:
        n_components = np.argmax(cumulative >= t) + 1
        print(f"    {t:.0%} 方差 → 需要 {n_components} 个主成分")

    # 实际降维
    pca_95 = PCA(n_components=0.95)  # 保留 95% 方差
    X_reduced = pca_95.fit_transform(X)
    print(f"\n  保留 95% 方差: {X.shape[1]} 维 → {X_reduced.shape[1]} 维")


def demo_embedding_visualization() -> None:
    """模拟 Embedding 降维可视化（AI 实战场景）。"""
    print("\n" + "=" * 60)
    print("3. Embedding 降维可视化（AI 实战）")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # 模拟 4 个主题的文档 Embedding（768 维）
    topics = ["RAG", "微调", "部署", "安全"]
    n_per_topic = 30
    dim = 768

    centers = rng.standard_normal((4, dim)) * 3
    embeddings = []
    labels = []
    for i, topic in enumerate(topics):
        docs = centers[i] + rng.standard_normal((n_per_topic, dim))
        embeddings.append(docs)
        labels.extend([topic] * n_per_topic)

    X = np.vstack(embeddings).astype(np.float32)
    print(f"  文档数: {len(X)}, 原始维度: {dim}")

    # PCA 降到 2 维
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    print(f"  PCA 降维: {dim} → 2")
    print(f"  方差解释: {pca.explained_variance_ratio_.sum():.4f}")

    # 展示每个主题在 2D 空间的分布
    print(f"\n  各主题在 2D 空间的中心坐标:")
    for topic in topics:
        mask = [l == topic for l in labels]
        center = X_2d[mask].mean(axis=0)
        spread = X_2d[mask].std(axis=0).mean()
        print(f"    {topic}: center=({center[0]:.2f}, {center[1]:.2f}), spread={spread:.2f}")

    # PCA 降到 50 维（用于加速后续检索）
    pca_50 = PCA(n_components=50)
    X_50d = pca_50.fit_transform(X)
    variance_kept = pca_50.explained_variance_ratio_.sum()
    print(f"\n  PCA 降维用于加速检索:")
    print(f"    {dim} → 50 维，保留 {variance_kept:.1%} 方差")
    print(f"    内存节省: {(1 - 50/dim):.0%}")


def main() -> None:
    print("🐍 PCA 降维 — 主成分分析、方差解释、Embedding 可视化")
    print("=" * 60)
    demo_pca_basic()
    demo_variance_analysis()
    demo_embedding_visualization()
    print("\n✅ 完成！")
    print("💡 PCA 适合线性降维和特征压缩，可视化推荐用 t-SNE/UMAP。")
    print("💡 AI 场景：Embedding 降维可加速检索、减少存储。")


if __name__ == "__main__":
    main()
