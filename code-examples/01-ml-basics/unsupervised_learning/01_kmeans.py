"""
K-Means 聚类 — 算法实现、肘部法则、Embedding 聚类

知识点：K-Means 原理、scikit-learn 实现、K 值选择（肘部法则/轮廓系数）、
       Embedding 向量聚类、聚类结果分析

Python 版本：3.11+
依赖：scikit-learn>=1.4, numpy>=1.26
最后验证：2024-12-01
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs


def demo_kmeans_basic() -> None:
    """K-Means 基础用法。"""
    print("\n" + "=" * 60)
    print("1. K-Means 基础")
    print("=" * 60)

    # 生成 3 个簇的数据
    X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)
    print(f"  数据: {X.shape}, 真实簇数: {len(set(y_true))}")

    # K-Means 聚类
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    print(f"  簇分布: {np.bincount(labels)}")
    print(f"  簇中心:\n{kmeans.cluster_centers_.round(2)}")
    print(f"  惯性 (inertia): {kmeans.inertia_:.1f}")
    print(f"  轮廓系数: {silhouette_score(X, labels):.3f}")


def demo_elbow_method() -> None:
    """肘部法则选择 K 值。"""
    print("\n" + "=" * 60)
    print("2. 肘部法则选择 K 值")
    print("=" * 60)

    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

    k_range = range(2, 10)
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    print(f"  {'K':>3} {'Inertia':>10} {'轮廓系数':>10}")
    print(f"  {'-'*25}")
    for k, inertia, sil in zip(k_range, inertias, silhouettes):
        marker = " ◀" if sil == max(silhouettes) else ""
        print(f"  {k:>3} {inertia:>10.1f} {sil:>10.3f}{marker}")

    best_k = list(k_range)[np.argmax(silhouettes)]
    print(f"\n  💡 最佳 K = {best_k}（轮廓系数最高）")


def demo_embedding_clustering() -> None:
    """模拟 Embedding 向量聚类（AI 实战场景）。"""
    print("\n" + "=" * 60)
    print("3. Embedding 向量聚类（AI 实战）")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # 模拟 5 个主题的文档 Embedding
    topics = ["RAG", "微调", "部署", "Agent", "安全"]
    n_per_topic = 20
    dim = 128

    # 每个主题有不同的中心向量
    centers = rng.standard_normal((5, dim))
    embeddings = []
    true_labels = []
    for i, topic in enumerate(topics):
        docs = centers[i] + rng.standard_normal((n_per_topic, dim)) * 0.5
        embeddings.append(docs)
        true_labels.extend([i] * n_per_topic)

    X = np.vstack(embeddings).astype(np.float32)
    print(f"  文档数: {len(X)}, 维度: {dim}, 主题数: {len(topics)}")

    # 聚类
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(X)

    # 分析每个簇的主题分布
    print(f"\n  聚类结果:")
    for cluster_id in range(5):
        mask = pred_labels == cluster_id
        true_in_cluster = np.array(true_labels)[mask]
        topic_counts = np.bincount(true_in_cluster, minlength=5)
        dominant = topics[np.argmax(topic_counts)]
        purity = topic_counts.max() / topic_counts.sum()
        print(f"    簇 {cluster_id}: {topic_counts.sum()} 篇, "
              f"主导主题={dominant}, 纯度={purity:.0%}")

    sil = silhouette_score(X, pred_labels)
    print(f"\n  轮廓系数: {sil:.3f}")


def main() -> None:
    print("🐍 K-Means 聚类 — 算法实现、K 值选择、Embedding 聚类")
    print("=" * 60)
    demo_kmeans_basic()
    demo_elbow_method()
    demo_embedding_clustering()
    print("\n✅ 完成！K-Means 是最常用的聚类算法，简单高效。")
    print("💡 AI 场景：对 Embedding 聚类可以发现文档主题、用户分群。")


if __name__ == "__main__":
    main()
