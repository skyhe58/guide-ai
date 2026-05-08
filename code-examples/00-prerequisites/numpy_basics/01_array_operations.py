"""
NumPy 数组操作 — 创建、索引、切片、形状变换

知识点：数组创建方式、索引与切片（一维/二维/布尔/花式）、
       形状变换（reshape/flatten/transpose）、数组拼接与分割、
       AI 场景中的 Embedding 矩阵操作

Python 版本：3.11+
依赖：numpy>=1.26
最后验证：2024-12-01
"""

from __future__ import annotations

import numpy as np

# ============================================================
# 1. 数组创建
# ============================================================

def demo_array_creation() -> None:
    """演示 NumPy 数组的多种创建方式。"""
    print("\n" + "=" * 60)
    print("1. 数组创建")
    print("=" * 60)

    # 从列表创建
    embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    print(f"  一维数组: {embedding}, dtype={embedding.dtype}")

    matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    print(f"  二维数组: shape={matrix.shape}, dtype={matrix.dtype}")

    # 常用创建函数
    # 模拟 100 个文档的 768 维 Embedding（AI 中最常见的数据结构）
    zeros_emb = np.zeros((100, 768), dtype=np.float32)
    print(f"\n  零向量矩阵: shape={zeros_emb.shape}, 内存={zeros_emb.nbytes / 1024:.0f}KB")

    ones = np.ones((3, 3))
    print(f"  全 1 矩阵:\n{ones}")

    identity = np.eye(3)
    print(f"  单位矩阵:\n{identity}")

    # 随机数组（模拟 Embedding）
    rng = np.random.default_rng(42)  # 推荐使用新版随机数生成器
    random_emb = rng.standard_normal((5, 4))
    print(f"\n  随机 Embedding (5x4):\n{random_emb.round(3)}")

    # 等差/等分数组
    sequence = np.arange(0, 1, 0.2)
    print(f"\n  等差数组: {sequence}")
    linspace = np.linspace(0, 1, 5)
    print(f"  等分数组: {linspace}")


# ============================================================
# 2. 索引与切片
# ============================================================

def demo_indexing() -> None:
    """演示数组索引与切片。"""
    print("\n" + "=" * 60)
    print("2. 索引与切片")
    print("=" * 60)

    # --- 一维索引 ---
    scores = np.array([0.92, 0.45, 0.78, 0.15, 0.88, 0.67])
    print(f"  相似度分数: {scores}")
    print(f"  第一个: {scores[0]}, 最后一个: {scores[-1]}")
    print(f"  前三个: {scores[:3]}")
    print(f"  步长为 2: {scores[::2]}")

    # --- 二维索引（Embedding 矩阵操作）---
    print("\n  --- 二维索引 ---")
    # 模拟 6 个文档的 4 维 Embedding
    embeddings = np.array([
        [0.1, 0.2, 0.3, 0.4],  # doc 0
        [0.5, 0.6, 0.7, 0.8],  # doc 1
        [0.9, 1.0, 1.1, 1.2],  # doc 2
        [1.3, 1.4, 1.5, 1.6],  # doc 3
        [1.7, 1.8, 1.9, 2.0],  # doc 4
        [2.1, 2.2, 2.3, 2.4],  # doc 5
    ])
    print(f"  Embedding 矩阵: shape={embeddings.shape}")
    print(f"  第 0 个文档向量: {embeddings[0]}")
    print(f"  前 3 个文档: shape={embeddings[:3].shape}")
    print(f"  所有文档的前 2 维: shape={embeddings[:, :2].shape}")

    # --- 布尔索引（筛选高分文档）---
    print("\n  --- 布尔索引 ---")
    mask = scores > 0.7
    print(f"  分数 > 0.7 的掩码: {mask}")
    print(f"  高分文档: {scores[mask]}")
    print(f"  高分文档的 Embedding: shape={embeddings[mask].shape}")

    # --- 花式索引（按 Top-K 索引取值）---
    print("\n  --- 花式索引 ---")
    top_k_indices = np.argsort(scores)[-3:][::-1]  # Top-3 索引
    print(f"  Top-3 索引: {top_k_indices}")
    print(f"  Top-3 分数: {scores[top_k_indices]}")
    print(f"  Top-3 Embedding: shape={embeddings[top_k_indices].shape}")


# ============================================================
# 3. 形状变换
# ============================================================

def demo_reshape() -> None:
    """演示数组形状变换。"""
    print("\n" + "=" * 60)
    print("3. 形状变换")
    print("=" * 60)

    arr = np.arange(12)
    print(f"  原始数组: {arr}, shape={arr.shape}")

    # reshape — 改变形状（不改变数据）
    reshaped = arr.reshape(3, 4)
    print(f"  reshape(3,4):\n{reshaped}")

    reshaped2 = arr.reshape(2, -1)  # -1 自动推断
    print(f"  reshape(2,-1): shape={reshaped2.shape}")

    # flatten / ravel — 展平
    flat = reshaped.flatten()  # 返回副本
    print(f"  flatten: {flat}")

    # transpose — 转置（在注意力机制中常用）
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"\n  原矩阵: shape={matrix.shape}")
    print(f"  转置后: shape={matrix.T.shape}")

    # expand_dims / squeeze — 增减维度
    vec = np.array([1, 2, 3])
    expanded = np.expand_dims(vec, axis=0)  # (3,) → (1, 3)
    print(f"\n  expand_dims: {vec.shape} → {expanded.shape}")
    squeezed = np.squeeze(expanded)  # (1, 3) → (3,)
    print(f"  squeeze: {expanded.shape} → {squeezed.shape}")


# ============================================================
# 4. 数组拼接与分割
# ============================================================

def demo_concat_split() -> None:
    """演示数组拼接与分割。"""
    print("\n" + "=" * 60)
    print("4. 数组拼接与分割")
    print("=" * 60)

    # 模拟从多个数据源获取的 Embedding
    batch1 = np.random.randn(3, 4)  # 数据源 1：3 个文档
    batch2 = np.random.randn(2, 4)  # 数据源 2：2 个文档

    # 垂直拼接（沿行方向，合并文档）
    combined = np.vstack([batch1, batch2])  # 等价于 np.concatenate([...], axis=0)
    print(f"  batch1: {batch1.shape}, batch2: {batch2.shape}")
    print(f"  vstack 合并: {combined.shape}")

    # 水平拼接（沿列方向，拼接特征）
    features1 = np.random.randn(5, 3)  # 特征组 1
    features2 = np.random.randn(5, 2)  # 特征组 2
    all_features = np.hstack([features1, features2])
    print(f"  hstack 拼接特征: {features1.shape} + {features2.shape} → {all_features.shape}")

    # 分割
    chunks = np.array_split(combined, 3)  # 分成 3 份（允许不等分）
    print(f"  分割为 {len(chunks)} 份: {[c.shape for c in chunks]}")


# ============================================================
# 5. 实战模式 — Embedding 相似度检索
# ============================================================

def demo_similarity_search() -> None:
    """演示基于 NumPy 的 Embedding 相似度检索。

    这是 RAG 系统中向量检索的核心逻辑：
    1. 计算查询向量与所有文档向量的余弦相似度
    2. 按相似度排序，返回 Top-K 结果
    """
    print("\n" + "=" * 60)
    print("5. 实战模式 — Embedding 相似度检索")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # 模拟数据
    n_docs = 1000
    dim = 768
    documents = rng.standard_normal((n_docs, dim)).astype(np.float32)
    query = rng.standard_normal(dim).astype(np.float32)

    # L2 归一化（余弦相似度的前置步骤）
    doc_norms = np.linalg.norm(documents, axis=1, keepdims=True)
    documents_normalized = documents / doc_norms
    query_normalized = query / np.linalg.norm(query)

    # 计算余弦相似度（归一化后的点积）
    similarities = documents_normalized @ query_normalized

    # Top-5 检索
    top_k = 5
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    print(f"  文档数: {n_docs}, 向量维度: {dim}")
    print(f"  查询向量: shape={query.shape}, dtype={query.dtype}")
    print(f"\n  Top-{top_k} 检索结果:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"    #{rank} 文档 {idx}: 相似度={similarities[idx]:.4f}")

    # 性能统计
    print(f"\n  内存占用: {documents.nbytes / 1024 / 1024:.1f}MB")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行所有演示。"""
    print("🐍 NumPy 数组操作 — 创建、索引、切片、形状变换")
    print("=" * 60)

    demo_array_creation()
    demo_indexing()
    demo_reshape()
    demo_concat_split()
    demo_similarity_search()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. ndarray 是 NumPy 的核心，所有 AI 数据最终都是数组")
    print("  2. 布尔索引和花式索引是筛选数据的利器")
    print("  3. reshape/transpose 在模型输入输出转换中频繁使用")
    print("  4. vstack/hstack 用于合并多批次数据")
    print("  5. 向量化的相似度计算是 RAG 检索的核心")


if __name__ == "__main__":
    main()
