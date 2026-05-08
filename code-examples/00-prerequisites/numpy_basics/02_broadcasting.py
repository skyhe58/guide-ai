"""
NumPy 广播机制 — 广播规则、向量化运算、性能对比

知识点：广播规则与形状匹配、向量化运算替代循环、
       常用数学函数（线性代数/统计/Softmax）、
       AI 场景中的批量运算（归一化/相似度/距离计算）

Python 版本：3.11+
依赖：numpy>=1.26
最后验证：2024-12-01
"""

from __future__ import annotations

import time

import numpy as np

# ============================================================
# 1. 广播规则
# ============================================================

def demo_broadcasting_rules() -> None:
    """演示 NumPy 广播规则。

    广播规则（从右向左逐维度比较）：
    1. 维度相同 → 直接运算
    2. 其中一个维度为 1 → 扩展到另一个的大小
    3. 维度数量不同 → 在左侧补 1
    4. 以上都不满足 → 报错
    """
    print("\n" + "=" * 60)
    print("1. 广播规则")
    print("=" * 60)

    # 标量 + 数组
    arr = np.array([1, 2, 3, 4, 5])
    print(f"  数组 + 标量: {arr} + 10 = {arr + 10}")

    # 向量 + 矩阵（最常见的广播场景）
    # 模拟：对 Embedding 矩阵的每个维度减去均值
    embeddings = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])
    mean = embeddings.mean(axis=0)  # (3,) — 每列均值
    centered = embeddings - mean     # (3,3) - (3,) → 广播
    print(f"\n  Embedding 矩阵:\n{embeddings}")
    print(f"  列均值: {mean}")
    print(f"  中心化后:\n{centered}")

    # 列向量 + 行向量 → 矩阵（外积效果）
    col = np.array([[1], [2], [3]])  # (3, 1)
    row = np.array([10, 20, 30])     # (3,)
    result = col + row                # (3, 1) + (3,) → (3, 3)
    print(f"\n  列向量 (3,1) + 行向量 (3,) → 矩阵 (3,3):")
    print(f"{result}")


# ============================================================
# 2. 向量化运算 vs Python 循环
# ============================================================

def demo_vectorization_benchmark() -> None:
    """对比向量化运算和 Python 循环的性能。"""
    print("\n" + "=" * 60)
    print("2. 向量化运算 vs Python 循环")
    print("=" * 60)

    n = 100_000
    rng = np.random.default_rng(42)
    data = rng.standard_normal(n)

    # --- Python 循环 ---
    start = time.perf_counter()
    result_loop = []
    for x in data:
        result_loop.append(x ** 2 + 2 * x + 1)
    loop_time = time.perf_counter() - start

    # --- NumPy 向量化 ---
    start = time.perf_counter()
    result_vec = data ** 2 + 2 * data + 1
    vec_time = time.perf_counter() - start

    speedup = loop_time / vec_time if vec_time > 0 else float("inf")
    print(f"  数据量: {n:,}")
    print(f"  Python 循环: {loop_time:.4f}s")
    print(f"  NumPy 向量化: {vec_time:.6f}s")
    print(f"  加速比: {speedup:.0f}x")

    # 验证结果一致
    assert np.allclose(result_loop, result_vec), "结果不一致！"
    print("  ✅ 结果验证一致")


# ============================================================
# 3. 常用数学函数
# ============================================================

def demo_math_functions() -> None:
    """演示 AI 开发中常用的 NumPy 数学函数。"""
    print("\n" + "=" * 60)
    print("3. 常用数学函数")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # --- 线性代数 ---
    print("\n  --- 线性代数 ---")
    a = rng.standard_normal((3, 4))
    b = rng.standard_normal((4, 2))

    # 矩阵乘法（注意力机制的核心运算）
    c = a @ b  # 等价于 np.matmul(a, b)
    print(f"  矩阵乘法: ({a.shape}) @ ({b.shape}) = {c.shape}")

    # 向量点积
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([4.0, 5.0, 6.0])
    dot = np.dot(v1, v2)
    print(f"  点积: {v1} · {v2} = {dot}")

    # 向量范数
    norm = np.linalg.norm(v1)
    print(f"  L2 范数: ||{v1}|| = {norm:.4f}")

    # --- 统计函数 ---
    print("\n  --- 统计函数 ---")
    data = rng.standard_normal((5, 3))
    print(f"  数据 shape: {data.shape}")
    print(f"  全局均值: {data.mean():.4f}")
    print(f"  列均值 (axis=0): {data.mean(axis=0).round(4)}")
    print(f"  行均值 (axis=1): {data.mean(axis=1).round(4)}")
    print(f"  标准差: {data.std():.4f}")
    print(f"  最大值索引: {np.argmax(data)}")

    # --- Softmax（LLM 输出概率分布）---
    print("\n  --- Softmax ---")
    logits = np.array([2.0, 1.0, 0.1, -1.0, 3.0])

    def softmax(x: np.ndarray) -> np.ndarray:
        """数值稳定的 Softmax 实现。"""
        exp_x = np.exp(x - np.max(x))  # 减去最大值防止溢出
        return exp_x / exp_x.sum()

    probs = softmax(logits)
    print(f"  Logits: {logits}")
    print(f"  Softmax: {probs.round(4)}")
    print(f"  概率之和: {probs.sum():.6f}")
    print(f"  最大概率索引: {np.argmax(probs)} (值={probs.max():.4f})")


# ============================================================
# 4. AI 实战 — 批量归一化与距离计算
# ============================================================

def demo_ai_operations() -> None:
    """演示 AI 场景中的批量运算。"""
    print("\n" + "=" * 60)
    print("4. AI 实战 — 批量归一化与距离计算")
    print("=" * 60)

    rng = np.random.default_rng(42)
    n_docs = 500
    dim = 128

    # 模拟 Embedding 矩阵
    embeddings = rng.standard_normal((n_docs, dim)).astype(np.float32)

    # --- L2 归一化（余弦相似度的前置步骤）---
    print("\n  --- L2 归一化 ---")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)  # (500, 1)
    normalized = embeddings / norms  # 广播：(500, 128) / (500, 1)

    # 验证：归一化后每个向量的范数应为 1
    check_norms = np.linalg.norm(normalized, axis=1)
    print(f"  归一化前范数范围: [{norms.min():.2f}, {norms.max():.2f}]")
    print(f"  归一化后范数范围: [{check_norms.min():.6f}, {check_norms.max():.6f}]")

    # --- 批量余弦相似度 ---
    print("\n  --- 批量余弦相似度 ---")
    query = rng.standard_normal(dim).astype(np.float32)
    query_norm = query / np.linalg.norm(query)

    # 归一化后的点积 = 余弦相似度
    similarities = normalized @ query_norm  # (500,)
    top_5 = np.argsort(similarities)[-5:][::-1]
    print(f"  Top-5 相似度: {similarities[top_5].round(4)}")

    # --- 批量欧氏距离 ---
    print("\n  --- 批量欧氏距离 ---")
    # 利用广播计算查询向量与所有文档的欧氏距离
    diff = embeddings - query  # (500, 128) - (128,) → 广播
    distances = np.linalg.norm(diff, axis=1)  # (500,)
    nearest_5 = np.argsort(distances)[:5]
    print(f"  最近 5 个文档距离: {distances[nearest_5].round(4)}")

    # --- Min-Max 归一化 ---
    print("\n  --- Min-Max 归一化 ---")
    raw_scores = rng.uniform(0, 100, size=10)
    min_val, max_val = raw_scores.min(), raw_scores.max()
    normalized_scores = (raw_scores - min_val) / (max_val - min_val)
    print(f"  原始分数: {raw_scores.round(1)}")
    print(f"  归一化后: {normalized_scores.round(4)}")
    print(f"  范围: [{normalized_scores.min():.1f}, {normalized_scores.max():.1f}]")


# ============================================================
# 5. NumPy 与 PyTorch 互转
# ============================================================

def demo_numpy_pytorch() -> None:
    """演示 NumPy 与 PyTorch 的互转。"""
    print("\n" + "=" * 60)
    print("5. NumPy 与 PyTorch 互转")
    print("=" * 60)

    try:
        import torch

        # NumPy → PyTorch（共享内存，零拷贝）
        np_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = torch.from_numpy(np_array)
        print(f"  NumPy → Tensor: {tensor}")

        # PyTorch → NumPy（共享内存）
        tensor2 = torch.randn(3, 4)
        np_array2 = tensor2.numpy()
        print(f"  Tensor → NumPy: shape={np_array2.shape}")

        # 注意：共享内存意味着修改一个会影响另一个
        np_array[0] = 999
        print(f"  修改 NumPy 后 Tensor 也变了: {tensor}")

        # 如果不想共享内存，用 .copy()
        np_copy = tensor2.numpy().copy()
        print(f"  独立副本: 修改不会互相影响")

    except ImportError:
        print("  ⚠️ PyTorch 未安装，跳过互转演示")
        print("  💡 安装: pip install torch")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行所有演示。"""
    print("🐍 NumPy 广播机制 — 广播规则、向量化运算、性能对比")
    print("=" * 60)

    demo_broadcasting_rules()
    demo_vectorization_benchmark()
    demo_math_functions()
    demo_ai_operations()
    demo_numpy_pytorch()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. 广播规则：从右向左逐维度比较，1 可以扩展")
    print("  2. 向量化运算比 Python 循环快 10-100 倍")
    print("  3. Softmax 实现要减去最大值防止数值溢出")
    print("  4. L2 归一化 + 点积 = 余弦相似度（RAG 检索核心）")
    print("  5. NumPy 和 PyTorch 可以零拷贝互转（共享内存）")


if __name__ == "__main__":
    main()
