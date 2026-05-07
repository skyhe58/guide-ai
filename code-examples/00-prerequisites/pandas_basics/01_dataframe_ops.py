"""
Pandas DataFrame 操作 — 创建、索引、筛选、排序、变换

知识点：DataFrame 创建方式、loc/iloc 索引、条件筛选、
       排序与去重、列操作（新增/重命名/删除）、类型转换

Python 版本：3.11+
依赖：pandas>=2.1
最后验证：2024-12-01
"""

from __future__ import annotations

import pandas as pd
import numpy as np


# ============================================================
# 1. DataFrame 创建
# ============================================================

def demo_create_dataframe() -> None:
    """演示 DataFrame 的多种创建方式。"""
    print("\n" + "=" * 60)
    print("1. DataFrame 创建")
    print("=" * 60)

    # 从字典创建（最常用）
    df = pd.DataFrame({
        "model": ["qwen2-7b", "llama3-8b", "deepseek-v2", "gpt-4o", "claude-3.5"],
        "params_b": [7, 8, 236, None, None],
        "mmlu": [65.3, 63.1, 78.5, 86.1, 88.7],
        "humaneval": [52.0, 48.5, 65.2, 87.1, 92.0],
        "license": ["apache-2.0", "llama3", "deepseek", "proprietary", "proprietary"],
        "cost_per_1m": [0.5, 0.0, 0.3, 5.0, 3.0],
    })
    print(f"  从字典创建:\n{df}")
    print(f"\n  shape: {df.shape}")
    print(f"  dtypes:\n{df.dtypes}")

    # 从 NumPy 数组创建
    np_data = np.random.randn(3, 4)
    df_np = pd.DataFrame(np_data, columns=["dim_0", "dim_1", "dim_2", "dim_3"])
    print(f"\n  从 NumPy 创建:\n{df_np.round(3)}")

    # 数据概览
    print(f"\n  --- 数据概览 ---")
    print(f"  info:")
    df.info()
    print(f"\n  describe:\n{df.describe()}")


# ============================================================
# 2. 索引与筛选
# ============================================================

def demo_indexing_filtering() -> None:
    """演示 DataFrame 索引和条件筛选。"""
    print("\n" + "=" * 60)
    print("2. 索引与筛选")
    print("=" * 60)

    df = pd.DataFrame({
        "model": ["qwen2-7b", "llama3-8b", "deepseek-v2", "gpt-4o", "claude-3.5"],
        "mmlu": [65.3, 63.1, 78.5, 86.1, 88.7],
        "license": ["apache-2.0", "llama3", "deepseek", "proprietary", "proprietary"],
        "cost_per_1m": [0.5, 0.0, 0.3, 5.0, 3.0],
    })

    # --- loc（按标签）---
    print("  --- loc 索引 ---")
    print(f"  第 0 行 model: {df.loc[0, 'model']}")
    print(f"  前 3 行 model+mmlu:\n{df.loc[:2, ['model', 'mmlu']]}")

    # --- iloc（按位置）---
    print(f"\n  --- iloc 索引 ---")
    print(f"  前 2 行前 2 列:\n{df.iloc[:2, :2]}")

    # --- 条件筛选 ---
    print(f"\n  --- 条件筛选 ---")
    # 高分模型
    high_score = df[df["mmlu"] > 70]
    print(f"  MMLU > 70:\n{high_score[['model', 'mmlu']]}")

    # 多条件组合（& 且，| 或）
    affordable_good = df[(df["mmlu"] > 70) & (df["cost_per_1m"] < 1.0)]
    print(f"\n  高分且便宜:\n{affordable_good[['model', 'mmlu', 'cost_per_1m']]}")

    # 开源模型
    open_source = df[df["license"] != "proprietary"]
    print(f"\n  开源模型:\n{open_source[['model', 'license']]}")

    # query 语法（更接近 SQL）
    result = df.query("mmlu > 70 and cost_per_1m < 5")
    print(f"\n  query 筛选:\n{result[['model', 'mmlu']]}")

    # isin 筛选
    target_licenses = ["apache-2.0", "llama3"]
    result = df[df["license"].isin(target_licenses)]
    print(f"\n  isin 筛选:\n{result[['model', 'license']]}")


# ============================================================
# 3. 排序、去重、列操作
# ============================================================

def demo_sort_transform() -> None:
    """演示排序、去重和列操作。"""
    print("\n" + "=" * 60)
    print("3. 排序、去重、列操作")
    print("=" * 60)

    df = pd.DataFrame({
        "model": ["qwen2-7b", "llama3-8b", "qwen2-7b", "gpt-4o", "llama3-8b"],
        "benchmark": ["mmlu", "mmlu", "humaneval", "mmlu", "humaneval"],
        "score": [65.3, 63.1, 52.0, 86.1, 48.5],
    })
    print(f"  原始数据:\n{df}")

    # 排序
    sorted_df = df.sort_values("score", ascending=False)
    print(f"\n  按分数降序:\n{sorted_df}")

    # 多列排序
    sorted_df2 = df.sort_values(["model", "score"], ascending=[True, False])
    print(f"\n  按模型升序+分数降序:\n{sorted_df2}")

    # 去重
    unique_models = df.drop_duplicates(subset=["model"], keep="first")
    print(f"\n  按模型去重（保留第一个）:\n{unique_models}")

    # 新增列
    df["rank"] = df["score"].rank(ascending=False).astype(int)
    df["is_top3"] = df["rank"] <= 3
    print(f"\n  新增排名列:\n{df}")

    # 重命名列
    renamed = df.rename(columns={"score": "分数", "model": "模型"})
    print(f"\n  重命名列: {list(renamed.columns)}")

    # 删除列
    dropped = df.drop(columns=["rank", "is_top3"])
    print(f"\n  删除列后: {list(dropped.columns)}")

    # apply — 对每个元素应用函数
    df["model_upper"] = df["model"].apply(str.upper)
    print(f"\n  apply 转大写:\n{df[['model', 'model_upper']]}")


# ============================================================
# 4. 缺失值处理
# ============================================================

def demo_missing_values() -> None:
    """演示缺失值检测和处理。"""
    print("\n" + "=" * 60)
    print("4. 缺失值处理")
    print("=" * 60)

    df = pd.DataFrame({
        "model": ["qwen2-7b", "llama3-8b", "deepseek-v2", "gpt-4o"],
        "params_b": [7.0, 8.0, 236.0, np.nan],
        "mmlu": [65.3, np.nan, 78.5, 86.1],
        "cost": [0.5, 0.0, np.nan, 5.0],
    })
    print(f"  原始数据:\n{df}")

    # 检查缺失值
    print(f"\n  缺失值统计:\n{df.isnull().sum()}")
    print(f"  缺失值比例:\n{(df.isnull().sum() / len(df) * 100).round(1)}%")

    # 填充缺失值
    df_filled = df.copy()
    df_filled["mmlu"] = df_filled["mmlu"].fillna(df_filled["mmlu"].mean())
    df_filled["cost"] = df_filled["cost"].fillna(0)
    df_filled["params_b"] = df_filled["params_b"].fillna(-1)  # 用 -1 标记未知
    print(f"\n  填充后:\n{df_filled}")

    # 删除含缺失值的行
    df_dropped = df.dropna(subset=["mmlu", "cost"])
    print(f"\n  删除缺失行后:\n{df_dropped}")


# ============================================================
# 5. 实战模式 — LLM 评测数据分析
# ============================================================

def demo_llm_benchmark_analysis() -> None:
    """演示用 Pandas 分析 LLM 评测数据。"""
    print("\n" + "=" * 60)
    print("5. 实战模式 — LLM 评测数据分析")
    print("=" * 60)

    # 模拟评测数据
    np.random.seed(42)
    models = ["qwen2-7b", "llama3-8b", "deepseek-v2", "gpt-4o", "claude-3.5"]
    benchmarks = ["mmlu", "humaneval", "gsm8k", "arc"]

    rows = []
    for model in models:
        for bench in benchmarks:
            score = np.random.uniform(40, 95)
            rows.append({"model": model, "benchmark": bench, "score": round(score, 1)})

    df = pd.DataFrame(rows)
    print(f"  评测数据 ({len(df)} 条):\n{df.head(8)}")

    # 按模型分组，计算平均分
    avg_scores = df.groupby("model")["score"].mean().sort_values(ascending=False)
    print(f"\n  模型平均分排名:\n{avg_scores.round(1)}")

    # 透视表 — 模型 x 基准 的分数矩阵
    pivot = df.pivot_table(index="model", columns="benchmark", values="score")
    print(f"\n  透视表:\n{pivot.round(1)}")

    # 找出每个基准的最佳模型
    best_per_bench = df.loc[df.groupby("benchmark")["score"].idxmax()]
    print(f"\n  每个基准的最佳模型:\n{best_per_bench}")

    # 综合排名
    pivot["avg"] = pivot.mean(axis=1)
    pivot["rank"] = pivot["avg"].rank(ascending=False).astype(int)
    print(f"\n  综合排名:\n{pivot[['avg', 'rank']].sort_values('rank')}")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行所有演示。"""
    print("🐍 Pandas DataFrame 操作 — 创建、索引、筛选、排序、变换")
    print("=" * 60)

    demo_create_dataframe()
    demo_indexing_filtering()
    demo_sort_transform()
    demo_missing_values()
    demo_llm_benchmark_analysis()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. DataFrame 是 Pandas 的核心，类似数据库表")
    print("  2. loc 按标签索引，iloc 按位置索引")
    print("  3. 条件筛选用布尔索引，query() 更接近 SQL")
    print("  4. groupby + agg 是数据分析的核心模式")
    print("  5. pivot_table 快速生成交叉分析表")


if __name__ == "__main__":
    main()
