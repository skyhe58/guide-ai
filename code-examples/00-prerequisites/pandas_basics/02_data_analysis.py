"""
Pandas 数据分析 — 分组聚合、合并、数据清洗、格式转换

知识点：groupby 分组聚合、merge/concat 数据合并、
       数据清洗（去重/类型转换/字符串处理）、
       AI 场景中的数据格式转换（CSV → Alpaca JSON）

Python 版本：3.11+
依赖：pandas>=2.1
最后验证：2024-12-01
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

# ============================================================
# 1. 分组聚合
# ============================================================

def demo_groupby() -> None:
    """演示 groupby 分组聚合。"""
    print("\n" + "=" * 60)
    print("1. 分组聚合")
    print("=" * 60)

    # 模拟推理日志数据
    np.random.seed(42)
    df = pd.DataFrame({
        "model": np.random.choice(["qwen2", "llama3", "deepseek"], 20),
        "latency_ms": np.random.uniform(100, 2000, 20).round(0),
        "tokens": np.random.randint(50, 500, 20),
        "status": np.random.choice(["success", "success", "success", "error"], 20),
    })
    print(f"  推理日志 ({len(df)} 条):\n{df.head()}")

    # 按模型分组统计
    stats = df.groupby("model").agg(
        请求数=("latency_ms", "count"),
        平均延迟=("latency_ms", "mean"),
        P99延迟=("latency_ms", lambda x: x.quantile(0.99)),
        总Token=("tokens", "sum"),
        成功率=("status", lambda x: (x == "success").mean()),
    ).round(1)
    print(f"\n  按模型统计:\n{stats}")

    # value_counts
    print(f"\n  状态分布:\n{df['status'].value_counts()}")
    print(f"  模型分布:\n{df['model'].value_counts()}")


# ============================================================
# 2. 数据合并
# ============================================================

def demo_merge_concat() -> None:
    """演示数据合并操作。"""
    print("\n" + "=" * 60)
    print("2. 数据合并")
    print("=" * 60)

    # 模型基本信息
    models = pd.DataFrame({
        "model": ["qwen2-7b", "llama3-8b", "deepseek-v2"],
        "params_b": [7, 8, 236],
        "license": ["apache-2.0", "llama3", "deepseek"],
    })

    # 评测分数
    scores = pd.DataFrame({
        "model": ["qwen2-7b", "llama3-8b", "gpt-4o"],
        "mmlu": [65.3, 63.1, 86.1],
    })

    # merge — 类似 SQL JOIN
    # inner join（默认）：只保留两边都有的
    inner = models.merge(scores, on="model", how="inner")
    print(f"  inner join:\n{inner}")

    # left join：保留左表所有行
    left = models.merge(scores, on="model", how="left")
    print(f"\n  left join:\n{left}")

    # outer join：保留所有行
    outer = models.merge(scores, on="model", how="outer")
    print(f"\n  outer join:\n{outer}")

    # concat — 纵向拼接（合并多批数据）
    batch1 = pd.DataFrame({"model": ["a", "b"], "score": [80, 70]})
    batch2 = pd.DataFrame({"model": ["c", "d"], "score": [90, 60]})
    combined = pd.concat([batch1, batch2], ignore_index=True)
    print(f"\n  concat 纵向拼接:\n{combined}")


# ============================================================
# 3. 数据清洗
# ============================================================

def demo_data_cleaning() -> None:
    """演示数据清洗操作。"""
    print("\n" + "=" * 60)
    print("3. 数据清洗")
    print("=" * 60)

    # 模拟脏数据
    df = pd.DataFrame({
        "model": ["  Qwen2-7B ", "llama3-8b", "LLAMA3-8B", "deepseek-v2", "qwen2-7b"],
        "score": ["65.3", "63.1", "63.1", "78.5", "65.3"],
        "date": ["2024-01-15", "2024-02-20", "2024-02-20", "2024-03-10", "2024-01-15"],
        "notes": ["good", None, "", "excellent", "good"],
    })
    print(f"  脏数据:\n{df}")

    # 字符串清洗
    df["model"] = df["model"].str.strip().str.lower()
    print(f"\n  strip + lower:\n{df['model'].tolist()}")

    # 类型转换
    df["score"] = df["score"].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    print(f"\n  类型转换后:\n{df.dtypes}")

    # 去重
    before = len(df)
    df = df.drop_duplicates(subset=["model"], keep="first")
    print(f"\n  去重: {before} → {len(df)} 行")

    # 缺失值和空字符串处理
    df["notes"] = df["notes"].replace("", np.nan)
    df["notes"] = df["notes"].fillna("无备注")
    print(f"\n  清洗后:\n{df}")


# ============================================================
# 4. 实战模式 — 微调数据格式转换
# ============================================================

def demo_finetune_data_conversion() -> None:
    """演示将 CSV 数据转换为 LLM 微调格式。

    AI 应用中的常见需求：
    - CSV/Excel 原始数据 → Alpaca 格式 JSON
    - 用于 LoRA/QLoRA 微调
    """
    print("\n" + "=" * 60)
    print("4. 实战模式 — 微调数据格式转换")
    print("=" * 60)

    # 模拟原始 QA 数据（CSV 格式）
    qa_data = pd.DataFrame({
        "question": [
            "什么是 RAG？",
            "LoRA 的核心原理是什么？",
            "Transformer 的自注意力机制如何工作？",
            "向量数据库有哪些选择？",
        ],
        "answer": [
            "RAG（检索增强生成）是一种结合信息检索和文本生成的技术...",
            "LoRA 通过低秩分解，在预训练权重旁添加小规模可训练矩阵...",
            "自注意力机制通过 Q、K、V 三个矩阵计算 token 间的关联度...",
            "常见的向量数据库包括 Chroma、Pinecone、FAISS、Milvus...",
        ],
        "category": ["RAG", "微调", "Transformer", "RAG"],
    })
    print(f"  原始 QA 数据:\n{qa_data}")

    # 转换为 Alpaca 格式
    alpaca_data = []
    for _, row in qa_data.iterrows():
        alpaca_data.append({
            "instruction": row["question"],
            "input": "",
            "output": row["answer"],
        })

    print(f"\n  Alpaca 格式（前 2 条）:")
    for item in alpaca_data[:2]:
        print(f"  {json.dumps(item, ensure_ascii=False, indent=2)[:150]}...")

    # 转换为 ShareGPT 格式
    sharegpt_data = []
    for _, row in qa_data.iterrows():
        sharegpt_data.append({
            "conversations": [
                {"from": "human", "value": row["question"]},
                {"from": "gpt", "value": row["answer"]},
            ]
        })

    print(f"\n  ShareGPT 格式（前 1 条）:")
    print(f"  {json.dumps(sharegpt_data[0], ensure_ascii=False, indent=2)[:200]}...")

    # 按类别统计
    print(f"\n  按类别统计:")
    print(f"{qa_data['category'].value_counts()}")

    # 保存为 JSON（实际使用时取消注释）
    # with open("alpaca_train.json", "w", encoding="utf-8") as f:
    #     json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
    print("\n  💡 实际使用时，用 json.dump 保存为 .json 文件即可用于微调")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行所有演示。"""
    print("🐍 Pandas 数据分析 — 分组聚合、合并、数据清洗、格式转换")
    print("=" * 60)

    demo_groupby()
    demo_merge_concat()
    demo_data_cleaning()
    demo_finetune_data_conversion()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. groupby + agg 是数据分析的核心模式")
    print("  2. merge 对应 SQL JOIN，concat 对应 UNION")
    print("  3. 数据清洗三板斧：strip/lower + 类型转换 + 去重")
    print("  4. CSV → Alpaca/ShareGPT 格式转换是微调数据准备的常见需求")
    print("  5. 用 Pandas 做 EDA 后再决定数据处理策略")


if __name__ == "__main__":
    main()
