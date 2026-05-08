"""
里程碑项目 — CSV 数据处理脚本

整合模块 0 核心知识点：
- 异步编程（asyncio）：异步文件读取模拟
- 类型注解（Pydantic）：数据模型验证
- NumPy：数值统计和向量化运算
- Pandas：数据加载、清洗、分组聚合、导出

场景：处理 LLM 评测数据 CSV，清洗后生成统计报告

Python 版本：3.11+
依赖：pandas>=2.1, numpy>=1.26, pydantic>=2.5
最后验证：2024-12-01
"""

from __future__ import annotations

import asyncio
import io
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

# ============================================================
# 1. Pydantic 数据模型（类型注解 + 数据验证）
# ============================================================

class BenchmarkRecord(BaseModel):
    """单条评测记录 — Pydantic 自动验证。"""
    model: str = Field(min_length=1, description="模型名称")
    benchmark: Literal["mmlu", "humaneval", "gsm8k", "arc"]
    score: float = Field(ge=0, le=100, description="评测分数")
    latency_ms: float = Field(ge=0, description="推理延迟（毫秒）")
    cost_per_1m: float = Field(ge=0, description="每百万 Token 成本（美元）")

    @field_validator("model")
    @classmethod
    def normalize_model_name(cls, v: str) -> str:
        """统一模型名称格式：去空格、转小写。"""
        return v.strip().lower()


class ProcessingReport(BaseModel):
    """处理报告。"""
    total_records: int
    valid_records: int
    invalid_records: int
    models_count: int
    benchmarks_count: int
    top_model: str
    top_score: float


# ============================================================
# 2. 模拟 CSV 数据生成
# ============================================================

def generate_mock_csv() -> str:
    """生成模拟的 LLM 评测 CSV 数据。"""
    np.random.seed(42)
    models = ["qwen2-7b", "llama3-8b", "deepseek-v2", "gpt-4o", "claude-3.5"]
    benchmarks = ["mmlu", "humaneval", "gsm8k", "arc"]
    costs = {"qwen2-7b": 0.5, "llama3-8b": 0.0, "deepseek-v2": 0.3,
             "gpt-4o": 5.0, "claude-3.5": 3.0}

    rows = ["model,benchmark,score,latency_ms,cost_per_1m"]
    for model in models:
        for bench in benchmarks:
            score = round(np.random.uniform(40, 95), 1)
            latency = round(np.random.uniform(100, 3000), 0)
            cost = costs[model]
            rows.append(f"{model},{bench},{score},{latency},{cost}")

    # 添加一些脏数据
    rows.append("  Qwen2-7B ,mmlu,72.5,500,0.5")  # 需要 strip + lower
    rows.append(",humaneval,80,200,0")              # 空模型名（无效）
    rows.append("test-model,mmlu,150,100,0")        # 分数超范围（无效）

    return "\n".join(rows)


# ============================================================
# 3. Pandas 数据处理流水线
# ============================================================

def load_and_clean(csv_content: str) -> tuple[pd.DataFrame, int]:
    """加载 CSV 并清洗数据。

    返回：(清洗后的 DataFrame, 无效记录数)
    """
    # 读取 CSV
    df = pd.read_csv(io.StringIO(csv_content))
    total = len(df)
    print(f"  📥 加载 {total} 条记录")

    # 用 Pydantic 逐行验证
    valid_rows = []
    invalid_count = 0

    for _, row in df.iterrows():
        try:
            record = BenchmarkRecord(**row.to_dict())
            valid_rows.append(record.model_dump())
        except Exception as e:
            invalid_count += 1
            print(f"  ⚠️ 无效记录: {row.to_dict()} → {e}")

    # 转回 DataFrame
    df_clean = pd.DataFrame(valid_rows)
    print(f"  ✅ 有效: {len(df_clean)}, 无效: {invalid_count}")

    return df_clean, invalid_count


def analyze(df: pd.DataFrame) -> dict:
    """数据分析 — 分组聚合、统计、排名。"""
    print("\n  📊 数据分析")
    print("  " + "-" * 50)

    # 按模型分组统计
    model_stats = df.groupby("model").agg(
        平均分=("score", "mean"),
        最高分=("score", "max"),
        平均延迟=("latency_ms", "mean"),
        成本=("cost_per_1m", "first"),
    ).round(1)
    model_stats["排名"] = model_stats["平均分"].rank(ascending=False).astype(int)
    model_stats = model_stats.sort_values("排名")
    print(f"\n  模型排名:\n{model_stats}")

    # 按基准分组
    bench_stats = df.groupby("benchmark")["score"].agg(["mean", "std", "min", "max"]).round(1)
    print(f"\n  基准统计:\n{bench_stats}")

    # 透视表
    pivot = df.pivot_table(index="model", columns="benchmark", values="score").round(1)
    print(f"\n  评测矩阵:\n{pivot}")

    # NumPy 向量化计算：性价比指数
    scores = df.groupby("model")["score"].mean().values
    costs = df.groupby("model")["cost_per_1m"].first().values
    # 性价比 = 分数 / (成本 + 0.1)，避免除零
    value_index = scores / (costs + 0.1)
    models = df.groupby("model")["score"].mean().index.tolist()

    print("\n  性价比指数（分数/成本）:")
    for m, v in sorted(zip(models, value_index), key=lambda x: -x[1]):
        print(f"    {m}: {v:.1f}")

    # 找出最佳模型
    best_idx = np.argmax(scores)
    return {
        "model_stats": model_stats.to_dict(),
        "best_model": models[best_idx],
        "best_score": float(scores[best_idx]),
    }


# ============================================================
# 4. 异步处理模拟
# ============================================================

async def async_process_pipeline(csv_content: str) -> ProcessingReport:
    """异步数据处理流水线。

    模拟实际场景：异步读取数据 → 清洗 → 分析 → 生成报告。
    """
    print("🚀 启动异步数据处理流水线")
    print("=" * 60)

    # 模拟异步读取（实际场景可能从远程 API 或数据库获取）
    print("\n  ⏳ 异步读取数据...")
    await asyncio.sleep(0.1)  # 模拟 I/O

    # 清洗
    df_clean, invalid_count = load_and_clean(csv_content)

    # 分析
    analysis = analyze(df_clean)

    # 生成报告
    report = ProcessingReport(
        total_records=len(df_clean) + invalid_count,
        valid_records=len(df_clean),
        invalid_records=invalid_count,
        models_count=df_clean["model"].nunique(),
        benchmarks_count=df_clean["benchmark"].nunique(),
        top_model=analysis["best_model"],
        top_score=analysis["best_score"],
    )

    print(f"\n  📋 处理报告:")
    print(f"  {report.model_dump_json(indent=2)}")

    return report


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行里程碑项目。"""
    print("🐍 里程碑项目 — CSV 数据处理脚本")
    print("=" * 60)
    print("整合：异步编程 + 类型注解(Pydantic) + NumPy + Pandas")
    print("=" * 60)

    # 生成模拟数据
    csv_content = generate_mock_csv()

    # 运行异步流水线
    report = asyncio.run(async_process_pipeline(csv_content))

    print("\n" + "=" * 60)
    print("✅ 里程碑项目完成！")
    print(f"  最佳模型: {report.top_model} (平均分: {report.top_score:.1f})")
    print(f"  处理: {report.valid_records}/{report.total_records} 条有效记录")


if __name__ == "__main__":
    main()
