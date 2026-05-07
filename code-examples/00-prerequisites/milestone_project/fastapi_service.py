"""
里程碑项目 — FastAPI 服务

整合模块 0 核心知识点：
- 异步编程（async def 路由）
- 类型注解（Pydantic 请求/响应模型）
- NumPy（向量相似度计算）
- Pandas（数据查询和统计）
- 错误处理（自定义异常 + 异常处理器）

场景：LLM 模型评测数据查询 API

Python 版本：3.11+
依赖：fastapi>=0.109, uvicorn>=0.27, pandas>=2.1, numpy>=1.26, pydantic>=2.5
最后验证：2024-12-01

运行方式：
  python fastapi_service.py
  # 访问 http://localhost:8000/docs 查看 API 文档
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field


# ============================================================
# 1. Pydantic 模型（请求/响应）
# ============================================================

class ModelInfo(BaseModel):
    """模型信息响应。"""
    model: str
    avg_score: float
    best_benchmark: str
    best_score: float
    cost_per_1m: float
    rank: int


class BenchmarkQuery(BaseModel):
    """评测查询请求。"""
    benchmark: Literal["mmlu", "humaneval", "gsm8k", "arc"]
    min_score: float = Field(default=0, ge=0, le=100)
    max_cost: float = Field(default=100, ge=0)


class SimilarityRequest(BaseModel):
    """相似度计算请求。"""
    vector_a: list[float] = Field(min_length=1)
    vector_b: list[float] = Field(min_length=1)


class SimilarityResponse(BaseModel):
    """相似度计算响应。"""
    cosine_similarity: float
    euclidean_distance: float
    dimensions: int


class StatsResponse(BaseModel):
    """统计响应。"""
    total_models: int
    total_benchmarks: int
    avg_score: float
    top_model: str
    top_score: float


# ============================================================
# 2. 模拟数据（Pandas DataFrame）
# ============================================================

def create_mock_data() -> pd.DataFrame:
    """创建模拟评测数据。"""
    np.random.seed(42)
    models = ["qwen2-7b", "llama3-8b", "deepseek-v2", "gpt-4o", "claude-3.5"]
    benchmarks = ["mmlu", "humaneval", "gsm8k", "arc"]
    costs = {"qwen2-7b": 0.5, "llama3-8b": 0.0, "deepseek-v2": 0.3,
             "gpt-4o": 5.0, "claude-3.5": 3.0}

    rows = []
    for model in models:
        for bench in benchmarks:
            rows.append({
                "model": model,
                "benchmark": bench,
                "score": round(np.random.uniform(45, 95), 1),
                "cost_per_1m": costs[model],
            })
    return pd.DataFrame(rows)


# 全局数据（实际项目中应该用数据库）
DATA = create_mock_data()


# ============================================================
# 3. FastAPI 应用
# ============================================================

app = FastAPI(
    title="LLM 评测数据 API",
    description="模块 0 里程碑项目 — 整合异步编程、类型注解、NumPy、Pandas",
    version="1.0.0",
)


@app.get("/", summary="健康检查")
async def health_check():
    """API 健康检查。"""
    return {"status": "ok", "models": DATA["model"].nunique()}


@app.get("/models", response_model=list[ModelInfo], summary="获取所有模型排名")
async def get_models():
    """获取所有模型的评测排名（异步路由 + Pandas 聚合）。"""
    # Pandas 分组聚合
    stats = DATA.groupby("model").agg(
        avg_score=("score", "mean"),
        best_score=("score", "max"),
        cost_per_1m=("cost_per_1m", "first"),
    ).round(1)

    # 找每个模型的最佳基准
    best_bench = DATA.loc[DATA.groupby("model")["score"].idxmax()]
    best_bench_map = dict(zip(best_bench["model"], best_bench["benchmark"]))

    # 排名
    stats["rank"] = stats["avg_score"].rank(ascending=False).astype(int)
    stats = stats.sort_values("rank")

    results = []
    for model, row in stats.iterrows():
        results.append(ModelInfo(
            model=str(model),
            avg_score=row["avg_score"],
            best_benchmark=best_bench_map.get(str(model), ""),
            best_score=row["best_score"],
            cost_per_1m=row["cost_per_1m"],
            rank=row["rank"],
        ))
    return results


@app.get("/models/{model_name}", response_model=ModelInfo, summary="获取单个模型信息")
async def get_model(model_name: str):
    """获取指定模型的评测信息。"""
    model_data = DATA[DATA["model"] == model_name.lower()]
    if model_data.empty:
        raise HTTPException(status_code=404, detail=f"模型 '{model_name}' 不存在")

    avg_score = round(model_data["score"].mean(), 1)
    best_row = model_data.loc[model_data["score"].idxmax()]

    return ModelInfo(
        model=model_name.lower(),
        avg_score=avg_score,
        best_benchmark=best_row["benchmark"],
        best_score=best_row["score"],
        cost_per_1m=best_row["cost_per_1m"],
        rank=0,  # 简化
    )


@app.post("/query", summary="按条件查询评测结果")
async def query_benchmarks(query: BenchmarkQuery):
    """按基准、最低分数、最高成本筛选（Pydantic 验证 + Pandas 筛选）。"""
    filtered = DATA[
        (DATA["benchmark"] == query.benchmark) &
        (DATA["score"] >= query.min_score) &
        (DATA["cost_per_1m"] <= query.max_cost)
    ].sort_values("score", ascending=False)

    return filtered.to_dict(orient="records")


@app.post("/similarity", response_model=SimilarityResponse, summary="计算向量相似度")
async def compute_similarity(req: SimilarityRequest):
    """计算两个向量的余弦相似度和欧氏距离（NumPy 向量化）。"""
    a = np.array(req.vector_a, dtype=np.float32)
    b = np.array(req.vector_b, dtype=np.float32)

    if a.shape != b.shape:
        raise HTTPException(status_code=400, detail="两个向量维度必须相同")

    # NumPy 计算
    cos_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    euc_dist = float(np.linalg.norm(a - b))

    return SimilarityResponse(
        cosine_similarity=round(cos_sim, 6),
        euclidean_distance=round(euc_dist, 6),
        dimensions=len(a),
    )


@app.get("/stats", response_model=StatsResponse, summary="获取统计概览")
async def get_stats():
    """获取评测数据统计概览。"""
    avg_by_model = DATA.groupby("model")["score"].mean()
    top_model = avg_by_model.idxmax()

    return StatsResponse(
        total_models=DATA["model"].nunique(),
        total_benchmarks=DATA["benchmark"].nunique(),
        avg_score=round(DATA["score"].mean(), 1),
        top_model=top_model,
        top_score=round(avg_by_model.max(), 1),
    )


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    import uvicorn

    print("🚀 启动 LLM 评测数据 API")
    print("📖 API 文档: http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
