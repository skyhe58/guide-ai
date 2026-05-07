---
title: "模块 0 速查卡片"
---

# 模块 0：前提准备 — 速查卡片

## asyncio

```python
import asyncio

async def main():
    result = await some_coroutine()           # 等待协程
    results = await asyncio.gather(a(), b())  # 并发执行
    task = asyncio.create_task(background())  # 后台任务

asyncio.run(main())                           # 启动事件循环

# Python 3.11+ TaskGroup
async with asyncio.TaskGroup() as tg:
    tg.create_task(a())
    tg.create_task(b())

# 超时控制（3.11+）
async with asyncio.timeout(5.0):
    await slow_operation()
```

## aiohttp

```python
import aiohttp

async with aiohttp.ClientSession() as session:
    # GET
    async with session.get(url) as resp:
        data = await resp.json()
    # POST
    async with session.post(url, json=payload) as resp:
        data = await resp.json()

# 超时
timeout = aiohttp.ClientTimeout(total=30, connect=5)
async with aiohttp.ClientSession(timeout=timeout) as session:
    ...
```

## 异常处理

```python
# 自定义异常
class AIServiceError(Exception):
    def __init__(self, msg, service="", retry_after=0):
        super().__init__(msg)
        self.service = service
        self.retry_after = retry_after

# 异常链
raise NewError("msg") from original_error

# try/except/else/finally
try:
    result = risky_call()
except SpecificError as e:
    handle(e)
else:
    on_success(result)      # 无异常时执行
finally:
    cleanup()               # 总是执行
```

## 类型注解

```python
from typing import Literal, Protocol, TypedDict

# 基础
def func(x: str, y: int = 0) -> list[str]: ...

# 联合类型（3.10+）
value: str | None = None

# Literal
Mode = Literal["train", "eval", "predict"]

# Protocol
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

# TypedDict
class Config(TypedDict):
    model: str
    temperature: float
```

## Pydantic

```python
from pydantic import BaseModel, Field, field_validator

class Request(BaseModel):
    prompt: str = Field(min_length=1, max_length=5000)
    temperature: float = Field(default=0.7, ge=0, le=2)

    @field_validator("prompt")
    @classmethod
    def not_blank(cls, v): return v.strip() or raise ValueError()

req = Request(prompt="hello")
req.model_dump()              # → dict
req.model_dump_json()         # → JSON string
Request.model_validate(data)  # dict → model
```

## NumPy

```python
import numpy as np

# 创建
arr = np.array([1, 2, 3])
zeros = np.zeros((100, 768), dtype=np.float32)
rand = np.random.randn(10, 768)

# 索引
arr[0], arr[-1], arr[1:4]
matrix[0]              # 第 0 行
matrix[:, :3]          # 所有行前 3 列
matrix[mask]           # 布尔索引

# 运算
np.dot(a, b)                          # 点积
a @ b                                 # 矩阵乘法
np.linalg.norm(v)                     # L2 范数
np.argsort(arr)[-5:][::-1]            # Top-5 索引

# 余弦相似度
sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Softmax
def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()
```

## Pandas

```python
import pandas as pd

# 创建
df = pd.DataFrame({"col1": [1,2], "col2": [3,4]})
df = pd.read_csv("file.csv")

# 筛选
df[df["score"] > 70]
df.query("score > 70 and cost < 1")
df[df["col"].isin(["a", "b"])]

# 分组聚合
df.groupby("model")["score"].mean()
df.groupby("model").agg(avg=("score","mean"), cnt=("score","count"))

# 合并
pd.merge(df1, df2, on="key", how="left")
pd.concat([df1, df2], ignore_index=True)

# 缺失值
df.isnull().sum()
df.fillna(0)
df.dropna(subset=["col"])
```

## 包管理

```bash
# venv
python -m venv .venv && source .venv/bin/activate

# pip
pip install -r requirements.txt
pip install ".[dev]"                    # pyproject.toml 可选依赖

# poetry
poetry install
poetry add torch transformers
poetry export -f requirements.txt -o requirements.txt
```

## Git

```bash
git checkout -b feature/xxx            # 创建分支
git add . && git commit -m "type(scope): msg"
git push origin feature/xxx            # 推送
# commit 类型：docs/code/feat/fix/ci/docker
```
