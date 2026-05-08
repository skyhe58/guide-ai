---
title: "Pandas 基础操作"
module: "prerequisites"
difficulty: "beginner"
interviewFrequency: "medium"
tags:
  - "Python"
  - "Pandas"
  - "DataFrame"
  - "数据分析"
  - "数据处理"
codeExample: "00-prerequisites/pandas_basics/"
relatedEntries:
  - "/0-prerequisites/05-numpy-basics"
  - "/1-ml-basics/01-supervised-learning"
prerequisites:
  - "/0-prerequisites/05-numpy-basics"
estimatedTime: "50min"
toolReferences:
  - name: "Cursor"
    usage: "辅助编写 Pandas 数据处理代码，自动补全 API"
    link: "/7-ai-tools/7.1-efficiency/ai-coding"
  - name: "ChatGPT"
    usage: "描述数据处理需求，生成 Pandas 代码"
    link: "/7-ai-tools/7.1-efficiency/ai-chat"
---

# Pandas 基础操作

## 概念说明

**Pandas** 是 Python 最流行的数据分析库，提供 `DataFrame`（二维表格）和 `Series`（一维序列）两种核心数据结构。它是数据预处理、特征工程和探索性数据分析（EDA）的标准工具。

### 为什么 AI 开发者需要掌握 Pandas？

- **数据预处理**：ML 模型训练前的数据清洗、缺失值处理、特征编码
- **训练数据管理**：微调数据集的加载、筛选、格式转换（Alpaca/ShareGPT 格式）
- **评估分析**：模型评估指标的汇总、对比、可视化
- **日志分析**：LLM 推理日志的解析和统计（延迟、Token 用量、错误率）
- **RAG 数据准备**：文档元数据管理、检索结果分析

## 核心原理

### 1. DataFrame 创建

```python
import pandas as pd

# 从字典创建
df = pd.DataFrame({
    "model": ["qwen2-7b", "llama3-8b", "deepseek-v2"],
    "params_b": [7, 8, 236],
    "license": ["apache-2.0", "llama3", "deepseek"],
    "score": [72.5, 68.3, 79.1],
})

# 从 CSV/JSON 读取
# df = pd.read_csv("models.csv")
# df = pd.read_json("models.json")
# df = pd.read_excel("models.xlsx")
```

### 2. 数据筛选与查询

```python
# 条件筛选
large_models = df[df["params_b"] > 10]
open_source = df[df["license"] != "proprietary"]

# 多条件组合
good_open = df[(df["score"] > 70) & (df["license"] == "apache-2.0")]

# loc — 按标签索引
df.loc[0, "model"]           # 第 0 行的 model 列
df.loc[:, ["model", "score"]] # 所有行的 model 和 score 列

# iloc — 按位置索引
df.iloc[0:2, 1:3]            # 前 2 行，第 1-2 列

# query — SQL 风格查询
df.query("score > 70 and params_b < 100")
```

### 3. 分组聚合

```python
# 按许可证分组，计算平均分数
df.groupby("license")["score"].mean()

# 多聚合函数
df.groupby("license").agg({
    "score": ["mean", "max", "min"],
    "params_b": "sum",
})

# value_counts — 统计频次
df["license"].value_counts()
```

### 4. 缺失值处理

```python
# 检查缺失值
df.isnull().sum()

# 填充缺失值
df["score"].fillna(df["score"].mean(), inplace=True)

# 删除含缺失值的行
df.dropna(subset=["model", "score"])

# 插值
df["score"].interpolate(method="linear")
```

### 5. 数据合并

```python
# merge — 类似 SQL JOIN
benchmarks = pd.DataFrame({
    "model": ["qwen2-7b", "llama3-8b"],
    "mmlu": [65.3, 63.1],
})
merged = df.merge(benchmarks, on="model", how="left")

# concat — 纵向/横向拼接
df_new = pd.concat([df1, df2], ignore_index=True)  # 纵向
df_wide = pd.concat([df_a, df_b], axis=1)           # 横向
```

### 6. 常用操作速查

| 操作 | 代码 | 说明 |
|------|------|------|
| 查看前 N 行 | `df.head(5)` | 快速预览数据 |
| 数据概览 | `df.info()` | 列名、类型、非空数量 |
| 统计摘要 | `df.describe()` | 均值、标准差、分位数 |
| 排序 | `df.sort_values("score", ascending=False)` | 按分数降序 |
| 去重 | `df.drop_duplicates(subset=["model"])` | 按模型名去重 |
| 重命名列 | `df.rename(columns={"params_b": "参数量"})` | 列名映射 |
| 新增列 | `df["rank"] = range(1, len(df)+1)` | 添加排名列 |
| 类型转换 | `df["score"].astype(int)` | 浮点转整数 |
| 应用函数 | `df["model"].apply(str.upper)` | 对每个元素应用函数 |

## 代码示例

> 💻 完整可运行代码：[code-examples/00-prerequisites/pandas_basics/](https://github.com/skyhe58/guide-ai/tree/main/code-examples/00-prerequisites/pandas_basics/)
> 🐍 Python 版本：3.11+
> 📦 依赖：pandas>=2.1

```python
import pandas as pd

# 模拟 LLM 评测数据
df = pd.DataFrame({
    "model": ["qwen2-7b", "llama3-8b", "deepseek-v2", "gpt-4o"],
    "mmlu": [65.3, 63.1, 78.5, 86.1],
    "humaneval": [52.0, 48.5, 65.2, 87.1],
    "cost_per_1m": [0.5, 0.0, 0.3, 5.0],  # 每百万 Token 成本（美元）
})

# 筛选性价比高的模型
affordable = df[df["cost_per_1m"] < 1.0].sort_values("mmlu", ascending=False)
print(affordable[["model", "mmlu", "cost_per_1m"]])
```

## 实战要点

**AI 数据处理常见模式：**
- 微调数据集格式转换：CSV → Alpaca JSON 格式
- 评测结果对比：多模型多指标的 pivot table
- 日志分析：按时间窗口聚合推理延迟和错误率

**性能优化：**
- 大数据集用 `dtype` 参数指定列类型，减少内存占用
- 用 `pd.read_csv(..., chunksize=10000)` 分块读取大文件
- 向量化操作优先于 `apply()`，`apply()` 优先于 `iterrows()`

**常见陷阱：**
- `inplace=True` 已不推荐使用，建议用赋值方式 `df = df.dropna()`
- 链式索引 `df[col][row]` 可能返回副本，用 `df.loc[row, col]` 更安全
- `merge` 默认是 inner join，注意数据丢失

## 常见面试题

### Q1: Pandas 的 apply、map、applymap 有什么区别？

**难度**：⭐⭐ | **频率**：🔥🔥

**答题思路**：
1. 作用范围不同（元素/列/行/整个 DataFrame）
2. 返回值不同
3. 性能差异

**标准答案**：

- `Series.map(func)`：对 Series 的每个元素应用函数，返回 Series
- `Series.apply(func)`：类似 map，但支持更复杂的函数（可接收额外参数）
- `DataFrame.apply(func, axis)`：对 DataFrame 的每行（axis=1）或每列（axis=0）应用函数
- `DataFrame.map(func)`：（Pandas 2.1+，替代旧版 applymap）对 DataFrame 的每个元素应用函数

性能排序：向量化操作 > map > apply > iterrows

**深入追问**：
- 什么时候应该用向量化操作替代 apply？（能用 NumPy/Pandas 内置函数实现的都应该向量化）
- 如何用 `pd.eval()` 加速大 DataFrame 的运算？

## 推荐工具

> 📌 以下工具可帮助你更高效地学习和实践本知识点，详见 [模块 7：AI 使用与实践](/7-ai-tools/)

| 工具 | 用途 | 详情 |
|------|------|------|
| Cursor | 辅助编写 Pandas 数据处理代码 | [AI 编程辅助](/7-ai-tools/7.1-efficiency/ai-coding) |
| ChatGPT | 描述数据处理需求，生成 Pandas 代码 | [AI 对话助手](/7-ai-tools/7.1-efficiency/ai-chat) |

## 参考资料

- [Pandas 官方文档](https://pandas.pydata.org/docs/)
- [Pandas 快速入门](https://pandas.pydata.org/docs/getting_started/intro_tutorials/)
- [Real Python — Pandas Tutorial](https://realpython.com/pandas-python-explore-dataset/)
- [Kaggle — Pandas 微课程](https://www.kaggle.com/learn/pandas)
