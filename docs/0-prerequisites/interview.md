---
title: "模块 0 面试指南"
---

# 模块 0：前提准备 — 面试指南

> 本指南覆盖 Python 异步编程、类型注解、NumPy 向量化等高频面试题。每道题标注难度和出现频率，方便按优先级复习。

## Python 异步编程

### Q1: asyncio 事件循环的工作原理

**难度**：⭐⭐⭐ | **频率**：🔥🔥🔥 | **关联**：[异步编程](/0-prerequisites/01-async-programming)

**答题思路**：单线程 → 协作式多任务 → I/O 多路复用 → await 让出控制权

**标准答案**：asyncio 事件循环运行在单线程中，维护任务队列。协程遇到 `await`（I/O 操作）时挂起，控制权交还事件循环，事件循环执行其他就绪协程。I/O 完成后通过 epoll/kqueue 通知事件循环恢复协程。关键优势：单线程避免锁竞争，协作式调度实现高效 I/O 并发。

**追问**：与 Node.js 事件循环的异同？如何处理 CPU 密集型任务？

---

### Q2: async/await 与多线程的区别

**难度**：⭐⭐⭐ | **频率**：🔥🔥🔥 | **关联**：[异步编程](/0-prerequisites/01-async-programming)

**答题思路**：并发模型 → 资源开销 → 适用场景 → GIL 影响

**标准答案**：async/await 是协作式单线程并发，开销极低（用户态切换），适合 I/O 密集型；多线程是抢占式并发，开销较高（内核态切换），受 GIL 限制无法真正并行 CPU 任务。AI 应用中优先用 async/await 处理网络请求，CPU 密集型用多进程。

**追问**：GIL 是什么？`asyncio.to_thread()` 的用途？

---

## 类型注解与 Pydantic

### Q3: Python 类型注解是强制的吗？

**难度**：⭐⭐ | **频率**：🔥🔥🔥 | **关联**：[类型注解](/0-prerequisites/03-type-annotations)

**答题思路**：可选性 → 不影响运行时 → 静态检查工具 → Pydantic 例外

**标准答案**：类型注解完全可选，Python 解释器不做运行时检查。价值在于：静态类型检查（mypy/pyright）、IDE 智能提示、文档作用。Pydantic 是例外——它利用类型注解在运行时做数据验证。

**追问**：Protocol 和 ABC 的区别？`from __future__ import annotations` 的作用？

---

### Q4: Pydantic v2 相比 v1 的重要变化

**难度**：⭐⭐⭐ | **频率**：🔥🔥 | **关联**：[类型注解](/0-prerequisites/03-type-annotations)

**答题思路**：Rust 重写 → API 变化 → 性能提升

**标准答案**：核心用 Rust 重写，速度提升 5-50 倍。API 变化：`.dict()` → `.model_dump()`，`@validator` → `@field_validator`，`Config` 类 → `model_config` 字典。新增严格模式 `strict=True`。

**追问**：如何从 v1 迁移到 v2？

---

## 错误处理

### Q5: try/except/else/finally 执行顺序

**难度**：⭐⭐ | **频率**：🔥🔥🔥 | **关联**：[错误处理](/0-prerequisites/02-error-handling)

**答题思路**：四个块的作用 → 有/无异常两种路径 → finally 的特殊性

**标准答案**：无异常：try → else → finally。有异常且被捕获：try → except → finally。finally 一定执行，即使有 return。else 中的异常不会被 except 捕获。

**追问**：finally 中 return 会怎样？finally 中抛异常会怎样？

---

## NumPy

### Q6: NumPy 向量化运算为什么比 Python 循环快？

**难度**：⭐⭐ | **频率**：🔥🔥🔥 | **关联**：[NumPy 基础](/0-prerequisites/05-numpy-basics)

**答题思路**：C 实现 → 连续内存 → SIMD → 无类型检查

**标准答案**：四个原因：(1) 底层 C/Fortran 实现，避免 Python 解释器开销；(2) ndarray 连续内存布局，CPU 缓存命中率高；(3) 利用 SIMD 指令集并行处理；(4) 元素类型固定，无需逐次类型检查。通常快 10-100 倍。

**追问**：ndarray 的 view 和 copy 区别？AI 场景推荐什么 dtype？

---

### Q7: 如何用 NumPy 实现余弦相似度？

**难度**：⭐⭐ | **频率**：🔥🔥 | **关联**：[NumPy 基础](/0-prerequisites/05-numpy-basics)

**答题思路**：公式 → L2 归一化 → 点积

**标准答案**：
```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 批量计算：先 L2 归一化，再矩阵乘法
norms = np.linalg.norm(docs, axis=1, keepdims=True)
normalized = docs / norms
similarities = normalized @ query_normalized
```

**追问**：为什么 RAG 检索用余弦相似度而非欧氏距离？

---

## 复习优先级

| 优先级 | 知识点 | 面试频率 |
|:------:|--------|:--------:|
| P0 | asyncio 事件循环、async/await vs 多线程 | 🔥🔥🔥 |
| P0 | 类型注解、Pydantic | 🔥🔥🔥 |
| P0 | NumPy 向量化、余弦相似度 | 🔥🔥🔥 |
| P1 | try/except/else/finally | 🔥🔥🔥 |
| P1 | 自定义异常设计 | 🔥🔥 |
| P2 | Pandas groupby/merge | 🔥🔥 |
| P2 | pyproject.toml vs requirements.txt | 🔥🔥 |
