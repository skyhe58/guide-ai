---
title: "Jupyter Notebook"
module: "prerequisites"
difficulty: "beginner"
interviewFrequency: "low"
tags:
  - "Jupyter"
  - "Notebook"
  - "交互式编程"
  - "数据探索"
codeExample: "00-prerequisites/"
relatedEntries:
  - "/0-prerequisites/09-virtual-env"
  - "/0-prerequisites/05-numpy-basics"
prerequisites: []
estimatedTime: "20min"
toolReferences:
  - name: "Cursor"
    usage: "VS Code 集成 Jupyter，AI 辅助编写 Notebook"
    link: "/7-ai-tools/7.1-efficiency/ai-coding"
---

# Jupyter Notebook

## 概念说明

**Jupyter Notebook** 是一个交互式计算环境，支持在浏览器中编写和运行代码，同时穿插 Markdown 文本、图表和公式。它是 AI/ML 领域最常用的开发工具之一。

### AI 开发中的 Jupyter 使用场景

- **数据探索（EDA）**：加载数据集，可视化分布，发现模式
- **模型实验**：快速迭代不同的模型架构和超参数
- **教学演示**：代码 + 说明 + 输出结果一体化展示
- **RAG 调试**：逐步执行检索 → Embedding → 生成流程，观察中间结果
- **Prompt 调试**：交互式测试不同的 Prompt，观察 LLM 输出

## 核心原理

### 1. 安装与启动

```bash
# 安装
pip install jupyter

# 启动（浏览器自动打开）
jupyter notebook

# 或使用 JupyterLab（更现代的界面）
pip install jupyterlab
jupyter lab
```

### 2. 核心概念

| 概念 | 说明 |
|------|------|
| Cell（单元格） | Notebook 的基本执行单元，分为 Code 和 Markdown 两种 |
| Kernel（内核） | 执行代码的后端进程（Python/R/Julia 等） |
| Output（输出） | 代码执行结果，支持文本、图表、HTML、交互式组件 |

### 3. 常用快捷键

| 快捷键 | 功能 |
|--------|------|
| `Shift+Enter` | 运行当前 Cell 并跳到下一个 |
| `Ctrl+Enter` | 运行当前 Cell（不跳转） |
| `A` / `B` | 在上方/下方插入新 Cell（命令模式） |
| `M` / `Y` | 切换为 Markdown / Code 模式 |
| `DD` | 删除当前 Cell |
| `Tab` | 代码补全 |

### 4. 魔法命令

```python
# 计时
%timeit np.random.randn(1000)          # 单行计时
%%timeit                                # 整个 Cell 计时

# 系统命令
!pip install transformers
!nvidia-smi                             # 查看 GPU 状态

# 环境信息
%who                                    # 列出所有变量
%whos                                   # 详细变量信息
%env                                    # 环境变量

# 自动重载（开发时修改了导入的模块）
%load_ext autoreload
%autoreload 2
```

### 5. 与 VS Code 集成

VS Code 原生支持 Jupyter Notebook（`.ipynb` 文件）：
- 安装 Python 和 Jupyter 扩展
- 直接在 VS Code 中打开 `.ipynb` 文件
- 支持 IntelliSense 代码补全
- 支持变量查看器和数据查看器
- Cursor/Kiro 等 AI IDE 也支持 Notebook

## 实战要点

**Notebook 最佳实践：**
- 用 Markdown Cell 写清楚每一步的目的和思路
- Cell 保持短小，每个 Cell 做一件事
- 重要的函数和类抽取到 `.py` 文件中，Notebook 中 import 使用
- 用 `%autoreload` 魔法命令实现热重载

**Notebook vs Python 脚本：**
- Notebook 适合：数据探索、实验、教学、调试
- Python 脚本适合：生产代码、CI/CD、自动化流水线
- 最终代码应该从 Notebook 整理到 `.py` 文件中

## 常见面试题

### Q1: Jupyter Notebook 的 Kernel 是什么？

**难度**：⭐ | **频率**：🔥

**标准答案**：

Kernel 是 Jupyter 的计算后端，负责执行 Cell 中的代码。每个 Notebook 关联一个 Kernel 进程（通常是 Python 解释器）。Kernel 维护代码执行的状态（变量、导入的模块等）。重启 Kernel 会清除所有状态。

**深入追问**：
- 如何在 Notebook 中使用不同的 Python 虚拟环境？（`python -m ipykernel install --user --name myenv`）

## 推荐工具

> 📌 以下工具可帮助你更高效地学习和实践本知识点，详见 [模块 7：AI 使用与实践](/7-ai-tools/)

| 工具 | 用途 | 详情 |
|------|------|------|
| Cursor | VS Code 集成 Jupyter，AI 辅助编写 Notebook | [AI 编程辅助](/7-ai-tools/7.1-efficiency/ai-coding) |

## 参考资料

- [Jupyter 官方文档](https://jupyter.org/documentation)
- [JupyterLab 文档](https://jupyterlab.readthedocs.io/)
- [VS Code Jupyter 扩展](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
