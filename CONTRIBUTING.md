# 贡献指南

感谢你对 guide-ai 知识库的关注！本文档说明如何为项目贡献内容。

## 📁 文档结构说明

```
docs/                            # 知识文档
├── {模块编号-模块名}/            # 如 0-prerequisites/、3-ai-apps/
│   ├── index.md                 # 模块索引（列出所有知识点）
│   ├── {序号-知识点}.md          # 知识条目（使用统一模板）
│   ├── interview.md             # 面试指南
│   └── cheatsheet.md            # 速查卡片
└── templates/
    └── entry-template.md        # 知识条目模板

code-examples/                   # 代码示例
├── {模块编号-模块名}/            # 如 00-prerequisites/、03-ai-apps/
│   └── {知识点子目录}/           # 如 async_programming/
│       └── 01_xxx.py            # 可独立运行的 Python 脚本
├── pyproject.toml               # 统一依赖管理
└── requirements.txt             # pip 兼容依赖
```

## 📝 如何添加新知识模块

1. 在 `docs/` 下创建模块目录，命名格式：`{编号}-{英文名}/`
2. 创建模块必需文件：
   - `index.md` — 模块索引，列出所有知识点及代码示例链接
   - `interview.md` — 面试指南
   - `cheatsheet.md` — 速查卡片
3. 在 `code-examples/` 下创建对应代码目录
4. 更新 `docs/.vitepress/sidebar.mts` 侧边栏配置
5. 更新 `README.md` 中的模块导航表和完成度追踪表

## 📄 如何添加新知识点

1. 复制模板文件 `docs/templates/entry-template.md`
2. 按模板结构填写内容，必须包含以下章节：
   - **概念说明**：用通俗语言解释知识点
   - **核心原理**：深入分析，复杂流程配 Mermaid 图
   - **代码示例**：链接到 `code-examples/` 对应代码
   - **推荐工具**：链接到模块 7 对应小节（编码模块 0-6 必须）
   - **常见面试题**：标注难度（⭐）和频率（🔥）
   - **参考资料**：外部学习资源
3. 填写 YAML frontmatter 元数据（title、module、difficulty 等）
4. 将文件放入对应模块目录，命名格式：`{序号}-{英文名}.md`
5. 更新模块 `index.md` 索引

## 💻 如何添加代码示例

### 命名规范

- 目录名使用 `snake_case`：如 `async_programming/`、`rag_retrieval/`
- 文件名使用 `{序号}_{描述}.py`：如 `01_asyncio_basics.py`、`02_async_context.py`
- 与文档文件名保持一致的命名风格，方便互相查找

### 代码要求

1. **中文注释**：每个文件必须包含中文注释，解释关键逻辑
2. **文件头注释**：包含 Python 版本要求、依赖库、最后验证日期
3. **可独立运行**：每个脚本必须包含 `if __name__ == "__main__":` 块
4. **双模式运行**：依赖外部服务的示例需支持内存模式和服务模式

### 文件头注释模板

```python
"""
{知识点标题}

Python 版本：3.11+
依赖：{主要依赖库}
外部服务：{如有，标注服务名}
启动命令：{如有，标注 Docker 启动命令}
免费替代：{如有付费 API，标注免费替代方案}
最后验证：{YYYY-MM-DD}
"""
```

### 代码示例结构

```python
"""
RAG 向量检索示例

Python 版本：3.11+
依赖：chromadb, langchain
外部服务：Chroma 向量数据库
启动命令：docker compose -f docker/docker-compose.yml up -d chroma
免费替代：使用内存模式（无需 Docker）
最后验证：2024-12-01
"""

import sys


def demo_in_memory():
    """内存模式：无需外部服务，直接运行理解原理"""
    # 核心逻辑...
    pass


def demo_with_server():
    """服务模式：连接真实服务，需 Docker"""
    # 连接外部服务...
    pass


if __name__ == "__main__":
    # 默认运行内存模式
    demo_in_memory()

    # 传入 'server' 参数运行服务模式
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        demo_with_server()
```

## 📌 提交规范

### Commit Message 格式

```
<type>(<scope>): <description>

[可选正文]
```

### Type 类型

| 类型 | 说明 | 示例 |
|------|------|------|
| `docs` | 文档变更 | `docs(ml-basics): 添加 Transformer 架构详解` |
| `code` | 代码示例变更 | `code(rag): 添加向量检索代码示例` |
| `feat` | 新功能/新模块 | `feat(cv): 添加计算机视觉模块` |
| `fix` | 修复错误 | `fix(llm): 修复 LoRA 示例代码 bug` |
| `style` | 格式调整 | `style: 统一代码注释格式` |
| `ci` | CI/CD 变更 | `ci: 添加 Python 代码语法检查` |
| `docker` | Docker 配置变更 | `docker: 更新 Ollama 镜像版本` |
| `chore` | 其他杂项 | `chore: 更新依赖版本` |

### Scope 范围

使用模块名作为 scope：`prerequisites`、`ml-basics`、`llm`、`ai-apps`、`cv`、`ai-engineering`、`ai-frontier`、`ai-tools`

### 示例

```
docs(ai-apps): 添加 RAG 检索策略知识条目

- 新增 docs/3-ai-apps/09-retrieval-strategies.md
- 包含相似度检索、MMR、混合检索等策略对比
- 添加 Mermaid 流程图说明检索流程
```

## ❓ 常见问题

- **Q: 模块 7 需要代码示例吗？**
  A: 模块 7 是使用线（工具参考手册），不需要 `code-examples/` 目录下的代码示例，但可以在文档中提供 Prompt 模板和使用截图。

- **Q: 如何更新模块 7 的工具信息？**
  A: AI 工具迭代快速，建议在工具条目中标注"最后更新日期"，定期检查工具的版本和功能变化。
