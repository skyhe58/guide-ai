# 使用指南

## 学习路径选择

本知识库提供三条学习路线，根据你的目标选择：

| 路线 | 时间 | 适合人群 | 覆盖模块 |
|------|------|----------|----------|
| [4 个月速成](/learning-paths/fast-track) | 4 个月 | 想快速掌握 LLM 应用开发的后端开发者 | 0 → 1 → 2 → 3 → 5 |
| [6-8 个月全栈](/learning-paths/full-stack) | 6-8 个月 | 想全面掌握 AI 全栈能力的开发者 | 0 → 1 → 2 → 3 → 4 → 5 → 6 |
| [AI 工具使用者](/learning-paths/tool-user) | 2-4 周 | 想用 AI 工具提效或变现的任何人 | 仅模块 7 |

> 💡 无论选择哪条路线，模块 0（前提准备）都是必修的第一步。

## 文档结构说明

### 双线并行架构

- **编码线（模块 0-6）**：培养 AI 工程师能力，每篇文档底部有"推荐工具"区块
- **使用线（模块 7）**：独立参考手册，无前置依赖，可随时查阅

### 每篇知识文档包含

1. **概念说明** — 通俗易懂的知识点介绍
2. **核心原理** — 深入分析，配合 Mermaid 图表
3. **代码示例** — 链接到可运行的 Python 代码
4. **实战要点** — 项目中的注意事项和最佳实践
5. **常见面试题** — 带难度和频率标注
6. **推荐工具** — 链接到模块 7 对应的 AI 工具
7. **参考资料** — 延伸学习资源

## 代码示例运行方式

### 内存模式 vs 服务模式

所有代码示例默认支持**内存模式**，直接运行即可理解核心原理：

```bash
python code-examples/03-ai-apps/rag/04_vector_store.py
```

需要连接真实服务时，传入 `server` 参数切换到**服务模式**：

```bash
python code-examples/03-ai-apps/rag/04_vector_store.py server
```

### Docker 服务启动

服务模式需要先启动对应的 Docker 服务：

| Docker Compose 文件 | 包含服务 | 对应模块 |
|---------------------|----------|----------|
| `docker/docker-compose.yml` | Ollama、Chroma | 模块 2-3 |
| `docker/docker-compose.ml.yml` | MLflow、Jupyter | 模块 5 |
| `docker/docker-compose.llm.yml` | vLLM、TGI | 模块 2、5 |
| `docker/docker-compose.monitor.yml` | Prometheus、Grafana | 模块 5 |

```bash
# 启动基础服务
docker compose -f docker/docker-compose.yml up -d

# 启动 ML 服务
docker compose -f docker/docker-compose.ml.yml up -d

# 按需启动，避免占用过多资源
```

> ⚠️ 建议按需启动服务，不要一次性启动所有 Docker Compose 文件。

## 面试备战

- 每个模块都有独立的 `interview.md` 面试指南
- [面试汇总](/interview/) 提供按岗位、公司、难度的分类视图
- [知识图谱](/interview/knowledge-map) 展示知识点关联和追问路径
