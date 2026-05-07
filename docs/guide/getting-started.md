# 快速开始

## 环境要求

| 工具 | 版本 | 说明 |
|------|------|------|
| Python | 3.11+ | 代码示例运行环境 |
| Node.js | 18+ | VitePress 文档站点构建 |
| pnpm | 8+ | Node.js 包管理器 |
| Git | 2.x | 版本控制 |
| Docker | 24+ | 可选，运行 AI 服务依赖 |

## 克隆仓库

```bash
git clone https://github.com/your-username/guide-ai.git
cd guide-ai
```

## 安装依赖

### Python 代码示例依赖

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install -r code-examples/requirements.txt

# 安装开发依赖（可选，用于运行测试）
pip install -r code-examples/requirements-dev.txt
```

### VitePress 文档站点依赖

```bash
cd docs
pnpm install
```

## 启动开发服务器

```bash
# 在 docs/ 目录下
pnpm run dev
```

浏览器访问 `http://localhost:5173` 即可查看文档站点。

## 运行代码示例

代码示例支持两种运行模式：

### 内存模式（默认）

直接运行，无需外部服务，适合理解原理：

```bash
python code-examples/00-prerequisites/async_programming/01_asyncio_basics.py
```

### 服务模式

连接真实 AI 服务，需要先通过 Docker 启动依赖：

```bash
# 启动基础服务（Ollama + Chroma）
docker compose -f docker/docker-compose.yml up -d

# 运行服务模式示例
python code-examples/03-ai-apps/rag/04_vector_store.py server
```

## 下一步

- 📖 阅读 [使用指南](/guide/how-to-use) 了解学习路径选择
- 🚀 从 [模块 0：前提准备](/0-prerequisites/) 开始学习
- 🛤️ 查看 [学习路径](/learning-paths/fast-track) 规划你的学习计划
