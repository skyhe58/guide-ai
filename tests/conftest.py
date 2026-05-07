"""
pytest 配置和 fixtures

为 guide-ai 知识库提供测试基础设施，包括：
- 项目根目录定位
- 文档和代码示例目录的 fixtures
- 模块目录列表 fixtures
- 文件收集辅助函数
"""

import os
from pathlib import Path

import pytest


def get_project_root() -> Path:
    """获取项目根目录（向上查找直到找到 README.md）"""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "README.md").exists() and (current / "docs").exists():
            return current
        current = current.parent
    # 回退：假设 tests/ 在项目根目录下
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = get_project_root()
DOCS_DIR = PROJECT_ROOT / "docs"
CODE_EXAMPLES_DIR = PROJECT_ROOT / "code-examples"
DOCKER_DIR = PROJECT_ROOT / "docker"

# 编码模块目录名（模块 0-6）
CODING_MODULE_DIRS = [
    "0-prerequisites",
    "1-ml-basics",
    "2-llm",
    "3-ai-apps",
    "4-cv",
    "5-ai-engineering",
    "6-ai-frontier",
]

# 代码示例模块目录名
CODE_MODULE_DIRS = [
    "00-prerequisites",
    "01-ml-basics",
    "02-llm",
    "03-ai-apps",
    "04-cv",
    "05-ai-engineering",
    "06-ai-frontier",
]

# 辅助文件（每个编码模块必须包含）
REQUIRED_MODULE_FILES = ["index.md", "interview.md", "cheatsheet.md"]


@pytest.fixture
def project_root() -> Path:
    """项目根目录"""
    return PROJECT_ROOT


@pytest.fixture
def docs_dir() -> Path:
    """文档目录"""
    return DOCS_DIR


@pytest.fixture
def code_examples_dir() -> Path:
    """代码示例目录"""
    return CODE_EXAMPLES_DIR


@pytest.fixture
def docker_dir() -> Path:
    """Docker 配置目录"""
    return DOCKER_DIR


@pytest.fixture
def coding_module_paths() -> list[Path]:
    """所有编码模块（0-6）的文档目录路径"""
    return [DOCS_DIR / d for d in CODING_MODULE_DIRS]


@pytest.fixture
def code_module_paths() -> list[Path]:
    """所有代码示例模块目录路径"""
    return [CODE_EXAMPLES_DIR / d for d in CODE_MODULE_DIRS]


@pytest.fixture
def all_knowledge_entry_files() -> list[Path]:
    """
    所有知识条目文档文件（编码模块下的非辅助 Markdown 文件）。
    排除 index.md、interview.md、cheatsheet.md。
    """
    entries = []
    for module_dir_name in CODING_MODULE_DIRS:
        module_path = DOCS_DIR / module_dir_name
        if module_path.is_dir():
            for f in sorted(module_path.glob("*.md")):
                if f.name not in REQUIRED_MODULE_FILES:
                    entries.append(f)
    return entries


@pytest.fixture
def all_interview_files() -> list[Path]:
    """所有模块的 interview.md 文件"""
    files = []
    for module_dir_name in CODING_MODULE_DIRS:
        interview_path = DOCS_DIR / module_dir_name / "interview.md"
        if interview_path.exists():
            files.append(interview_path)
    return files


@pytest.fixture
def all_python_files() -> list[Path]:
    """code-examples/ 下的所有 .py 文件（排除 __init__.py 和 __pycache__）"""
    files = []
    if CODE_EXAMPLES_DIR.is_dir():
        for f in sorted(CODE_EXAMPLES_DIR.rglob("*.py")):
            if f.name == "__init__.py":
                continue
            if "__pycache__" in str(f):
                continue
            files.append(f)
    return files


@pytest.fixture
def all_code_subdirs() -> list[Path]:
    """code-examples/ 下的所有知识点子目录（包含 .py 文件的最深层目录）"""
    dirs = set()
    if CODE_EXAMPLES_DIR.is_dir():
        for f in CODE_EXAMPLES_DIR.rglob("*.py"):
            if f.name == "__init__.py" or "__pycache__" in str(f):
                continue
            dirs.add(f.parent)
    return sorted(dirs)


# 外部服务关键词（用于 Property 8 检测）
EXTERNAL_SERVICE_IMPORTS = [
    "chromadb",
    "ollama",
    "vllm",
    "mlflow",
    "pinecone",
    "prometheus_client",
    "wandb",
]
