"""
Property-based tests 骨架

使用 pytest + hypothesis 验证 guide-ai 知识库的文档和代码文件的
格式一致性与结构完整性。每个 property 对应设计文档中定义的一个属性。

运行方式：
    pytest tests/test_properties.py -v
    pytest tests/test_properties.py -v -k "property"
"""

import re
from pathlib import Path

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from conftest import (
    CODE_EXAMPLES_DIR,
    CODING_MODULE_DIRS,
    DOCS_DIR,
    EXTERNAL_SERVICE_IMPORTS,
    REQUIRED_MODULE_FILES,
)


# ============================================================
# 辅助函数：收集测试输入空间
# ============================================================

def collect_coding_module_paths() -> list[Path]:
    """收集所有编码模块（0-6）的文档目录路径"""
    paths = []
    for d in CODING_MODULE_DIRS:
        p = DOCS_DIR / d
        if p.is_dir():
            paths.append(p)
    return paths


def collect_knowledge_entry_files() -> list[Path]:
    """收集所有知识条目文档（排除 index.md、interview.md、cheatsheet.md）"""
    entries = []
    for d in CODING_MODULE_DIRS:
        module_path = DOCS_DIR / d
        if module_path.is_dir():
            for f in sorted(module_path.glob("*.md")):
                if f.name not in REQUIRED_MODULE_FILES:
                    entries.append(f)
    return entries


def collect_interview_files() -> list[Path]:
    """收集所有 interview.md 文件"""
    files = []
    for d in CODING_MODULE_DIRS:
        p = DOCS_DIR / d / "interview.md"
        if p.exists():
            files.append(p)
    return files


def collect_python_files() -> list[Path]:
    """收集 code-examples/ 下所有 .py 文件（排除 __init__.py）"""
    files = []
    if CODE_EXAMPLES_DIR.is_dir():
        for f in sorted(CODE_EXAMPLES_DIR.rglob("*.py")):
            if f.name == "__init__.py" or "__pycache__" in str(f):
                continue
            files.append(f)
    return files


def collect_code_subdirs() -> list[Path]:
    """收集 code-examples/ 下所有包含 .py 文件的子目录"""
    dirs = set()
    if CODE_EXAMPLES_DIR.is_dir():
        for f in CODE_EXAMPLES_DIR.rglob("*.py"):
            if f.name == "__init__.py" or "__pycache__" in str(f):
                continue
            dirs.add(f.parent)
    return sorted(dirs)


def collect_external_service_files() -> list[Path]:
    """收集依赖外部服务的 .py 文件"""
    files = []
    for f in collect_python_files():
        try:
            content = f.read_text(encoding="utf-8")
            for service in EXTERNAL_SERVICE_IMPORTS:
                if f"import {service}" in content or f"from {service}" in content:
                    files.append(f)
                    break
        except Exception:
            continue
    return files


# ============================================================
# 预收集输入空间（避免 hypothesis 每次重新扫描文件系统）
# ============================================================

_CODING_MODULES = collect_coding_module_paths()
_KNOWLEDGE_ENTRIES = collect_knowledge_entry_files()
_INTERVIEW_FILES = collect_interview_files()
_PYTHON_FILES = collect_python_files()
_CODE_SUBDIRS = collect_code_subdirs()
_EXTERNAL_SERVICE_FILES = collect_external_service_files()


# ============================================================
# Property 1: 知识模块辅助文件完整性
# Feature: ai-knowledge-base, Property 1: 知识模块辅助文件完整性
# **Validates: Requirements 1.3, 2.8, 3.9, 4.9, 5.9, 6.8, 7.9, 8.12, 12.1**
# ============================================================

@pytest.mark.skipif(not _CODING_MODULES, reason="无编码模块目录")
@given(data=st.data())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_1_module_auxiliary_files(data):
    """
    Property 1: 知识模块辅助文件完整性

    For any 知识模块目录（编码模块 0-6），该目录下应同时存在
    index.md、interview.md 和 cheatsheet.md。

    **Validates: Requirements 1.3, 2.8, 3.9, 4.9, 5.9, 6.8, 7.9, 8.12, 12.1**
    """
    module_path = data.draw(st.sampled_from(_CODING_MODULES))

    for required_file in REQUIRED_MODULE_FILES:
        file_path = module_path / required_file
        assert file_path.exists(), (
            f"模块 {module_path.name} 缺少 {required_file}"
        )


# ============================================================
# Property 2: 知识条目文档格式完整性
# Feature: ai-knowledge-base, Property 2: 知识条目文档格式完整性
# **Validates: Requirements 1.4**
# ============================================================

@pytest.mark.skipif(not _KNOWLEDGE_ENTRIES, reason="无知识条目文件")
@given(data=st.data())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_2_entry_format_completeness(data):
    """
    Property 2: 知识条目文档格式完整性

    For any 知识条目文档，该文档应包含必需章节：
    概念说明、核心原理、代码示例、参考资料。

    **Validates: Requirements 1.4**
    """
    entry_file = data.draw(st.sampled_from(_KNOWLEDGE_ENTRIES))
    content = entry_file.read_text(encoding="utf-8")

    # 检查必需章节（通过标题匹配）
    required_sections = ["概念说明", "核心原理", "代码示例", "参考资料"]
    for section in required_sections:
        assert section in content, (
            f"文件 {entry_file.relative_to(DOCS_DIR)} 缺少「{section}」章节"
        )


# ============================================================
# Property 3: 编码模块工具引用完整性
# Feature: ai-knowledge-base, Property 3: 编码模块工具引用完整性
# **Validates: Requirements 1.10**
# ============================================================

@pytest.mark.skipif(not _KNOWLEDGE_ENTRIES, reason="无知识条目文件")
@given(data=st.data())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_3_tool_reference_completeness(data):
    """
    Property 3: 编码模块工具引用完整性

    For any 编码模块（0-6）下的知识条目文档，该文档应包含"推荐工具"区块，
    且区块中包含至少一个指向模块 7（/7-ai-tools/）的链接。

    **Validates: Requirements 1.10**
    """
    entry_file = data.draw(st.sampled_from(_KNOWLEDGE_ENTRIES))
    content = entry_file.read_text(encoding="utf-8")

    assert "推荐工具" in content, (
        f"文件 {entry_file.relative_to(DOCS_DIR)} 缺少「推荐工具」区块"
    )
    assert "/7-ai-tools/" in content or "7-ai-tools" in content, (
        f"文件 {entry_file.relative_to(DOCS_DIR)} 的推荐工具区块缺少模块 7 链接"
    )


# ============================================================
# Property 4: 代码示例可运行性
# Feature: ai-knowledge-base, Property 4: 代码示例可运行性
# **Validates: Requirements 2.4, 3.5, 4.6, 5.6, 6.6, 7.7**
# ============================================================

@pytest.mark.skipif(not _CODE_SUBDIRS, reason="无代码示例子目录")
@given(data=st.data())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_4_code_example_runnable(data):
    """
    Property 4: 代码示例可运行性

    For any 代码示例目录，该目录下应存在至少一个包含 `if __name__` 块的
    .py 文件或至少一个 .ipynb Notebook 文件。

    **Validates: Requirements 2.4, 3.5, 4.6, 5.6, 6.6, 7.7**
    """
    code_dir = data.draw(st.sampled_from(_CODE_SUBDIRS))

    # 检查是否有 .ipynb 文件
    notebooks = list(code_dir.glob("*.ipynb"))
    if notebooks:
        return  # 有 Notebook 即可

    # 检查是否有包含 if __name__ 的 .py 文件
    py_files = [f for f in code_dir.glob("*.py") if f.name != "__init__.py"]
    has_main = False
    for py_file in py_files:
        try:
            content = py_file.read_text(encoding="utf-8")
            if 'if __name__' in content:
                has_main = True
                break
        except Exception:
            continue

    assert has_main, (
        f"目录 {code_dir.relative_to(CODE_EXAMPLES_DIR)} 缺少包含 "
        f"'if __name__' 块的 .py 文件或 .ipynb Notebook"
    )


# ============================================================
# Property 5: 代码示例中文注释
# Feature: ai-knowledge-base, Property 5: 代码示例中文注释
# **Validates: Requirements 2.5, 14.5**
# ============================================================

# 匹配中文字符的正则
_CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]")


@pytest.mark.skipif(not _PYTHON_FILES, reason="无 Python 文件")
@given(data=st.data())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_5_chinese_comments(data):
    """
    Property 5: 代码示例中文注释

    For any .py 文件，该文件应包含至少一行中文注释。

    **Validates: Requirements 2.5, 14.5**
    """
    py_file = data.draw(st.sampled_from(_PYTHON_FILES))
    content = py_file.read_text(encoding="utf-8")

    assert _CHINESE_PATTERN.search(content), (
        f"文件 {py_file.relative_to(CODE_EXAMPLES_DIR)} 缺少中文注释"
    )


# ============================================================
# Property 6: 代码示例元数据完整性
# Feature: ai-knowledge-base, Property 6: 代码示例元数据完整性
# **Validates: Requirements 13.3**
# ============================================================

_PYTHON_VERSION_PATTERN = re.compile(r"Python\s*(?:版本[：:]?\s*)?3\.\d+", re.IGNORECASE)
_DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")


@pytest.mark.skipif(not _PYTHON_FILES, reason="无 Python 文件")
@given(data=st.data())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_6_code_metadata(data):
    """
    Property 6: 代码示例元数据完整性

    For any .py 文件，其文件头注释应包含 Python 版本要求和最后验证日期。

    **Validates: Requirements 13.3**
    """
    py_file = data.draw(st.sampled_from(_PYTHON_FILES))
    content = py_file.read_text(encoding="utf-8")

    # 检查文件头部分（前 500 字符或第一个 docstring）
    header = content[:500]

    assert _PYTHON_VERSION_PATTERN.search(header), (
        f"文件 {py_file.relative_to(CODE_EXAMPLES_DIR)} 头注释缺少 Python 版本要求"
    )
    assert _DATE_PATTERN.search(header), (
        f"文件 {py_file.relative_to(CODE_EXAMPLES_DIR)} 头注释缺少最后验证日期"
    )


# ============================================================
# Property 7: 面试题标注完整性
# Feature: ai-knowledge-base, Property 7: 面试题标注完整性
# **Validates: Requirements 12.4**
# ============================================================

@pytest.mark.skipif(not _INTERVIEW_FILES, reason="无 interview.md 文件")
@given(data=st.data())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_7_interview_annotations(data):
    """
    Property 7: 面试题标注完整性

    For any interview.md，其中的面试题条目应包含难度标注（⭐）、
    频率标注（🔥）和答题思路。

    **Validates: Requirements 12.4**
    """
    interview_file = data.draw(st.sampled_from(_INTERVIEW_FILES))
    content = interview_file.read_text(encoding="utf-8")

    # 查找面试题（以 ## Q 或 ### Q 开头的标题）
    question_pattern = re.compile(r"^#{2,3}\s+Q\d+", re.MULTILINE)
    questions = question_pattern.findall(content)

    if not questions:
        pytest.skip(f"{interview_file.name} 中未找到面试题")

    # 检查是否包含难度标注、频率标注和答题思路
    assert "⭐" in content, (
        f"{interview_file.relative_to(DOCS_DIR)} 缺少难度标注（⭐）"
    )
    assert "🔥" in content, (
        f"{interview_file.relative_to(DOCS_DIR)} 缺少频率标注（🔥）"
    )
    assert "答题思路" in content, (
        f"{interview_file.relative_to(DOCS_DIR)} 缺少答题思路"
    )


# ============================================================
# Property 8: 外部依赖示例规范性
# Feature: ai-knowledge-base, Property 8: 外部依赖示例规范性
# **Validates: Requirements 14.6, 14.7**
# ============================================================

@pytest.mark.skipif(not _EXTERNAL_SERVICE_FILES, reason="无依赖外部服务的文件")
@given(data=st.data())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_8_external_dependency_docs(data):
    """
    Property 8: 外部依赖示例规范性

    For any 依赖外部服务的 .py 文件，该文件头注释中应包含
    Docker 启动命令或服务配置说明，以及免费替代方案说明。

    **Validates: Requirements 14.6, 14.7**
    """
    py_file = data.draw(st.sampled_from(_EXTERNAL_SERVICE_FILES))
    content = py_file.read_text(encoding="utf-8")

    # 检查文件头部分
    header = content[:800]

    has_docker = "docker" in header.lower() or "启动命令" in header
    has_alternative = "免费替代" in header or "替代方案" in header or "内存模式" in header

    assert has_docker, (
        f"文件 {py_file.relative_to(CODE_EXAMPLES_DIR)} 头注释缺少 Docker 启动命令"
    )
    assert has_alternative, (
        f"文件 {py_file.relative_to(CODE_EXAMPLES_DIR)} 头注释缺少免费替代方案说明"
    )
