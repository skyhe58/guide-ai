"""
目录结构和文件存在性单元测试

验证 guide-ai 知识库的目录结构、关键文件存在性和内容完整性。
"""

from pathlib import Path

import pytest

from conftest import (
    CODE_EXAMPLES_DIR,
    CODE_MODULE_DIRS,
    CODING_MODULE_DIRS,
    DOCKER_DIR,
    DOCS_DIR,
    PROJECT_ROOT,
)


class TestTopLevelStructure:
    """验证顶层目录结构 — Requirements 1.1"""

    def test_docs_directory_exists(self):
        assert DOCS_DIR.is_dir(), "docs/ 目录不存在"

    def test_code_examples_directory_exists(self):
        assert CODE_EXAMPLES_DIR.is_dir(), "code-examples/ 目录不存在"

    def test_docker_directory_exists(self):
        assert DOCKER_DIR.is_dir(), "docker/ 目录不存在"

    def test_github_directory_exists(self):
        assert (PROJECT_ROOT / ".github").is_dir(), ".github/ 目录不存在"


class TestReadme:
    """验证 README.md 内容 — Requirements 1.2, 13.5"""

    @pytest.fixture(autouse=True)
    def _load_readme(self):
        self.readme_path = PROJECT_ROOT / "README.md"
        assert self.readme_path.exists(), "README.md 不存在"
        self.content = self.readme_path.read_text(encoding="utf-8")

    def test_readme_has_learning_path_diagram(self):
        assert "mermaid" in self.content.lower() or "```mermaid" in self.content, (
            "README.md 缺少学习路径图（Mermaid）"
        )

    def test_readme_has_module_navigation_table(self):
        assert "模块导航" in self.content or "模块 | 名称" in self.content, (
            "README.md 缺少模块导航表"
        )

    def test_readme_has_completion_tracking(self):
        assert "完成度" in self.content or "追踪" in self.content, (
            "README.md 缺少完成度追踪表"
        )


class TestVitePressConfig:
    """验证 VitePress 配置文件 — Requirements 1.5, 1.6"""

    def test_config_exists(self):
        config_path = DOCS_DIR / ".vitepress" / "config.mts"
        assert config_path.exists(), ".vitepress/config.mts 不存在"

    def test_sidebar_exists(self):
        sidebar_path = DOCS_DIR / ".vitepress" / "sidebar.mts"
        assert sidebar_path.exists(), ".vitepress/sidebar.mts 不存在"


class TestGitHubActions:
    """验证 GitHub Actions 配置 — Requirements 1.7"""

    def test_deploy_workflow_exists(self):
        workflow_path = PROJECT_ROOT / ".github" / "workflows" / "deploy.yml"
        assert workflow_path.exists(), ".github/workflows/deploy.yml 不存在"


class TestDockerCompose:
    """验证 Docker Compose 文件 — Requirements 15.1"""

    @pytest.mark.parametrize(
        "filename",
        [
            "docker-compose.yml",
            "docker-compose.ml.yml",
            "docker-compose.llm.yml",
            "docker-compose.monitor.yml",
        ],
    )
    def test_docker_compose_file_exists(self, filename: str):
        filepath = DOCKER_DIR / filename
        assert filepath.exists(), f"docker/{filename} 不存在"


class TestContributing:
    """验证贡献指南 — Requirements 13.2"""

    def test_contributing_exists(self):
        assert (PROJECT_ROOT / "CONTRIBUTING.md").exists(), "CONTRIBUTING.md 不存在"


class TestEntryTemplate:
    """验证知识条目模板 — Requirements 13.1"""

    def test_entry_template_exists(self):
        template_path = DOCS_DIR / "templates" / "entry-template.md"
        assert template_path.exists(), "docs/templates/entry-template.md 不存在"


class TestPyprojectToml:
    """验证 pyproject.toml — Requirements 13.4"""

    def test_pyproject_toml_exists(self):
        pyproject_path = CODE_EXAMPLES_DIR / "pyproject.toml"
        assert pyproject_path.exists(), "code-examples/pyproject.toml 不存在"


class TestCodeModuleStructure:
    """验证代码分组结构 — Requirements 14.1, 14.2"""

    @pytest.mark.parametrize("module_dir", CODE_MODULE_DIRS)
    def test_code_module_directory_exists(self, module_dir: str):
        path = CODE_EXAMPLES_DIR / module_dir
        assert path.is_dir(), f"code-examples/{module_dir}/ 目录不存在"


class TestInterviewDocs:
    """验证面试汇总文档 — Requirements 12.2, 12.3, 12.5, 12.6"""

    def test_interview_index_exists(self):
        """验证面试总入口索引页存在 — Requirements 12.2"""
        assert (DOCS_DIR / "interview" / "index.md").exists(), (
            "docs/interview/index.md 不存在"
        )

    def test_by_difficulty_exists(self):
        """验证按难度分类面试索引页存在 — Requirements 12.2"""
        assert (DOCS_DIR / "interview" / "by-difficulty.md").exists(), (
            "docs/interview/by-difficulty.md 不存在"
        )

    def test_by_position_exists(self):
        """验证按岗位分类面试索引页存在 — Requirements 12.3"""
        assert (DOCS_DIR / "interview" / "by-position.md").exists(), (
            "docs/interview/by-position.md 不存在"
        )

    def test_by_company_exists(self):
        """验证按公司分类面试索引页存在 — Requirements 12.6"""
        assert (DOCS_DIR / "interview" / "by-company.md").exists(), (
            "docs/interview/by-company.md 不存在"
        )

    def test_knowledge_map_exists(self):
        """验证面试知识图谱存在 — Requirements 12.5"""
        assert (DOCS_DIR / "interview" / "knowledge-map.md").exists(), (
            "docs/interview/knowledge-map.md 不存在"
        )

    def test_knowledge_map_has_mermaid(self):
        """验证面试知识图谱包含 Mermaid 图 — Requirements 12.5"""
        km_path = DOCS_DIR / "interview" / "knowledge-map.md"
        if km_path.exists():
            content = km_path.read_text(encoding="utf-8")
            assert "```mermaid" in content, (
                "knowledge-map.md 缺少 Mermaid 图"
            )
