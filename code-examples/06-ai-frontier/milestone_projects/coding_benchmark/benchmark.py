"""
AI Coding 工具评测 — 里程碑项目

知识点：AI IDE 评测方法论、评测指标设计、自动化评测框架、
       多维度对比分析、评测报告生成

Python 版本：3.11+
依赖：标准库
最后验证：2024-12-01

项目说明：
  本项目实现 AI Coding 工具的自动化评测框架。
  通过标准化的编程任务，评估不同 AI IDE 的代码生成质量、
  速度、正确性和可维护性。

运行方式：
  python benchmark.py
"""

from __future__ import annotations

import json
import random
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


# ============================================================
# 1. 评测数据结构
# ============================================================

class TaskDifficulty(Enum):
    """任务难度"""
    EASY = "easy"           # 简单（单函数）
    MEDIUM = "medium"       # 中等（多函数/类）
    HARD = "hard"           # 困难（多文件/架构）


class TaskCategory(Enum):
    """任务类别"""
    ALGORITHM = "algorithm"         # 算法实现
    API_ENDPOINT = "api_endpoint"   # API 端点
    DATA_PROCESSING = "data_processing"  # 数据处理
    TESTING = "testing"             # 测试编写
    REFACTORING = "refactoring"     # 代码重构
    BUG_FIX = "bug_fix"            # Bug 修复


@dataclass
class BenchmarkTask:
    """评测任务定义"""
    id: str                         # 任务 ID
    name: str                       # 任务名称
    description: str                # 任务描述
    category: TaskCategory          # 任务类别
    difficulty: TaskDifficulty      # 难度
    prompt: str                     # 给 AI 的 Prompt
    expected_features: list[str]    # 期望的功能特征
    test_cases: list[dict]          # 测试用例
    max_time_seconds: int = 60      # 最大允许时间


@dataclass
class TaskResult:
    """单个任务的评测结果"""
    task_id: str
    ide_name: str
    generation_time_ms: float       # 代码生成时间
    code_length: int                # 生成代码行数
    correctness_score: float        # 正确性分数 (0-1)
    completeness_score: float       # 完整性分数 (0-1)
    code_quality_score: float       # 代码质量分数 (0-1)
    has_type_hints: bool            # 是否有类型注解
    has_docstrings: bool            # 是否有文档字符串
    has_error_handling: bool        # 是否有错误处理
    passed_tests: int               # 通过的测试数
    total_tests: int                # 总测试数


@dataclass
class IDEProfile:
    """AI IDE 配置"""
    name: str                       # IDE 名称
    version: str                    # 版本
    model: str                      # 使用的 AI 模型
    features: list[str]             # 特色功能
    price_monthly: float            # 月费（美元）


# ============================================================
# 2. 评测任务库
# ============================================================

class TaskLibrary:
    """评测任务库 — 管理标准化的编程任务"""

    def __init__(self):
        self._tasks: list[BenchmarkTask] = []
        self._load_default_tasks()

    def _load_default_tasks(self):
        """加载默认评测任务"""

        # 任务 1：算法实现 — 快速排序
        self._tasks.append(BenchmarkTask(
            id="algo-001",
            name="快速排序实现",
            description="实现快速排序算法，支持自定义比较函数",
            category=TaskCategory.ALGORITHM,
            difficulty=TaskDifficulty.EASY,
            prompt="实现一个快速排序函数，支持升序和降序排列，包含类型注解和文档字符串",
            expected_features=["类型注解", "文档字符串", "自定义排序", "边界处理"],
            test_cases=[
                {"input": [3, 1, 4, 1, 5], "expected": [1, 1, 3, 4, 5]},
                {"input": [], "expected": []},
                {"input": [1], "expected": [1]},
            ],
        ))

        # 任务 2：API 端点 — CRUD
        self._tasks.append(BenchmarkTask(
            id="api-001",
            name="FastAPI CRUD 端点",
            description="创建一个 FastAPI 的用户 CRUD API",
            category=TaskCategory.API_ENDPOINT,
            difficulty=TaskDifficulty.MEDIUM,
            prompt="用 FastAPI 创建用户管理 API，包含创建、读取、更新、删除端点，使用 Pydantic 模型",
            expected_features=["Pydantic 模型", "CRUD 端点", "错误处理", "状态码"],
            test_cases=[
                {"method": "POST", "path": "/users", "status": 201},
                {"method": "GET", "path": "/users/1", "status": 200},
                {"method": "PUT", "path": "/users/1", "status": 200},
                {"method": "DELETE", "path": "/users/1", "status": 204},
            ],
        ))

        # 任务 3：数据处理 — CSV 分析
        self._tasks.append(BenchmarkTask(
            id="data-001",
            name="CSV 数据分析",
            description="读取 CSV 文件并进行统计分析",
            category=TaskCategory.DATA_PROCESSING,
            difficulty=TaskDifficulty.EASY,
            prompt="编写一个 CSV 数据分析脚本，计算各列的统计指标（均值、中位数、标准差），支持缺失值处理",
            expected_features=["CSV 读取", "统计计算", "缺失值处理", "结果输出"],
            test_cases=[
                {"input": "test.csv", "expected_columns": ["mean", "median", "std"]},
            ],
        ))

        # 任务 4：测试编写
        self._tasks.append(BenchmarkTask(
            id="test-001",
            name="单元测试编写",
            description="为给定函数编写完整的单元测试",
            category=TaskCategory.TESTING,
            difficulty=TaskDifficulty.MEDIUM,
            prompt="为一个用户注册函数编写单元测试，覆盖正常流程、边界条件和异常情况",
            expected_features=["正常测试", "边界测试", "异常测试", "Mock 使用"],
            test_cases=[
                {"test_type": "normal", "expected": "pass"},
                {"test_type": "edge", "expected": "pass"},
                {"test_type": "error", "expected": "pass"},
            ],
        ))

        # 任务 5：代码重构
        self._tasks.append(BenchmarkTask(
            id="refactor-001",
            name="遗留代码重构",
            description="重构一段包含代码异味的遗留代码",
            category=TaskCategory.REFACTORING,
            difficulty=TaskDifficulty.HARD,
            prompt="重构以下代码：提取函数、消除重复、添加类型注解、改善命名",
            expected_features=["函数提取", "消除重复", "类型注解", "命名改善"],
            test_cases=[
                {"check": "no_duplicate_code", "expected": True},
                {"check": "has_type_hints", "expected": True},
            ],
        ))

        # 任务 6：Bug 修复
        self._tasks.append(BenchmarkTask(
            id="bug-001",
            name="并发 Bug 修复",
            description="修复一个多线程竞态条件 Bug",
            category=TaskCategory.BUG_FIX,
            difficulty=TaskDifficulty.HARD,
            prompt="修复以下代码中的竞态条件：多线程同时修改共享计数器导致结果不正确",
            expected_features=["锁机制", "线程安全", "正确性验证"],
            test_cases=[
                {"threads": 10, "expected_count": 1000},
            ],
        ))

    def get_tasks(self, difficulty: TaskDifficulty = None,
                  category: TaskCategory = None) -> list[BenchmarkTask]:
        """获取评测任务"""
        tasks = self._tasks
        if difficulty:
            tasks = [t for t in tasks if t.difficulty == difficulty]
        if category:
            tasks = [t for t in tasks if t.category == category]
        return tasks


# ============================================================
# 3. AI IDE 模拟器
# ============================================================

class IDESimulator:
    """
    AI IDE 模拟器

    模拟不同 AI IDE 的代码生成行为，用于评测框架演示。
    实际评测应使用真实 IDE 的 API 或手动测试。
    """

    def __init__(self, profile: IDEProfile):
        self.profile = profile
        # 不同 IDE 的模拟参数
        self._params = self._get_ide_params()

    def _get_ide_params(self) -> dict:
        """获取 IDE 模拟参数"""
        params = {
            "GitHub Copilot": {
                "speed_factor": 0.8,        # 生成速度因子
                "correctness_base": 0.85,   # 基础正确率
                "quality_base": 0.80,       # 基础质量分
                "type_hint_prob": 0.7,      # 类型注解概率
                "docstring_prob": 0.6,      # 文档字符串概率
                "error_handling_prob": 0.5, # 错误处理概率
            },
            "Cursor": {
                "speed_factor": 0.9,
                "correctness_base": 0.90,
                "quality_base": 0.85,
                "type_hint_prob": 0.8,
                "docstring_prob": 0.7,
                "error_handling_prob": 0.7,
            },
            "Kiro": {
                "speed_factor": 1.2,        # Spec 驱动稍慢但更准确
                "correctness_base": 0.92,
                "quality_base": 0.90,
                "type_hint_prob": 0.9,
                "docstring_prob": 0.85,
                "error_handling_prob": 0.8,
            },
            "Trae": {
                "speed_factor": 0.85,
                "correctness_base": 0.82,
                "quality_base": 0.78,
                "type_hint_prob": 0.65,
                "docstring_prob": 0.6,
                "error_handling_prob": 0.5,
            },
        }
        return params.get(self.profile.name, params["GitHub Copilot"])

    def generate_code(self, task: BenchmarkTask) -> TaskResult:
        """模拟代码生成"""
        random.seed(hash(f"{self.profile.name}{task.id}"))
        p = self._params

        # 难度影响因子
        difficulty_factor = {
            TaskDifficulty.EASY: 1.0,
            TaskDifficulty.MEDIUM: 0.9,
            TaskDifficulty.HARD: 0.75,
        }[task.difficulty]

        # 模拟生成时间
        base_time = random.uniform(500, 3000)
        gen_time = base_time * p["speed_factor"]

        # 模拟代码行数
        code_length = random.randint(20, 100) * (1 + ["easy", "medium", "hard"].index(task.difficulty.value))

        # 模拟各项分数
        correctness = min(1.0, p["correctness_base"] * difficulty_factor + random.uniform(-0.1, 0.1))
        completeness = min(1.0, correctness * random.uniform(0.85, 1.0))
        quality = min(1.0, p["quality_base"] * difficulty_factor + random.uniform(-0.1, 0.1))

        # 模拟特征
        has_types = random.random() < p["type_hint_prob"]
        has_docs = random.random() < p["docstring_prob"]
        has_errors = random.random() < p["error_handling_prob"]

        # 模拟测试通过率
        total_tests = len(task.test_cases)
        passed_tests = int(total_tests * correctness)

        return TaskResult(
            task_id=task.id,
            ide_name=self.profile.name,
            generation_time_ms=gen_time,
            code_length=code_length,
            correctness_score=round(correctness, 3),
            completeness_score=round(completeness, 3),
            code_quality_score=round(quality, 3),
            has_type_hints=has_types,
            has_docstrings=has_docs,
            has_error_handling=has_errors,
            passed_tests=passed_tests,
            total_tests=total_tests,
        )


# ============================================================
# 4. 评测引擎
# ============================================================

class BenchmarkEngine:
    """
    AI Coding 评测引擎

    功能：
    - 管理评测任务和 IDE 配置
    - 执行自动化评测
    - 计算综合评分
    - 生成对比报告
    """

    def __init__(self):
        self.task_library = TaskLibrary()
        self._ide_profiles: list[IDEProfile] = []
        self._results: list[TaskResult] = []
        self._load_default_ides()

    def _load_default_ides(self):
        """加载默认 IDE 配置"""
        self._ide_profiles = [
            IDEProfile("GitHub Copilot", "2025.1", "GPT-4o",
                       ["代码补全", "Chat", "Agent"], 10.0),
            IDEProfile("Cursor", "0.45", "Claude 3.5 Sonnet",
                       ["Composer", "Agent", "@codebase"], 20.0),
            IDEProfile("Kiro", "1.0", "Claude Sonnet",
                       ["Spec 驱动", "Steering", "Hooks", "MCP"], 0.0),
            IDEProfile("Trae", "1.0", "Claude/GPT/豆包",
                       ["Builder", "中文优化", "多模型"], 0.0),
        ]

    def run_benchmark(self) -> dict:
        """执行完整评测"""
        tasks = self.task_library.get_tasks()
        print(f"\n开始评测：{len(self._ide_profiles)} 个 IDE × {len(tasks)} 个任务\n")

        for ide_profile in self._ide_profiles:
            simulator = IDESimulator(ide_profile)
            print(f"  评测 {ide_profile.name}...")

            for task in tasks:
                result = simulator.generate_code(task)
                self._results.append(result)

        return self._generate_report()

    def _generate_report(self) -> dict:
        """生成评测报告"""
        # 按 IDE 分组统计
        ide_stats = {}
        for profile in self._ide_profiles:
            ide_results = [r for r in self._results if r.ide_name == profile.name]
            if not ide_results:
                continue

            ide_stats[profile.name] = {
                "model": profile.model,
                "price": f"${profile.price_monthly}/月",
                "avg_correctness": round(statistics.mean(
                    r.correctness_score for r in ide_results), 3),
                "avg_quality": round(statistics.mean(
                    r.code_quality_score for r in ide_results), 3),
                "avg_completeness": round(statistics.mean(
                    r.completeness_score for r in ide_results), 3),
                "avg_generation_time_ms": round(statistics.mean(
                    r.generation_time_ms for r in ide_results), 0),
                "type_hint_rate": round(sum(
                    1 for r in ide_results if r.has_type_hints) / len(ide_results), 2),
                "docstring_rate": round(sum(
                    1 for r in ide_results if r.has_docstrings) / len(ide_results), 2),
                "error_handling_rate": round(sum(
                    1 for r in ide_results if r.has_error_handling) / len(ide_results), 2),
                "test_pass_rate": round(sum(
                    r.passed_tests for r in ide_results) / sum(
                    r.total_tests for r in ide_results), 3) if sum(
                    r.total_tests for r in ide_results) > 0 else 0,
            }

            # 计算综合评分（加权平均）
            stats = ide_stats[profile.name]
            overall = (
                stats["avg_correctness"] * 0.35 +
                stats["avg_quality"] * 0.25 +
                stats["avg_completeness"] * 0.20 +
                stats["type_hint_rate"] * 0.10 +
                stats["error_handling_rate"] * 0.10
            )
            ide_stats[profile.name]["overall_score"] = round(overall, 3)

        # 排名
        ranking = sorted(ide_stats.items(),
                         key=lambda x: x[1]["overall_score"], reverse=True)

        return {
            "benchmark_date": datetime.now().isoformat(),
            "total_tasks": len(self.task_library.get_tasks()),
            "total_ides": len(self._ide_profiles),
            "ide_stats": ide_stats,
            "ranking": [{"rank": i + 1, "ide": name, "score": stats["overall_score"]}
                        for i, (name, stats) in enumerate(ranking)],
        }


# ============================================================
# 5. 运行演示
# ============================================================

def main():
    """运行 AI Coding 工具评测"""
    print("=" * 60)
    print("AI Coding 工具评测 — 里程碑项目")
    print("=" * 60)

    engine = BenchmarkEngine()
    report = engine.run_benchmark()

    # 显示排名
    print("\n" + "=" * 60)
    print("评测结果排名")
    print("=" * 60)
    print(f"\n{'排名':<6}{'IDE':<20}{'综合评分':<12}{'正确性':<10}{'质量':<10}{'价格'}")
    print("-" * 70)

    for item in report["ranking"]:
        stats = report["ide_stats"][item["ide"]]
        print(f"  {item['rank']:<4}{item['ide']:<20}"
              f"{item['score']:<12.3f}"
              f"{stats['avg_correctness']:<10.3f}"
              f"{stats['avg_quality']:<10.3f}"
              f"{stats['price']}")

    # 详细对比
    print("\n" + "=" * 60)
    print("详细对比")
    print("=" * 60)

    for ide_name, stats in report["ide_stats"].items():
        print(f"\n  {ide_name} ({stats['model']}):")
        print(f"    综合评分: {stats['overall_score']:.3f}")
        print(f"    正确性: {stats['avg_correctness']:.3f}")
        print(f"    代码质量: {stats['avg_quality']:.3f}")
        print(f"    完整性: {stats['avg_completeness']:.3f}")
        print(f"    类型注解率: {stats['type_hint_rate']:.0%}")
        print(f"    文档字符串率: {stats['docstring_rate']:.0%}")
        print(f"    错误处理率: {stats['error_handling_rate']:.0%}")
        print(f"    测试通过率: {stats['test_pass_rate']:.0%}")
        print(f"    平均生成时间: {stats['avg_generation_time_ms']:.0f}ms")

    print("\n" + "=" * 60)
    print("评测完成！")
    print("注意：以上数据为模拟结果，实际评测请使用真实 IDE 进行测试。")


if __name__ == "__main__":
    main()
