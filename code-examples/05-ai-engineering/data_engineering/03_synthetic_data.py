"""
合成数据生成模拟

知识点：Self-Instruct、Evol-Instruct、领域 QA 生成、
       质量评分、多样性分析、数据去重、合成数据管道

Python 版本：3.11+
依赖：标准库（默认模式）
最后验证：2024-12-01
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

# ============================================================
# 1. 合成数据结构
# ============================================================

class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class SyntheticSample:
    """合成数据样本"""
    sample_id: str
    instruction: str
    input_text: str = ""
    output: str = ""
    difficulty: Difficulty = Difficulty.MEDIUM
    source_method: str = ""  # self_instruct / evol_instruct / domain_qa
    quality_score: float = 0.0
    tags: list[str] = field(default_factory=list)


# ============================================================
# 2. Self-Instruct 模拟
# ============================================================

class SelfInstructGenerator:
    """Self-Instruct 合成数据生成器"""

    # 种子指令池
    SEED_INSTRUCTIONS = [
        "解释什么是机器学习",
        "用 Python 实现快速排序",
        "总结以下文章的核心观点",
        "将以下英文翻译成中文",
        "分析这段代码的时间复杂度",
        "设计一个 RESTful API",
        "解释 Transformer 的注意力机制",
        "比较 SQL 和 NoSQL 数据库",
    ]

    # 指令模板
    TEMPLATES = [
        "请{action}{topic}",
        "用简单的语言解释{topic}",
        "列出{topic}的{num}个{aspect}",
        "比较{topic_a}和{topic_b}的区别",
        "设计一个{topic}的解决方案",
        "分析{topic}的优缺点",
    ]

    TOPICS = [
        "RAG 系统", "向量数据库", "LoRA 微调", "Prompt Engineering",
        "LLM 推理优化", "数据清洗", "模型评估", "分布式训练",
        "API 网关设计", "缓存策略", "GPU 显存优化", "混合精度训练",
    ]

    ACTIONS = ["解释", "描述", "分析", "设计", "实现", "优化", "对比"]
    ASPECTS = ["优势", "步骤", "注意事项", "最佳实践", "常见问题"]

    def __init__(self):
        self.instruction_pool = list(self.SEED_INSTRUCTIONS)
        self.generated: list[SyntheticSample] = []
        self.seen_hashes: set[str] = set()

    def generate_instruction(self) -> str:
        """生成新指令"""
        template = random.choice(self.TEMPLATES)
        instruction = template.format(
            action=random.choice(self.ACTIONS),
            topic=random.choice(self.TOPICS),
            topic_a=random.choice(self.TOPICS),
            topic_b=random.choice(self.TOPICS),
            num=random.randint(3, 7),
            aspect=random.choice(self.ASPECTS),
        )
        return instruction

    def generate_response(self, instruction: str) -> str:
        """模拟生成回答"""
        # 实际应调用 LLM 生成
        responses = [
            f"关于「{instruction[:20]}」，以下是详细说明：\n\n"
            f"1. 核心概念：这是一个重要的技术领域...\n"
            f"2. 实现方法：可以通过以下步骤实现...\n"
            f"3. 注意事项：在实践中需要注意...\n"
            f"4. 最佳实践：建议采用以下方案...",
        ]
        return random.choice(responses)

    def _is_duplicate(self, instruction: str) -> bool:
        """检查是否重复"""
        h = hashlib.md5(instruction.strip().lower().encode()).hexdigest()
        if h in self.seen_hashes:
            return True
        self.seen_hashes.add(h)
        return False

    def generate_batch(self, count: int) -> list[SyntheticSample]:
        """批量生成"""
        print(f"[Self-Instruct] 开始生成 {count} 条数据")
        samples = []
        attempts = 0
        max_attempts = count * 3

        while len(samples) < count and attempts < max_attempts:
            attempts += 1
            instruction = self.generate_instruction()

            if self._is_duplicate(instruction):
                continue

            response = self.generate_response(instruction)
            sample = SyntheticSample(
                sample_id=f"si_{len(self.generated) + len(samples)}",
                instruction=instruction,
                output=response,
                difficulty=random.choice(list(Difficulty)),
                source_method="self_instruct",
                quality_score=random.uniform(0.6, 0.95),
                tags=["self-instruct"],
            )
            samples.append(sample)
            self.instruction_pool.append(instruction)

        self.generated.extend(samples)
        print(f"[Self-Instruct] 生成完成: {len(samples)} 条 (尝试: {attempts})")
        return samples


# ============================================================
# 3. Evol-Instruct 模拟
# ============================================================

class EvolInstructGenerator:
    """Evol-Instruct 指令进化生成器"""

    DEPTH_EVOLUTIONS = [
        "添加约束: {instruction}，要求{constraint}",
        "增加推理: {instruction}，并分析{analysis}",
        "具体化: {instruction}，以{context}为例",
        "增加复杂度: {instruction}，同时考虑{factor}",
    ]

    BREADTH_EVOLUTIONS = [
        "改写: 用不同的方式表达「{instruction}」",
        "换领域: 将「{instruction}」应用到{domain}领域",
    ]

    CONSTRAINTS = ["时间复杂度 O(nlogn)", "不使用第三方库", "支持并发", "内存限制 1GB"]
    ANALYSES = ["性能瓶颈", "安全风险", "可扩展性", "成本效益"]
    CONTEXTS = ["电商推荐系统", "医疗问答", "金融风控", "智能客服"]
    FACTORS = ["高并发场景", "数据隐私", "多语言支持", "离线部署"]
    DOMAINS = ["医疗", "金融", "教育", "制造业", "物流"]

    def __init__(self):
        self.generated: list[SyntheticSample] = []

    def evolve_depth(self, instruction: str) -> str:
        """深度进化"""
        template = random.choice(self.DEPTH_EVOLUTIONS)
        return template.format(
            instruction=instruction,
            constraint=random.choice(self.CONSTRAINTS),
            analysis=random.choice(self.ANALYSES),
            context=random.choice(self.CONTEXTS),
            factor=random.choice(self.FACTORS),
        )

    def evolve_breadth(self, instruction: str) -> str:
        """广度进化"""
        template = random.choice(self.BREADTH_EVOLUTIONS)
        return template.format(
            instruction=instruction,
            domain=random.choice(self.DOMAINS),
        )

    def evolve(self, seed_instructions: list[str], rounds: int = 2) -> list[SyntheticSample]:
        """多轮进化"""
        print(f"[Evol-Instruct] 开始进化: {len(seed_instructions)} 条种子, {rounds} 轮")
        current = list(seed_instructions)
        all_samples = []

        for round_idx in range(rounds):
            evolved = []
            for instruction in current:
                # 50% 深度进化，50% 广度进化
                if random.random() < 0.5:
                    new_instruction = self.evolve_depth(instruction)
                    evolution_type = "depth"
                else:
                    new_instruction = self.evolve_breadth(instruction)
                    evolution_type = "breadth"

                sample = SyntheticSample(
                    sample_id=f"ei_{len(all_samples)}",
                    instruction=new_instruction,
                    output=f"[模拟回答] {new_instruction[:50]}...",
                    difficulty=Difficulty.HARD if round_idx > 0 else Difficulty.MEDIUM,
                    source_method=f"evol_instruct_{evolution_type}",
                    quality_score=random.uniform(0.7, 0.98),
                    tags=["evol-instruct", evolution_type, f"round-{round_idx}"],
                )
                evolved.append(new_instruction)
                all_samples.append(sample)

            current = evolved
            print(f"  轮次 {round_idx + 1}: 生成 {len(evolved)} 条")

        self.generated.extend(all_samples)
        return all_samples


# ============================================================
# 4. 质量评估
# ============================================================

class QualityEvaluator:
    """合成数据质量评估器"""

    def evaluate_sample(self, sample: SyntheticSample) -> dict[str, float]:
        """评估单个样本"""
        scores = {
            "instruction_clarity": self._score_clarity(sample.instruction),
            "response_completeness": self._score_completeness(sample.output),
            "diversity": random.uniform(0.5, 1.0),
        }
        scores["overall"] = sum(scores.values()) / len(scores)
        return scores

    def _score_clarity(self, text: str) -> float:
        """评估指令清晰度"""
        score = 0.5
        if len(text) > 20:
            score += 0.2
        if "?" in text or "请" in text or "如何" in text:
            score += 0.15
        if len(text) < 200:
            score += 0.15
        return min(1.0, score)

    def _score_completeness(self, text: str) -> float:
        """评估回答完整度"""
        score = 0.3
        if len(text) > 100:
            score += 0.3
        if any(marker in text for marker in ["1.", "2.", "首先", "其次"]):
            score += 0.2
        if len(text) > 200:
            score += 0.2
        return min(1.0, score)

    def evaluate_dataset(self, samples: list[SyntheticSample]) -> dict:
        """评估整个数据集"""
        if not samples:
            return {"count": 0}

        all_scores = [self.evaluate_sample(s) for s in samples]
        avg_overall = sum(s["overall"] for s in all_scores) / len(all_scores)

        # 多样性分析
        instructions = [s.instruction for s in samples]
        unique_words = set()
        for inst in instructions:
            unique_words.update(inst.split())

        difficulty_dist = Counter(s.difficulty.value for s in samples)
        method_dist = Counter(s.source_method for s in samples)

        return {
            "count": len(samples),
            "avg_quality": round(avg_overall, 3),
            "unique_words": len(unique_words),
            "difficulty_distribution": dict(difficulty_dist),
            "method_distribution": dict(method_dist),
        }


# ============================================================
# 5. 主程序入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("合成数据生成模拟演示")
    print("=" * 60)

    # --- Self-Instruct ---
    print("\n--- Self-Instruct ---")
    si_gen = SelfInstructGenerator()
    si_samples = si_gen.generate_batch(10)
    for s in si_samples[:3]:
        print(f"  [{s.difficulty.value}] {s.instruction[:60]}...")

    # --- Evol-Instruct ---
    print("\n--- Evol-Instruct ---")
    ei_gen = EvolInstructGenerator()
    seeds = ["解释什么是 RAG", "用 Python 实现排序", "设计一个 API 网关"]
    ei_samples = ei_gen.evolve(seeds, rounds=2)
    for s in ei_samples[:3]:
        print(f"  [{s.source_method}] {s.instruction[:60]}...")

    # --- 质量评估 ---
    print("\n--- 质量评估 ---")
    evaluator = QualityEvaluator()
    all_samples = si_samples + ei_samples
    report = evaluator.evaluate_dataset(all_samples)
    print(f"  总样本数: {report['count']}")
    print(f"  平均质量: {report['avg_quality']}")
    print(f"  唯一词数: {report['unique_words']}")
    print(f"  难度分布: {report['difficulty_distribution']}")
    print(f"  方法分布: {report['method_distribution']}")

    print("\n✅ 合成数据生成模拟演示完成！")
