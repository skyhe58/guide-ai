"""
Chain-of-Thought 思维链 — Zero-shot CoT、Few-shot CoT、Self-Consistency 模拟

知识点：Zero-shot CoT（"让我们一步一步思考"）、Few-shot CoT（提供推理示例）、
       Self-Consistency（多路径投票）、数学推理 CoT、逻辑推理 CoT、
       CoT 效果对比、与 Ollama API 集成

Python 版本：3.11+
依赖：标准库（默认模式）、ollama>=0.1（服务模式）
最后验证：2024-12-01

外部服务（可选）：
  Ollama 本地 LLM 推理服务
  启动命令：docker compose -f docker/docker-compose.yml up -d ollama
  模型下载：docker exec guide-ai-ollama ollama pull qwen2
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from typing import Any

# ============================================================
# 1. Zero-shot CoT — 只需一句"魔法咒语"
# ============================================================

class ZeroShotCoT:
    """Zero-shot Chain-of-Thought 提示构建器。

    核心思想：在问题后面加上"让我们一步一步思考"，
    引导 LLM 先输出推理过程再给出最终答案。
    适用于大模型（GPT-4/Claude），小模型效果有限。
    """

    # 常用的 CoT 触发短语（不同语言/风格）
    TRIGGERS = {
        "standard": "让我们一步一步思考：",
        "english": "Let's think step by step:",
        "detailed": "请按以下步骤详细分析，每一步都要写出计算过程：",
        "verify": "让我们一步一步思考，并在最后验证答案是否正确：",
    }

    def __init__(self, trigger_style: str = "standard"):
        if trigger_style not in self.TRIGGERS:
            raise ValueError(f"不支持的触发风格: {trigger_style}，可选: {list(self.TRIGGERS.keys())}")
        self.trigger = self.TRIGGERS[trigger_style]

    def build_prompt(self, question: str) -> str:
        """构建 Zero-shot CoT Prompt。"""
        return f"{question}\n\n{self.trigger}"

    def build_with_format(self, question: str, output_format: str = "number") -> str:
        """构建带输出格式要求的 Zero-shot CoT Prompt。

        Args:
            question: 用户问题
            output_format: 输出格式，"number" 表示数字答案，"choice" 表示选择题
        """
        format_instructions = {
            "number": "最终答案请用「答案：数字」的格式给出。",
            "choice": "最终答案请用「答案：A/B/C/D」的格式给出。",
            "yesno": "最终答案请用「答案：是/否」的格式给出。",
            "text": "最终答案请用「答案：你的回答」的格式给出。",
        }
        instruction = format_instructions.get(output_format, format_instructions["text"])
        return f"{question}\n\n{self.trigger}\n\n{instruction}"


# ============================================================
# 2. Few-shot CoT — 提供推理示例
# ============================================================

class FewShotCoT:
    """Few-shot Chain-of-Thought 提示构建器。

    核心思想：在 Prompt 中提供带推理过程的示例，
    让 LLM "学会"分步推理的模式。
    适用于所有模型，尤其是小模型（7B/8B）。
    """

    def __init__(self, task_instruction: str = ""):
        self.task_instruction = task_instruction
        self.examples: list[dict[str, str]] = []

    def add_example(self, question: str, reasoning: str, answer: str) -> FewShotCoT:
        """添加一个推理示例（链式调用）。"""
        self.examples.append({
            "question": question,
            "reasoning": reasoning,
            "answer": answer,
        })
        return self

    def build_prompt(self, question: str) -> str:
        """构建 Few-shot CoT Prompt。"""
        parts = []

        # 任务指令
        if self.task_instruction:
            parts.append(self.task_instruction)
            parts.append("")

        # 示例
        for i, ex in enumerate(self.examples, 1):
            parts.append(f"问题 {i}：{ex['question']}")
            parts.append(f"推理过程：\n{ex['reasoning']}")
            parts.append(f"最终答案：{ex['answer']}")
            parts.append("")

        # 用户问题
        parts.append(f"问题 {len(self.examples) + 1}：{question}")
        parts.append("推理过程：")

        return "\n".join(parts)


# ============================================================
# 3. 预定义 CoT 示例库 — 数学推理 & 逻辑推理
# ============================================================

def create_math_cot() -> FewShotCoT:
    """创建数学推理 Few-shot CoT 构建器。"""
    cot = FewShotCoT(
        task_instruction="请解决以下数学问题。先展示详细的推理过程，再给出最终答案。"
    )

    # 示例 1：百分比计算
    cot.add_example(
        question="一个书店有 120 本书，卖出了 40%，又进货 30 本，现在有多少本？",
        reasoning=(
            "1. 卖出的数量：120 × 40% = 48 本\n"
            "2. 卖出后剩余：120 - 48 = 72 本\n"
            "3. 进货后总数：72 + 30 = 102 本"
        ),
        answer="102 本",
    )

    # 示例 2：工程问题
    cot.add_example(
        question="一个水池有两个水管，A 管 3 小时注满，B 管 6 小时注满，同时开多久注满？",
        reasoning=(
            "1. A 管每小时注入：1/3 池\n"
            "2. B 管每小时注入：1/6 池\n"
            "3. 同时开每小时注入：1/3 + 1/6 = 2/6 + 1/6 = 3/6 = 1/2 池\n"
            "4. 注满时间：1 ÷ (1/2) = 2 小时"
        ),
        answer="2 小时",
    )

    # 示例 3：利润计算
    cot.add_example(
        question="一件商品进价 80 元，标价 120 元，打八折出售，利润率是多少？",
        reasoning=(
            "1. 售价：120 × 0.8 = 96 元\n"
            "2. 利润：96 - 80 = 16 元\n"
            "3. 利润率：16 / 80 × 100% = 20%"
        ),
        answer="20%",
    )

    return cot


def create_logic_cot() -> FewShotCoT:
    """创建逻辑推理 Few-shot CoT 构建器。"""
    cot = FewShotCoT(
        task_instruction="请解决以下逻辑推理问题。先展示推理过程，再给出最终答案。"
    )

    # 示例 1：排列推理
    cot.add_example(
        question="小明比小红高，小红比小刚高，小刚比小李高。谁最矮？",
        reasoning=(
            "1. 小明 > 小红（小明比小红高）\n"
            "2. 小红 > 小刚（小红比小刚高）\n"
            "3. 小刚 > 小李（小刚比小李高）\n"
            "4. 排列：小明 > 小红 > 小刚 > 小李\n"
            "5. 最矮的是排在最后的"
        ),
        answer="小李最矮",
    )

    # 示例 2：条件推理
    cot.add_example(
        question="如果下雨就带伞，如果带伞就不会淋湿。今天下雨了，小明会淋湿吗？",
        reasoning=(
            "1. 条件 1：下雨 → 带伞\n"
            "2. 条件 2：带伞 → 不淋湿\n"
            "3. 已知：今天下雨了\n"
            "4. 由条件 1：下雨 → 小明带伞\n"
            "5. 由条件 2：带伞 → 小明不会淋湿"
        ),
        answer="小明不会淋湿",
    )

    return cot


# ============================================================
# 4. Self-Consistency — 多路径投票
# ============================================================

class SelfConsistency:
    """Self-Consistency 自一致性推理。

    核心思想：对同一个问题用 CoT 生成多个推理路径，
    提取每个路径的最终答案，取多数投票的答案。
    通过 temperature > 0 引入随机性，获得不同推理路径。
    """

    def __init__(self, n_samples: int = 5, temperature: float = 0.7):
        """
        Args:
            n_samples: 采样次数（推理路径数量），通常 3-5 即可
            temperature: 采样温度，越高随机性越大
        """
        self.n_samples = n_samples
        self.temperature = temperature

    @staticmethod
    def extract_answer(response: str) -> str:
        """从 CoT 推理输出中提取最终答案。

        支持多种答案格式：
        - "最终答案：xxx"
        - "答案：xxx"
        - "答案是 xxx"
        - "所以答案是 xxx"
        """
        patterns = [
            r"最终答案[：:]\s*(.+?)(?:\n|$)",
            r"答案[：:]\s*(.+?)(?:\n|$)",
            r"答案是\s*(.+?)(?:\n|$)",
            r"所以[，,]?\s*(.+?)(?:\n|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1).strip()
        # 如果没有匹配到格式化答案，返回最后一行
        lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
        return lines[-1] if lines else "无法提取答案"

    @staticmethod
    def majority_vote(answers: list[str]) -> dict[str, Any]:
        """多数投票，返回最终答案和投票详情。"""
        counter = Counter(answers)
        most_common_answer, most_common_count = counter.most_common(1)[0]
        return {
            "final_answer": most_common_answer,
            "confidence": most_common_count / len(answers),
            "vote_count": most_common_count,
            "total_samples": len(answers),
            "vote_distribution": dict(counter),
        }

    def simulate_reasoning(self, question: str) -> dict[str, Any]:
        """模拟 Self-Consistency 推理过程（不调用真实 LLM）。

        通过模拟不同推理路径展示 Self-Consistency 的工作原理。
        """
        # 模拟多条推理路径（实际应用中由 LLM 生成）
        simulated_paths = self._generate_simulated_paths(question)

        answers = []
        paths_detail = []
        for i, path in enumerate(simulated_paths[:self.n_samples]):
            answer = self.extract_answer(path["response"])
            answers.append(answer)
            paths_detail.append({
                "path_id": i + 1,
                "reasoning": path["response"],
                "extracted_answer": answer,
            })

        vote_result = self.majority_vote(answers)
        return {
            "question": question,
            "paths": paths_detail,
            "vote_result": vote_result,
        }

    @staticmethod
    def _generate_simulated_paths(question: str) -> list[dict[str, str]]:
        """生成模拟的推理路径（用于演示）。"""
        # 针对经典数学问题的模拟路径
        if "鸡兔同笼" in question or ("35" in question and "94" in question):
            return [
                {"response": "设鸡 x 只，兔 y 只\nx + y = 35\n2x + 4y = 94\n解方程：x = 23, y = 12\n最终答案：鸡 23 只，兔 12 只"},
                {"response": "假设全是鸡：35×2=70 只脚\n差：94-70=24 只脚\n每只兔比鸡多 2 只脚\n兔：24÷2=12 只\n鸡：35-12=23 只\n最终答案：鸡 23 只，兔 12 只"},
                {"response": "假设全是兔：35×4=140 只脚\n多了：140-94=46 只脚\n每只鸡比兔少 2 只脚\n鸡：46÷2=23 只\n兔：35-23=12 只\n最终答案：鸡 23 只，兔 12 只"},
                {"response": "用代入法\n兔 = 35 - 鸡\n2×鸡 + 4×(35-鸡) = 94\n2鸡 + 140 - 4鸡 = 94\n-2鸡 = -46\n鸡 = 23, 兔 = 12\n最终答案：鸡 23 只，兔 12 只"},
                {"response": "列表尝试法\n鸡20兔15: 40+60=100 ✗\n鸡25兔10: 50+40=90 ✗\n鸡23兔12: 46+48=94 ✓\n最终答案：鸡 23 只，兔 12 只"},
            ]
        # 通用模拟路径
        return [
            {"response": f"分析问题：{question}\n步骤 1：理解题意\n步骤 2：列出条件\n步骤 3：求解\n最终答案：需要更多信息"},
        ] * 5


# ============================================================
# 5. CoT 效果对比演示
# ============================================================

def demo_zero_shot_cot() -> None:
    """演示 Zero-shot CoT 不同触发风格。"""
    print("\n" + "=" * 60)
    print("1. Zero-shot CoT — 不同触发风格")
    print("=" * 60)

    question = "一个班有 45 人，男生比女生多 5 人，男生和女生各有多少人？"

    for style in ZeroShotCoT.TRIGGERS:
        cot = ZeroShotCoT(trigger_style=style)
        prompt = cot.build_prompt(question)
        print(f"\n  📋 风格: {style}")
        print(f"  Prompt 长度: {len(prompt)} 字符")
        print(f"  触发语: {cot.trigger}")

    # 带格式要求的 Zero-shot CoT
    cot = ZeroShotCoT(trigger_style="verify")
    prompt = cot.build_with_format(question, output_format="number")
    print(f"\n  📋 带格式要求的 Zero-shot CoT:")
    print(f"  {prompt}")


def demo_few_shot_math_cot() -> None:
    """演示数学推理 Few-shot CoT。"""
    print("\n" + "=" * 60)
    print("2. Few-shot CoT — 数学推理")
    print("=" * 60)

    math_cot = create_math_cot()
    question = "小明有 200 元，花了 35% 买书，又花了剩余的 20% 买文具，还剩多少元？"
    prompt = math_cot.build_prompt(question)

    print(f"  示例数量: {len(math_cot.examples)}")
    print(f"  Prompt 总长度: {len(prompt)} 字符")
    print(f"\n  --- 生成的 Prompt ---")
    print(f"  {prompt[:300]}...")

    # 模拟 LLM 输出
    mock_response = (
        "1. 买书花费：200 × 35% = 70 元\n"
        "2. 买书后剩余：200 - 70 = 130 元\n"
        "3. 买文具花费：130 × 20% = 26 元\n"
        "4. 最终剩余：130 - 26 = 104 元\n"
        "最终答案：104 元"
    )
    print(f"\n  --- 模拟 LLM 输出 ---")
    print(f"  {mock_response}")

    # 提取答案
    answer = SelfConsistency.extract_answer(mock_response)
    print(f"\n  提取的答案: {answer}")


def demo_few_shot_logic_cot() -> None:
    """演示逻辑推理 Few-shot CoT。"""
    print("\n" + "=" * 60)
    print("3. Few-shot CoT — 逻辑推理")
    print("=" * 60)

    logic_cot = create_logic_cot()
    question = "所有的猫都是动物，有些动物会游泳。能否推出有些猫会游泳？"
    prompt = logic_cot.build_prompt(question)

    print(f"  示例数量: {len(logic_cot.examples)}")
    print(f"  Prompt 总长度: {len(prompt)} 字符")
    print(f"\n  --- 生成的 Prompt（前 300 字符）---")
    print(f"  {prompt[:300]}...")

    # 模拟 LLM 输出
    mock_response = (
        "1. 前提 1：所有的猫都是动物（猫 ⊂ 动物）\n"
        "2. 前提 2：有些动物会游泳（动物 ∩ 会游泳 ≠ ∅）\n"
        "3. 分析：会游泳的动物不一定是猫\n"
        "4. 例如：鱼是动物且会游泳，但鱼不是猫\n"
        "5. 结论：不能推出有些猫会游泳\n"
        "最终答案：不能推出。这是一个典型的三段论谬误，'有些动物会游泳'不能传递到猫这个子集。"
    )
    print(f"\n  --- 模拟 LLM 输出 ---")
    print(f"  {mock_response}")


def demo_self_consistency() -> None:
    """演示 Self-Consistency 多路径投票。"""
    print("\n" + "=" * 60)
    print("4. Self-Consistency — 多路径投票")
    print("=" * 60)

    sc = SelfConsistency(n_samples=5, temperature=0.7)
    question = "鸡兔同笼，共 35 个头，94 只脚，鸡和兔各有多少只？"

    result = sc.simulate_reasoning(question)

    print(f"  问题: {result['question']}")
    print(f"  采样数: {sc.n_samples}")
    print(f"  温度: {sc.temperature}")

    print(f"\n  --- 各推理路径 ---")
    for path in result["paths"]:
        reasoning_preview = path["reasoning"][:60].replace("\n", " | ")
        print(f"  路径 {path['path_id']}: {reasoning_preview}...")
        print(f"         答案: {path['extracted_answer']}")

    vote = result["vote_result"]
    print(f"\n  --- 投票结果 ---")
    print(f"  最终答案: {vote['final_answer']}")
    print(f"  置信度: {vote['confidence']:.0%} ({vote['vote_count']}/{vote['total_samples']})")
    print(f"  投票分布: {vote['vote_distribution']}")


def demo_cot_comparison() -> None:
    """对比不同 CoT 策略的 Prompt 长度和特点。"""
    print("\n" + "=" * 60)
    print("5. CoT 策略对比")
    print("=" * 60)

    question = "一个工厂第一天生产 100 件产品，之后每天比前一天多生产 10%，三天共生产多少件？"

    # Zero-shot CoT
    zs_cot = ZeroShotCoT()
    zs_prompt = zs_cot.build_prompt(question)

    # Few-shot CoT
    fs_cot = create_math_cot()
    fs_prompt = fs_cot.build_prompt(question)

    # 对比
    strategies = [
        ("直接提问（无 CoT）", question, "最快，但复杂问题容易出错"),
        ("Zero-shot CoT", zs_prompt, "零成本，大模型效果好"),
        ("Few-shot CoT", fs_prompt, "效果最好，但 Prompt 更长"),
        ("Self-Consistency", f"[{zs_prompt}] × 5 次采样", "最准确，但成本 = 5 倍"),
    ]

    print(f"\n  {'策略':<20} {'Prompt 长度':<15} {'特点'}")
    print(f"  {'-'*65}")
    for name, prompt, note in strategies:
        length = len(prompt) if not prompt.startswith("[") else f"~{len(zs_prompt)}×5"
        print(f"  {name:<20} {str(length):<15} {note}")


# ============================================================
# 6. 服务模式 — 调用 Ollama API
# ============================================================

async def demo_ollama_cot() -> None:
    """服务模式：用不同 CoT 策略调用 Ollama。"""
    print("\n" + "=" * 60)
    print("6. 服务模式 — 调用 Ollama API")
    print("=" * 60)

    try:
        import ollama
    except ImportError:
        print("  ⚠️ ollama 未安装: pip install ollama")
        return

    try:
        ollama.list()
    except Exception:
        print("  ❌ 无法连接 Ollama 服务")
        print("  💡 启动: docker compose -f docker/docker-compose.yml up -d ollama")
        return

    model = "qwen2"
    question = "小明有 200 元，花了 35% 买书，又花了剩余的 20% 买文具，还剩多少元？"

    # --- 对比：无 CoT vs Zero-shot CoT vs Few-shot CoT ---
    print(f"\n  问题: {question}")

    # 无 CoT
    print(f"\n  --- 无 CoT ---")
    response = ollama.generate(model=model, prompt=question, options={"num_predict": 200})
    print(f"  输出: {response['response'][:200]}...")

    # Zero-shot CoT
    print(f"\n  --- Zero-shot CoT ---")
    zs_cot = ZeroShotCoT(trigger_style="verify")
    zs_prompt = zs_cot.build_prompt(question)
    response = ollama.generate(model=model, prompt=zs_prompt, options={"num_predict": 300})
    print(f"  输出: {response['response'][:300]}...")

    # Few-shot CoT
    print(f"\n  --- Few-shot CoT ---")
    fs_cot = create_math_cot()
    fs_prompt = fs_cot.build_prompt(question)
    response = ollama.generate(model=model, prompt=fs_prompt, options={"num_predict": 300})
    print(f"  输出: {response['response'][:300]}...")

    # Self-Consistency（真实 LLM 调用）
    print(f"\n  --- Self-Consistency (3 次采样) ---")
    answers = []
    for i in range(3):
        response = ollama.generate(
            model=model,
            prompt=zs_prompt,
            options={"num_predict": 300, "temperature": 0.7},
        )
        answer = SelfConsistency.extract_answer(response["response"])
        answers.append(answer)
        print(f"  路径 {i+1} 答案: {answer}")

    vote = SelfConsistency.majority_vote(answers)
    print(f"  投票结果: {vote['final_answer']} (置信度: {vote['confidence']:.0%})")


# ============================================================
# 主入口
# ============================================================

def main(server_mode: bool = False) -> None:
    """运行所有 CoT 演示。"""
    print("🐍 Chain-of-Thought 思维链 — Zero-shot CoT、Few-shot CoT、Self-Consistency")
    print("=" * 60)

    demo_zero_shot_cot()
    demo_few_shot_math_cot()
    demo_few_shot_logic_cot()
    demo_self_consistency()
    demo_cot_comparison()

    if server_mode:
        import asyncio
        asyncio.run(demo_ollama_cot())

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. Zero-shot CoT：加一句'让我们一步一步思考'，零成本提升推理能力")
    print("  2. Few-shot CoT：提供推理示例，小模型也能做好复杂推理")
    print("  3. Self-Consistency：多路径投票，用成本换准确率")
    print("  4. 简单任务不需要 CoT，复杂推理任务才用")
    print("  5. 大模型用 Zero-shot，小模型用 Few-shot")

    if not server_mode:
        print("\n💡 要测试真实 LLM 调用: python 02_chain_of_thought.py server")


if __name__ == "__main__":
    is_server = len(sys.argv) > 1 and sys.argv[1].lower() == "server"
    main(server_mode=is_server)
