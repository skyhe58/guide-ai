"""
微调数据准备 — CSV → Alpaca/ShareGPT 格式转换

知识点：指令微调数据格式、Alpaca 格式、ShareGPT 格式、
       数据清洗、质量控制、数据集构建

Python 版本：3.11+
依赖：无额外依赖（仅使用标准库）
最后验证：2024-12-01
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass, field
from typing import Literal

# ============================================================
# 1. 数据格式定义
# ============================================================

@dataclass
class AlpacaItem:
    """Alpaca 格式 — 最常用的指令微调格式。

    格式:
    {
        "instruction": "任务指令",
        "input": "可选的输入上下文",
        "output": "期望的输出"
    }

    适用场景: 单轮指令微调（SFT）
    """
    instruction: str
    input: str = ""
    output: str = ""


@dataclass
class ShareGPTMessage:
    """ShareGPT 消息格式。"""
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class ShareGPTItem:
    """ShareGPT 格式 — 多轮对话微调格式。

    格式:
    {
        "conversations": [
            {"role": "system", "content": "你是一个助手"},
            {"role": "user", "content": "用户问题"},
            {"role": "assistant", "content": "助手回答"},
            {"role": "user", "content": "追问"},
            {"role": "assistant", "content": "再次回答"}
        ]
    }

    适用场景: 多轮对话微调
    """
    conversations: list[ShareGPTMessage] = field(default_factory=list)


# ============================================================
# 2. CSV → Alpaca 格式转换
# ============================================================

def csv_to_alpaca(
    csv_content: str,
    instruction_col: str = "question",
    input_col: str | None = None,
    output_col: str = "answer",
) -> list[dict]:
    """将 CSV 数据转换为 Alpaca 格式。

    Args:
        csv_content: CSV 文件内容
        instruction_col: 指令列名
        input_col: 输入列名（可选）
        output_col: 输出列名

    Returns:
        Alpaca 格式数据列表
    """
    reader = csv.DictReader(io.StringIO(csv_content))
    items = []

    for row in reader:
        item = AlpacaItem(
            instruction=row.get(instruction_col, "").strip(),
            input=row.get(input_col, "").strip() if input_col else "",
            output=row.get(output_col, "").strip(),
        )
        # 数据质量检查
        if item.instruction and item.output:
            items.append(asdict(item))

    return items


# ============================================================
# 3. CSV → ShareGPT 格式转换
# ============================================================

def csv_to_sharegpt(
    csv_content: str,
    question_col: str = "question",
    answer_col: str = "answer",
    system_prompt: str = "你是一个专业的 AI 助手，请用中文回答问题。",
) -> list[dict]:
    """将 CSV 数据转换为 ShareGPT 多轮对话格式。

    Args:
        csv_content: CSV 文件内容
        question_col: 问题列名
        answer_col: 回答列名
        system_prompt: 系统提示词

    Returns:
        ShareGPT 格式数据列表
    """
    reader = csv.DictReader(io.StringIO(csv_content))
    items = []

    for row in reader:
        question = row.get(question_col, "").strip()
        answer = row.get(answer_col, "").strip()

        if question and answer:
            item = {
                "conversations": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            }
            items.append(item)

    return items


# ============================================================
# 4. 数据清洗与质量控制
# ============================================================

@dataclass
class QualityReport:
    """数据质量报告。"""
    total: int = 0
    valid: int = 0
    too_short: int = 0
    too_long: int = 0
    duplicates: int = 0
    empty_fields: int = 0


def clean_dataset(
    data: list[dict],
    format_type: Literal["alpaca", "sharegpt"] = "alpaca",
    min_length: int = 10,
    max_length: int = 4096,
    remove_duplicates: bool = True,
) -> tuple[list[dict], QualityReport]:
    """清洗微调数据集。

    Args:
        data: 原始数据列表
        format_type: 数据格式类型
        min_length: 最小文本长度
        max_length: 最大文本长度
        remove_duplicates: 是否去重

    Returns:
        (清洗后数据, 质量报告)
    """
    report = QualityReport(total=len(data))
    cleaned = []
    seen = set()

    for item in data:
        # 提取文本内容
        if format_type == "alpaca":
            text = f"{item.get('instruction', '')} {item.get('output', '')}"
        else:
            convs = item.get("conversations", [])
            text = " ".join(m.get("content", "") for m in convs)

        text = text.strip()

        # 空字段检查
        if not text:
            report.empty_fields += 1
            continue

        # 长度检查
        if len(text) < min_length:
            report.too_short += 1
            continue
        if len(text) > max_length:
            report.too_long += 1
            continue

        # 去重
        if remove_duplicates:
            text_hash = hash(text)
            if text_hash in seen:
                report.duplicates += 1
                continue
            seen.add(text_hash)

        cleaned.append(item)
        report.valid += 1

    return cleaned, report


# ============================================================
# 5. 演示
# ============================================================

def demo_csv_to_alpaca() -> None:
    """演示 CSV → Alpaca 格式转换。"""
    print("\n" + "=" * 60)
    print("1. CSV → Alpaca 格式转换")
    print("=" * 60)

    # 模拟 CSV 数据
    csv_data = """question,answer
什么是机器学习？,机器学习是人工智能的一个分支，通过数据和算法让计算机自动学习和改进。
解释什么是过拟合,过拟合是模型在训练数据上表现很好但在新数据上表现差的现象，通常因为模型过于复杂。
什么是梯度下降？,梯度下降是一种优化算法，通过计算损失函数的梯度来迭代更新模型参数，使损失最小化。"""

    alpaca_data = csv_to_alpaca(csv_data)
    print(f"  转换了 {len(alpaca_data)} 条数据")
    print(f"  示例:\n{json.dumps(alpaca_data[0], ensure_ascii=False, indent=4)}")


def demo_csv_to_sharegpt() -> None:
    """演示 CSV → ShareGPT 格式转换。"""
    print("\n" + "=" * 60)
    print("2. CSV → ShareGPT 格式转换")
    print("=" * 60)

    csv_data = """question,answer
什么是 Transformer？,Transformer 是一种基于注意力机制的神经网络架构，由 Google 在 2017 年提出。
LoRA 微调的原理是什么？,LoRA 通过低秩分解来近似权重更新，只训练两个小矩阵 A 和 B，大幅减少可训练参数。"""

    sharegpt_data = csv_to_sharegpt(csv_data)
    print(f"  转换了 {len(sharegpt_data)} 条数据")
    print(f"  示例:\n{json.dumps(sharegpt_data[0], ensure_ascii=False, indent=4)}")


def demo_data_cleaning() -> None:
    """演示数据清洗。"""
    print("\n" + "=" * 60)
    print("3. 数据清洗与质量控制")
    print("=" * 60)

    # 模拟含噪声的数据
    raw_data = [
        {"instruction": "什么是 AI？", "output": "人工智能是模拟人类智能的技术。"},
        {"instruction": "短", "output": "短"},  # 太短
        {"instruction": "", "output": ""},  # 空字段
        {"instruction": "什么是 AI？", "output": "人工智能是模拟人类智能的技术。"},  # 重复
        {"instruction": "解释深度学习", "output": "深度学习是机器学习的子集，使用多层神经网络。"},
    ]

    cleaned, report = clean_dataset(raw_data, format_type="alpaca", min_length=10)

    print(f"  原始数据: {report.total} 条")
    print(f"  有效数据: {report.valid} 条")
    print(f"  太短: {report.too_short} 条")
    print(f"  空字段: {report.empty_fields} 条")
    print(f"  重复: {report.duplicates} 条")

    print("\n  💡 数据质量建议:")
    print("    - 指令多样性: 避免重复相似的指令")
    print("    - 回答质量: 确保回答准确、完整")
    print("    - 数据量: 高质量 1000 条 > 低质量 10000 条")
    print("    - 格式一致: 统一使用 Alpaca 或 ShareGPT 格式")


def main() -> None:
    """运行所有数据准备演示。"""
    print("📊 微调数据准备 — CSV → Alpaca/ShareGPT 格式转换")
    print("=" * 60)

    demo_csv_to_alpaca()
    demo_csv_to_sharegpt()
    demo_data_cleaning()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. Alpaca 格式: instruction/input/output，适合单轮指令微调")
    print("  2. ShareGPT 格式: 多轮对话，适合对话模型微调")
    print("  3. 数据质量 > 数据数量")
    print("  4. 清洗步骤: 去空 → 去短 → 去长 → 去重")


if __name__ == "__main__":
    main()
