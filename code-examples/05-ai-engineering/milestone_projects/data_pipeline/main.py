"""
自动化数据管道 — 里程碑项目

功能：数据采集 → 清洗 → 标注 → 增强 → 训练数据集生成
模拟完整的 LLM 训练数据准备流水线。

Python 版本：3.11+
依赖：标准库（默认模式）
最后验证：2024-12-01

运行说明：
  python main.py
  管道将模拟从数据采集到训练数据集生成的完整流程。
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

# ============================================================
# 1. 数据管道配置
# ============================================================

@dataclass
class PipelineConfig:
    """数据管道配置"""
    source_name: str = "web_crawl"
    target_format: str = "instruction"  # instruction / qa / preference
    min_quality_score: float = 0.7
    dedup_threshold: float = 0.9
    augmentation_ratio: float = 1.5  # 增强后数据量 = 原始 × ratio
    max_samples: int = 1000
    output_path: str = "training_data.jsonl"


# ============================================================
# 2. 数据采集阶段
# ============================================================

class DataCollector:
    """数据采集器"""

    # 模拟数据源
    SAMPLE_TEXTS = [
        "Transformer 架构通过自注意力机制实现了并行计算，是现代 NLP 的基础。",
        "RAG 系统结合检索和生成，让 LLM 能基于外部知识回答问题。",
        "LoRA 通过低秩分解减少微调参数，只需训练 0.1% 的参数。",
        "vLLM 的 PagedAttention 技术显著提升了推理吞吐量。",
        "混合精度训练使用 FP16 加速计算，FP32 保证精度。",
        "DeepSpeed ZeRO 通过分片优化器状态减少显存占用。",
        "向量数据库存储文本的 embedding 向量，支持语义搜索。",
        "Prompt Engineering 是设计和优化 LLM 输入提示词的技术。",
        "联系我 zhangsan@test.com 电话 13812345678",  # 含 PII
        "点击链接领取免费优惠券！关注公众号！",  # 广告
        "短",  # 太短
        "Transformer 架构通过自注意力机制实现了并行计算，是现代 NLP 的基础。",  # 重复
        "Agent 系统让 LLM 能够使用工具、规划任务、自主完成复杂目标。",
        "KV Cache 缓存已计算的 Key/Value 矩阵，避免重复计算。",
        "数据标注是为原始数据添加标签的过程，是监督学习的基础。",
    ]

    def collect(self, count: int) -> list[dict]:
        """采集数据"""
        print(f"  📥 采集数据: {count} 条")
        data = []
        for i in range(count):
            text = random.choice(self.SAMPLE_TEXTS)
            data.append({
                "id": f"raw_{i:04d}",
                "text": text,
                "source": "web_crawl",
                "timestamp": datetime.now().isoformat(),
            })
        return data


# ============================================================
# 3. 数据清洗阶段
# ============================================================

class DataCleaner:
    """数据清洗器"""

    PII_PATTERNS = {
        "email": r"\b[\w.-]+@[\w.-]+\.\w{2,}\b",
        "phone": r"\b1[3-9]\d{9}\b",
    }
    SPAM_PATTERNS = [r"点击.*链接", r"关注.*公众号", r"免费.*领取"]

    def clean(self, data: list[dict]) -> tuple[list[dict], dict]:
        """清洗数据"""
        print(f"  🧹 清洗数据: {len(data)} 条")
        cleaned = []
        stats = {"dedup": 0, "too_short": 0, "spam": 0, "pii_cleaned": 0}
        seen_hashes = set()

        for item in data:
            text = item["text"].strip()

            # 去重
            text_hash = hashlib.md5(text.lower().encode()).hexdigest()
            if text_hash in seen_hashes:
                stats["dedup"] += 1
                continue
            seen_hashes.add(text_hash)

            # 长度过滤
            if len(text) < 20:
                stats["too_short"] += 1
                continue

            # 垃圾内容过滤
            is_spam = any(re.search(p, text) for p in self.SPAM_PATTERNS)
            if is_spam:
                stats["spam"] += 1
                continue

            # PII 脱敏
            for pii_type, pattern in self.PII_PATTERNS.items():
                if re.search(pattern, text):
                    text = re.sub(pattern, f"[{pii_type.upper()}]", text)
                    stats["pii_cleaned"] += 1

            item["text"] = text
            cleaned.append(item)

        print(f"     清洗后: {len(cleaned)} 条 (去重: {stats['dedup']}, "
              f"过短: {stats['too_short']}, 垃圾: {stats['spam']})")
        return cleaned, stats


# ============================================================
# 4. 数据标注阶段
# ============================================================

class DataLabeler:
    """数据标注器（LLM 辅助）"""

    CATEGORIES = ["概念解释", "技术对比", "实践指南", "代码示例", "其他"]

    def label(self, data: list[dict]) -> list[dict]:
        """LLM 辅助标注"""
        print(f"  🏷️ LLM 辅助标注: {len(data)} 条")
        labeled = []
        for item in data:
            text = item["text"]
            # 模拟 LLM 分类
            if any(kw in text for kw in ["是什么", "解释", "定义", "概念"]):
                category = "概念解释"
            elif any(kw in text for kw in ["对比", "区别", "vs", "比较"]):
                category = "技术对比"
            elif any(kw in text for kw in ["如何", "步骤", "方法", "实践"]):
                category = "实践指南"
            elif any(kw in text for kw in ["代码", "实现", "Python", "函数"]):
                category = "代码示例"
            else:
                category = random.choice(self.CATEGORIES)

            item["category"] = category
            item["quality_score"] = random.uniform(0.5, 1.0)
            labeled.append(item)

        # 统计分布
        dist = Counter(item["category"] for item in labeled)
        print(f"     类别分布: {dict(dist)}")
        return labeled


# ============================================================
# 5. 数据增强阶段
# ============================================================

class DataAugmenter:
    """数据增强器"""

    SYNONYMS = {
        "技术": ["方法", "方案", "手段"],
        "提升": ["提高", "增强", "改善"],
        "减少": ["降低", "缩减", "节省"],
        "显著": ["明显", "大幅", "极大"],
    }

    def augment(self, data: list[dict], ratio: float = 1.5) -> list[dict]:
        """数据增强"""
        target_count = int(len(data) * ratio)
        augment_count = target_count - len(data)
        print(f"  🔄 数据增强: {len(data)} → {target_count} 条 (增强 {augment_count} 条)")

        augmented = list(data)
        for i in range(augment_count):
            source = random.choice(data)
            new_text = self._synonym_replace(source["text"])
            new_item = {
                **source,
                "id": f"aug_{i:04d}",
                "text": new_text,
                "augmented": True,
                "source_id": source["id"],
            }
            augmented.append(new_item)

        return augmented

    def _synonym_replace(self, text: str) -> str:
        """同义词替换"""
        for word, synonyms in self.SYNONYMS.items():
            if word in text and random.random() > 0.5:
                text = text.replace(word, random.choice(synonyms), 1)
        return text


# ============================================================
# 6. 训练数据集生成
# ============================================================

class DatasetGenerator:
    """训练数据集生成器"""

    def generate_instruction_format(self, data: list[dict]) -> list[dict]:
        """生成指令微调格式"""
        print(f"  📦 生成训练数据集: {len(data)} 条")
        dataset = []
        for item in data:
            text = item["text"]
            category = item.get("category", "其他")

            # 根据类别生成指令
            if category == "概念解释":
                instruction = f"请解释以下技术概念"
            elif category == "技术对比":
                instruction = f"请对比分析以下技术"
            elif category == "实践指南":
                instruction = f"请提供以下技术的实践指南"
            else:
                instruction = f"请详细说明以下内容"

            dataset.append({
                "instruction": instruction,
                "input": text[:50] + "..." if len(text) > 50 else text,
                "output": text,
                "category": category,
                "quality_score": item.get("quality_score", 0.5),
            })

        return dataset

    def quality_filter(self, dataset: list[dict], min_score: float) -> list[dict]:
        """质量过滤"""
        filtered = [d for d in dataset if d.get("quality_score", 0) >= min_score]
        print(f"     质量过滤: {len(dataset)} → {len(filtered)} 条 (阈值: {min_score})")
        return filtered

    def split_dataset(self, dataset: list[dict], train_ratio: float = 0.9) -> dict:
        """划分训练集和验证集"""
        random.shuffle(dataset)
        split_idx = int(len(dataset) * train_ratio)
        return {
            "train": dataset[:split_idx],
            "validation": dataset[split_idx:],
        }


# ============================================================
# 7. 数据管道编排
# ============================================================

class DataPipeline:
    """自动化数据管道"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.collector = DataCollector()
        self.cleaner = DataCleaner()
        self.labeler = DataLabeler()
        self.augmenter = DataAugmenter()
        self.generator = DatasetGenerator()

    def run(self) -> dict:
        """执行完整管道"""
        print(f"\n{'='*60}")
        print(f"🔄 自动化数据管道启动")
        print(f"   目标格式: {self.config.target_format}")
        print(f"   最大样本数: {self.config.max_samples}")
        print(f"{'='*60}")

        # 阶段 1: 采集
        print("\n[1/5] 数据采集")
        raw_data = self.collector.collect(self.config.max_samples)

        # 阶段 2: 清洗
        print("\n[2/5] 数据清洗")
        cleaned_data, clean_stats = self.cleaner.clean(raw_data)

        # 阶段 3: 标注
        print("\n[3/5] 数据标注")
        labeled_data = self.labeler.label(cleaned_data)

        # 阶段 4: 增强
        print("\n[4/5] 数据增强")
        augmented_data = self.augmenter.augment(labeled_data, self.config.augmentation_ratio)

        # 阶段 5: 生成数据集
        print("\n[5/5] 生成训练数据集")
        dataset = self.generator.generate_instruction_format(augmented_data)
        dataset = self.generator.quality_filter(dataset, self.config.min_quality_score)
        splits = self.generator.split_dataset(dataset)

        # 报告
        report = {
            "raw_count": len(raw_data),
            "cleaned_count": len(cleaned_data),
            "labeled_count": len(labeled_data),
            "augmented_count": len(augmented_data),
            "final_count": len(dataset),
            "train_count": len(splits["train"]),
            "val_count": len(splits["validation"]),
            "cleaning_stats": clean_stats,
        }

        print(f"\n{'='*60}")
        print(f"📊 管道执行报告")
        print(f"   原始数据: {report['raw_count']} 条")
        print(f"   清洗后: {report['cleaned_count']} 条")
        print(f"   增强后: {report['augmented_count']} 条")
        print(f"   最终数据集: {report['final_count']} 条")
        print(f"   训练集: {report['train_count']} 条")
        print(f"   验证集: {report['val_count']} 条")
        print(f"{'='*60}")

        return report


# ============================================================
# 8. 主程序入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("自动化数据管道 — 里程碑项目")
    print("=" * 60)

    config = PipelineConfig(
        source_name="web_crawl",
        target_format="instruction",
        min_quality_score=0.6,
        augmentation_ratio=1.5,
        max_samples=50,
    )

    pipeline = DataPipeline(config)
    report = pipeline.run()

    # 展示样本
    print("\n--- 样本展示 ---")
    collector = DataCollector()
    raw = collector.collect(5)
    cleaner = DataCleaner()
    cleaned, _ = cleaner.clean(raw)
    labeler = DataLabeler()
    labeled = labeler.label(cleaned)
    gen = DatasetGenerator()
    samples = gen.generate_instruction_format(labeled[:3])
    for s in samples:
        print(f"\n  指令: {s['instruction']}")
        print(f"  输入: {s['input']}")
        print(f"  输出: {s['output'][:60]}...")
        print(f"  类别: {s['category']}, 质量: {s['quality_score']:.2f}")

    print("\n✅ 自动化数据管道里程碑项目完成！")
