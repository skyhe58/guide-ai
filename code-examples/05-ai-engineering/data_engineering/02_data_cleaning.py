"""
数据清洗模拟

知识点：文本去重（Hash/SimHash）、质量过滤、PII 脱敏、
       格式标准化、清洗流水线、质量报告生成

Python 版本：3.11+
依赖：标准库（默认模式）
最后验证：2024-12-01
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum

# ============================================================
# 1. 清洗步骤定义
# ============================================================

class CleaningStep(Enum):
    FORMAT = "format_standardize"
    DEDUP = "deduplication"
    QUALITY = "quality_filter"
    PII = "pii_anonymize"
    CONTENT = "content_filter"


@dataclass
class CleaningResult:
    """清洗结果"""
    original_count: int
    final_count: int
    removed_by_step: dict[str, int] = field(default_factory=dict)
    pii_found: dict[str, int] = field(default_factory=dict)

    @property
    def retention_rate(self) -> float:
        return self.final_count / max(self.original_count, 1)


# ============================================================
# 2. 格式标准化
# ============================================================

class FormatStandardizer:
    """格式标准化器"""

    def standardize(self, text: str) -> str:
        """标准化文本格式"""
        # Unicode 标准化（NFKC）
        text = unicodedata.normalize("NFKC", text)
        # 统一换行符
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # 去除多余空白
        text = re.sub(r"[ \t]+", " ", text)
        # 去除多余换行
        text = re.sub(r"\n{3,}", "\n\n", text)
        # 去除首尾空白
        text = text.strip()
        # 全角转半角（数字和字母）
        text = self._fullwidth_to_halfwidth(text)
        return text

    def _fullwidth_to_halfwidth(self, text: str) -> str:
        """全角转半角"""
        result = []
        for char in text:
            code = ord(char)
            if 0xFF01 <= code <= 0xFF5E:
                result.append(chr(code - 0xFEE0))
            elif code == 0x3000:
                result.append(" ")
            else:
                result.append(char)
        return "".join(result)


# ============================================================
# 3. 文本去重
# ============================================================

class TextDeduplicator:
    """文本去重器"""

    def __init__(self):
        self.seen_hashes: set[str] = set()
        self.seen_simhashes: dict[int, list[str]] = {}

    def exact_dedup(self, text: str) -> bool:
        """精确去重（MD5 Hash）"""
        normalized = text.strip().lower()
        text_hash = hashlib.md5(normalized.encode("utf-8")).hexdigest()
        if text_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(text_hash)
        return False

    def simhash(self, text: str, hash_bits: int = 64) -> int:
        """计算 SimHash"""
        tokens = text.split()
        v = [0] * hash_bits
        for token in tokens:
            token_hash = int(hashlib.md5(token.encode()).hexdigest(), 16)
            for i in range(hash_bits):
                if token_hash & (1 << i):
                    v[i] += 1
                else:
                    v[i] -= 1
        fingerprint = 0
        for i in range(hash_bits):
            if v[i] > 0:
                fingerprint |= (1 << i)
        return fingerprint

    def hamming_distance(self, hash1: int, hash2: int) -> int:
        """计算汉明距离"""
        xor = hash1 ^ hash2
        distance = 0
        while xor:
            distance += xor & 1
            xor >>= 1
        return distance

    def fuzzy_dedup(self, text: str, doc_id: str, threshold: int = 3) -> bool:
        """模糊去重（SimHash + 汉明距离）"""
        sh = self.simhash(text)
        for existing_hash, ids in self.seen_simhashes.items():
            if self.hamming_distance(sh, existing_hash) <= threshold:
                return True
        if sh not in self.seen_simhashes:
            self.seen_simhashes[sh] = []
        self.seen_simhashes[sh].append(doc_id)
        return False


# ============================================================
# 4. 质量过滤
# ============================================================

@dataclass
class QualityConfig:
    """质量过滤配置"""
    min_length: int = 50
    max_length: int = 100000
    min_word_count: int = 10
    max_repeat_ratio: float = 0.3
    min_unique_ratio: float = 0.3
    max_special_char_ratio: float = 0.3
    blocked_patterns: list[str] = field(default_factory=lambda: [
        r"点击.*链接", r"关注.*公众号", r"扫码.*二维码",
        r"免费.*领取", r"限时.*优惠",
    ])


class QualityFilter:
    """质量过滤器"""

    def __init__(self, config: QualityConfig | None = None):
        self.config = config or QualityConfig()

    def filter(self, text: str) -> tuple[bool, str]:
        """过滤低质量文本"""
        # 长度检查
        if len(text) < self.config.min_length:
            return False, f"文本过短 ({len(text)} < {self.config.min_length})"
        if len(text) > self.config.max_length:
            return False, f"文本过长 ({len(text)} > {self.config.max_length})"

        # 词数检查
        words = text.split()
        if len(words) < self.config.min_word_count:
            return False, f"词数过少 ({len(words)})"

        # 重复率检查
        unique_words = set(words)
        unique_ratio = len(unique_words) / len(words)
        if unique_ratio < self.config.min_unique_ratio:
            return False, f"重复率过高 (唯一词比例: {unique_ratio:.2%})"

        # 特殊字符比例
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_ratio = special_chars / max(len(text), 1)
        if special_ratio > self.config.max_special_char_ratio:
            return False, f"特殊字符过多 ({special_ratio:.2%})"

        # 广告/垃圾内容检测
        for pattern in self.config.blocked_patterns:
            if re.search(pattern, text):
                return False, f"匹配屏蔽模式: {pattern}"

        return True, "通过"


# ============================================================
# 5. PII 脱敏
# ============================================================

class PIIAnonymizer:
    """个人敏感信息脱敏器"""

    PATTERNS = {
        "email": (r"\b[\w.-]+@[\w.-]+\.\w{2,}\b", "[EMAIL]"),
        "phone_cn": (r"\b1[3-9]\d{9}\b", "[PHONE]"),
        "id_card": (r"\b\d{17}[\dXx]\b", "[ID_CARD]"),
        "ip_address": (r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "[IP]"),
        "bank_card": (r"\b\d{16,19}\b", "[BANK_CARD]"),
        "api_key": (r"\b(?:sk-|ak-|key-)[a-zA-Z0-9]{20,}\b", "[API_KEY]"),
        "url": (r"https?://\S+", "[URL]"),
    }

    def anonymize(self, text: str) -> tuple[str, dict[str, int]]:
        """脱敏处理"""
        stats: dict[str, int] = {}
        for name, (pattern, replacement) in self.PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                stats[name] = len(matches)
                text = re.sub(pattern, replacement, text)
        return text, stats


# ============================================================
# 6. 清洗流水线
# ============================================================

class CleaningPipeline:
    """数据清洗流水线"""

    def __init__(self, config: QualityConfig | None = None):
        self.standardizer = FormatStandardizer()
        self.deduplicator = TextDeduplicator()
        self.quality_filter = QualityFilter(config)
        self.pii_anonymizer = PIIAnonymizer()
        self.stats: dict[str, int] = {step.value: 0 for step in CleaningStep}
        self.pii_stats: dict[str, int] = {}

    def process(self, texts: list[str]) -> tuple[list[str], CleaningResult]:
        """处理文本列表"""
        original_count = len(texts)
        cleaned = []

        for i, text in enumerate(texts):
            # 步骤 1: 格式标准化
            text = self.standardizer.standardize(text)

            # 步骤 2: 去重
            if self.deduplicator.exact_dedup(text):
                self.stats[CleaningStep.DEDUP.value] += 1
                continue

            # 步骤 3: 质量过滤
            passed, reason = self.quality_filter.filter(text)
            if not passed:
                self.stats[CleaningStep.QUALITY.value] += 1
                continue

            # 步骤 4: PII 脱敏
            text, pii_found = self.pii_anonymizer.anonymize(text)
            for pii_type, count in pii_found.items():
                self.pii_stats[pii_type] = self.pii_stats.get(pii_type, 0) + count

            cleaned.append(text)

        result = CleaningResult(
            original_count=original_count,
            final_count=len(cleaned),
            removed_by_step=dict(self.stats),
            pii_found=dict(self.pii_stats),
        )
        return cleaned, result


# ============================================================
# 7. 主程序入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("数据清洗模拟演示")
    print("=" * 60)

    # 模拟原始数据
    raw_texts = [
        "这是一篇关于机器学习的高质量文章，介绍了深度学习的基本原理和应用场景，包括计算机视觉和自然语言处理等领域。",
        "这是一篇关于机器学习的高质量文章，介绍了深度学习的基本原理和应用场景，包括计算机视觉和自然语言处理等领域。",  # 重复
        "短文",  # 太短
        "联系我：zhangsan@example.com，电话 13812345678，身份证 110101199001011234",  # 含 PII
        "点击链接领取免费优惠券！关注公众号获取更多福利！",  # 广告
        "Transformer 架构是现代 NLP 的基础，它通过自注意力机制实现了并行计算，大幅提升了训练效率。BERT、GPT 等模型都基于 Transformer。",
        "啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊",  # 重复内容
        "RAG（检索增强生成）系统通过结合信息检索和文本生成技术，让 LLM 能够基于外部知识库回答问题，有效减少幻觉问题。",
        "API Key: sk-abc123def456ghi789jkl012mno345 请妥善保管",  # 含 API Key
        "LoRA 微调技术通过低秩分解大幅减少了微调参数量，只需训练原始参数的 0.1% 即可达到接近全参数微调的效果，显著降低了训练成本。",
    ]

    # 创建清洗流水线
    pipeline = CleaningPipeline()

    # 执行清洗
    cleaned_texts, result = pipeline.process(raw_texts)

    # 打印结果
    print(f"\n清洗结果:")
    print(f"  原始数据: {result.original_count} 条")
    print(f"  清洗后: {result.final_count} 条")
    print(f"  保留率: {result.retention_rate:.0%}")
    print(f"\n各步骤移除:")
    for step, count in result.removed_by_step.items():
        if count > 0:
            print(f"  {step}: {count} 条")
    print(f"\nPII 检测:")
    for pii_type, count in result.pii_found.items():
        print(f"  {pii_type}: {count} 处")

    print(f"\n清洗后的文本:")
    for i, text in enumerate(cleaned_texts):
        print(f"  [{i+1}] {text[:80]}...")

    print("\n✅ 数据清洗模拟演示完成！")
