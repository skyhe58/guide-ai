"""
BPE 算法基础实现 — Byte Pair Encoding 分词

知识点：BPE 算法原理、词表构建、编码/解码、
       与 WordPiece/Unigram 对比

Python 版本：3.11+
依赖：无额外依赖（纯 Python 实现）
最后验证：2024-12-01
"""

from __future__ import annotations

from collections import Counter


# ============================================================
# 1. BPE 算法实现（从零实现）
# ============================================================

class SimpleBPE:
    """简化版 BPE（Byte Pair Encoding）分词器。

    BPE 算法步骤：
    1. 将所有单词拆分为字符序列
    2. 统计所有相邻字符对的频率
    3. 合并频率最高的字符对为新 Token
    4. 重复步骤 2-3，直到达到目标词表大小

    GPT-2/GPT-3/GPT-4 使用 Byte-level BPE。
    """

    def __init__(self, vocab_size: int = 50):
        self.vocab_size = vocab_size
        self.merges: list[tuple[str, str]] = []
        self.vocab: dict[str, int] = {}

    def _get_pairs(self, tokens: list[str]) -> Counter:
        """统计所有相邻 Token 对的频率。"""
        pairs = Counter()
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += 1
        return pairs

    def _merge_pair(
        self, tokens: list[str], pair: tuple[str, str]
    ) -> list[str]:
        """合并指定的 Token 对。"""
        merged = []
        i = 0
        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and tokens[i] == pair[0]
                and tokens[i + 1] == pair[1]
            ):
                merged.append(pair[0] + pair[1])
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    def train(self, corpus: list[str]) -> None:
        """训练 BPE 分词器。

        Args:
            corpus: 训练语料（单词列表）
        """
        # 初始化：每个字符作为一个 Token
        # 添加特殊的词尾标记 </w>
        word_freqs: dict[str, int] = Counter()
        for word in corpus:
            word_freqs[" ".join(list(word) + ["</w>"])] += 1

        # 构建初始词表
        self.vocab = {}
        for word in word_freqs:
            for token in word.split():
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

        print(f"  初始词表大小: {len(self.vocab)}")
        print(f"  初始词表: {list(self.vocab.keys())}")

        # 迭代合并
        num_merges = self.vocab_size - len(self.vocab)
        for step in range(max(0, num_merges)):
            # 统计所有相邻对的频率
            pair_freqs: Counter = Counter()
            for word, freq in word_freqs.items():
                tokens = word.split()
                for i in range(len(tokens) - 1):
                    pair_freqs[(tokens[i], tokens[i + 1])] += freq

            if not pair_freqs:
                break

            # 找到频率最高的对
            best_pair = pair_freqs.most_common(1)[0][0]
            self.merges.append(best_pair)

            # 合并
            new_word_freqs: dict[str, int] = {}
            for word, freq in word_freqs.items():
                tokens = word.split()
                merged = self._merge_pair(tokens, best_pair)
                new_word_freqs[" ".join(merged)] = freq
            word_freqs = new_word_freqs

            # 更新词表
            new_token = best_pair[0] + best_pair[1]
            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)

            if step < 5:  # 只打印前 5 步
                print(f"  Step {step + 1}: 合并 {best_pair} → '{new_token}' (频率: {pair_freqs[best_pair]})")

        print(f"  最终词表大小: {len(self.vocab)}")

    def encode(self, text: str) -> list[str]:
        """编码文本为 Token 序列。"""
        tokens = list(text) + ["</w>"]

        for pair in self.merges:
            tokens = self._merge_pair(tokens, pair)

        return tokens

    def decode(self, tokens: list[str]) -> str:
        """解码 Token 序列为文本。"""
        text = "".join(tokens)
        text = text.replace("</w>", "")
        return text


# ============================================================
# 2. 演示
# ============================================================

def demo_bpe_training() -> None:
    """演示 BPE 训练过程。"""
    print("\n" + "=" * 60)
    print("1. BPE 训练过程")
    print("=" * 60)

    # 训练语料
    corpus = [
        "low", "low", "low", "low", "low",
        "lower", "lower",
        "newest", "newest", "newest", "newest", "newest", "newest",
        "widest", "widest", "widest",
    ]

    bpe = SimpleBPE(vocab_size=30)
    bpe.train(corpus)

    # 编码测试
    print("\n  --- 编码测试 ---")
    test_words = ["low", "lower", "newest", "new"]
    for word in test_words:
        tokens = bpe.encode(word)
        print(f"  '{word}' → {tokens}")


def demo_bpe_concept() -> None:
    """BPE 概念说明。"""
    print("\n" + "=" * 60)
    print("2. BPE 核心概念")
    print("=" * 60)

    print("""
    BPE（Byte Pair Encoding）分词算法：

    核心思想：从字符级别开始，逐步合并高频字符对

    示例：
    训练语料: ["hello", "help", "held"]

    Step 0: h e l l o, h e l p, h e l d
    Step 1: 合并 (h, e) → he    → he l l o, he l p, he l d
    Step 2: 合并 (he, l) → hel  → hel l o, hel p, hel d
    Step 3: 合并 (hel, l) → hell → hell o, hel p, hel d

    分词器对比：
    ┌──────────────┬──────────────┬──────────────┐
    │ 算法         │ 代表模型     │ 特点         │
    ├──────────────┼──────────────┼──────────────┤
    │ BPE          │ GPT 系列     │ 贪心合并     │
    │ WordPiece    │ BERT         │ 似然最大化   │
    │ Unigram      │ T5, LLaMA    │ 概率模型     │
    │ SentencePiece│ LLaMA, Qwen  │ 语言无关     │
    └──────────────┴──────────────┴──────────────┘

    💡 现代 LLM 趋势：
    - GPT-4: Byte-level BPE（tiktoken）
    - LLaMA/Qwen: SentencePiece（Unigram）
    - 词表大小: 32K-150K
    """)


def main() -> None:
    """运行所有 BPE 演示。"""
    print("🔤 BPE 算法基础实现 — Byte Pair Encoding 分词")
    print("=" * 60)

    demo_bpe_training()
    demo_bpe_concept()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. BPE: 从字符开始，逐步合并高频对")
    print("  2. 子词分词: 平衡词表大小和覆盖率")
    print("  3. GPT 系列使用 Byte-level BPE")
    print("  4. LLaMA/Qwen 使用 SentencePiece")


if __name__ == "__main__":
    main()
