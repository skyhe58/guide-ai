"""
Hugging Face Tokenizers 库使用 — 高性能分词器

知识点：tokenizers 库使用、预训练分词器加载、
       编码/解码、特殊 Token、Token 统计

Python 版本：3.11+
依赖：无额外依赖（演示模式使用模拟数据）
可选依赖：transformers>=4.36, tokenizers>=0.15
最后验证：2024-12-01
"""

from __future__ import annotations


# ============================================================
# 1. Tokenizer 基础操作（模拟）
# ============================================================

class MockTokenizer:
    """模拟 Tokenizer（无需安装 transformers）。

    模拟 Qwen2 Tokenizer 的基本行为，用于演示概念。
    实际使用请安装 transformers 库。
    """

    def __init__(self):
        # 模拟词表（简化版）
        self.vocab = {
            "你": 100, "好": 101, "世": 102, "界": 103,
            "什": 104, "么": 105, "是": 106,
            "机": 107, "器": 108, "学": 109, "习": 110,
            "人": 111, "工": 112, "智": 113, "能": 114,
            "Trans": 200, "former": 201,
            "Attention": 202, "is": 203, "all": 204,
            "you": 205, "need": 206,
            "hello": 207, "world": 208,
            " ": 209, "，": 210, "。": 211,
            "<|im_start|>": 1, "<|im_end|>": 2,
            "<|endoftext|>": 0,
        }
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = 151936  # Qwen2 实际词表大小

    def encode(self, text: str) -> list[int]:
        """简化编码（实际 Tokenizer 使用 BPE/Unigram）。"""
        ids = []
        i = 0
        while i < len(text):
            matched = False
            # 尝试匹配最长 Token
            for length in range(min(10, len(text) - i), 0, -1):
                substr = text[i : i + length]
                if substr in self.vocab:
                    ids.append(self.vocab[substr])
                    i += length
                    matched = True
                    break
            if not matched:
                ids.append(hash(text[i]) % 1000 + 300)  # 未知 Token
                i += 1
        return ids

    def decode(self, ids: list[int]) -> str:
        """解码 Token ID 为文本。"""
        return "".join(self.id_to_token.get(i, f"[UNK:{i}]") for i in ids)


def demo_basic_tokenization() -> None:
    """演示基础分词操作。"""
    print("\n" + "=" * 60)
    print("1. Tokenizer 基础操作")
    print("=" * 60)

    tokenizer = MockTokenizer()

    # 编码
    texts = ["你好世界", "什么是机器学习", "Transformer"]
    for text in texts:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        print(f"  '{text}'")
        print(f"    → Token IDs: {ids}")
        print(f"    → 解码: '{decoded}'")
        print(f"    → Token 数: {len(ids)}")
        print()

    print(f"  词表大小（Qwen2 实际）: {tokenizer.vocab_size:,}")


# ============================================================
# 2. Hugging Face Transformers 使用（伪代码）
# ============================================================

def demo_hf_tokenizer() -> None:
    """展示 Hugging Face Tokenizer 使用方式。"""
    print("\n" + "=" * 60)
    print("2. Hugging Face Tokenizer 使用（伪代码）")
    print("=" * 60)

    code = '''
    from transformers import AutoTokenizer

    # 加载预训练 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")

    # 基础编码/解码
    text = "什么是 Transformer？"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    tokens = tokenizer.tokenize(text)  # 查看 Token 文本

    print(f"Token IDs: {ids}")
    print(f"Tokens: {tokens}")
    print(f"解码: {decoded}")

    # 批量编码（带 padding 和 truncation）
    batch = tokenizer(
        ["你好", "什么是机器学习？"],
        padding=True,           # 填充到相同长度
        truncation=True,        # 截断超长文本
        max_length=512,
        return_tensors="pt",    # 返回 PyTorch 张量
    )
    print(f"input_ids: {batch['input_ids'].shape}")
    print(f"attention_mask: {batch['attention_mask'].shape}")

    # Chat 模板（对话模型专用）
    messages = [
        {"role": "system", "content": "你是一个助手"},
        {"role": "user", "content": "你好"},
    ]
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(f"Chat 模板:\\n{chat_text}")

    # 特殊 Token
    print(f"BOS: {tokenizer.bos_token} ({tokenizer.bos_token_id})")
    print(f"EOS: {tokenizer.eos_token} ({tokenizer.eos_token_id})")
    print(f"PAD: {tokenizer.pad_token} ({tokenizer.pad_token_id})")
    print(f"词表大小: {tokenizer.vocab_size}")
    '''
    print(code)


# ============================================================
# 3. Token 计数与成本估算
# ============================================================

def demo_token_counting() -> None:
    """演示 Token 计数与成本估算。"""
    print("\n" + "=" * 60)
    print("3. Token 计数与成本估算")
    print("=" * 60)

    # 经验法则
    print("""
    Token 计数经验法则：
    - 英文: ~1 Token ≈ 4 个字符 ≈ 0.75 个单词
    - 中文: ~1 Token ≈ 1-2 个汉字
    - 代码: Token 数通常比自然语言多 50-100%

    API 成本估算（2024 年参考价格）：
    ┌──────────────────┬──────────────┬──────────────┐
    │ 模型             │ 输入价格     │ 输出价格     │
    ├──────────────────┼──────────────┼──────────────┤
    │ GPT-4o           │ $2.5/1M      │ $10/1M       │
    │ GPT-4o-mini      │ $0.15/1M     │ $0.6/1M      │
    │ Claude 3.5 Sonnet│ $3/1M        │ $15/1M       │
    │ Qwen2-72B (API)  │ ¥4/1M        │ ¥12/1M       │
    │ DeepSeek V2      │ ¥1/1M        │ ¥2/1M        │
    │ 本地部署         │ 免费         │ 免费         │
    └──────────────────┴──────────────┴──────────────┘

    💡 省钱技巧：
    1. 用 GPT-4o-mini / DeepSeek 处理简单任务
    2. 精简 System Prompt（减少输入 Token）
    3. 本地部署开源模型（Ollama + Qwen2）
    """)


# ============================================================
# 4. 分词器对比
# ============================================================

def demo_tokenizer_comparison() -> None:
    """分词器对比。"""
    print("\n" + "=" * 60)
    print("4. 主流模型分词器对比")
    print("=" * 60)

    print("""
    ┌──────────────┬──────────────┬──────────┬──────────────┐
    │ 模型         │ 分词算法     │ 词表大小 │ 中文支持     │
    ├──────────────┼──────────────┼──────────┼──────────────┤
    │ GPT-4        │ BPE(tiktoken)│ ~100K    │ 一般         │
    │ Claude 3     │ BPE          │ ~100K    │ 一般         │
    │ LLaMA 3      │ BPE(tiktoken)│ 128K     │ 较好         │
    │ Qwen2        │ BPE          │ 152K     │ 优秀         │
    │ DeepSeek V2  │ BPE          │ 100K     │ 优秀         │
    │ BERT         │ WordPiece    │ 30K      │ 差           │
    │ T5           │ SentencePiece│ 32K      │ 差           │
    └──────────────┴──────────────┴──────────┴──────────────┘

    💡 中文场景推荐 Qwen2 / DeepSeek 的分词器（中文 Token 效率高）
    """)


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行所有 Tokenizer 演示。"""
    print("🔤 Hugging Face Tokenizers 库使用")
    print("=" * 60)

    demo_basic_tokenization()
    demo_hf_tokenizer()
    demo_token_counting()
    demo_tokenizer_comparison()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. Tokenizer 将文本转为 Token ID 序列")
    print("  2. BPE 是最主流的分词算法（GPT/Qwen/DeepSeek）")
    print("  3. 中文约 1-2 字 = 1 Token")
    print("  4. Chat 模板是对话模型的关键（apply_chat_template）")
    print("  5. Token 数直接影响 API 成本和推理速度")


if __name__ == "__main__":
    main()
