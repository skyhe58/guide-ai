"""
Few-shot Learning 少样本学习 — 静态 Few-shot、动态 Few-shot、示例数量对比

知识点：静态 Few-shot（固定示例集）、动态 Few-shot（基于相似度选择示例）、
       模拟 Embedding 相似度计算、示例数量对比实验、示例排序影响、
       Few-shot Prompt 模板管理、与 Ollama API 集成

Python 版本：3.11+
依赖：标准库（默认模式）、ollama>=0.1（服务模式）
最后验证：2024-12-01

外部服务（可选）：
  Ollama 本地 LLM 推理服务
  启动命令：docker compose -f docker/docker-compose.yml up -d ollama
  模型下载：docker exec guide-ai-ollama ollama pull qwen2
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field

# ============================================================
# 1. 数据结构 — Few-shot 示例
# ============================================================

@dataclass
class Example:
    """一个 Few-shot 示例。

    Attributes:
        input_text: 输入文本
        output_text: 期望的输出文本
        category: 示例类别（用于分类选择）
        embedding: 模拟的 Embedding 向量（用于相似度计算）
    """
    input_text: str
    output_text: str
    category: str = ""
    embedding: list[float] = field(default_factory=list)


# ============================================================
# 2. 模拟 Embedding 相似度计算
# ============================================================

class SimpleEmbedding:
    """模拟 Embedding 相似度计算。

    在真实场景中应使用：
    - OpenAI text-embedding-3-small
    - 开源 bge-m3 / sentence-transformers
    这里用基于关键词的 TF 向量模拟，展示动态选择的工作原理。
    """

    def __init__(self, vocabulary: list[str] | None = None):
        """
        Args:
            vocabulary: 词汇表，如果不提供则自动构建
        """
        self._vocabulary = vocabulary or []
        self._vocab_index: dict[str, int] = {}
        if vocabulary:
            self._build_index()

    def _build_index(self) -> None:
        """构建词汇索引。"""
        self._vocab_index = {word: i for i, word in enumerate(self._vocabulary)}

    def fit(self, texts: list[str]) -> SimpleEmbedding:
        """从文本集合中构建词汇表。"""
        vocab_set: set[str] = set()
        for text in texts:
            tokens = self._tokenize(text)
            vocab_set.update(tokens)
        self._vocabulary = sorted(vocab_set)
        self._build_index()
        return self

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """简单分词：中文按字分割，英文按单词分割。"""
        import re
        tokens: list[str] = []
        # 先提取英文单词和数字
        for match in re.finditer(r'[\u4e00-\u9fff]|[a-zA-Z]+|[0-9]+', text.lower()):
            tokens.append(match.group())
        return tokens

    def encode(self, text: str) -> list[float]:
        """将文本编码为向量（基于词频的简单 Embedding）。"""
        if not self._vocabulary:
            raise ValueError("词汇表为空，请先调用 fit() 构建词汇表")

        tokens = self._tokenize(text)
        vector = [0.0] * len(self._vocabulary)

        for token in tokens:
            if token in self._vocab_index:
                vector[self._vocab_index[token]] += 1.0

        # L2 归一化
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]

        return vector

    @staticmethod
    def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """计算两个向量的余弦相似度。"""
        if len(vec_a) != len(vec_b):
            raise ValueError("向量维度不匹配")

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))

        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)


# ============================================================
# 3. 静态 Few-shot — 固定示例集
# ============================================================

class StaticFewShot:
    """静态 Few-shot Prompt 构建器。

    使用预定义的固定示例集，所有输入共用相同的示例。
    适用于任务类型单一、输入变化不大的场景。
    """

    def __init__(self, task_instruction: str, examples: list[Example]):
        self.task_instruction = task_instruction
        self.examples = examples

    def build_prompt(self, query: str, max_examples: int | None = None) -> str:
        """构建 Few-shot Prompt。

        Args:
            query: 用户输入
            max_examples: 最大示例数量，None 表示使用全部
        """
        selected = self.examples[:max_examples] if max_examples else self.examples
        parts = [self.task_instruction, ""]

        for i, ex in enumerate(selected, 1):
            parts.append(f"示例 {i}：")
            parts.append(f"输入：{ex.input_text}")
            parts.append(f"输出：{ex.output_text}")
            parts.append("")

        parts.append("现在请处理以下输入：")
        parts.append(f"输入：{query}")
        parts.append("输出：")

        return "\n".join(parts)


# ============================================================
# 4. 动态 Few-shot — 基于相似度选择示例
# ============================================================

class DynamicFewShot:
    """动态 Few-shot Prompt 构建器。

    根据用户输入动态选择最相关的示例。
    使用 Embedding 相似度计算找到最匹配的示例。
    """

    def __init__(self, task_instruction: str, examples: list[Example]):
        self.task_instruction = task_instruction
        self.examples = examples
        self.embedder = SimpleEmbedding()

        # 构建词汇表并计算所有示例的 Embedding
        # 注意：词汇表需要足够大以覆盖查询中可能出现的词
        all_texts = [ex.input_text for ex in examples] + [ex.output_text for ex in examples]
        self.embedder.fit(all_texts)

        for ex in self.examples:
            ex.embedding = self.embedder.encode(ex.input_text)

    def select_examples(self, query: str, top_k: int = 3) -> list[tuple[float, Example]]:
        """根据输入选择最相关的示例。

        Args:
            query: 用户输入
            top_k: 返回的示例数量

        Returns:
            按相似度降序排列的 (相似度, 示例) 列表
        """
        query_embedding = self.embedder.encode(query)

        scored: list[tuple[float, Example]] = []
        for ex in self.examples:
            sim = SimpleEmbedding.cosine_similarity(query_embedding, ex.embedding)
            scored.append((sim, ex))

        # 按相似度降序排列
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def build_prompt(self, query: str, top_k: int = 3) -> str:
        """构建动态 Few-shot Prompt。

        选择最相关的示例，并按最佳排序策略排列：
        - 最相关的示例放在最后（利用 Recency Bias）
        - 其余示例按相似度降序排列
        """
        selected = self.select_examples(query, top_k)

        # 排序策略：最相关的放最后（Recency Bias）
        if len(selected) > 1:
            most_relevant = selected[0]
            others = selected[1:]
            ordered = others + [most_relevant]
        else:
            ordered = selected

        parts = [self.task_instruction, ""]

        for i, (sim, ex) in enumerate(ordered, 1):
            parts.append(f"示例 {i}（相似度: {sim:.2f}）：")
            parts.append(f"输入：{ex.input_text}")
            parts.append(f"输出：{ex.output_text}")
            parts.append("")

        parts.append("现在请处理以下输入：")
        parts.append(f"输入：{query}")
        parts.append("输出：")

        return "\n".join(parts)


# ============================================================
# 5. 预定义示例库
# ============================================================

def create_sentiment_examples() -> list[Example]:
    """创建情感分析示例库。"""
    return [
        Example("这个手机拍照效果太棒了，夜景模式清晰度很高", "正面", "电子产品"),
        Example("电池续航太差了，一天要充三次电", "负面", "电子产品"),
        Example("屏幕显示效果一般，价格还行", "中性", "电子产品"),
        Example("这家餐厅的牛排非常嫩，服务态度也很好", "正面", "餐饮"),
        Example("等了一个小时才上菜，味道也很一般", "负面", "餐饮"),
        Example("菜品种类丰富，但环境有点吵", "中性", "餐饮"),
        Example("这本书写得深入浅出，非常适合入门学习", "正面", "图书"),
        Example("内容太浅了，全是概念没有实战", "负面", "图书"),
        Example("快递速度很快，第二天就到了，包装也很好", "正面", "物流"),
        Example("包裹被压坏了，客服态度还很差", "负面", "物流"),
        Example("物流速度正常，三天到货", "中性", "物流"),
        Example("AI 课程内容很前沿，老师讲解清晰", "正面", "教育"),
    ]


def create_tag_examples() -> list[Example]:
    """创建标签生成示例库。"""
    return [
        Example(
            "LoRA 通过低秩分解减少微调参数，只需训练 0.1% 的参数",
            '["LoRA", "微调", "参数高效", "低秩分解"]',
            "LLM",
        ),
        Example(
            "RAG 系统先从知识库检索相关文档，再将检索结果作为上下文输入 LLM",
            '["RAG", "检索增强", "知识库", "LLM 应用"]',
            "应用",
        ),
        Example(
            "vLLM 通过 PagedAttention 将 KV Cache 分页管理，显存利用率提升到 95%",
            '["vLLM", "推理优化", "PagedAttention", "KV Cache"]',
            "部署",
        ),
        Example(
            "Transformer 的自注意力机制计算复杂度为 O(n²)，限制了长序列处理",
            '["Transformer", "自注意力", "计算复杂度", "长序列"]',
            "模型",
        ),
        Example(
            "Agent 通过 Function Calling 调用外部工具，实现搜索和数据库查询",
            '["Agent", "Function Calling", "工具调用", "LLM 应用"]',
            "应用",
        ),
        Example(
            "RLHF 使用人类反馈训练奖励模型，再用 PPO 算法优化 LLM",
            '["RLHF", "人类反馈", "PPO", "对齐"]',
            "训练",
        ),
    ]


# ============================================================
# 6. 演示函数
# ============================================================

def demo_static_few_shot() -> None:
    """演示静态 Few-shot。"""
    print("\n" + "=" * 60)
    print("1. 静态 Few-shot — 固定示例集")
    print("=" * 60)

    examples = create_sentiment_examples()[:5]  # 取前 5 个示例
    fs = StaticFewShot(
        task_instruction="请判断以下评论的情感倾向（正面/负面/中性）。",
        examples=examples,
    )

    query = "这个耳机音质很好，但佩戴时间长了耳朵会疼"
    prompt = fs.build_prompt(query, max_examples=3)

    print(f"  示例数量: {len(examples)}")
    print(f"  使用示例: 3 个")
    print(f"  查询: {query}")
    print(f"  Prompt 长度: {len(prompt)} 字符")
    print(f"\n  --- 生成的 Prompt ---")
    print(f"  {prompt}")


def demo_dynamic_few_shot() -> None:
    """演示动态 Few-shot — 基于相似度选择示例。"""
    print("\n" + "=" * 60)
    print("2. 动态 Few-shot — 基于相似度选择")
    print("=" * 60)

    examples = create_sentiment_examples()
    dfs = DynamicFewShot(
        task_instruction="请判断以下评论的情感倾向（正面/负面/中性）。",
        examples=examples,
    )

    queries = [
        "这个平板电脑屏幕很大，看视频体验很好",
        "这家火锅店味道不错，就是价格有点贵",
        "这本 Python 教程讲得很详细，适合零基础",
    ]

    for query in queries:
        print(f"\n  查询: {query}")
        selected = dfs.select_examples(query, top_k=3)
        print(f"  选中的示例（按相似度排序）:")
        for sim, ex in selected:
            print(f"    [{sim:.3f}] {ex.input_text[:40]}... → {ex.output_text}")

        prompt = dfs.build_prompt(query, top_k=3)
        print(f"  Prompt 长度: {len(prompt)} 字符")


def demo_example_count_comparison() -> None:
    """演示示例数量对 Prompt 的影响。"""
    print("\n" + "=" * 60)
    print("3. 示例数量对比实验")
    print("=" * 60)

    examples = create_tag_examples()
    query = "Embedding 模型将文本转为向量，支持语义搜索和相似度计算"

    print(f"  查询: {query}")
    print(f"  总示例数: {len(examples)}")
    print(f"\n  {'示例数':<8} {'Prompt 长度':<15} {'Token 估算':<15} {'建议'}")
    print(f"  {'-'*60}")

    for n in [0, 1, 2, 3, 5, len(examples)]:
        fs = StaticFewShot(
            task_instruction="根据文档内容生成 3-5 个标签，以 JSON 数组格式输出。",
            examples=examples,
        )
        prompt = fs.build_prompt(query, max_examples=n if n > 0 else 0)

        # 粗略估算 Token 数（中文约 1.5 字符/token）
        estimated_tokens = int(len(prompt) / 1.5)

        if n == 0:
            suggestion = "Zero-shot，简单任务可用"
        elif n <= 2:
            suggestion = "基础 Few-shot"
        elif n <= 3:
            suggestion = "⭐ 推荐平衡点"
        elif n <= 5:
            suggestion = "复杂任务可用"
        else:
            suggestion = "⚠️ 通常不值得"

        print(f"  {n:<8} {len(prompt):<15} ~{estimated_tokens:<14} {suggestion}")


def demo_example_ordering() -> None:
    """演示示例排序对输出的影响。"""
    print("\n" + "=" * 60)
    print("4. 示例排序影响")
    print("=" * 60)

    examples = create_sentiment_examples()[:4]
    query = "这个产品质量还可以，但售后服务需要改进"

    # 原始顺序
    fs_original = StaticFewShot(
        task_instruction="请判断情感倾向。",
        examples=examples,
    )

    # 反转顺序
    fs_reversed = StaticFewShot(
        task_instruction="请判断情感倾向。",
        examples=list(reversed(examples)),
    )

    print(f"  查询: {query}")
    print(f"\n  --- 原始顺序（最后一个示例: {examples[-1].output_text}）---")
    print(f"  示例顺序: {[ex.output_text for ex in examples]}")
    print(f"  💡 LLM 倾向于输出与最后一个示例相似的结果")

    print(f"\n  --- 反转顺序（最后一个示例: {examples[0].output_text}）---")
    print(f"  示例顺序: {[ex.output_text for ex in reversed(examples)]}")
    print(f"  💡 改变顺序可能改变 LLM 的输出倾向")

    print(f"\n  📋 最佳排序策略:")
    print(f"  1. 第一个示例：最典型的案例（设定基调）")
    print(f"  2. 中间示例：覆盖不同类型")
    print(f"  3. 最后一个示例：与当前输入最相似的（利用 Recency Bias）")


def demo_dynamic_vs_static() -> None:
    """对比动态 Few-shot 和静态 Few-shot。"""
    print("\n" + "=" * 60)
    print("5. 动态 vs 静态 Few-shot 对比")
    print("=" * 60)

    examples = create_sentiment_examples()
    query = "这本机器学习的书内容很全面，但排版有点乱"

    # 静态：固定前 3 个示例
    static_fs = StaticFewShot(
        task_instruction="请判断情感倾向。",
        examples=examples,
    )
    static_prompt = static_fs.build_prompt(query, max_examples=3)

    # 动态：选择最相关的 3 个示例
    dynamic_fs = DynamicFewShot(
        task_instruction="请判断情感倾向。",
        examples=examples,
    )
    selected = dynamic_fs.select_examples(query, top_k=3)
    dynamic_prompt = dynamic_fs.build_prompt(query, top_k=3)

    print(f"  查询: {query}")

    print(f"\n  --- 静态 Few-shot ---")
    print(f"  使用示例（固定前 3 个）:")
    for ex in examples[:3]:
        print(f"    • {ex.input_text[:40]}... [{ex.category}]")

    print(f"\n  --- 动态 Few-shot ---")
    print(f"  使用示例（相似度最高的 3 个）:")
    for sim, ex in selected:
        print(f"    • [{sim:.3f}] {ex.input_text[:40]}... [{ex.category}]")

    print(f"\n  📊 对比:")
    print(f"  {'指标':<20} {'静态':<20} {'动态'}")
    print(f"  {'-'*55}")
    print(f"  {'Prompt 长度':<20} {len(static_prompt):<20} {len(dynamic_prompt)}")
    print(f"  {'示例相关性':<20} {'随机（可能不相关）':<20} {'高（基于相似度）'}")
    print(f"  {'额外计算':<20} {'无':<20} {'Embedding 计算'}")
    print(f"  {'适用场景':<20} {'任务单一':<20} {'输入多样化'}")


# ============================================================
# 7. 服务模式 — 调用 Ollama API
# ============================================================

async def demo_ollama_few_shot() -> None:
    """服务模式：用不同 Few-shot 策略调用 Ollama。"""
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
    query = "这个 AI 编程助手代码补全很准确，但有时候会生成多余的代码"

    # --- Zero-shot vs Few-shot 对比 ---
    print(f"\n  查询: {query}")

    # Zero-shot
    print(f"\n  --- Zero-shot ---")
    zero_prompt = f"请判断以下评论的情感倾向（正面/负面/中性）：\n{query}\n情感倾向："
    response = ollama.generate(model=model, prompt=zero_prompt, options={"num_predict": 50})
    print(f"  输出: {response['response'][:100]}...")

    # 静态 Few-shot (3 个示例)
    print(f"\n  --- 静态 Few-shot (3 示例) ---")
    examples = create_sentiment_examples()[:3]
    static_fs = StaticFewShot(
        task_instruction="请判断以下评论的情感倾向（正面/负面/中性）。",
        examples=examples,
    )
    static_prompt = static_fs.build_prompt(query)
    response = ollama.generate(model=model, prompt=static_prompt, options={"num_predict": 50})
    print(f"  输出: {response['response'][:100]}...")

    # 动态 Few-shot (3 个最相关示例)
    print(f"\n  --- 动态 Few-shot (3 最相关示例) ---")
    all_examples = create_sentiment_examples()
    dynamic_fs = DynamicFewShot(
        task_instruction="请判断以下评论的情感倾向（正面/负面/中性）。",
        examples=all_examples,
    )
    dynamic_prompt = dynamic_fs.build_prompt(query, top_k=3)
    response = ollama.generate(model=model, prompt=dynamic_prompt, options={"num_predict": 50})
    print(f"  输出: {response['response'][:100]}...")

    # 示例数量对比
    print(f"\n  --- 示例数量对比 ---")
    for n in [1, 3, 5]:
        prompt = static_fs.build_prompt(query, max_examples=n)
        response = ollama.generate(model=model, prompt=prompt, options={"num_predict": 50})
        output = response["response"].strip()[:80]
        print(f"  {n} 个示例: {output}")


# ============================================================
# 主入口
# ============================================================

def main(server_mode: bool = False) -> None:
    """运行所有 Few-shot 演示。"""
    print("🐍 Few-shot Learning — 静态 Few-shot、动态 Few-shot、示例数量对比")
    print("=" * 60)

    demo_static_few_shot()
    demo_dynamic_few_shot()
    demo_example_count_comparison()
    demo_example_ordering()
    demo_dynamic_vs_static()

    if server_mode:
        import asyncio
        asyncio.run(demo_ollama_few_shot())

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. 静态 Few-shot：简单稳定，适合任务类型单一的场景")
    print("  2. 动态 Few-shot：基于相似度选择示例，效果更好但需要额外计算")
    print("  3. 示例数量：3 个高质量示例通常是最佳平衡点")
    print("  4. 示例排序：最后一个示例影响最大（Recency Bias）")
    print("  5. 质量 > 数量：3 个好示例胜过 10 个差示例")

    if not server_mode:
        print("\n💡 要测试真实 LLM 调用: python 03_few_shot.py server")


if __name__ == "__main__":
    is_server = len(sys.argv) > 1 and sys.argv[1].lower() == "server"
    main(server_mode=is_server)
