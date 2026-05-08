"""
Agent 记忆 — 对话历史管理、向量记忆模拟

知识点：短期记忆（对话历史）、长期记忆（向量记忆）、工作记忆、
       滑动窗口策略、摘要压缩策略、Token 预算管理、
       向量存储与语义检索、记忆遗忘与更新

Python 版本：3.11+
依赖：标准库（默认模式）、ollama>=0.1（服务模式）
最后验证：2024-12-01

外部服务（可选）：
  Ollama 本地 LLM 推理服务
  启动命令：docker compose -f docker/docker-compose.yml up -d ollama
  模型下载：docker exec guide-ai-ollama ollama pull qwen2
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# ============================================================
# 1. 基础数据结构
# ============================================================

@dataclass
class ChatMessage:
    """对话消息。"""
    role: str  # system, user, assistant, tool
    content: str
    timestamp: float = field(default_factory=time.time)
    token_count: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.token_count == 0:
            # 简单估算 token 数（中文约 1.5 token/字，英文约 0.75 token/词）
            self.token_count = self._estimate_tokens(self.content)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """估算 token 数量。"""
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return int(chinese_chars * 1.5 + other_chars * 0.3)


@dataclass
class MemoryEntry:
    """长期记忆条目。"""
    id: str
    content: str
    embedding: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    importance: float = 0.5  # 重要性评分 0-1
    access_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)


# ============================================================
# 2. 短期记忆 — 对话历史管理
# ============================================================

class BaseShortTermMemory(ABC):
    """短期记忆基类。"""

    @abstractmethod
    def add(self, message: ChatMessage) -> None:
        """添加消息。"""

    @abstractmethod
    def get_messages(self) -> list[ChatMessage]:
        """获取当前有效的消息列表。"""

    @abstractmethod
    def get_token_count(self) -> int:
        """获取当前总 token 数。"""

    @abstractmethod
    def clear(self) -> None:
        """清空记忆。"""


class WindowBufferMemory(BaseShortTermMemory):
    """滑动窗口记忆 — 保留最近 K 轮对话。"""

    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self._messages: list[ChatMessage] = []
        self._system_message: ChatMessage | None = None

    def add(self, message: ChatMessage) -> None:
        """添加消息，超出窗口时丢弃最早的消息。"""
        if message.role == "system":
            self._system_message = message
            return
        self._messages.append(message)
        # 保留最近 max_messages 条
        if len(self._messages) > self.max_messages:
            removed = self._messages.pop(0)
            # 记录被丢弃的消息数
            self._messages[0].metadata["dropped_before"] = True

    def get_messages(self) -> list[ChatMessage]:
        """获取消息列表（系统消息 + 最近消息）。"""
        result = []
        if self._system_message:
            result.append(self._system_message)
        result.extend(self._messages)
        return result

    def get_token_count(self) -> int:
        """获取总 token 数。"""
        total = sum(m.token_count for m in self._messages)
        if self._system_message:
            total += self._system_message.token_count
        return total

    def clear(self) -> None:
        """清空（保留系统消息）。"""
        self._messages.clear()


class SummaryBufferMemory(BaseShortTermMemory):
    """摘要缓存记忆 — 超过阈值时压缩早期对话为摘要。"""

    def __init__(self, max_tokens: int = 4000, summary_threshold: int = 3000):
        self.max_tokens = max_tokens
        self.summary_threshold = summary_threshold
        self._messages: list[ChatMessage] = []
        self._system_message: ChatMessage | None = None
        self._summary: str = ""
        self._summary_count: int = 0  # 被摘要的消息数

    def add(self, message: ChatMessage) -> None:
        """添加消息，超过阈值时触发摘要压缩。"""
        if message.role == "system":
            self._system_message = message
            return
        self._messages.append(message)

        # 检查是否需要压缩
        if self.get_token_count() > self.summary_threshold:
            self._compress()

    def _compress(self) -> None:
        """压缩早期消息为摘要。"""
        if len(self._messages) <= 4:
            return

        # 取前半部分消息进行摘要
        half = len(self._messages) // 2
        to_summarize = self._messages[:half]
        self._messages = self._messages[half:]

        # 模拟 LLM 摘要（实际应调用 LLM）
        summary_parts = []
        for msg in to_summarize:
            if msg.role == "user":
                summary_parts.append(f"用户询问了: {msg.content[:30]}")
            elif msg.role == "assistant":
                summary_parts.append(f"助手回答了: {msg.content[:30]}")

        new_summary = "；".join(summary_parts)
        if self._summary:
            self._summary = f"{self._summary}。此后，{new_summary}"
        else:
            self._summary = f"对话摘要：{new_summary}"

        self._summary_count += len(to_summarize)

    def get_messages(self) -> list[ChatMessage]:
        """获取消息列表（系统消息 + 摘要 + 最近消息）。"""
        result = []
        if self._system_message:
            result.append(self._system_message)
        if self._summary:
            result.append(ChatMessage(
                role="system",
                content=f"[对话历史摘要，包含 {self._summary_count} 条消息] {self._summary}",
            ))
        result.extend(self._messages)
        return result

    def get_token_count(self) -> int:
        """获取总 token 数。"""
        total = sum(m.token_count for m in self._messages)
        if self._system_message:
            total += self._system_message.token_count
        if self._summary:
            total += ChatMessage._estimate_tokens(self._summary)
        return total

    def clear(self) -> None:
        """清空。"""
        self._messages.clear()
        self._summary = ""
        self._summary_count = 0


class TokenBudgetMemory(BaseShortTermMemory):
    """Token 预算记忆 — 精确控制 token 消耗。"""

    def __init__(self, max_tokens: int = 4000, reserve_for_response: int = 1000):
        self.max_tokens = max_tokens
        self.reserve_for_response = reserve_for_response
        self.available_tokens = max_tokens - reserve_for_response
        self._messages: list[ChatMessage] = []
        self._system_message: ChatMessage | None = None

    def add(self, message: ChatMessage) -> None:
        """添加消息，超出预算时从最早的消息开始删除。"""
        if message.role == "system":
            self._system_message = message
            self.available_tokens -= message.token_count
            return

        self._messages.append(message)

        # 检查是否超出预算
        while self.get_token_count() > self.available_tokens and len(self._messages) > 2:
            self._messages.pop(0)

    def get_messages(self) -> list[ChatMessage]:
        """获取消息列表。"""
        result = []
        if self._system_message:
            result.append(self._system_message)
        result.extend(self._messages)
        return result

    def get_token_count(self) -> int:
        """获取总 token 数。"""
        total = sum(m.token_count for m in self._messages)
        if self._system_message:
            total += self._system_message.token_count
        return total

    def clear(self) -> None:
        """清空。"""
        self._messages.clear()


# ============================================================
# 3. 长期记忆 — 向量记忆
# ============================================================

class SimpleVectorMemory:
    """简单向量记忆 — 模拟向量数据库的语义检索。

    使用简单的词袋模型模拟 embedding，演示向量记忆的核心流程。
    生产环境应使用真实的 embedding 模型和向量数据库。
    """

    def __init__(self, max_entries: int = 1000):
        self.max_entries = max_entries
        self._entries: dict[str, MemoryEntry] = {}
        self._vocabulary: set[str] = set()

    def _simple_embedding(self, text: str) -> list[float]:
        """简单的词袋 embedding（模拟）。

        生产环境应使用 text-embedding-3-small 等模型。
        """
        # 分词（简单按字符和空格分割）
        words = set()
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                words.add(char)
        words.update(text.lower().split())
        self._vocabulary.update(words)

        # 构建词袋向量
        vocab_list = sorted(self._vocabulary)
        vector = [1.0 if w in words else 0.0 for w in vocab_list]

        # 归一化
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def _cosine_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """计算余弦相似度。"""
        # 对齐向量长度
        max_len = max(len(vec_a), len(vec_b))
        a = vec_a + [0.0] * (max_len - len(vec_a))
        b = vec_b + [0.0] * (max_len - len(vec_b))

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
        norm_b = math.sqrt(sum(x * x for x in b)) or 1.0
        return dot / (norm_a * norm_b)

    def add(self, content: str, metadata: dict | None = None,
            importance: float = 0.5) -> str:
        """添加记忆条目。"""
        entry_id = hashlib.md5(content.encode()).hexdigest()[:12]

        # 检查去重
        if entry_id in self._entries:
            self._entries[entry_id].access_count += 1
            return entry_id

        # 容量检查
        if len(self._entries) >= self.max_entries:
            self._evict()

        entry = MemoryEntry(
            id=entry_id,
            content=content,
            embedding=self._simple_embedding(content),
            metadata=metadata or {},
            importance=importance,
        )
        self._entries[entry_id] = entry
        return entry_id

    def search(self, query: str, top_k: int = 5,
               min_similarity: float = 0.1) -> list[dict]:
        """语义搜索。"""
        if not self._entries:
            return []

        query_embedding = self._simple_embedding(query)
        results = []

        for entry in self._entries.values():
            similarity = self._cosine_similarity(query_embedding, entry.embedding)
            if similarity >= min_similarity:
                # 综合评分：相似度 * 0.7 + 重要性 * 0.2 + 时间衰减 * 0.1
                time_decay = 1.0 / (1.0 + (time.time() - entry.created_at) / 86400)
                score = similarity * 0.7 + entry.importance * 0.2 + time_decay * 0.1

                results.append({
                    "id": entry.id,
                    "content": entry.content,
                    "similarity": round(similarity, 4),
                    "importance": entry.importance,
                    "score": round(score, 4),
                    "metadata": entry.metadata,
                })

                # 更新访问记录
                entry.access_count += 1
                entry.last_accessed = time.time()

        # 按综合评分排序
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _evict(self) -> None:
        """淘汰最不重要的记忆。"""
        if not self._entries:
            return
        # 按重要性和访问频率排序，淘汰最低的
        entries = sorted(
            self._entries.values(),
            key=lambda e: e.importance * 0.5 + (e.access_count / 10) * 0.5,
        )
        to_remove = entries[0]
        del self._entries[to_remove.id]

    def get_stats(self) -> dict:
        """获取记忆统计。"""
        return {
            "total_entries": len(self._entries),
            "max_entries": self.max_entries,
            "vocabulary_size": len(self._vocabulary),
            "avg_importance": round(
                sum(e.importance for e in self._entries.values()) / max(len(self._entries), 1), 2
            ),
        }

    def clear(self) -> None:
        """清空所有记忆。"""
        self._entries.clear()
        self._vocabulary.clear()


# ============================================================
# 4. 工作记忆 — 任务状态管理
# ============================================================

class WorkingMemory:
    """工作记忆 — 管理当前任务的中间状态。"""

    def __init__(self):
        self._scratchpad: dict[str, Any] = {}
        self._tool_cache: dict[str, str] = {}

    def set(self, key: str, value: Any) -> None:
        """设置工作记忆。"""
        self._scratchpad[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """获取工作记忆。"""
        return self._scratchpad.get(key, default)

    def cache_tool_result(self, tool_call: str, result: str) -> None:
        """缓存工具调用结果。"""
        cache_key = hashlib.md5(tool_call.encode()).hexdigest()[:16]
        self._tool_cache[cache_key] = result

    def get_cached_result(self, tool_call: str) -> str | None:
        """获取缓存的工具结果。"""
        cache_key = hashlib.md5(tool_call.encode()).hexdigest()[:16]
        return self._tool_cache.get(cache_key)

    def clear(self) -> None:
        """清空工作记忆。"""
        self._scratchpad.clear()
        self._tool_cache.clear()

    def summary(self) -> dict:
        """获取工作记忆摘要。"""
        return {
            "scratchpad_keys": list(self._scratchpad.keys()),
            "cached_tools": len(self._tool_cache),
        }


# ============================================================
# 5. 综合记忆管理器
# ============================================================

class AgentMemoryManager:
    """Agent 记忆管理器 — 统一管理短期、长期、工作记忆。"""

    def __init__(
        self,
        short_term: BaseShortTermMemory | None = None,
        long_term: SimpleVectorMemory | None = None,
        working: WorkingMemory | None = None,
    ):
        self.short_term = short_term or WindowBufferMemory()
        self.long_term = long_term or SimpleVectorMemory()
        self.working = working or WorkingMemory()

    def build_context(self, current_query: str, max_long_term: int = 3) -> list[ChatMessage]:
        """构建完整的上下文（短期 + 长期记忆）。"""
        messages = []

        # 1. 系统消息（从短期记忆获取）
        short_messages = self.short_term.get_messages()
        system_msgs = [m for m in short_messages if m.role == "system"]
        messages.extend(system_msgs)

        # 2. 长期记忆检索结果
        long_term_results = self.long_term.search(current_query, top_k=max_long_term)
        if long_term_results:
            memory_context = "\n".join(
                f"- {r['content']}" for r in long_term_results
            )
            messages.append(ChatMessage(
                role="system",
                content=f"[相关历史记忆]\n{memory_context}",
            ))

        # 3. 对话历史（非系统消息）
        non_system = [m for m in short_messages if m.role != "system"]
        messages.extend(non_system)

        return messages

    def after_interaction(self, user_msg: str, assistant_msg: str,
                          save_to_long_term: bool = False,
                          importance: float = 0.5) -> None:
        """交互后更新记忆。"""
        # 更新短期记忆
        self.short_term.add(ChatMessage(role="user", content=user_msg))
        self.short_term.add(ChatMessage(role="assistant", content=assistant_msg))

        # 可选：保存到长期记忆
        if save_to_long_term:
            self.long_term.add(
                content=f"用户: {user_msg}\n助手: {assistant_msg}",
                metadata={"type": "conversation"},
                importance=importance,
            )

    def get_stats(self) -> dict:
        """获取记忆系统统计。"""
        return {
            "short_term_tokens": self.short_term.get_token_count(),
            "short_term_messages": len(self.short_term.get_messages()),
            "long_term": self.long_term.get_stats(),
            "working": self.working.summary(),
        }


# ============================================================
# 6. 演示函数
# ============================================================

def demo_window_buffer() -> None:
    """演示滑动窗口记忆。"""
    print("\n" + "=" * 60)
    print("1. 滑动窗口记忆 (WindowBufferMemory)")
    print("=" * 60)

    memory = WindowBufferMemory(max_messages=6)
    memory.add(ChatMessage(role="system", content="你是一个 AI 助手"))

    # 模拟多轮对话
    conversations = [
        ("什么是 RAG？", "RAG 是检索增强生成技术..."),
        ("它有什么优势？", "RAG 的优势包括减少幻觉、知识可更新..."),
        ("如何实现？", "实现 RAG 需要：文档加载、切分、向量化、检索、生成..."),
        ("推荐什么向量数据库？", "推荐 Chroma（原型）、Milvus（生产）..."),
        ("LangChain 怎么用？", "LangChain 提供了完整的 RAG 链..."),
    ]

    for user_msg, assistant_msg in conversations:
        memory.add(ChatMessage(role="user", content=user_msg))
        memory.add(ChatMessage(role="assistant", content=assistant_msg))

    messages = memory.get_messages()
    print(f"  总消息数: {len(messages)}（含系统消息）")
    print(f"  Token 数: {memory.get_token_count()}")
    print(f"  最早消息: {messages[1].content[:30]}...")
    print(f"  最新消息: {messages[-1].content[:30]}...")


def demo_summary_buffer() -> None:
    """演示摘要缓存记忆。"""
    print("\n" + "=" * 60)
    print("2. 摘要缓存记忆 (SummaryBufferMemory)")
    print("=" * 60)

    memory = SummaryBufferMemory(max_tokens=500, summary_threshold=300)
    memory.add(ChatMessage(role="system", content="你是一个 AI 技术专家"))

    conversations = [
        ("什么是 Transformer？", "Transformer 是一种基于自注意力机制的神经网络架构..."),
        ("什么是 BERT？", "BERT 是基于 Transformer 的预训练语言模型..."),
        ("什么是 GPT？", "GPT 是生成式预训练 Transformer 模型..."),
        ("什么是 RAG？", "RAG 是检索增强生成技术..."),
        ("什么是 Agent？", "Agent 是能自主决策和行动的 AI 系统..."),
    ]

    for i, (user_msg, assistant_msg) in enumerate(conversations):
        memory.add(ChatMessage(role="user", content=user_msg))
        memory.add(ChatMessage(role="assistant", content=assistant_msg))
        print(f"  第 {i + 1} 轮后: {memory.get_token_count()} tokens, "
              f"{len(memory.get_messages())} 条消息")

    # 查看摘要
    messages = memory.get_messages()
    for msg in messages:
        if "摘要" in msg.content:
            print(f"  📝 摘要: {msg.content[:100]}...")
            break


def demo_vector_memory() -> None:
    """演示向量记忆。"""
    print("\n" + "=" * 60)
    print("3. 向量记忆 (SimpleVectorMemory)")
    print("=" * 60)

    memory = SimpleVectorMemory(max_entries=100)

    # 添加记忆
    entries = [
        ("用户偏好 Python 编程语言，使用 MacBook 开发", {"type": "preference"}, 0.8),
        ("用户正在学习 RAG 技术，关注检索增强生成", {"type": "learning"}, 0.9),
        ("用户之前问过 LangChain 的使用方法", {"type": "history"}, 0.6),
        ("用户对向量数据库选型感兴趣", {"type": "interest"}, 0.7),
        ("用户是后端开发者，有 3 年 Java 经验", {"type": "profile"}, 0.8),
        ("用户想了解 Agent 的记忆机制", {"type": "learning"}, 0.9),
    ]

    for content, metadata, importance in entries:
        entry_id = memory.add(content, metadata, importance)
        print(f"  ✅ 添加: {content[:40]}... (id={entry_id})")

    # 语义搜索
    print(f"\n  🔍 搜索: 'RAG 检索技术'")
    results = memory.search("RAG 检索技术", top_k=3)
    for r in results:
        print(f"     [{r['score']:.3f}] {r['content'][:50]}...")

    print(f"\n  🔍 搜索: 'Python 开发'")
    results = memory.search("Python 开发", top_k=3)
    for r in results:
        print(f"     [{r['score']:.3f}] {r['content'][:50]}...")

    print(f"\n  📊 统计: {memory.get_stats()}")


def demo_working_memory() -> None:
    """演示工作记忆。"""
    print("\n" + "=" * 60)
    print("4. 工作记忆 (WorkingMemory)")
    print("=" * 60)

    working = WorkingMemory()

    # 模拟任务执行过程中的中间状态
    working.set("current_task", "分析 RAG 系统性能")
    working.set("step", 1)
    working.set("findings", ["检索延迟 200ms", "生成延迟 500ms"])

    # 缓存工具调用结果
    working.cache_tool_result("search(RAG 性能优化)", "优化方案：1. 缓存 2. 并行 3. 预计算")
    working.cache_tool_result("calculate(200+500)", "700")

    print(f"  当前任务: {working.get('current_task')}")
    print(f"  当前步骤: {working.get('step')}")
    print(f"  发现: {working.get('findings')}")

    # 检查缓存
    cached = working.get_cached_result("search(RAG 性能优化)")
    print(f"  缓存命中: {cached}")

    print(f"  摘要: {working.summary()}")


def demo_memory_manager() -> None:
    """演示综合记忆管理器。"""
    print("\n" + "=" * 60)
    print("5. 综合记忆管理器 (AgentMemoryManager)")
    print("=" * 60)

    manager = AgentMemoryManager(
        short_term=WindowBufferMemory(max_messages=10),
        long_term=SimpleVectorMemory(),
        working=WorkingMemory(),
    )

    # 设置系统消息
    manager.short_term.add(ChatMessage(role="system", content="你是一个 AI 学习助手"))

    # 添加一些长期记忆
    manager.long_term.add("用户偏好 Python，正在学习 AI", importance=0.8)
    manager.long_term.add("用户之前学习了 Transformer 和 BERT", importance=0.7)
    manager.long_term.add("用户对 RAG 技术特别感兴趣", importance=0.9)

    # 模拟多轮交互
    interactions = [
        ("什么是 RAG？", "RAG 是检索增强生成技术...", True, 0.8),
        ("如何选择向量数据库？", "推荐 Chroma 用于原型...", True, 0.7),
        ("谢谢！", "不客气，有问题随时问！", False, 0.3),
    ]

    for user_msg, assistant_msg, save, importance in interactions:
        manager.after_interaction(user_msg, assistant_msg, save, importance)
        print(f"  👤 {user_msg}")
        print(f"  🤖 {assistant_msg[:40]}...")

    # 构建上下文
    print(f"\n  --- 构建上下文 ---")
    context = manager.build_context("RAG 的最佳实践是什么？")
    for msg in context:
        role_icon = {"system": "⚙️", "user": "👤", "assistant": "🤖"}.get(msg.role, "?")
        print(f"  {role_icon} [{msg.role}] {msg.content[:60]}...")

    print(f"\n  📊 记忆统计: {json.dumps(manager.get_stats(), ensure_ascii=False, indent=2)}")


# ============================================================
# 服务模式 — 调用 Ollama API
# ============================================================

def demo_ollama_memory() -> None:
    """服务模式：用 Ollama 演示带记忆的对话。"""
    print("\n" + "=" * 60)
    print("服务模式 — 调用 Ollama API 演示带记忆的对话")
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

    # 模拟多轮对话，展示记忆管理
    conversation = [
        "我叫小明，是一名 Python 开发者",
        "我最近在学习 RAG 技术",
        "你还记得我叫什么名字吗？我在学什么？",
    ]

    history: list[str] = []
    for i, user_msg in enumerate(conversation, 1):
        print(f"\n  --- 第 {i} 轮 ---")
        print(f"  👤 用户: {user_msg}")

        # 构建带历史的 Prompt
        history_text = "\n".join(history) if history else "（无历史对话）"
        prompt = f"""你是一个有记忆的 AI 助手。请根据对话历史回答用户问题。

对话历史：
{history_text}

用户：{user_msg}
助手："""

        response = ollama.generate(model=model, prompt=prompt, options={"num_predict": 150})
        answer = response["response"].strip()
        print(f"  🤖 助手: {answer[:200]}...")

        # 更新历史
        history.append(f"用户：{user_msg}")
        history.append(f"助手：{answer[:100]}")

    print(f"\n  💡 对话历史长度: {len(history)} 条，实际生产中需要滑动窗口或摘要压缩")


# ============================================================
# 主入口
# ============================================================

def main(server_mode: bool = False) -> None:
    """运行所有 Agent 记忆演示。"""
    print("🐍 Agent 记忆 — 对话历史管理、向量记忆模拟")
    print("=" * 60)

    demo_window_buffer()
    demo_summary_buffer()
    demo_vector_memory()
    demo_working_memory()
    demo_memory_manager()

    if server_mode:
        demo_ollama_memory()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. 短期记忆管理对话历史，策略：滑动窗口 / 摘要压缩 / Token 预算")
    print("  2. 长期记忆用向量数据库存储，支持语义检索跨对话信息")
    print("  3. 工作记忆管理当前任务状态和工具调用缓存")
    print("  4. 记忆管理器统一协调三种记忆，构建完整上下文")
    print("  5. 记忆需要遗忘机制：时间衰减、重要性评分、容量淘汰")
    print("  6. 生产环境用 tiktoken 精确计算 token，用真实 embedding 模型")

    if not server_mode:
        print("\n💡 要测试真实 LLM 调用: python 05_agent_memory.py server")


if __name__ == "__main__":
    is_server = len(sys.argv) > 1 and sys.argv[1].lower() == "server"
    main(server_mode=is_server)
