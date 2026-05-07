"""
智能客服系统 — RAG + Agent + 对话管理

知识点：完整的智能客服系统实现，包括意图识别、知识库检索（RAG）、
       工具调用（Agent）、对话历史管理、多轮对话、工单创建

Python 版本：3.11+
依赖：标准库（默认模式）、fastapi + uvicorn（API 模式）
最后验证：2024-12-01
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


# ============================================================
# 1. 数据模型
# ============================================================

class Intent(Enum):
    """用户意图类型。"""
    GREETING = "greeting"
    FAQ = "faq"
    PRODUCT = "product"
    COMPLAINT = "complaint"
    TICKET = "ticket"
    TRANSFER = "transfer"
    GOODBYE = "goodbye"
    UNKNOWN = "unknown"


@dataclass
class Message:
    """对话消息。"""
    role: str  # user / assistant / system
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """对话会话。"""
    session_id: str = ""
    user_id: str = ""
    messages: list[Message] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.session_id:
            self.session_id = str(uuid.uuid4())[:8]

    def add_message(self, role: str, content: str, **metadata) -> None:
        self.messages.append(Message(role, content, metadata=metadata))

    def get_history(self, max_turns: int = 10) -> list[dict]:
        recent = self.messages[-max_turns * 2:] if len(self.messages) > max_turns * 2 else self.messages
        return [{"role": m.role, "content": m.content} for m in recent]


@dataclass
class Ticket:
    """工单。"""
    ticket_id: str = ""
    user_id: str = ""
    title: str = ""
    description: str = ""
    priority: str = "normal"
    status: str = "open"
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.ticket_id:
            self.ticket_id = f"TK-{str(uuid.uuid4())[:6].upper()}"


# ============================================================
# 2. 意图识别
# ============================================================

class IntentClassifier:
    """意图识别器 — 基于关键词匹配。"""

    INTENT_KEYWORDS: dict[Intent, list[str]] = {
        Intent.GREETING: ["你好", "您好", "hi", "hello", "早上好", "下午好"],
        Intent.FAQ: ["怎么", "如何", "什么是", "为什么", "能不能", "可以吗", "帮我查"],
        Intent.PRODUCT: ["产品", "功能", "价格", "版本", "升级", "套餐"],
        Intent.COMPLAINT: ["投诉", "不满", "差评", "退款", "问题", "故障", "bug"],
        Intent.TICKET: ["工单", "提交", "反馈", "报修", "申请"],
        Intent.TRANSFER: ["转人工", "人工客服", "真人", "转接"],
        Intent.GOODBYE: ["再见", "拜拜", "谢谢", "感谢", "bye"],
    }

    def classify(self, text: str) -> Intent:
        """识别用户意图。"""
        text_lower = text.lower()
        scores: dict[Intent, int] = {}
        for intent, keywords in self.INTENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[intent] = score
        if scores:
            return max(scores, key=scores.get)
        return Intent.UNKNOWN


# ============================================================
# 3. 知识库（RAG）
# ============================================================

@dataclass
class KBArticle:
    """知识库文章。"""
    title: str
    content: str
    category: str
    embedding: list[float] = field(default_factory=list)


class KnowledgeBase:
    """知识库 — 简化的 RAG 检索。"""

    def __init__(self):
        self.articles: list[KBArticle] = []
        self._load_articles()

    def _load_articles(self) -> None:
        """加载知识库文章。"""
        articles_data = [
            ("如何重置密码", "访问登录页面，点击'忘记密码'，输入注册邮箱，系统会发送重置链接到您的邮箱。点击链接后设置新密码即可。密码要求：8 位以上，包含大小写字母和数字。", "账户"),
            ("如何申请退款", "在订单详情页点击'申请退款'，选择退款原因，提交申请。退款将在 3-5 个工作日内原路返回。注意：超过 30 天的订单不支持退款。", "订单"),
            ("产品功能介绍", "我们的智能客服系统支持：多轮对话、意图识别、知识库问答、工单管理、数据分析。支持接入微信、网页、APP 等多个渠道。", "产品"),
            ("VIP 套餐说明", "VIP 套餐分为基础版（99 元/月）、专业版（299 元/月）和企业版（999 元/月）。基础版支持 1000 次/天对话，专业版 10000 次/天，企业版不限量。", "产品"),
            ("API 接入指南", "1. 注册开发者账号 2. 获取 API Key 3. 阅读 API 文档 4. 调用 /api/chat 接口 5. 处理返回结果。详细文档：docs.example.com/api", "技术"),
            ("常见故障排查", "Q: 系统响应慢？A: 检查网络连接，清除浏览器缓存。Q: 无法登录？A: 确认账号密码正确，检查是否被锁定。Q: 消息发送失败？A: 检查消息长度是否超过 4000 字符限制。", "技术"),
            ("隐私政策", "我们严格保护用户隐私。用户数据加密存储，不会分享给第三方。用户可以随时申请删除个人数据。详见隐私政策页面。", "法务"),
            ("联系方式", "客服热线：400-123-4567（工作日 9:00-18:00）。邮箱：support@example.com。在线客服：网站右下角聊天窗口。", "联系"),
        ]
        for title, content, category in articles_data:
            emb = self._simple_embedding(title + " " + content)
            self.articles.append(KBArticle(title, content, category, emb))

    def _simple_embedding(self, text: str, dim: int = 8) -> list[float]:
        h = int(hashlib.md5(text.encode()).hexdigest(), 16)
        vec = [((h >> (i * 4)) & 0xF) / 15.0 - 0.5 for i in range(dim)]
        norm = math.sqrt(sum(v * v for v in vec))
        return [round(v / norm, 4) for v in vec] if norm > 0 else vec

    def search(self, query: str, top_k: int = 3) -> list[tuple[KBArticle, float]]:
        """检索相关文章。"""
        query_emb = self._simple_embedding(query)
        scored = []
        for article in self.articles:
            # 向量相似度 + 关键词匹配
            vec_sim = sum(a * b for a, b in zip(query_emb, article.embedding))
            kw_sim = sum(1 for w in query.split() if w in article.title or w in article.content)
            score = vec_sim * 0.3 + kw_sim * 0.7
            scored.append((article, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ============================================================
# 4. 工具集
# ============================================================

class ToolRegistry:
    """工具注册中心。"""

    def __init__(self):
        self.tools: dict[str, Callable] = {}
        self.tickets: list[Ticket] = []
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        self.tools["create_ticket"] = self._create_ticket
        self.tools["check_order"] = self._check_order
        self.tools["transfer_human"] = self._transfer_human

    def _create_ticket(self, title: str, description: str, user_id: str = "") -> str:
        ticket = Ticket(user_id=user_id, title=title, description=description)
        self.tickets.append(ticket)
        return f"工单已创建，编号：{ticket.ticket_id}。我们会在 24 小时内处理。"

    def _check_order(self, order_id: str = "", **kwargs) -> str:
        return f"订单 {order_id or 'ORD-001'} 状态：已发货，预计 3 天内送达。"

    def _transfer_human(self, **kwargs) -> str:
        return "正在为您转接人工客服，请稍候...当前排队人数：3 人，预计等待 2 分钟。"

    def execute(self, tool_name: str, **kwargs) -> str:
        if tool_name in self.tools:
            return self.tools[tool_name](**kwargs)
        return f"未找到工具：{tool_name}"


# ============================================================
# 5. 对话管理器
# ============================================================

class ConversationManager:
    """对话管理器 — 管理多个会话。"""

    def __init__(self):
        self.sessions: dict[str, Session] = {}

    def get_or_create_session(self, session_id: str = "", user_id: str = "") -> Session:
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        session = Session(session_id=session_id, user_id=user_id)
        self.sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)


# ============================================================
# 6. 智能客服引擎
# ============================================================

class CustomerServiceEngine:
    """智能客服引擎 — 核心处理逻辑。"""

    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.knowledge_base = KnowledgeBase()
        self.tools = ToolRegistry()
        self.conversation_mgr = ConversationManager()

    def chat(self, message: str, session_id: str = "", user_id: str = "") -> dict:
        """处理用户消息。"""
        start_time = time.time()

        # 获取或创建会话
        session = self.conversation_mgr.get_or_create_session(session_id, user_id)
        session.add_message("user", message)

        # 意图识别
        intent = self.intent_classifier.classify(message)

        # 根据意图路由处理
        response = self._route_by_intent(intent, message, session)

        # 记录回复
        session.add_message("assistant", response["answer"], intent=intent.value)

        latency = (time.time() - start_time) * 1000
        return {
            "session_id": session.session_id,
            "answer": response["answer"],
            "intent": intent.value,
            "sources": response.get("sources", []),
            "confidence": response.get("confidence", 0.8),
            "latency_ms": round(latency, 2),
        }

    def _route_by_intent(self, intent: Intent, message: str, session: Session) -> dict:
        """根据意图路由到不同处理逻辑。"""
        if intent == Intent.GREETING:
            return {"answer": "您好！我是智能客服助手，请问有什么可以帮您的？", "confidence": 1.0}

        elif intent == Intent.GOODBYE:
            return {"answer": "感谢您的咨询，祝您生活愉快！如有其他问题随时联系我们。", "confidence": 1.0}

        elif intent == Intent.TRANSFER:
            result = self.tools.execute("transfer_human")
            return {"answer": result, "confidence": 1.0}

        elif intent == Intent.TICKET:
            result = self.tools.execute("create_ticket", title=message[:30], description=message, user_id=session.user_id)
            return {"answer": result, "confidence": 0.9}

        elif intent in (Intent.FAQ, Intent.PRODUCT, Intent.COMPLAINT, Intent.UNKNOWN):
            # RAG 检索
            results = self.knowledge_base.search(message, top_k=2)
            if results and results[0][1] > 0:
                article, score = results[0]
                answer = f"{article.content}"
                sources = [{"title": a.title, "category": a.category} for a, _ in results[:2]]
                return {"answer": answer, "sources": sources, "confidence": min(score / 3, 0.95)}
            return {"answer": "抱歉，我暂时无法回答这个问题。建议您联系人工客服获取帮助。", "confidence": 0.3}

        return {"answer": "请问您能详细描述一下您的问题吗？", "confidence": 0.5}


# ============================================================
# 演示
# ============================================================

def demo_basic_chat() -> None:
    """演示基础对话。"""
    print("\n" + "=" * 60)
    print("1. 基础对话演示")
    print("=" * 60)

    engine = CustomerServiceEngine()
    conversations = [
        "你好",
        "如何重置密码？",
        "你们有什么套餐？",
        "我要投诉，系统太慢了",
        "帮我创建一个工单",
        "转人工客服",
        "谢谢，再见",
    ]

    session_id = ""
    for msg in conversations:
        result = engine.chat(msg, session_id=session_id)
        session_id = result["session_id"]
        print(f"  👤 用户: {msg}")
        print(f"  🤖 客服: {result['answer'][:80]}...")
        print(f"     意图: {result['intent']} | 置信度: {result['confidence']:.2f} | 延迟: {result['latency_ms']:.0f}ms")
        print()


def demo_multi_turn() -> None:
    """演示多轮对话。"""
    print("\n" + "=" * 60)
    print("2. 多轮对话演示")
    print("=" * 60)

    engine = CustomerServiceEngine()
    session_id = ""

    turns = [
        "你好，我想了解一下你们的产品",
        "有哪些套餐可以选择？",
        "API 怎么接入？",
        "好的，谢谢",
    ]

    for msg in turns:
        result = engine.chat(msg, session_id=session_id)
        session_id = result["session_id"]
        print(f"  👤: {msg}")
        print(f"  🤖: {result['answer'][:80]}...")
        print()

    # 查看会话历史
    session = engine.conversation_mgr.get_session(session_id)
    if session:
        print(f"  📝 会话历史: {len(session.messages)} 条消息")


def demo_tool_usage() -> None:
    """演示工具调用。"""
    print("\n" + "=" * 60)
    print("3. 工具调用演示")
    print("=" * 60)

    engine = CustomerServiceEngine()

    # 创建工单
    result = engine.chat("帮我提交一个工单，系统登录异常")
    print(f"  👤: 帮我提交一个工单，系统登录异常")
    print(f"  🤖: {result['answer']}")
    print(f"  📋 工单数: {len(engine.tools.tickets)}")

    # 转人工
    result = engine.chat("转人工客服")
    print(f"\n  👤: 转人工客服")
    print(f"  🤖: {result['answer']}")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行智能客服系统演示。"""
    print("智能客服系统 — RAG + Agent + 对话管理")
    print("=" * 60)

    demo_basic_chat()
    demo_multi_turn()
    demo_tool_usage()

    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("\n关键要点:")
    print("  1. 意图识别是客服系统的第一步，决定后续处理逻辑")
    print("  2. RAG 检索知识库回答常见问题，减少人工压力")
    print("  3. 工具调用（创建工单、转人工）处理操作类请求")
    print("  4. 对话管理维护会话上下文，支持多轮对话")
    print("  5. 置信度评分帮助决定是否需要转人工")
    print("  6. 生产环境需要增加：用户认证、限流、监控、日志")


if __name__ == "__main__":
    main()
