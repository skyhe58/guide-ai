"""
Decoder-Only Transformer 完整实现 — 类 GPT 架构

知识点：Pre-Norm Decoder Block、SwiGLU FFN、RMSNorm、
       因果掩码、自回归生成、完整前向传播

Python 版本：3.11+
依赖：torch>=2.1, numpy>=1.26
最后验证：2024-12-01
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1. RMSNorm（现代 LLM 替代 LayerNorm）
# ============================================================

class RMSNorm(nn.Module):
    """RMSNorm — 均方根归一化。

    相比 LayerNorm 去掉了均值中心化，只做缩放，计算更快。
    LLaMA、Qwen、Mistral 等现代 LLM 标配。
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x²))
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ============================================================
# 2. SwiGLU FFN（现代 LLM 标配激活函数）
# ============================================================

class SwiGLUFFN(nn.Module):
    """SwiGLU 前馈网络。

    SwiGLU(x) = (xW₁ ⊙ Swish(xW_gate)) W₂
    相比 ReLU/GELU，SwiGLU 在大模型中表现更好。
    LLaMA、Qwen、Mistral 等均使用此结构。
    """

    def __init__(self, d_model: int, d_ff: int | None = None):
        super().__init__()
        # 默认 d_ff = 8/3 * d_model（LLaMA 风格）
        if d_ff is None:
            d_ff = int(8 / 3 * d_model)
            # 对齐到 256 的倍数（GPU 友好）
            d_ff = ((d_ff + 255) // 256) * 256

        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w_gate(x)) * self.w1(x))


# ============================================================
# 3. 多头注意力（带因果掩码）
# ============================================================

class CausalSelfAttention(nn.Module):
    """因果自注意力（Decoder-Only 专用）。"""

    def __init__(self, d_model: int, num_heads: int, max_len: int = 2048):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_QKV = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        # 预计算因果掩码
        mask = torch.tril(torch.ones(max_len, max_len))
        self.register_buffer("causal_mask", mask.view(1, 1, max_len, max_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # 一次投影得到 Q/K/V
        qkv = self.W_QKV(x)
        Q, K, V = qkv.split(C, dim=-1)

        # 分头
        Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # 注意力计算
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)

        output = weights @ V
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_O(output)


# ============================================================
# 4. Decoder Block（Pre-Norm 风格）
# ============================================================

class DecoderBlock(nn.Module):
    """Decoder Block — Pre-Norm 风格。

    结构（现代 LLM 标准）:
        x → RMSNorm → CausalSelfAttention → + → RMSNorm → SwiGLU FFN → +
        └──────────────────────────────────────┘  └──────────────────────┘
                    残差连接                              残差连接
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int | None = None):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm: 先归一化再计算（训练更稳定）
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# 5. 完整 Decoder-Only Transformer
# ============================================================

class DecoderOnlyTransformer(nn.Module):
    """Decoder-Only Transformer（类 GPT/LLaMA 架构）。

    完整结构:
        Token IDs → Embedding → [DecoderBlock × N] → RMSNorm → LM Head → logits
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        max_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model

        # Token Embedding（不使用位置编码，实际模型用 RoPE）
        self.token_emb = nn.Embedding(vocab_size, d_model)
        # 简化：使用可学习位置编码（实际模型用 RoPE）
        self.pos_emb = nn.Embedding(max_len, d_model)

        # Decoder Blocks
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads) for _ in range(num_layers)
        ])

        # 最终归一化
        self.norm = RMSNorm(d_model)

        # LM Head: 映射到词表大小
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 权重共享: Embedding 和 LM Head 共享权重（减少参数）
        self.lm_head.weight = self.token_emb.weight

        # 初始化
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            input_ids: Token ID 序列 (batch, seq_len)

        Returns:
            logits: 下一个 Token 的概率分布 (batch, seq_len, vocab_size)
        """
        B, T = input_ids.shape

        # 1. Embedding + 位置编码
        tok_emb = self.token_emb(input_ids)
        pos = torch.arange(T, device=input_ids.device)
        pos_emb = self.pos_emb(pos)
        x = tok_emb + pos_emb

        # 2. 通过所有 Decoder Block
        for block in self.blocks:
            x = block(x)

        # 3. 最终归一化 + LM Head
        x = self.norm(x)
        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """自回归生成。

        Args:
            input_ids: 初始 Token 序列 (1, seq_len)
            max_new_tokens: 最大生成 Token 数
            temperature: 温度参数（越高越随机）
            top_k: Top-K 采样

        Returns:
            生成的完整序列
        """
        self.eval()
        for _ in range(max_new_tokens):
            # 截断到最大长度
            idx_cond = input_ids[:, -512:]

            # 前向传播
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # 只取最后一个位置

            # Top-K 采样
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


# ============================================================
# 演示
# ============================================================

def demo_model() -> None:
    """演示完整 Decoder-Only Transformer。"""
    print("\n" + "=" * 60)
    print("Decoder-Only Transformer 完整实现")
    print("=" * 60)

    # 模型配置（小型演示模型）
    config = {
        "vocab_size": 1000,
        "d_model": 256,
        "num_heads": 8,
        "num_layers": 4,
        "max_len": 512,
    }

    model = DecoderOnlyTransformer(**config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  模型配置: {config}")
    print(f"  总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")

    # 前向传播
    input_ids = torch.randint(0, config["vocab_size"], (2, 10))
    logits = model(input_ids)
    print(f"\n  输入: {input_ids.shape} (batch=2, seq_len=10)")
    print(f"  输出 logits: {logits.shape} (batch=2, seq_len=10, vocab={config['vocab_size']})")

    # 自回归生成
    print("\n  --- 自回归生成 ---")
    prompt = torch.randint(0, config["vocab_size"], (1, 5))
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=50)
    print(f"  Prompt: {prompt[0].tolist()}")
    print(f"  生成序列: {generated[0].tolist()}")
    print(f"  生成长度: {generated.size(1) - prompt.size(1)} 个新 Token")

    # 模型结构概览
    print("\n  --- 模型结构 ---")
    for name, param in model.named_parameters():
        if "blocks.0" in name or "token_emb" in name or "lm_head" in name or "norm" in name:
            if "blocks.0" not in name or "blocks" in name:
                print(f"    {name}: {list(param.shape)}")


def main() -> None:
    """运行演示。"""
    print("🏗️ Decoder-Only Transformer 完整实现（类 GPT/LLaMA 架构）")
    print("=" * 60)

    demo_model()

    print("\n" + "=" * 60)
    print("✅ 演示完成！")
    print("\n💡 关键要点:")
    print("  1. Pre-Norm: RMSNorm 在注意力/FFN 之前（训练更稳定）")
    print("  2. SwiGLU: 现代 LLM 标配激活函数（优于 ReLU/GELU）")
    print("  3. 因果掩码: 只看左边，防止信息泄露")
    print("  4. 权重共享: Embedding 和 LM Head 共享权重")
    print("  5. 自回归生成: 逐 Token 生成，支持 Top-K 采样")


if __name__ == "__main__":
    main()
