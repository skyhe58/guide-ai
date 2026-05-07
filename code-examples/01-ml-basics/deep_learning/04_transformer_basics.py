"""
Transformer 基础实现 — 自注意力、多头注意力、位置编码

知识点：缩放点积注意力、多头注意力、正弦位置编码、
       Transformer Encoder Block、完整前向传播

Python 版本：3.11+
依赖：torch>=2.1, numpy>=1.26
最后验证：2024-12-01
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 缩放点积注意力
# ============================================================

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """缩放点积注意力。

    Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V

    Args:
        Q: 查询矩阵 (batch, ..., seq_len, d_k)
        K: 键矩阵 (batch, ..., seq_len, d_k)
        V: 值矩阵 (batch, ..., seq_len, d_v)
        mask: 注意力掩码（可选）

    Returns:
        (输出, 注意力权重)
    """
    d_k = Q.size(-1)

    # 1. Q @ K^T → 注意力分数
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)

    # 2. 应用掩码（因果掩码用于 Decoder）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # 3. Softmax 归一化
    weights = F.softmax(scores, dim=-1)

    # 4. 加权求和
    output = weights @ V

    return output, weights


def demo_attention() -> None:
    """演示缩放点积注意力。"""
    print("\n" + "=" * 60)
    print("1. 缩放点积注意力")
    print("=" * 60)

    batch_size = 1
    seq_len = 4
    d_model = 8

    # 模拟输入（4 个 Token 的表示）
    X = torch.randn(batch_size, seq_len, d_model)

    # Q, K, V 投影（简化：直接用输入）
    Q = K = V = X

    output, weights = scaled_dot_product_attention(Q, K, V)
    print(f"  输入: {X.shape}")
    print(f"  输出: {output.shape}")
    print(f"  注意力权重:\n{weights[0].detach().numpy().round(3)}")
    print(f"  每行之和: {weights[0].sum(dim=-1).detach().numpy().round(3)}")

    # 因果掩码（Decoder 用，防止看到未来 Token）
    print("\n  --- 因果掩码 ---")
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    output_masked, weights_masked = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
    print(f"  因果掩码:\n{causal_mask.numpy().astype(int)}")
    print(f"  掩码后注意力权重:\n{weights_masked[0].detach().numpy().round(3)}")


# ============================================================
# 2. 多头注意力
# ============================================================

class MultiHeadAttention(nn.Module):
    """多头注意力。

    将 Q/K/V 分成 num_heads 个头，每个头独立计算注意力，最后拼接。
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Q/K/V 投影矩阵
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)  # 输出投影

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = Q.size(0)

        # 1. 线性投影
        Q = self.W_Q(Q)  # (batch, seq, d_model)
        K = self.W_K(K)
        V = self.W_V(V)

        # 2. 分头：(batch, seq, d_model) → (batch, num_heads, seq, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 3. 每个头独立计算注意力
        output, _ = scaled_dot_product_attention(Q, K, V, mask)

        # 4. 拼接所有头：(batch, num_heads, seq, d_k) → (batch, seq, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 5. 输出投影
        return self.W_O(output)


def demo_multi_head_attention() -> None:
    """演示多头注意力。"""
    print("\n" + "=" * 60)
    print("2. 多头注意力")
    print("=" * 60)

    d_model = 64
    num_heads = 8
    seq_len = 10
    batch_size = 2

    mha = MultiHeadAttention(d_model, num_heads)
    X = torch.randn(batch_size, seq_len, d_model)

    output = mha(X, X, X)  # 自注意力：Q=K=V=X
    print(f"  d_model={d_model}, num_heads={num_heads}, d_k={d_model // num_heads}")
    print(f"  输入: {X.shape}")
    print(f"  输出: {output.shape}")
    print(f"  参数量: {sum(p.numel() for p in mha.parameters()):,}")


# ============================================================
# 3. 位置编码
# ============================================================

class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码（原始 Transformer）。"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


def demo_positional_encoding() -> None:
    """演示位置编码。"""
    print("\n" + "=" * 60)
    print("3. 正弦位置编码")
    print("=" * 60)

    d_model = 32
    seq_len = 8
    pe = SinusoidalPositionalEncoding(d_model)

    x = torch.zeros(1, seq_len, d_model)
    encoded = pe(x)

    print(f"  d_model={d_model}, seq_len={seq_len}")
    print(f"  位置 0 编码（前 8 维）: {encoded[0, 0, :8].numpy().round(3)}")
    print(f"  位置 1 编码（前 8 维）: {encoded[0, 1, :8].numpy().round(3)}")
    print(f"  位置 7 编码（前 8 维）: {encoded[0, 7, :8].numpy().round(3)}")


# ============================================================
# 4. Transformer Encoder Block
# ============================================================

class TransformerEncoderBlock(nn.Module):
    """Transformer 编码器块。

    结构：Multi-Head Attention → Add & Norm → FFN → Add & Norm
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 自注意力 + 残差连接 + LayerNorm
        attn_out = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN + 残差连接 + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


def demo_encoder_block() -> None:
    """演示 Transformer Encoder Block。"""
    print("\n" + "=" * 60)
    print("4. Transformer Encoder Block")
    print("=" * 60)

    d_model = 64
    num_heads = 8
    d_ff = 256
    seq_len = 10
    batch_size = 2

    block = TransformerEncoderBlock(d_model, num_heads, d_ff)
    pe = SinusoidalPositionalEncoding(d_model)

    # 模拟输入
    x = torch.randn(batch_size, seq_len, d_model)
    x = pe(x)  # 加位置编码

    output = block(x)
    print(f"  配置: d_model={d_model}, heads={num_heads}, d_ff={d_ff}")
    print(f"  输入: {x.shape}")
    print(f"  输出: {output.shape}")
    print(f"  参数量: {sum(p.numel() for p in block.parameters()):,}")

    # 堆叠多层
    encoder = nn.Sequential(*[
        TransformerEncoderBlock(d_model, num_heads, d_ff) for _ in range(6)
    ])
    deep_output = encoder(pe(torch.randn(batch_size, seq_len, d_model)))
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\n  6 层 Encoder:")
    print(f"    输出: {deep_output.shape}")
    print(f"    总参数量: {total_params:,}")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行所有演示。"""
    print("🐍 Transformer 基础实现 — 注意力、多头、位置编码")
    print("=" * 60)

    demo_attention()
    demo_multi_head_attention()
    demo_positional_encoding()
    demo_encoder_block()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. 自注意力: softmax(Q@K^T/√d_k) @ V")
    print("  2. 多头注意力: 分头并行计算，捕捉不同关联")
    print("  3. 位置编码: 注入位置信息（注意力本身不感知顺序）")
    print("  4. Encoder Block = MHA + Add&Norm + FFN + Add&Norm")
    print("  5. 残差连接 + LayerNorm 是训练深层网络的关键")


if __name__ == "__main__":
    main()
