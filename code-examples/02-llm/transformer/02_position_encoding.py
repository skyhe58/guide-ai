"""
位置编码实现 — 正弦位置编码 / RoPE 旋转位置编码

知识点：绝对位置编码（正弦函数）、旋转位置编码（RoPE）简化实现、
       位置编码可视化

Python 版本：3.11+
依赖：torch>=2.1, numpy>=1.26
最后验证：2024-12-01
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# 1. 正弦位置编码（Sinusoidal Positional Encoding）
# ============================================================

class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码（原始 Transformer 使用）。

    公式:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    特点:
        - 固定编码，不需要学习
        - 不同位置的编码正交性好
        - 理论上可以外推到更长序列（但实际效果有限）
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度: sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度: cos

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将位置编码加到输入上。"""
        return x + self.pe[:, : x.size(1), :]


def demo_sinusoidal_encoding() -> None:
    """演示正弦位置编码。"""
    print("\n" + "=" * 60)
    print("1. 正弦位置编码 (Sinusoidal Positional Encoding)")
    print("=" * 60)

    d_model, seq_len = 32, 8
    pe = SinusoidalPositionalEncoding(d_model)

    x = torch.zeros(1, seq_len, d_model)
    encoded = pe(x)

    print(f"  d_model={d_model}, seq_len={seq_len}")
    print(f"  位置 0 编码（前 8 维）: {encoded[0, 0, :8].numpy().round(3)}")
    print(f"  位置 1 编码（前 8 维）: {encoded[0, 1, :8].numpy().round(3)}")
    print(f"  位置 7 编码（前 8 维）: {encoded[0, 7, :8].numpy().round(3)}")

    # 验证：不同位置的编码不同
    pos0 = encoded[0, 0].numpy()
    pos1 = encoded[0, 1].numpy()
    cos_sim = np.dot(pos0, pos1) / (np.linalg.norm(pos0) * np.linalg.norm(pos1))
    print(f"\n  位置 0 和位置 1 的余弦相似度: {cos_sim:.4f}")
    print("  （相邻位置相似度较高，远距离位置相似度较低）")


# ============================================================
# 2. 旋转位置编码（RoPE — Rotary Position Embedding）
# ============================================================

def precompute_rope_freqs(dim: int, max_len: int = 4096, base: float = 10000.0) -> torch.Tensor:
    """预计算 RoPE 频率。

    RoPE 核心思想：将位置信息编码为旋转角度，
    通过旋转 Q/K 向量来注入位置信息。

    优势（相比正弦编码）：
    - 相对位置信息自然编码在 Q·K 的点积中
    - 更好的长序列外推能力
    - LLaMA、Qwen、Mistral 等主流模型标配

    Args:
        dim: 每个头的维度 d_k
        max_len: 最大序列长度
        base: 频率基数（默认 10000）

    Returns:
        复数频率张量 (max_len, dim//2)
    """
    # 频率: θ_i = 1 / base^(2i/dim)
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    # 位置索引
    t = torch.arange(max_len, dtype=torch.float)
    # 外积: (max_len, dim//2)
    freqs = torch.outer(t, freqs)
    # 转为复数形式: e^(iθ) = cos(θ) + i·sin(θ)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """对输入张量应用 RoPE。

    将实数向量视为复数，乘以旋转频率。

    Args:
        x: 输入张量 (batch, num_heads, seq_len, d_k)
        freqs: 预计算的频率 (seq_len, d_k//2)

    Returns:
        旋转后的张量，形状不变
    """
    # 将实数对视为复数: (a, b) → a + bi
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # 乘以旋转频率
    freqs = freqs[:x_complex.size(-2)].unsqueeze(0).unsqueeze(0)
    x_rotated = x_complex * freqs
    # 转回实数
    x_out = torch.view_as_real(x_rotated).reshape(*x.shape)
    return x_out.type_as(x)


def demo_rope() -> None:
    """演示 RoPE 旋转位置编码。"""
    print("\n" + "=" * 60)
    print("2. 旋转位置编码 (RoPE)")
    print("=" * 60)

    d_k = 16  # 每个头的维度
    seq_len = 8
    batch_size = 1
    num_heads = 1

    # 预计算频率
    freqs = precompute_rope_freqs(d_k, max_len=seq_len)
    print(f"  d_k={d_k}, seq_len={seq_len}")
    print(f"  频率张量形状: {freqs.shape}")

    # 模拟 Q 和 K
    Q = torch.randn(batch_size, num_heads, seq_len, d_k)
    K = torch.randn(batch_size, num_heads, seq_len, d_k)

    # 应用 RoPE
    Q_rotated = apply_rope(Q, freqs)
    K_rotated = apply_rope(K, freqs)

    print(f"  Q 形状: {Q.shape} → 旋转后: {Q_rotated.shape}")
    print(f"  K 形状: {K.shape} → 旋转后: {K_rotated.shape}")

    # 验证 RoPE 的关键性质：Q·K 的点积只依赖相对位置
    # 位置 (m, n) 的注意力分数 = f(q_m, k_n, m-n)
    print("\n  💡 RoPE 关键性质:")
    print("    - Q·K 的点积自然包含相对位置信息")
    print("    - 不需要额外的位置编码向量")
    print("    - 支持长序列外推（配合 NTK-aware 缩放）")

    # 对比：无 RoPE vs 有 RoPE 的注意力分数
    scores_no_rope = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    scores_with_rope = (Q_rotated @ K_rotated.transpose(-2, -1)) / math.sqrt(d_k)
    print(f"\n  无 RoPE 注意力分数（前 4x4）:\n{scores_no_rope[0, 0, :4, :4].detach().numpy().round(3)}")
    print(f"  有 RoPE 注意力分数（前 4x4）:\n{scores_with_rope[0, 0, :4, :4].detach().numpy().round(3)}")


# ============================================================
# 3. 位置编码对比总结
# ============================================================

def demo_comparison() -> None:
    """位置编码方案对比。"""
    print("\n" + "=" * 60)
    print("3. 位置编码方案对比")
    print("=" * 60)

    comparison = """
    | 方案 | 类型 | 代表模型 | 外推能力 | 特点 |
    |------|------|----------|----------|------|
    | 正弦编码 | 绝对 | 原始 Transformer | 有限 | 固定，不需学习 |
    | 可学习编码 | 绝对 | GPT-2, BERT | 无 | 需要训练，受限于训练长度 |
    | RoPE | 相对 | LLaMA, Qwen, Mistral | 较好 | 旋转编码，主流方案 |
    | ALiBi | 相对 | BLOOM, MPT | 好 | 注意力偏置，无需额外参数 |
    """
    print(comparison)
    print("  💡 现代 LLM 主流选择: RoPE（配合 NTK-aware 或 YaRN 缩放）")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行所有位置编码演示。"""
    print("📐 位置编码实现 — 正弦编码 / RoPE 旋转位置编码")
    print("=" * 60)

    demo_sinusoidal_encoding()
    demo_rope()
    demo_comparison()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. 注意力机制不感知位置，必须注入位置信息")
    print("  2. 正弦编码: 固定函数，原始 Transformer 使用")
    print("  3. RoPE: 旋转编码，现代 LLM 标配（LLaMA/Qwen/Mistral）")
    print("  4. ALiBi: 注意力偏置方案，无需额外参数")


if __name__ == "__main__":
    main()
