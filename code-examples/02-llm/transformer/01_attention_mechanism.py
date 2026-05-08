"""
注意力机制实现 — Scaled Dot-Product / Multi-Head / GQA / KV Cache 模拟

知识点：缩放点积注意力、多头注意力、分组查询注意力（GQA）、
       KV Cache 推理加速模拟

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
# 1. 缩放点积注意力（Scaled Dot-Product Attention）
# ============================================================

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """缩放点积注意力。

    公式: Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V

    Args:
        Q: 查询矩阵 (batch, ..., seq_len, d_k)
        K: 键矩阵   (batch, ..., kv_len, d_k)
        V: 值矩阵   (batch, ..., kv_len, d_v)
        mask: 注意力掩码，0 表示屏蔽位置

    Returns:
        (输出张量, 注意力权重)
    """
    d_k = Q.size(-1)

    # 1. 计算注意力分数：Q @ K^T
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)

    # 2. 应用掩码（因果掩码用于 Decoder-Only 模型）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # 3. Softmax 归一化为概率分布
    weights = F.softmax(scores, dim=-1)

    # 4. 加权求和得到输出
    output = weights @ V

    return output, weights


def demo_scaled_attention() -> None:
    """演示缩放点积注意力。"""
    print("\n" + "=" * 60)
    print("1. 缩放点积注意力 (Scaled Dot-Product Attention)")
    print("=" * 60)

    batch_size, seq_len, d_model = 1, 4, 8
    X = torch.randn(batch_size, seq_len, d_model)
    Q = K = V = X  # 自注意力：Q=K=V

    output, weights = scaled_dot_product_attention(Q, K, V)
    print(f"  输入形状: {X.shape}")
    print(f"  输出形状: {output.shape}")
    print(f"  注意力权重:\n{weights[0].detach().numpy().round(3)}")
    print(f"  每行权重之和: {weights[0].sum(dim=-1).detach().numpy().round(3)}")

    # 因果掩码（Decoder-Only 模型使用，防止看到未来 Token）
    print("\n  --- 因果掩码（Causal Mask）---")
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    output_masked, weights_masked = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
    print(f"  因果掩码:\n{causal_mask.numpy().astype(int)}")
    print(f"  掩码后注意力权重:\n{weights_masked[0].detach().numpy().round(3)}")


# ============================================================
# 2. 多头注意力（Multi-Head Attention）
# ============================================================

class MultiHeadAttention(nn.Module):
    """标准多头注意力（MHA）。

    将 Q/K/V 分成 num_heads 个头，每个头独立计算注意力后拼接。
    这是 GPT-2/GPT-3 使用的标准注意力。
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Q/K/V/O 投影矩阵
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = Q.size(0)

        # 1. 线性投影
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)

        # 2. 分头: (batch, seq, d_model) → (batch, num_heads, seq, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 3. 每个头独立计算注意力
        output, _ = scaled_dot_product_attention(Q, K, V, mask)

        # 4. 拼接所有头: (batch, num_heads, seq, d_k) → (batch, seq, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 5. 输出投影
        return self.W_O(self.dropout(output))


def demo_multi_head_attention() -> None:
    """演示多头注意力。"""
    print("\n" + "=" * 60)
    print("2. 多头注意力 (Multi-Head Attention)")
    print("=" * 60)

    d_model, num_heads = 64, 8
    seq_len, batch_size = 10, 2

    mha = MultiHeadAttention(d_model, num_heads)
    X = torch.randn(batch_size, seq_len, d_model)

    output = mha(X, X, X)  # 自注意力
    print(f"  配置: d_model={d_model}, num_heads={num_heads}, d_k={d_model // num_heads}")
    print(f"  输入: {X.shape}")
    print(f"  输出: {output.shape}")
    print(f"  参数量: {sum(p.numel() for p in mha.parameters()):,}")


# ============================================================
# 3. KV Cache 模拟（推理加速核心技术）
# ============================================================

class KVCacheAttention(nn.Module):
    """带 KV Cache 的注意力（推理加速）。

    原理：自回归生成时，已生成 Token 的 K/V 不变，
    缓存它们避免重复计算，每步只需计算新 Token 的 Q/K/V。

    无 KV Cache: 每步计算 O(n²) 的注意力
    有 KV Cache: 每步只计算 O(n) 的注意力（新 Token vs 所有历史）
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        # KV Cache
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None

    def reset_cache(self) -> None:
        """重置 KV Cache（新对话时调用）。"""
        self.k_cache = None
        self.v_cache = None

    def forward(self, x: torch.Tensor, use_cache: bool = True) -> torch.Tensor:
        batch_size = x.size(0)

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # 分头
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        if use_cache:
            # 拼接历史 K/V
            if self.k_cache is not None:
                K = torch.cat([self.k_cache, K], dim=2)
                V = torch.cat([self.v_cache, V], dim=2)
            # 更新缓存
            self.k_cache = K.detach()
            self.v_cache = V.detach()

        # 计算注意力
        output, _ = scaled_dot_product_attention(Q, K, V)

        # 合并头
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_O(output)


def demo_kv_cache() -> None:
    """演示 KV Cache 推理加速。"""
    print("\n" + "=" * 60)
    print("3. KV Cache 推理加速模拟")
    print("=" * 60)

    d_model, num_heads = 32, 4
    kv_attn = KVCacheAttention(d_model, num_heads)
    kv_attn.eval()

    # 模拟自回归生成：逐 Token 生成
    print("  模拟自回归生成（每步只输入 1 个新 Token）:")
    kv_attn.reset_cache()

    for step in range(5):
        # 每步只输入 1 个新 Token
        new_token = torch.randn(1, 1, d_model)
        with torch.no_grad():
            output = kv_attn(new_token, use_cache=True)
        cache_len = kv_attn.k_cache.size(2) if kv_attn.k_cache is not None else 0
        print(f"    Step {step}: 输入 1 Token, KV Cache 长度={cache_len}, 输出={output.shape}")

    print("\n  💡 KV Cache 优势:")
    print("    - 无 Cache: 每步重新计算所有 Token 的 K/V → O(n²)")
    print("    - 有 Cache: 每步只计算新 Token 的 K/V → O(n)")
    print("    - 生成 1000 Token 时，加速约 500 倍")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行所有注意力机制演示。"""
    print("🧠 注意力机制实现 — Scaled Dot-Product / Multi-Head / KV Cache")
    print("=" * 60)

    demo_scaled_attention()
    demo_multi_head_attention()
    demo_kv_cache()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. Scaled Dot-Product: softmax(Q@K^T/√d_k) @ V")
    print("  2. Multi-Head: 分头并行计算，捕捉不同类型关联")
    print("  3. KV Cache: 缓存已生成 Token 的 K/V，避免重复计算")
    print("  4. GQA: 多个 Q 头共享 K/V 头，减少显存占用")


if __name__ == "__main__":
    main()
