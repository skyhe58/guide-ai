"""
RNN/LSTM 序列建模 — 文本分类示例

知识点：nn.RNN/nn.LSTM/nn.GRU 使用、序列数据处理、
       隐藏状态传递、文本分类模型、RNN vs LSTM 对比

Python 版本：3.11+
依赖：torch>=2.1, numpy>=1.26
最后验证：2024-12-01
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ============================================================
# 1. RNN/LSTM/GRU 基础用法
# ============================================================

def demo_rnn_basics() -> None:
    """演示 PyTorch RNN/LSTM/GRU 的基本用法。"""
    print("\n" + "=" * 60)
    print("1. RNN/LSTM/GRU 基础用法")
    print("=" * 60)

    batch_size = 2
    seq_len = 10
    input_size = 128   # 输入特征维度（如 Embedding 维度）
    hidden_size = 256   # 隐藏状态维度

    # 模拟输入：(batch, seq_len, input_size)
    x = torch.randn(batch_size, seq_len, input_size)

    # --- RNN ---
    rnn = nn.RNN(input_size, hidden_size, num_layers=1, batch_first=True)
    rnn_out, rnn_hidden = rnn(x)
    print(f"  RNN:")
    print(f"    输入: {x.shape}")
    print(f"    输出: {rnn_out.shape}  (所有时间步的隐藏状态)")
    print(f"    最终隐藏: {rnn_hidden.shape}  (最后一个时间步)")

    # --- LSTM ---
    lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
    lstm_out, (lstm_h, lstm_c) = lstm(x)
    print(f"\n  LSTM (2 层):")
    print(f"    输出: {lstm_out.shape}")
    print(f"    隐藏状态 h: {lstm_h.shape}  (num_layers, batch, hidden)")
    print(f"    细胞状态 c: {lstm_c.shape}  (LSTM 特有)")

    # --- GRU ---
    gru = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True)
    gru_out, gru_hidden = gru(x)
    print(f"\n  GRU (2 层):")
    print(f"    输出: {gru_out.shape}")
    print(f"    隐藏状态: {gru_hidden.shape}  (无细胞状态，比 LSTM 更轻量)")

    # 参数量对比
    rnn_params = sum(p.numel() for p in rnn.parameters())
    lstm_params = sum(p.numel() for p in lstm.parameters())
    gru_params = sum(p.numel() for p in gru.parameters())
    print(f"\n  参数量对比:")
    print(f"    RNN (1层):  {rnn_params:>8,}")
    print(f"    LSTM (2层): {lstm_params:>8,}")
    print(f"    GRU (2层):  {gru_params:>8,}")


# ============================================================
# 2. LSTM 文本分类模型
# ============================================================

class LSTMTextClassifier(nn.Module):
    """基于 LSTM 的文本分类模型。

    结构：Embedding → LSTM → 取最后时间步 → 全连接 → 分类
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) — Token ID 序列
        embedded = self.embedding(x)           # (batch, seq_len, embed_dim)
        lstm_out, (h_n, _) = self.lstm(embedded)  # h_n: (1, batch, hidden_dim)
        # 取最后一层最后一个时间步的隐藏状态
        last_hidden = h_n[-1]                  # (batch, hidden_dim)
        return self.classifier(last_hidden)    # (batch, num_classes)


def demo_text_classification() -> None:
    """演示 LSTM 文本分类。"""
    print("\n" + "=" * 60)
    print("2. LSTM 文本分类模型")
    print("=" * 60)

    # 模拟参数
    vocab_size = 5000
    embed_dim = 128
    hidden_dim = 256
    num_classes = 3  # 正面/中性/负面
    batch_size = 8
    seq_len = 50

    # 模型
    model = LSTMTextClassifier(vocab_size, embed_dim, hidden_dim, num_classes)
    params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {params:,}")

    # 模拟输入（Token ID 序列）
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    y = torch.randint(0, num_classes, (batch_size,))

    # 前向传播
    output = model(x)
    print(f"  输入: {x.shape} (batch, seq_len)")
    print(f"  输出: {output.shape} (batch, num_classes)")

    # 训练一步
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print(f"  损失: {loss.item():.4f}")

    # 预测
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x[:1]), dim=1)
        pred = probs.argmax(dim=1).item()
        labels = ["负面", "中性", "正面"]
        print(f"  预测: {labels[pred]} (概率: {probs[0].tolist()})")


# ============================================================
# 3. 梯度消失演示
# ============================================================

def demo_gradient_vanishing() -> None:
    """演示 RNN 的梯度消失问题。"""
    print("\n" + "=" * 60)
    print("3. 梯度消失演示 — RNN vs LSTM")
    print("=" * 60)

    input_size = 32
    hidden_size = 64
    seq_lengths = [10, 50, 100, 200]

    for seq_len in seq_lengths:
        x = torch.randn(1, seq_len, input_size, requires_grad=True)

        # RNN
        rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        rnn_out, _ = rnn(x)
        rnn_loss = rnn_out[:, -1, :].sum()  # 只看最后一个时间步
        rnn_loss.backward()
        rnn_grad = x.grad.abs().mean().item() if x.grad is not None else 0
        x.grad = None

        # LSTM
        lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        lstm_out, _ = lstm(x)
        lstm_loss = lstm_out[:, -1, :].sum()
        lstm_loss.backward()
        lstm_grad = x.grad.abs().mean().item() if x.grad is not None else 0
        x.grad = None

        print(f"  序列长度={seq_len:>3}: RNN 梯度={rnn_grad:.6f}, LSTM 梯度={lstm_grad:.6f}")

    print("\n  💡 序列越长，RNN 梯度衰减越严重，LSTM 相对稳定")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行所有演示。"""
    print("🐍 RNN/LSTM 序列建模 — 文本分类示例")
    print("=" * 60)

    demo_rnn_basics()
    demo_text_classification()
    demo_gradient_vanishing()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. RNN 通过隐藏状态在时间步间传递信息")
    print("  2. LSTM 用门控机制解决梯度消失，GRU 是简化版")
    print("  3. 文本分类：Embedding → LSTM → 取最后隐藏状态 → 分类")
    print("  4. RNN/LSTM 已被 Transformer 取代，但理解原理很重要")
    print("  5. 面试重点：LSTM 门控机制、为什么被 Transformer 取代")


if __name__ == "__main__":
    main()
