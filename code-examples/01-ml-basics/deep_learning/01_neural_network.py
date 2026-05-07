"""
PyTorch 神经网络 — MLP 分类器、训练循环、评估

知识点：nn.Module 模型定义、前向传播、反向传播、
       训练循环（train/eval）、损失函数、优化器、
       Dropout 正则化、学习率调度

Python 版本：3.11+
依赖：torch>=2.1, numpy>=1.26, scikit-learn>=1.4
最后验证：2024-12-01
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 1. 模型定义
# ============================================================

class MLPClassifier(nn.Module):
    """多层感知机分类器。

    结构：输入 → 隐藏层1(ReLU+Dropout) → 隐藏层2(ReLU+Dropout) → 输出
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。"""
        return self.network(x)


# ============================================================
# 2. 数据准备
# ============================================================

def create_dataset(n_samples: int = 1000, n_features: int = 20, n_classes: int = 3):
    """生成模拟分类数据集。"""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=n_features // 2, n_classes=n_classes,
        n_clusters_per_class=1, random_state=42,
    )

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 转为 PyTorch Tensor
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    # 创建 DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    return train_loader, X_test_t, y_test_t, n_features, n_classes


# ============================================================
# 3. 训练与评估
# ============================================================

def train_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer) -> float:
    """训练一个 epoch。"""
    model.train()  # 开启训练模式（Dropout 生效）
    total_loss = 0.0
    for X_batch, y_batch in loader:
        # 前向传播
        output = model(X_batch)
        loss = criterion(output, y_batch)

        # 反向传播
        optimizer.zero_grad()  # 清零梯度（必须！）
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新参数

        total_loss += loss.item() * len(y_batch)

    return total_loss / len(loader.dataset)


@torch.no_grad()  # 推理时不计算梯度（节省内存和计算）
def evaluate(model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor) -> tuple[float, float]:
    """评估模型。"""
    model.eval()  # 开启评估模式（Dropout 关闭）
    output = model(X_test)
    loss = nn.CrossEntropyLoss()(output, y_test).item()
    predictions = output.argmax(dim=1)
    accuracy = (predictions == y_test).float().mean().item()
    return loss, accuracy


# ============================================================
# 4. 完整训练流程
# ============================================================

def demo_training() -> None:
    """完整的 PyTorch 训练流程演示。"""
    print("\n" + "=" * 60)
    print("PyTorch MLP 分类器 — 完整训练流程")
    print("=" * 60)

    # 数据准备
    train_loader, X_test, y_test, n_features, n_classes = create_dataset()
    print(f"  特征维度: {n_features}, 类别数: {n_classes}")
    print(f"  训练集: {len(train_loader.dataset)}, 测试集: {len(y_test)}")

    # 模型
    model = MLPClassifier(
        input_dim=n_features,
        hidden_dim=64,
        num_classes=n_classes,
        dropout=0.3,
    )
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    # 训练
    epochs = 30
    best_acc = 0.0
    print(f"\n  {'Epoch':>5} {'Train Loss':>11} {'Test Loss':>10} {'Test Acc':>9} {'LR':>10}")
    print(f"  {'-'*47}")

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluate(model, X_test, y_test)
        scheduler.step(test_loss)

        lr = optimizer.param_groups[0]["lr"]
        best_acc = max(best_acc, test_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  {epoch:>5} {train_loss:>11.4f} {test_loss:>10.4f} {test_acc:>8.1%} {lr:>10.6f}")

    print(f"\n  最佳测试准确率: {best_acc:.1%}")

    # 模型保存和加载
    print("\n  --- 模型保存与加载 ---")
    torch.save(model.state_dict(), "/tmp/mlp_model.pth")
    print("  ✅ 模型已保存")

    # 加载
    model_loaded = MLPClassifier(n_features, 64, n_classes)
    model_loaded.load_state_dict(torch.load("/tmp/mlp_model.pth", weights_only=True))
    _, loaded_acc = evaluate(model_loaded, X_test, y_test)
    print(f"  ✅ 模型已加载，准确率: {loaded_acc:.1%}")


# ============================================================
# 5. 激活函数对比
# ============================================================

def demo_activation_comparison() -> None:
    """对比不同激活函数的效果。"""
    print("\n" + "=" * 60)
    print("激活函数对比")
    print("=" * 60)

    train_loader, X_test, y_test, n_features, n_classes = create_dataset()

    activations = {
        "ReLU": nn.ReLU(),
        "Sigmoid": nn.Sigmoid(),
        "Tanh": nn.Tanh(),
        "GELU": nn.GELU(),
    }

    print(f"  {'激活函数':<10} {'最终准确率':>10} {'最终损失':>10}")
    print(f"  {'-'*32}")

    for name, act_fn in activations.items():
        model = nn.Sequential(
            nn.Linear(n_features, 64), act_fn, nn.Dropout(0.2),
            nn.Linear(64, 32), act_fn, nn.Dropout(0.2),
            nn.Linear(32, n_classes),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        for _ in range(20):
            train_epoch(model, train_loader, criterion, optimizer)

        loss, acc = evaluate(model, X_test, y_test)
        print(f"  {name:<10} {acc:>9.1%} {loss:>10.4f}")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行所有演示。"""
    print("🐍 PyTorch 神经网络 — MLP 分类器、训练循环、评估")
    print("=" * 60)

    demo_training()
    demo_activation_comparison()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. nn.Module 定义模型，forward() 定义前向传播")
    print("  2. 训练循环：forward → loss → backward → step → zero_grad")
    print("  3. model.train() 开启 Dropout，model.eval() 关闭")
    print("  4. torch.no_grad() 推理时节省内存")
    print("  5. 隐藏层用 ReLU/GELU，输出层不加激活（CrossEntropyLoss 内含）")


if __name__ == "__main__":
    main()
