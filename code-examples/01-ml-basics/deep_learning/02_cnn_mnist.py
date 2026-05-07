"""
CNN 手写数字识别 — MNIST 分类

知识点：CNN 模型定义（Conv2d/MaxPool2d/Flatten）、
       MNIST 数据集加载、训练循环、模型评估、
       卷积特征可视化

Python 版本：3.11+
依赖：torch>=2.1, torchvision>=0.16
最后验证：2024-12-01
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ============================================================
# 1. CNN 模型定义
# ============================================================

class MNISTClassifier(nn.Module):
    """MNIST 手写数字分类 CNN。

    结构：
    - Conv(1→32, 3×3) → ReLU → MaxPool(2×2)  → 输出 32×14×14
    - Conv(32→64, 3×3) → ReLU → MaxPool(2×2)  → 输出 64×7×7
    - Flatten → Linear(3136→128) → ReLU → Dropout → Linear(128→10)
    """

    def __init__(self):
        super().__init__()
        # 特征提取器（卷积层）
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # (1, 28, 28) → (32, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2),                               # (32, 28, 28) → (32, 14, 14)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # (32, 14, 14) → (64, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),                               # (64, 14, 14) → (64, 7, 7)
        )
        # 分类器（全连接层）
        self.classifier = nn.Sequential(
            nn.Flatten(),                                  # (64, 7, 7) → (3136,)
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),                            # 10 个数字类别
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# 2. 数据加载
# ============================================================

def load_mnist(batch_size: int = 64) -> tuple[DataLoader, DataLoader]:
    """加载 MNIST 数据集。

    首次运行会自动下载（约 12MB）。
    """
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),           # PIL → Tensor, 值域 [0, 1]
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST 均值和标准差
    ])

    train_dataset = datasets.MNIST(
        root="/tmp/mnist_data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="/tmp/mnist_data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# ============================================================
# 3. 训练与评估
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    """训练一个 epoch。"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        output = model(images)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct += (output.argmax(1) == labels).sum().item()
        total += len(labels)

    return correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    """评估模型。"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        output = model(images)
        total_loss += criterion(output, labels).item() * len(labels)
        correct += (output.argmax(1) == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


# ============================================================
# 4. 完整训练流程
# ============================================================

def demo_mnist_training() -> None:
    """MNIST CNN 完整训练流程。"""
    print("\n" + "=" * 60)
    print("CNN MNIST 手写数字识别")
    print("=" * 60)

    # 加载数据
    print("  📥 加载 MNIST 数据集...")
    train_loader, test_loader = load_mnist(batch_size=64)
    print(f"  训练集: {len(train_loader.dataset)}, 测试集: {len(test_loader.dataset)}")

    # 模型
    model = MNISTClassifier()
    params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {params:,}")

    # 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 5  # MNIST 5 个 epoch 就能达到 99%+

    # 训练
    print(f"\n  {'Epoch':>5} {'Train Acc':>10} {'Test Loss':>10} {'Test Acc':>10}")
    print(f"  {'-'*37}")

    for epoch in range(1, epochs + 1):
        train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluate(model, test_loader)
        print(f"  {epoch:>5} {train_acc:>9.1%} {test_loss:>10.4f} {test_acc:>9.1%}")

    print(f"\n  🎯 最终测试准确率: {test_acc:.1%}")

    # 查看模型结构
    print(f"\n  模型结构:")
    print(f"  {model}")


# ============================================================
# 5. 模型结构分析
# ============================================================

def demo_model_analysis() -> None:
    """分析 CNN 各层的输出形状。"""
    print("\n" + "=" * 60)
    print("CNN 各层输出形状分析")
    print("=" * 60)

    model = MNISTClassifier()
    x = torch.randn(1, 1, 28, 28)  # 单张 MNIST 图像

    print(f"  输入: {x.shape}")

    # 逐层查看
    for i, layer in enumerate(model.features):
        x = layer(x)
        print(f"  {layer.__class__.__name__:>12}: {x.shape}")

    x = model.classifier[0](x)  # Flatten
    print(f"  {'Flatten':>12}: {x.shape}")

    for layer in model.classifier[1:]:
        x = layer(x)
        print(f"  {layer.__class__.__name__:>12}: {x.shape}")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行所有演示。"""
    print("🐍 CNN 手写数字识别 — MNIST 分类")
    print("=" * 60)

    demo_model_analysis()
    demo_mnist_training()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. CNN = 卷积层（特征提取）+ 池化层（降维）+ 全连接层（分类）")
    print("  2. 3×3 卷积核 + padding=1 保持空间尺寸不变")
    print("  3. MaxPool2d(2) 将空间尺寸减半")
    print("  4. MNIST 是入门级数据集，5 个 epoch 可达 99%+")
    print("  5. 实际项目中用预训练 ResNet 做迁移学习更高效")


if __name__ == "__main__":
    main()
