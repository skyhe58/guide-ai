"""
MNIST CNN 训练脚本 — 训练并保存模型

Python 版本：3.11+
依赖：torch>=2.1, torchvision>=0.16
最后验证：2024-12-01
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64*7*7, 128), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

def train():
    print("🐍 MNIST CNN 训练")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST("/tmp/mnist", train=True, download=True, transform=transform)
    test_data = datasets.MNIST("/tmp/mnist", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64)

    model = MNISTNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        model.train()
        for images, labels in train_loader:
            loss = criterion(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        correct = sum((model(img).argmax(1) == lab).sum().item() for img, lab in test_loader)
        acc = correct / len(test_data)
        print(f"  Epoch {epoch+1}: test_acc={acc:.4f}")

    torch.save(model.state_dict(), "/tmp/mnist_model.pth")
    print(f"  ✅ 模型已保存到 /tmp/mnist_model.pth")
    return model

if __name__ == "__main__":
    train()
