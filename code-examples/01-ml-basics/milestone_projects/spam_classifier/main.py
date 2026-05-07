"""
里程碑项目 — 垃圾邮件分类器（scikit-learn + PyTorch）

整合模块 1 核心知识点：
- 监督学习：分类任务、数据划分
- 经典算法：逻辑回归、随机森林
- 深度学习：PyTorch MLP
- 评估指标：准确率、F1、混淆矩阵
- 特征工程：TF-IDF 文本向量化

Python 版本：3.11+
依赖：scikit-learn>=1.4, torch>=2.1, numpy>=1.26
最后验证：2024-12-01
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


# ============================================================
# 1. 数据准备（模拟垃圾邮件：用 20newsgroups 的 2 个类别）
# ============================================================

def load_data():
    """加载并预处理数据。"""
    print("  📥 加载数据集...")
    # 用 20newsgroups 的 2 个类别模拟垃圾邮件检测
    categories = ["rec.sport.baseball", "sci.space"]
    data = fetch_20newsgroups(subset="all", categories=categories, remove=("headers", "footers"))
    
    # TF-IDF 特征提取
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X = vectorizer.fit_transform(data.data).toarray().astype(np.float32)
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"  训练集: {X_train.shape}, 测试集: {X_test.shape}")
    print(f"  类别: {categories}")
    return X_train, X_test, y_train, y_test, categories


# ============================================================
# 2. scikit-learn 模型
# ============================================================

def train_sklearn_models(X_train, X_test, y_train, y_test, categories):
    """训练 scikit-learn 模型。"""
    print("\n" + "=" * 60)
    print("scikit-learn 模型对比")
    print("=" * 60)

    models = {
        "逻辑回归": LogisticRegression(max_iter=200),
        "随机森林": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"\n  {name}: Accuracy={acc:.4f}, F1={f1:.4f}")
        print(classification_report(y_test, y_pred, target_names=categories))


# ============================================================
# 3. PyTorch MLP 模型
# ============================================================

class SpamMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 2),
        )
    def forward(self, x):
        return self.net(x)


def train_pytorch_model(X_train, X_test, y_train, y_test, categories):
    """训练 PyTorch MLP 模型。"""
    print("\n" + "=" * 60)
    print("PyTorch MLP 模型")
    print("=" * 60)

    X_tr = torch.FloatTensor(X_train)
    y_tr = torch.LongTensor(y_train)
    X_te = torch.FloatTensor(X_test)
    y_te = torch.LongTensor(y_test)

    model = SpamMLP(X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练
    model.train()
    for epoch in range(20):
        output = model(X_tr)
        loss = criterion(output, y_tr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            acc = (output.argmax(1) == y_tr).float().mean()
            print(f"  Epoch {epoch+1:>2}: loss={loss.item():.4f}, train_acc={acc:.4f}")

    # 评估
    model.eval()
    with torch.no_grad():
        y_pred = model(X_te).argmax(1).numpy()
    acc = accuracy_score(y_te.numpy(), y_pred)
    f1 = f1_score(y_te.numpy(), y_pred)
    print(f"\n  PyTorch MLP: Accuracy={acc:.4f}, F1={f1:.4f}")


# ============================================================
# 主入口
# ============================================================

def main():
    print("🐍 里程碑项目 — 垃圾邮件分类器")
    print("=" * 60)
    X_train, X_test, y_train, y_test, categories = load_data()
    train_sklearn_models(X_train, X_test, y_train, y_test, categories)
    train_pytorch_model(X_train, X_test, y_train, y_test, categories)
    print("\n✅ 完成！对比了逻辑回归、随机森林和 PyTorch MLP 三种方法。")

if __name__ == "__main__":
    main()
