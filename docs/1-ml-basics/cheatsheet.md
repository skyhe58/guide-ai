---
title: "模块 1 速查卡片"
---

# 模块 1：AI/ML 基础理论 — 速查卡片

## scikit-learn 常用 API

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # 注意：用 transform 不是 fit_transform

# 交叉验证
scores = cross_val_score(model, X, y, cv=5, scoring="f1")

# 常用模型
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
```

## PyTorch 训练模板

```python
import torch
import torch.nn as nn

# 模型定义
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )
    def forward(self, x):
        return self.layers(x)

# 训练循环
model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(epochs):
    model.train()
    output = model(X_train)
    loss = criterion(output, y_train)
    optimizer.zero_grad()   # 必须！清零梯度
    loss.backward()         # 反向传播
    optimizer.step()        # 更新参数

# 推理
model.eval()
with torch.no_grad():
    pred = model(X_test).argmax(dim=1)
```

## 算法选择速查

| 数据量 | 推荐算法 |
|--------|----------|
| < 1K | KNN、SVM、决策树 |
| 1K-100K | 随机森林、XGBoost、SVM |
| > 100K | 深度学习、XGBoost |

| 需求 | 推荐 |
|------|------|
| 可解释性 | 决策树、线性模型 |
| 最高精度 | 随机森林、XGBoost、深度学习 |
| 快速原型 | KNN、逻辑回归 |

## 评估指标速查

| 场景 | 指标 |
|------|------|
| 类别平衡 | Accuracy |
| 类别不平衡 | F1、AUC-ROC |
| 关注误报 | Precision |
| 关注漏报 | Recall |
| 回归 | MSE、MAE、R² |

## 激活函数速查

| 函数 | 公式 | 用途 |
|------|------|------|
| ReLU | max(0, x) | 隐藏层首选 |
| GELU | x·Φ(x) | Transformer |
| Sigmoid | 1/(1+e⁻ˣ) | 二分类输出 |
| Softmax | eˣⁱ/Σeˣʲ | 多分类输出 |

## 损失函数速查

| 任务 | 损失函数 | PyTorch |
|------|----------|---------|
| 回归 | MSE | `nn.MSELoss()` |
| 二分类 | BCE | `nn.BCEWithLogitsLoss()` |
| 多分类 | 交叉熵 | `nn.CrossEntropyLoss()` |
| 不平衡 | Focal Loss | 手动实现 |

## 优化器速查

| 优化器 | 适用场景 |
|--------|----------|
| Adam | 默认首选 |
| AdamW | Transformer/LLM |
| SGD+Momentum | CNN 训练 |

## Transformer 核心公式

```
Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V

Multi-Head: 分头计算 → 拼接 → 投影
Encoder Block: MHA → Add&Norm → FFN → Add&Norm
```

## 关键数字

| 概念 | 典型值 |
|------|--------|
| 学习率（小模型） | 1e-3 |
| 学习率（LLM 微调） | 1e-4 ~ 1e-5 |
| Batch Size | 32 / 64 / 128 |
| Dropout | 0.1 ~ 0.5 |
| 交叉验证 K | 5 或 10 |
| MNIST 准确率 | 99%+ (CNN) |
