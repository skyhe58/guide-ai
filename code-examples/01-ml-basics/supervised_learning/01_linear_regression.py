"""
线性回归 — scikit-learn 实现 + 手动梯度下降实现

知识点：线性回归原理、最小二乘法、梯度下降、
       scikit-learn API、模型评估（MSE/R²）

Python 版本：3.11+
依赖：scikit-learn>=1.4, numpy>=1.26, matplotlib>=3.8
最后验证：2024-12-01
"""

from __future__ import annotations

import numpy as np

# ============================================================
# 1. scikit-learn 线性回归
# ============================================================

def demo_sklearn_linear_regression() -> None:
    """使用 scikit-learn 实现线性回归。"""
    print("\n" + "=" * 60)
    print("1. scikit-learn 线性回归")
    print("=" * 60)

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    # 生成模拟数据：y = 2*x1 - 1*x2 + 0.5*x3 + 噪声
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 3)
    true_weights = np.array([2.0, -1.0, 0.5])
    y = X @ true_weights + np.random.randn(n_samples) * 0.3

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  训练集: {X_train.shape}, 测试集: {X_test.shape}")

    # 训练模型
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 评估
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"  真实权重: {true_weights}")
    print(f"  学到的权重: {model.coef_.round(4)}")
    print(f"  截距: {model.intercept_:.4f}")
    print(f"  MSE: {mse:.4f}")
    print(f"  R²: {r2:.4f}")


# ============================================================
# 2. 手动实现梯度下降
# ============================================================

def demo_manual_gradient_descent() -> None:
    """手动实现梯度下降训练线性回归。

    帮助理解梯度下降的核心机制：
    1. 前向传播：计算预测值
    2. 计算损失：MSE
    3. 反向传播：计算梯度
    4. 更新参数：参数 -= 学习率 * 梯度
    """
    print("\n" + "=" * 60)
    print("2. 手动实现梯度下降")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 100
    n_features = 3

    # 生成数据
    X = np.random.randn(n_samples, n_features)
    true_w = np.array([2.0, -1.0, 0.5])
    true_b = 0.3
    y = X @ true_w + true_b + np.random.randn(n_samples) * 0.1

    # 初始化参数
    w = np.zeros(n_features)
    b = 0.0
    learning_rate = 0.01
    epochs = 100

    print(f"  初始权重: {w}")
    print(f"  学习率: {learning_rate}, 迭代次数: {epochs}")

    # 梯度下降训练
    for epoch in range(epochs):
        # 前向传播
        y_pred = X @ w + b

        # 计算损失（MSE）
        loss = np.mean((y_pred - y) ** 2)

        # 计算梯度
        # dL/dw = (2/n) * X^T @ (y_pred - y)
        # dL/db = (2/n) * sum(y_pred - y)
        error = y_pred - y
        dw = (2 / n_samples) * X.T @ error
        db = (2 / n_samples) * np.sum(error)

        # 更新参数
        w -= learning_rate * dw
        b -= learning_rate * db

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}: loss={loss:.4f}, w={w.round(3)}, b={b:.3f}")

    print(f"\n  最终权重: {w.round(4)} (真实: {true_w})")
    print(f"  最终截距: {b:.4f} (真实: {true_b})")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行所有演示。"""
    print("🐍 线性回归 — scikit-learn + 手动梯度下降")
    print("=" * 60)

    demo_sklearn_linear_regression()
    demo_manual_gradient_descent()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. 线性回归是最基础的监督学习算法")
    print("  2. 梯度下降通过迭代更新参数最小化损失")
    print("  3. 学习率控制每次更新的步长")
    print("  4. MSE 和 R² 是回归任务的标准评估指标")


if __name__ == "__main__":
    main()
