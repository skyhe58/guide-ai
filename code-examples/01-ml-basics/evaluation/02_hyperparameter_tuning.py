"""
超参调优 — Grid Search、Random Search、Optuna

Python 版本：3.11+
依赖：scikit-learn>=1.4, optuna>=3.5
最后验证：2024-12-01
"""
from __future__ import annotations

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score


def demo_grid_search() -> None:
    """Grid Search 网格搜索。"""
    print("\n" + "=" * 60)
    print("1. Grid Search")
    print("=" * 60)
    X, y = make_classification(n_samples=300, n_features=10, random_state=42)
    param_grid = {"n_estimators": [50, 100], "max_depth": [3, 5, 10], "min_samples_split": [2, 5]}
    gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring="f1")
    gs.fit(X, y)
    print(f"  最佳参数: {gs.best_params_}")
    print(f"  最佳 F1: {gs.best_score_:.4f}")
    print(f"  搜索组合数: {len(gs.cv_results_['params'])}")

def demo_random_search() -> None:
    """Random Search 随机搜索。"""
    print("\n" + "=" * 60)
    print("2. Random Search")
    print("=" * 60)
    X, y = make_classification(n_samples=300, n_features=10, random_state=42)
    param_dist = {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]}
    rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_dist, n_iter=10, cv=3, scoring="f1", random_state=42)
    rs.fit(X, y)
    print(f"  最佳参数: {rs.best_params_}")
    print(f"  最佳 F1: {rs.best_score_:.4f}")
    print(f"  搜索次数: 10（比 Grid Search 更高效）")

def demo_optuna() -> None:
    """Optuna 贝叶斯优化。"""
    print("\n" + "=" * 60)
    print("3. Optuna 贝叶斯优化")
    print("=" * 60)
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  ⚠️ optuna 未安装，跳过。pip install optuna")
        return

    X, y = make_classification(n_samples=300, n_features=10, random_state=42)

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 200)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
        return cross_val_score(model, X, y, cv=3, scoring="f1").mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print(f"  最佳参数: {study.best_params}")
    print(f"  最佳 F1: {study.best_value:.4f}")
    print(f"  💡 Optuna 智能搜索，20 次试验通常优于 Grid Search 的穷举")

def main() -> None:
    print("🐍 超参调优 — Grid/Random/Optuna")
    demo_grid_search()
    demo_random_search()
    demo_optuna()
    print("\n✅ 完成！推荐使用 Optuna 做超参调优。")

if __name__ == "__main__":
    main()
