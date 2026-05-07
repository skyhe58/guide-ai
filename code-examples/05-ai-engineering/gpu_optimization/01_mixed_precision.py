"""
混合精度训练模拟

知识点：AMP 自动混合精度、GradScaler 梯度缩放、FP16/BF16 对比、
       Loss Scaling 原理、Tensor Core 加速、精度与速度权衡

Python 版本：3.11+
依赖：标准库（默认模式）、torch>=2.0（GPU 模式）
最后验证：2024-12-01
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ============================================================
# 1. 数值精度模拟
# ============================================================

class Precision(Enum):
    """数值精度"""
    FP32 = "fp32"      # 32 位浮点
    FP16 = "fp16"      # 16 位浮点
    BF16 = "bf16"      # Brain Float 16
    FP8 = "fp8"        # 8 位浮点


@dataclass
class PrecisionSpec:
    """精度规格"""
    name: str
    bits: int
    bytes_per_param: float
    max_value: float
    min_positive: float
    significant_digits: int
    needs_loss_scaling: bool
    tensor_core_speedup: float  # 相对 FP32 的加速比

    def memory_for_params(self, num_params_billions: float) -> float:
        """计算参数显存（GB）"""
        return num_params_billions * self.bytes_per_param


# 精度规格表
PRECISION_SPECS = {
    Precision.FP32: PrecisionSpec("FP32", 32, 4.0, 3.4e38, 1.2e-38, 7, False, 1.0),
    Precision.FP16: PrecisionSpec("FP16", 16, 2.0, 65504, 6.1e-5, 3, True, 2.0),
    Precision.BF16: PrecisionSpec("BF16", 16, 2.0, 3.4e38, 1.2e-38, 2, False, 2.0),
    Precision.FP8: PrecisionSpec("FP8", 8, 1.0, 448, 0.015625, 1, True, 4.0),
}


# ============================================================
# 2. 模拟张量和梯度
# ============================================================

@dataclass
class SimTensor:
    """模拟张量"""
    data: list[float]
    precision: Precision = Precision.FP32
    name: str = ""

    @property
    def size(self) -> int:
        return len(self.data)

    @property
    def memory_bytes(self) -> int:
        spec = PRECISION_SPECS[self.precision]
        return self.size * int(spec.bytes_per_param)

    def cast(self, target: Precision) -> "SimTensor":
        """精度转换"""
        spec = PRECISION_SPECS[target]
        new_data = []
        for val in self.data:
            # 模拟精度损失
            if target == Precision.FP16:
                # FP16 范围限制
                val = max(-spec.max_value, min(spec.max_value, val))
                # 模拟精度损失（保留 3 位有效数字）
                if val != 0:
                    magnitude = 10 ** (math.floor(math.log10(abs(val))) - 2)
                    val = round(val / magnitude) * magnitude
            elif target == Precision.BF16:
                # BF16 精度更低但范围大
                if val != 0:
                    magnitude = 10 ** (math.floor(math.log10(abs(val))) - 1)
                    val = round(val / magnitude) * magnitude
            new_data.append(val)
        return SimTensor(new_data, target, self.name)

    def has_overflow(self) -> bool:
        """检查是否溢出"""
        spec = PRECISION_SPECS[self.precision]
        return any(abs(v) > spec.max_value for v in self.data)

    def has_underflow(self) -> bool:
        """检查是否下溢"""
        spec = PRECISION_SPECS[self.precision]
        return any(0 < abs(v) < spec.min_positive for v in self.data)


# ============================================================
# 3. GradScaler 模拟
# ============================================================

class GradScaler:
    """
    梯度缩放器模拟

    解决 FP16 训练中的梯度下溢问题。
    """

    def __init__(
        self,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._growth_tracker = 0
        self._found_inf_count = 0
        self._total_steps = 0

    def scale_loss(self, loss: float) -> float:
        """缩放损失值"""
        return loss * self.scale

    def unscale_gradients(self, gradients: list[float]) -> list[float]:
        """反缩放梯度"""
        return [g / self.scale for g in gradients]

    def check_for_inf_nan(self, gradients: list[float]) -> bool:
        """检查梯度是否有 inf/nan"""
        for g in gradients:
            if math.isinf(g) or math.isnan(g):
                return True
        return False

    def step(self, gradients: list[float]) -> tuple[list[float] | None, dict]:
        """执行一步更新"""
        self._total_steps += 1

        # 反缩放梯度
        unscaled = self.unscale_gradients(gradients)

        # 检查 inf/nan
        has_inf = self.check_for_inf_nan(unscaled)

        if has_inf:
            # 发现 inf/nan，跳过更新，减小缩放因子
            self.scale *= self.backoff_factor
            self._found_inf_count += 1
            self._growth_tracker = 0
            return None, {
                "skipped": True,
                "scale": self.scale,
                "reason": "发现 inf/nan，减小缩放因子",
            }

        # 正常更新
        self._growth_tracker += 1
        if self._growth_tracker >= self.growth_interval:
            # 连续多步没有 inf/nan，增大缩放因子
            self.scale *= self.growth_factor
            self._growth_tracker = 0

        return unscaled, {
            "skipped": False,
            "scale": self.scale,
            "growth_tracker": self._growth_tracker,
        }

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "current_scale": self.scale,
            "total_steps": self._total_steps,
            "inf_count": self._found_inf_count,
            "skip_rate": self._found_inf_count / max(self._total_steps, 1),
        }


# ============================================================
# 4. 混合精度训练器
# ============================================================

@dataclass
class TrainingConfig:
    """训练配置"""
    model_params_billions: float = 0.1
    learning_rate: float = 1e-4
    epochs: int = 5
    batch_size: int = 32
    steps_per_epoch: int = 100
    precision: Precision = Precision.FP16
    use_grad_scaler: bool = True


class MixedPrecisionTrainer:
    """
    混合精度训练器模拟

    模拟 PyTorch AMP 的训练流程。
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.scaler = GradScaler() if config.use_grad_scaler else None

        # 模拟模型参数（FP32 主权重）
        num_params = int(config.model_params_billions * 1e6)  # 简化
        self.master_weights = SimTensor(
            [random.gauss(0, 0.02) for _ in range(min(num_params, 1000))],
            Precision.FP32,
            "master_weights",
        )

        # 训练历史
        self.history: list[dict] = []

        print(f"[Trainer] 初始化混合精度训练器")
        print(f"  精度: {config.precision.value}")
        print(f"  GradScaler: {'启用' if config.use_grad_scaler else '禁用'}")
        self._print_memory_comparison()

    def _print_memory_comparison(self) -> None:
        """打印显存对比"""
        params_b = self.config.model_params_billions
        print(f"\n  显存对比 ({params_b}B 参数):")
        for prec in [Precision.FP32, Precision.FP16, Precision.BF16]:
            spec = PRECISION_SPECS[prec]
            mem = spec.memory_for_params(params_b)
            print(f"    {spec.name}: {mem:.1f} GB")

    def train_step(self, step: int) -> dict:
        """执行一步训练"""
        # 1. 将主权重转换为低精度
        low_prec_weights = self.master_weights.cast(self.config.precision)

        # 2. 前向传播（低精度）
        loss = self._forward(low_prec_weights, step)

        # 3. 损失缩放（FP16 需要）
        if self.scaler and self.config.precision == Precision.FP16:
            scaled_loss = self.scaler.scale_loss(loss)
        else:
            scaled_loss = loss

        # 4. 反向传播（低精度）
        gradients = self._backward(scaled_loss, low_prec_weights)

        # 5. 梯度处理
        if self.scaler:
            unscaled_grads, scaler_info = self.scaler.step(gradients)
            if unscaled_grads is None:
                return {"step": step, "loss": loss, "skipped": True, **scaler_info}
            gradients = unscaled_grads
        else:
            scaler_info = {}

        # 6. 参数更新（FP32）
        self._update_weights(gradients)

        return {"step": step, "loss": round(loss, 6), "skipped": False, **scaler_info}

    def _forward(self, weights: SimTensor, step: int) -> float:
        """模拟前向传播"""
        base_loss = 2.0 * math.exp(-0.01 * step) + 0.1
        noise = random.gauss(0, 0.05)
        return max(0.01, base_loss + noise)

    def _backward(self, loss: float, weights: SimTensor) -> list[float]:
        """模拟反向传播"""
        gradients = []
        for w in weights.data[:100]:  # 简化
            grad = -loss * w * random.uniform(0.8, 1.2)
            # 模拟 FP16 下溢风险
            if self.config.precision == Precision.FP16 and random.random() < 0.001:
                grad = float("inf")  # 模拟溢出
            gradients.append(grad)
        return gradients

    def _update_weights(self, gradients: list[float]) -> None:
        """更新主权重（FP32）"""
        lr = self.config.learning_rate
        for i in range(min(len(gradients), len(self.master_weights.data))):
            self.master_weights.data[i] -= lr * gradients[i]

    def train(self) -> list[dict]:
        """完整训练循环"""
        print(f"\n[Trainer] 开始训练: {self.config.epochs} epochs × {self.config.steps_per_epoch} steps")

        all_results = []
        for epoch in range(self.config.epochs):
            epoch_losses = []
            skipped = 0

            for step in range(self.config.steps_per_epoch):
                global_step = epoch * self.config.steps_per_epoch + step
                result = self.train_step(global_step)
                all_results.append(result)

                if result.get("skipped"):
                    skipped += 1
                else:
                    epoch_losses.append(result["loss"])

            avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            print(f"  Epoch {epoch}: avg_loss={avg_loss:.4f}, skipped={skipped}")

            self.history.append({
                "epoch": epoch,
                "avg_loss": round(avg_loss, 4),
                "skipped_steps": skipped,
            })

        if self.scaler:
            print(f"\n  GradScaler 统计: {self.scaler.get_stats()}")

        return all_results


# ============================================================
# 5. 精度对比实验
# ============================================================

def compare_precisions() -> None:
    """对比不同精度的训练效果"""
    print("\n" + "=" * 60)
    print("精度对比实验")
    print("=" * 60)

    results = {}
    for precision in [Precision.FP32, Precision.FP16, Precision.BF16]:
        config = TrainingConfig(
            model_params_billions=0.01,
            epochs=3,
            steps_per_epoch=50,
            precision=precision,
            use_grad_scaler=(precision == Precision.FP16),
        )
        trainer = MixedPrecisionTrainer(config)
        trainer.train()

        spec = PRECISION_SPECS[precision]
        results[precision.value] = {
            "final_loss": trainer.history[-1]["avg_loss"],
            "memory_gb": spec.memory_for_params(config.model_params_billions),
            "speedup": spec.tensor_core_speedup,
            "total_skipped": sum(h["skipped_steps"] for h in trainer.history),
        }

    print(f"\n{'精度':<8} {'最终损失':<12} {'显存(GB)':<10} {'加速比':<8} {'跳过步数'}")
    print("-" * 50)
    for prec, data in results.items():
        print(f"  {prec:<8} {data['final_loss']:<12.4f} {data['memory_gb']:<10.2f} "
              f"{data['speedup']:<8.1f}x {data['total_skipped']}")


# ============================================================
# 6. 主程序入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("混合精度训练模拟演示")
    print("=" * 60)

    # --- 演示 1: FP16 混合精度训练 ---
    print("\n--- FP16 混合精度训练 ---")
    config = TrainingConfig(
        model_params_billions=0.01,
        epochs=3,
        steps_per_epoch=50,
        precision=Precision.FP16,
        use_grad_scaler=True,
    )
    trainer = MixedPrecisionTrainer(config)
    trainer.train()

    # --- 演示 2: BF16 训练（无需 GradScaler）---
    print("\n--- BF16 训练 ---")
    config_bf16 = TrainingConfig(
        model_params_billions=0.01,
        epochs=3,
        steps_per_epoch=50,
        precision=Precision.BF16,
        use_grad_scaler=False,
    )
    trainer_bf16 = MixedPrecisionTrainer(config_bf16)
    trainer_bf16.train()

    # --- 演示 3: 精度对比 ---
    compare_precisions()

    # --- 演示 4: GradScaler 行为 ---
    print("\n--- GradScaler 行为演示 ---")
    scaler = GradScaler(init_scale=1024)
    for i in range(10):
        grads = [random.gauss(0, 0.01) for _ in range(10)]
        if i == 3:
            grads[0] = float("inf")  # 模拟溢出
        result_grads, info = scaler.step(grads)
        status = "跳过 ⚠️" if info["skipped"] else "正常 ✅"
        print(f"  Step {i}: {status}, scale={info['scale']:.0f}")

    print("\n✅ 混合精度训练模拟演示完成！")
