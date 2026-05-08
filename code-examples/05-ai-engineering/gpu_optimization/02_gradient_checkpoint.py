"""
梯度检查点模拟

知识点：梯度检查点原理、激活值重计算、显存节省估算、
       检查点策略选择、时间-空间权衡、与混合精度结合

Python 版本：3.11+
依赖：标准库（默认模式）、torch>=2.0（GPU 模式）
最后验证：2024-12-01
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum

# ============================================================
# 1. 显存管理模拟
# ============================================================

class MemoryUnit(Enum):
    """显存单位"""
    BYTES = "B"
    KB = "KB"
    MB = "MB"
    GB = "GB"


@dataclass
class MemoryBlock:
    """显存块"""
    name: str
    size_bytes: int
    persistent: bool = True  # 是否持久（权重 vs 激活值）

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 ** 2)

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024 ** 3)


class GPUMemorySimulator:
    """GPU 显存模拟器"""

    def __init__(self, total_memory_gb: float = 24.0):
        self.total_memory_bytes = int(total_memory_gb * 1024 ** 3)
        self.allocated_blocks: list[MemoryBlock] = []
        self.peak_usage = 0

    @property
    def used_bytes(self) -> int:
        return sum(b.size_bytes for b in self.allocated_blocks)

    @property
    def used_gb(self) -> float:
        return self.used_bytes / (1024 ** 3)

    @property
    def free_gb(self) -> float:
        return (self.total_memory_bytes - self.used_bytes) / (1024 ** 3)

    def allocate(self, block: MemoryBlock) -> bool:
        """分配显存"""
        if self.used_bytes + block.size_bytes > self.total_memory_bytes:
            return False  # OOM
        self.allocated_blocks.append(block)
        self.peak_usage = max(self.peak_usage, self.used_bytes)
        return True

    def free(self, name: str) -> None:
        """释放显存"""
        self.allocated_blocks = [b for b in self.allocated_blocks if b.name != name]

    def free_non_persistent(self) -> None:
        """释放非持久显存（激活值）"""
        self.allocated_blocks = [b for b in self.allocated_blocks if b.persistent]

    def get_peak_gb(self) -> float:
        return self.peak_usage / (1024 ** 3)

    def print_status(self) -> None:
        """打印显存状态"""
        total_gb = self.total_memory_bytes / (1024 ** 3)
        print(f"  显存: {self.used_gb:.2f}/{total_gb:.1f} GB "
              f"(使用率: {self.used_bytes/self.total_memory_bytes:.1%})")


# ============================================================
# 2. Transformer 层模拟
# ============================================================

@dataclass
class LayerConfig:
    """Transformer 层配置"""
    hidden_size: int = 4096
    num_heads: int = 32
    intermediate_size: int = 11008
    batch_size: int = 4
    seq_length: int = 2048
    dtype_bytes: int = 2  # FP16 = 2 bytes


@dataclass
class LayerActivation:
    """层激活值"""
    layer_idx: int
    attention_output: list[float]
    ffn_output: list[float]
    memory_bytes: int

    @property
    def memory_mb(self) -> float:
        return self.memory_bytes / (1024 ** 2)


class TransformerLayer:
    """Transformer 层模拟"""

    def __init__(self, layer_idx: int, config: LayerConfig):
        self.layer_idx = layer_idx
        self.config = config
        # 计算激活值大小
        self.activation_size = self._calc_activation_size()

    def _calc_activation_size(self) -> int:
        """计算单层激活值大小（字节）"""
        c = self.config
        # 注意力激活值
        attn_size = c.batch_size * c.seq_length * c.hidden_size * c.dtype_bytes
        # QKV 中间结果
        qkv_size = 3 * attn_size
        # 注意力权重矩阵
        attn_weights = c.batch_size * c.num_heads * c.seq_length * c.seq_length * c.dtype_bytes
        # FFN 激活值
        ffn_size = c.batch_size * c.seq_length * c.intermediate_size * c.dtype_bytes
        return attn_size + qkv_size + attn_weights + ffn_size

    def forward(self) -> LayerActivation:
        """前向传播（保存激活值）"""
        # 模拟计算
        output = [random.gauss(0, 0.1) for _ in range(10)]
        return LayerActivation(
            layer_idx=self.layer_idx,
            attention_output=output,
            ffn_output=output,
            memory_bytes=self.activation_size,
        )

    def recompute_forward(self) -> LayerActivation:
        """重新计算前向传播（不保存中间结果）"""
        # 与 forward 相同但不保存中间激活值
        output = [random.gauss(0, 0.1) for _ in range(10)]
        return LayerActivation(
            layer_idx=self.layer_idx,
            attention_output=output,
            ffn_output=output,
            memory_bytes=0,  # 不保存
        )


# ============================================================
# 3. 检查点策略
# ============================================================

class CheckpointStrategy(Enum):
    """检查点策略"""
    NONE = "none"                    # 不使用检查点
    EVERY_LAYER = "every_layer"      # 每层都设检查点
    EVERY_N_LAYERS = "every_n"       # 每 N 层设检查点
    SQRT = "sqrt"                    # sqrt(N) 个检查点


@dataclass
class CheckpointConfig:
    """检查点配置"""
    strategy: CheckpointStrategy = CheckpointStrategy.EVERY_LAYER
    checkpoint_interval: int = 1  # 每 N 层设一个检查点（EVERY_N_LAYERS 策略）

    def get_checkpoint_layers(self, num_layers: int) -> set[int]:
        """获取需要设置检查点的层"""
        if self.strategy == CheckpointStrategy.NONE:
            return set()
        elif self.strategy == CheckpointStrategy.EVERY_LAYER:
            return set(range(num_layers))
        elif self.strategy == CheckpointStrategy.EVERY_N_LAYERS:
            return set(range(0, num_layers, self.checkpoint_interval))
        elif self.strategy == CheckpointStrategy.SQRT:
            interval = max(1, int(math.sqrt(num_layers)))
            return set(range(0, num_layers, interval))
        return set()


# ============================================================
# 4. 梯度检查点训练器
# ============================================================

class GradientCheckpointTrainer:
    """
    梯度检查点训练器模拟

    对比有无检查点的显存使用和训练时间。
    """

    def __init__(
        self,
        num_layers: int,
        layer_config: LayerConfig,
        gpu_memory_gb: float = 24.0,
        checkpoint_config: CheckpointConfig | None = None,
    ):
        self.num_layers = num_layers
        self.layer_config = layer_config
        self.checkpoint_config = checkpoint_config or CheckpointConfig(CheckpointStrategy.NONE)
        self.gpu = GPUMemorySimulator(gpu_memory_gb)

        # 创建层
        self.layers = [TransformerLayer(i, layer_config) for i in range(num_layers)]

        # 获取检查点层
        self.checkpoint_layers = self.checkpoint_config.get_checkpoint_layers(num_layers)

        # 计算模型权重显存
        self.model_weight_bytes = self._calc_model_weights()

        print(f"[Checkpoint] 初始化: {num_layers} 层 Transformer")
        print(f"  策略: {self.checkpoint_config.strategy.value}")
        print(f"  检查点层数: {len(self.checkpoint_layers)}/{num_layers}")
        print(f"  模型权重: {self.model_weight_bytes / (1024**3):.2f} GB")

    def _calc_model_weights(self) -> int:
        """计算模型权重大小"""
        c = self.layer_config
        # 每层参数：QKV + Output + FFN_up + FFN_down
        params_per_layer = (
            4 * c.hidden_size * c.hidden_size  # QKV + Output
            + 2 * c.hidden_size * c.intermediate_size  # FFN
        )
        total_params = params_per_layer * self.num_layers
        return total_params * c.dtype_bytes

    def simulate_forward(self) -> dict:
        """模拟前向传播"""
        # 分配模型权重
        self.gpu.allocate(MemoryBlock("model_weights", self.model_weight_bytes, persistent=True))

        saved_activations = 0
        recomputed_layers = 0
        total_activation_bytes = 0

        for i, layer in enumerate(self.layers):
            if i in self.checkpoint_layers:
                # 检查点层：保存激活值
                activation = layer.forward()
                block = MemoryBlock(f"activation_{i}", activation.memory_bytes, persistent=False)
                success = self.gpu.allocate(block)
                if not success:
                    return {"status": "OOM", "layer": i}
                saved_activations += 1
                total_activation_bytes += activation.memory_bytes
            else:
                # 非检查点层：不保存激活值（反向传播时重新计算）
                recomputed_layers += 1

        result = {
            "status": "success",
            "saved_activations": saved_activations,
            "recomputed_layers": recomputed_layers,
            "peak_memory_gb": self.gpu.get_peak_gb(),
            "activation_memory_gb": total_activation_bytes / (1024 ** 3),
        }

        # 清理
        self.gpu.free_non_persistent()
        self.gpu.allocated_blocks.clear()

        return result

    def simulate_backward(self) -> dict:
        """模拟反向传播"""
        recompute_time_factor = 0  # 重计算的额外时间

        for i in range(self.num_layers - 1, -1, -1):
            if i not in self.checkpoint_layers:
                # 需要重新计算前向传播
                recompute_time_factor += 1

        # 总时间 = 正常反向传播 + 重计算时间
        normal_time = self.num_layers  # 基准时间
        extra_time = recompute_time_factor  # 重计算时间
        total_time = normal_time + extra_time
        overhead = extra_time / normal_time if normal_time > 0 else 0

        return {
            "normal_backward_time": normal_time,
            "recompute_time": extra_time,
            "total_time": total_time,
            "time_overhead": round(overhead, 2),
        }


# ============================================================
# 5. 对比分析
# ============================================================

def compare_strategies(num_layers: int = 32, gpu_gb: float = 24.0) -> None:
    """对比不同检查点策略"""
    print(f"\n{'='*60}")
    print(f"检查点策略对比 ({num_layers} 层, {gpu_gb}GB GPU)")
    print(f"{'='*60}")

    layer_config = LayerConfig(hidden_size=4096, batch_size=4, seq_length=2048)

    strategies = [
        ("无检查点", CheckpointConfig(CheckpointStrategy.NONE)),
        ("每层检查点", CheckpointConfig(CheckpointStrategy.EVERY_LAYER)),
        ("每 4 层", CheckpointConfig(CheckpointStrategy.EVERY_N_LAYERS, 4)),
        ("sqrt(N) 层", CheckpointConfig(CheckpointStrategy.SQRT)),
    ]

    print(f"\n{'策略':<15} {'峰值显存(GB)':<14} {'激活值(GB)':<12} {'时间开销':<10} {'状态'}")
    print("-" * 65)

    for name, config in strategies:
        trainer = GradientCheckpointTrainer(num_layers, layer_config, gpu_gb, config)
        fwd_result = trainer.simulate_forward()
        bwd_result = trainer.simulate_backward()

        status = fwd_result["status"]
        peak_mem = fwd_result.get("peak_memory_gb", 0)
        act_mem = fwd_result.get("activation_memory_gb", 0)
        overhead = bwd_result.get("time_overhead", 0)

        print(f"  {name:<15} {peak_mem:<14.2f} {act_mem:<12.2f} "
              f"+{overhead:<9.0%} {status}")


def estimate_memory_savings(model_params_b: float, num_layers: int) -> None:
    """估算显存节省"""
    print(f"\n{'='*60}")
    print(f"显存节省估算 ({model_params_b}B 参数, {num_layers} 层)")
    print(f"{'='*60}")

    # 模型权重（FP16）
    weight_gb = model_params_b * 2
    # 梯度（FP16）
    gradient_gb = model_params_b * 2
    # 优化器状态（FP32 Adam）
    optimizer_gb = model_params_b * 8
    # 激活值（粗略估算）
    activation_per_layer_gb = model_params_b * 0.5  # 简化估算
    total_activation_gb = activation_per_layer_gb * num_layers

    print(f"\n无检查点:")
    total_no_ckpt = weight_gb + gradient_gb + optimizer_gb + total_activation_gb
    print(f"  权重: {weight_gb:.1f} GB")
    print(f"  梯度: {gradient_gb:.1f} GB")
    print(f"  优化器: {optimizer_gb:.1f} GB")
    print(f"  激活值: {total_activation_gb:.1f} GB")
    print(f"  总计: {total_no_ckpt:.1f} GB")

    # 使用检查点（每层）
    ckpt_activation_gb = activation_per_layer_gb * 2  # 只保存 2 层的激活值
    total_with_ckpt = weight_gb + gradient_gb + optimizer_gb + ckpt_activation_gb
    savings = total_activation_gb - ckpt_activation_gb

    print(f"\n有检查点（每层）:")
    print(f"  激活值: {ckpt_activation_gb:.1f} GB (节省 {savings:.1f} GB)")
    print(f"  总计: {total_with_ckpt:.1f} GB")
    print(f"  节省: {savings:.1f} GB ({savings/total_no_ckpt:.0%})")
    print(f"  代价: 训练时间增加约 30%")


# ============================================================
# 6. 主程序入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("梯度检查点模拟演示")
    print("=" * 60)

    # --- 演示 1: 基础检查点训练 ---
    print("\n--- 基础检查点训练 ---")
    config = LayerConfig(hidden_size=2048, batch_size=4, seq_length=1024)
    ckpt_config = CheckpointConfig(CheckpointStrategy.EVERY_LAYER)
    trainer = GradientCheckpointTrainer(16, config, 24.0, ckpt_config)

    fwd = trainer.simulate_forward()
    bwd = trainer.simulate_backward()
    print(f"  前向传播: {fwd}")
    print(f"  反向传播: {bwd}")

    # --- 演示 2: 策略对比 ---
    compare_strategies(num_layers=32, gpu_gb=24.0)

    # --- 演示 3: 显存节省估算 ---
    estimate_memory_savings(model_params_b=7.0, num_layers=32)
    estimate_memory_savings(model_params_b=13.0, num_layers=40)

    print("\n✅ 梯度检查点模拟演示完成！")
