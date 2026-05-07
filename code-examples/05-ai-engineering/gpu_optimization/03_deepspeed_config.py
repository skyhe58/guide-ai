"""
DeepSpeed 配置模拟

知识点：DeepSpeed ZeRO 三阶段、配置文件生成、显存估算、
       CPU Offload、通信优化、与 Hugging Face Trainer 集成

Python 版本：3.11+
依赖：标准库（默认模式）、deepspeed>=0.12（分布式模式）
最后验证：2024-12-01
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ============================================================
# 1. DeepSpeed ZeRO 阶段
# ============================================================

class ZeROStage(Enum):
    """ZeRO 优化阶段"""
    STAGE_0 = 0  # 无优化（数据并行）
    STAGE_1 = 1  # 分片优化器状态
    STAGE_2 = 2  # 分片优化器状态 + 梯度
    STAGE_3 = 3  # 分片优化器状态 + 梯度 + 模型参数


@dataclass
class ZeROStageInfo:
    """ZeRO 阶段信息"""
    stage: ZeROStage
    description: str
    partitioned: list[str]  # 被分片的内容
    memory_savings: str     # 显存节省描述
    communication_overhead: str  # 通信开销


ZERO_STAGES = {
    ZeROStage.STAGE_0: ZeROStageInfo(
        ZeROStage.STAGE_0, "无优化（标准数据并行）",
        [], "无", "低（只同步梯度）",
    ),
    ZeROStage.STAGE_1: ZeROStageInfo(
        ZeROStage.STAGE_1, "分片优化器状态",
        ["优化器状态"], "~4x 优化器显存", "低",
    ),
    ZeROStage.STAGE_2: ZeROStageInfo(
        ZeROStage.STAGE_2, "分片优化器状态 + 梯度",
        ["优化器状态", "梯度"], "~8x", "中",
    ),
    ZeROStage.STAGE_3: ZeROStageInfo(
        ZeROStage.STAGE_3, "分片所有内容",
        ["优化器状态", "梯度", "模型参数"], "~Nx (N=GPU数)", "高",
    ),
}


# ============================================================
# 2. DeepSpeed 配置
# ============================================================

@dataclass
class OffloadConfig:
    """CPU/NVMe Offload 配置"""
    device: str = "none"           # none / cpu / nvme
    pin_memory: bool = True        # 锁页内存
    nvme_path: str = "/local_nvme" # NVMe 路径

    def to_dict(self) -> dict:
        if self.device == "none":
            return {}
        config = {"device": self.device, "pin_memory": self.pin_memory}
        if self.device == "nvme":
            config["nvme_path"] = self.nvme_path
        return config


@dataclass
class DeepSpeedConfig:
    """
    DeepSpeed 配置

    生成 DeepSpeed 训练所需的 JSON 配置文件。
    """
    # ZeRO 配置
    zero_stage: ZeROStage = ZeROStage.STAGE_2
    zero_allgather_partitions: bool = True
    zero_allgather_bucket_size: int = 500_000_000
    zero_reduce_bucket_size: int = 500_000_000
    zero_reduce_scatter: bool = True

    # Offload 配置
    optimizer_offload: OffloadConfig = field(default_factory=OffloadConfig)
    param_offload: OffloadConfig = field(default_factory=OffloadConfig)

    # 训练配置
    train_batch_size: int = 32
    train_micro_batch_size_per_gpu: int = 4
    gradient_accumulation_steps: int = 8
    gradient_clipping: float = 1.0

    # 混合精度
    fp16_enabled: bool = True
    bf16_enabled: bool = False
    fp16_loss_scale: float = 0       # 0 = 动态缩放
    fp16_initial_scale_power: int = 16

    # 优化器
    optimizer_type: str = "AdamW"
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # 学习率调度
    scheduler_type: str = "WarmupDecayLR"
    warmup_steps: int = 100
    total_steps: int = 10000

    # 通信
    communication_data_type: str = "fp16"

    # 激活检查点
    activation_checkpointing: bool = False
    contiguous_memory_optimization: bool = False

    def to_dict(self) -> dict:
        """生成 DeepSpeed JSON 配置"""
        config: dict[str, Any] = {
            "train_batch_size": self.train_batch_size,
            "train_micro_batch_size_per_gpu": self.train_micro_batch_size_per_gpu,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_clipping": self.gradient_clipping,
        }

        # ZeRO 配置
        zero_config: dict[str, Any] = {
            "stage": self.zero_stage.value,
            "allgather_partitions": self.zero_allgather_partitions,
            "allgather_bucket_size": self.zero_allgather_bucket_size,
            "reduce_bucket_size": self.zero_reduce_bucket_size,
            "reduce_scatter": self.zero_reduce_scatter,
        }

        # ZeRO-3 特有配置
        if self.zero_stage == ZeROStage.STAGE_3:
            zero_config["stage3_max_live_parameters"] = 1e9
            zero_config["stage3_max_reuse_distance"] = 1e9
            zero_config["stage3_prefetch_bucket_size"] = 5e8
            zero_config["stage3_param_persistence_threshold"] = 1e6

        # Offload 配置
        if self.optimizer_offload.device != "none":
            zero_config["offload_optimizer"] = self.optimizer_offload.to_dict()
        if self.param_offload.device != "none":
            zero_config["offload_param"] = self.param_offload.to_dict()

        config["zero_optimization"] = zero_config

        # 混合精度
        if self.fp16_enabled:
            config["fp16"] = {
                "enabled": True,
                "loss_scale": self.fp16_loss_scale,
                "initial_scale_power": self.fp16_initial_scale_power,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1,
            }
        elif self.bf16_enabled:
            config["bf16"] = {"enabled": True}

        # 优化器
        config["optimizer"] = {
            "type": self.optimizer_type,
            "params": {
                "lr": self.learning_rate,
                "weight_decay": self.weight_decay,
                "betas": [self.adam_beta1, self.adam_beta2],
                "eps": self.adam_epsilon,
            },
        }

        # 学习率调度
        config["scheduler"] = {
            "type": self.scheduler_type,
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": self.learning_rate,
                "warmup_num_steps": self.warmup_steps,
                "total_num_steps": self.total_steps,
            },
        }

        # 激活检查点
        if self.activation_checkpointing:
            config["activation_checkpointing"] = {
                "partition_activations": True,
                "contiguous_memory_optimization": self.contiguous_memory_optimization,
                "cpu_checkpointing": False,
            }

        # 通信
        config["communication_data_type"] = self.communication_data_type

        return config

    def to_json(self, indent: int = 2) -> str:
        """生成 JSON 字符串"""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str) -> None:
        """保存配置文件"""
        with open(path, "w") as f:
            f.write(self.to_json())
        print(f"[DeepSpeed] 配置已保存: {path}")


# ============================================================
# 3. 显存估算器
# ============================================================

class MemoryEstimator:
    """DeepSpeed 显存估算器"""

    def __init__(
        self,
        model_params_billions: float,
        num_gpus: int,
        zero_stage: ZeROStage,
        precision: str = "fp16",
        optimizer: str = "adam",
        batch_size_per_gpu: int = 4,
        seq_length: int = 2048,
        hidden_size: int = 4096,
    ):
        self.params_b = model_params_billions
        self.num_gpus = num_gpus
        self.zero_stage = zero_stage
        self.precision = precision
        self.optimizer = optimizer
        self.batch_size = batch_size_per_gpu
        self.seq_length = seq_length
        self.hidden_size = hidden_size

    def estimate(self) -> dict[str, float]:
        """估算每张 GPU 的显存需求"""
        bytes_per_param = 2 if self.precision in ("fp16", "bf16") else 4
        N = self.num_gpus

        # 模型权重
        model_gb = self.params_b * bytes_per_param

        # 梯度
        gradient_gb = self.params_b * bytes_per_param

        # 优化器状态（Adam: momentum + variance，FP32）
        optimizer_gb = self.params_b * 8  # 2 × 4 bytes

        # 根据 ZeRO 阶段分片
        if self.zero_stage == ZeROStage.STAGE_0:
            per_gpu_model = model_gb
            per_gpu_gradient = gradient_gb
            per_gpu_optimizer = optimizer_gb
        elif self.zero_stage == ZeROStage.STAGE_1:
            per_gpu_model = model_gb
            per_gpu_gradient = gradient_gb
            per_gpu_optimizer = optimizer_gb / N
        elif self.zero_stage == ZeROStage.STAGE_2:
            per_gpu_model = model_gb
            per_gpu_gradient = gradient_gb / N
            per_gpu_optimizer = optimizer_gb / N
        else:  # STAGE_3
            per_gpu_model = model_gb / N
            per_gpu_gradient = gradient_gb / N
            per_gpu_optimizer = optimizer_gb / N

        # 激活值（粗略估算）
        activation_gb = (
            self.batch_size * self.seq_length * self.hidden_size
            * bytes_per_param * 10  # 约 10 倍 hidden_size
        ) / (1024 ** 3)

        # 系统开销
        overhead_gb = 2.0

        total = per_gpu_model + per_gpu_gradient + per_gpu_optimizer + activation_gb + overhead_gb

        return {
            "model_weights_gb": round(per_gpu_model, 2),
            "gradients_gb": round(per_gpu_gradient, 2),
            "optimizer_states_gb": round(per_gpu_optimizer, 2),
            "activations_gb": round(activation_gb, 2),
            "overhead_gb": overhead_gb,
            "total_per_gpu_gb": round(total, 2),
            "zero_stage": self.zero_stage.value,
            "num_gpus": N,
        }


# ============================================================
# 4. 配置模板工厂
# ============================================================

class DeepSpeedConfigFactory:
    """DeepSpeed 配置模板工厂"""

    @staticmethod
    def for_7b_single_gpu() -> DeepSpeedConfig:
        """7B 模型单卡训练（ZeRO-2 + CPU Offload）"""
        return DeepSpeedConfig(
            zero_stage=ZeROStage.STAGE_2,
            optimizer_offload=OffloadConfig(device="cpu"),
            train_micro_batch_size_per_gpu=1,
            gradient_accumulation_steps=16,
            fp16_enabled=True,
            activation_checkpointing=True,
        )

    @staticmethod
    def for_7b_multi_gpu(num_gpus: int = 4) -> DeepSpeedConfig:
        """7B 模型多卡训练（ZeRO-2）"""
        return DeepSpeedConfig(
            zero_stage=ZeROStage.STAGE_2,
            train_batch_size=32,
            train_micro_batch_size_per_gpu=32 // num_gpus,
            gradient_accumulation_steps=1,
            fp16_enabled=True,
        )

    @staticmethod
    def for_70b_multi_gpu(num_gpus: int = 8) -> DeepSpeedConfig:
        """70B 模型多卡训练（ZeRO-3）"""
        return DeepSpeedConfig(
            zero_stage=ZeROStage.STAGE_3,
            train_batch_size=num_gpus,
            train_micro_batch_size_per_gpu=1,
            gradient_accumulation_steps=1,
            bf16_enabled=True,
            fp16_enabled=False,
            activation_checkpointing=True,
            contiguous_memory_optimization=True,
        )

    @staticmethod
    def for_70b_with_offload(num_gpus: int = 4) -> DeepSpeedConfig:
        """70B 模型 + CPU Offload（GPU 不够时）"""
        return DeepSpeedConfig(
            zero_stage=ZeROStage.STAGE_3,
            optimizer_offload=OffloadConfig(device="cpu"),
            param_offload=OffloadConfig(device="cpu"),
            train_micro_batch_size_per_gpu=1,
            gradient_accumulation_steps=8,
            bf16_enabled=True,
            fp16_enabled=False,
            activation_checkpointing=True,
        )


# ============================================================
# 5. 对比分析
# ============================================================

def compare_zero_stages(model_params_b: float, num_gpus: int) -> None:
    """对比不同 ZeRO 阶段的显存需求"""
    print(f"\n{'='*60}")
    print(f"ZeRO 阶段对比 ({model_params_b}B 参数, {num_gpus} GPUs)")
    print(f"{'='*60}")

    print(f"\n{'阶段':<10} {'权重(GB)':<10} {'梯度(GB)':<10} {'优化器(GB)':<12} {'总计/GPU(GB)':<14}")
    print("-" * 60)

    for stage in ZeROStage:
        estimator = MemoryEstimator(model_params_b, num_gpus, stage)
        est = estimator.estimate()
        info = ZERO_STAGES[stage]
        print(f"  ZeRO-{stage.value:<4} {est['model_weights_gb']:<10} "
              f"{est['gradients_gb']:<10} {est['optimizer_states_gb']:<12} "
              f"{est['total_per_gpu_gb']:<14}")

    print(f"\n分片内容说明:")
    for stage in ZeROStage:
        info = ZERO_STAGES[stage]
        partitioned = ", ".join(info.partitioned) if info.partitioned else "无"
        print(f"  ZeRO-{stage.value}: {info.description} (分片: {partitioned})")


# ============================================================
# 6. 主程序入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DeepSpeed 配置模拟演示")
    print("=" * 60)

    # --- 演示 1: 配置模板 ---
    print("\n--- 配置模板 ---")
    templates = {
        "7B 单卡": DeepSpeedConfigFactory.for_7b_single_gpu(),
        "7B 4卡": DeepSpeedConfigFactory.for_7b_multi_gpu(4),
        "70B 8卡": DeepSpeedConfigFactory.for_70b_multi_gpu(8),
        "70B + Offload": DeepSpeedConfigFactory.for_70b_with_offload(4),
    }

    for name, config in templates.items():
        print(f"\n{name}:")
        print(f"  ZeRO Stage: {config.zero_stage.value}")
        print(f"  精度: {'FP16' if config.fp16_enabled else 'BF16'}")
        print(f"  Offload: 优化器={config.optimizer_offload.device}, "
              f"参数={config.param_offload.device}")

    # --- 演示 2: 生成配置文件 ---
    print("\n--- 生成配置文件 ---")
    config = DeepSpeedConfigFactory.for_7b_multi_gpu(4)
    print(config.to_json())

    # --- 演示 3: ZeRO 阶段对比 ---
    compare_zero_stages(model_params_b=7.0, num_gpus=4)
    compare_zero_stages(model_params_b=70.0, num_gpus=8)

    # --- 演示 4: 显存估算 ---
    print("\n--- 显存估算 ---")
    for params in [7.0, 13.0, 70.0]:
        for gpus in [1, 4, 8]:
            est = MemoryEstimator(params, gpus, ZeROStage.STAGE_3).estimate()
            print(f"  {params}B / {gpus} GPUs / ZeRO-3: {est['total_per_gpu_gb']} GB/GPU")

    print("\n✅ DeepSpeed 配置模拟演示完成！")
