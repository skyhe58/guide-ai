"""
LoRA 微调示例 — 低秩适配原理与实现模拟

知识点：LoRA 原理（低秩分解）、秩选择、目标模块选择、
       参数效率对比、PEFT 库使用方式

Python 版本：3.11+
依赖：torch>=2.1, numpy>=1.26
可选依赖：peft>=0.7（Hugging Face PEFT 库，需 GPU）
最后验证：2024-12-01

⚠️ 免费替代方案：
  - Google Colab（免费 T4 GPU）可运行真实 LoRA 微调
  - Unsloth（2x 加速，见 04_unsloth_finetune.py）
  - LLaMA-Factory（WebUI 微调，零代码）
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. LoRA 核心实现（从零实现）
# ============================================================

class LoRALinear(nn.Module):
    """LoRA 线性层 — 低秩适配。

    原理: W' = W + ΔW = W + B @ A
    - W: 原始权重（冻结，不更新）
    - A: 低秩矩阵 (d_in, r)，高斯初始化
    - B: 低秩矩阵 (r, d_out)，零初始化
    - r: 秩（rank），通常 8-64

    参数量对比:
    - 全参数: d_in × d_out
    - LoRA:   d_in × r + r × d_out = r × (d_in + d_out)
    - 当 r << min(d_in, d_out) 时，参数量大幅减少
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # 缩放因子

        d_in = original_layer.in_features
        d_out = original_layer.out_features

        # 冻结原始权重
        original_layer.weight.requires_grad = False
        if original_layer.bias is not None:
            original_layer.bias.requires_grad = False

        # LoRA 低秩矩阵
        self.lora_A = nn.Parameter(torch.randn(d_in, rank) * (1 / math.sqrt(rank)))
        self.lora_B = nn.Parameter(torch.zeros(rank, d_out))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始输出 + LoRA 增量
        original_out = self.original_layer(x)
        lora_out = self.lora_dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        return original_out + lora_out

    def merge_weights(self) -> nn.Linear:
        """合并 LoRA 权重到原始层（推理时使用，无额外开销）。"""
        merged = nn.Linear(
            self.original_layer.in_features,
            self.original_layer.out_features,
            bias=self.original_layer.bias is not None,
        )
        merged.weight.data = (
            self.original_layer.weight.data
            + (self.lora_A @ self.lora_B * self.scaling).T
        )
        if self.original_layer.bias is not None:
            merged.bias.data = self.original_layer.bias.data
        return merged


# ============================================================
# 2. 对模型应用 LoRA
# ============================================================

def apply_lora_to_model(
    model: nn.Module,
    target_modules: list[str],
    rank: int = 8,
    alpha: float = 16.0,
) -> nn.Module:
    """对模型的指定模块应用 LoRA。

    Args:
        model: 原始模型
        target_modules: 要应用 LoRA 的模块名称列表
        rank: LoRA 秩
        alpha: LoRA 缩放因子

    Returns:
        应用 LoRA 后的模型
    """
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # 替换为 LoRA 层
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model
                if parent_name:
                    for part in parent_name.split("."):
                        parent = getattr(parent, part)
                setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha))

    return model


def count_parameters(model: nn.Module) -> dict[str, int]:
    """统计模型参数量。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {"total": total, "trainable": trainable, "frozen": frozen}


# ============================================================
# 3. 演示
# ============================================================

def demo_lora_basics() -> None:
    """演示 LoRA 基础原理。"""
    print("\n" + "=" * 60)
    print("1. LoRA 基础原理")
    print("=" * 60)

    d_in, d_out = 768, 768  # 模拟 BERT-base 的隐藏层
    rank = 8

    original = nn.Linear(d_in, d_out)
    lora_layer = LoRALinear(original, rank=rank, alpha=16.0)

    original_params = d_in * d_out + d_out  # weight + bias
    lora_params = d_in * rank + rank * d_out
    ratio = lora_params / original_params * 100

    print(f"  原始层: Linear({d_in}, {d_out})")
    print(f"  原始参数量: {original_params:,}")
    print(f"  LoRA 参数量: {lora_params:,} (rank={rank})")
    print(f"  参数比例: {ratio:.2f}%")

    # 前向传播
    x = torch.randn(2, 10, d_in)
    output = lora_layer(x)
    print(f"\n  输入: {x.shape}")
    print(f"  输出: {output.shape}")

    # 合并权重
    merged = lora_layer.merge_weights()
    print(f"\n  合并后: Linear({merged.in_features}, {merged.out_features})")
    print("  💡 合并后推理无额外开销，与原始模型速度相同")


def demo_lora_on_model() -> None:
    """演示对完整模型应用 LoRA。"""
    print("\n" + "=" * 60)
    print("2. 对模型应用 LoRA")
    print("=" * 60)

    # 模拟一个简单的 Transformer 模型
    class SimpleTransformer(nn.Module):
        def __init__(self, d_model: int = 256, num_layers: int = 4):
            super().__init__()
            self.layers = nn.ModuleList()
            for _ in range(num_layers):
                self.layers.append(nn.ModuleDict({
                    "q_proj": nn.Linear(d_model, d_model),
                    "k_proj": nn.Linear(d_model, d_model),
                    "v_proj": nn.Linear(d_model, d_model),
                    "o_proj": nn.Linear(d_model, d_model),
                    "ffn_up": nn.Linear(d_model, d_model * 4),
                    "ffn_down": nn.Linear(d_model * 4, d_model),
                }))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for layer in self.layers:
                q = layer["q_proj"](x)
                k = layer["k_proj"](x)
                v = layer["v_proj"](x)
                x = x + layer["o_proj"](v)  # 简化
                x = x + layer["ffn_down"](F.relu(layer["ffn_up"](x)))
            return x

    model = SimpleTransformer()
    before = count_parameters(model)
    print(f"  应用 LoRA 前: 总参数={before['total']:,}, 可训练={before['trainable']:,}")

    # 只对 Q/V 投影应用 LoRA（常见策略）
    model = apply_lora_to_model(model, target_modules=["q_proj", "v_proj"], rank=8)
    after = count_parameters(model)
    print(f"  应用 LoRA 后: 总参数={after['total']:,}, 可训练={after['trainable']:,}")
    print(f"  可训练参数比例: {after['trainable'] / after['total'] * 100:.2f}%")

    print("\n  💡 LoRA 秩选择建议:")
    print("    - r=8:  通用场景，效果好且参数少")
    print("    - r=16: 复杂任务（代码生成、数学推理）")
    print("    - r=64: 接近全参数微调效果")
    print("    - 目标模块: q_proj + v_proj（最常见）或全部注意力层")


def demo_peft_usage() -> None:
    """展示 PEFT 库的使用方式（伪代码）。"""
    print("\n" + "=" * 60)
    print("3. PEFT 库使用方式（伪代码）")
    print("=" * 60)

    code = '''
    # pip install peft transformers
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM

    # 1. 加载基座模型
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B")

    # 2. 配置 LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,                          # 秩
        lora_alpha=16,                # 缩放因子
        lora_dropout=0.05,            # Dropout
        target_modules=["q_proj", "v_proj"],  # 目标模块
    )

    # 3. 应用 LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # → trainable params: 4,194,304 || all params: 7,615,616,000 || 0.055%

    # 4. 训练（与普通训练相同）
    # optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    # ...

    # 5. 保存/加载 LoRA 权重（只有几 MB）
    # model.save_pretrained("./lora_weights")
    # model = PeftModel.from_pretrained(base_model, "./lora_weights")
    '''
    print(code)


def main() -> None:
    """运行所有 LoRA 演示。"""
    print("🔧 LoRA 微调示例 — 低秩适配原理与实现")
    print("=" * 60)

    demo_lora_basics()
    demo_lora_on_model()
    demo_peft_usage()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. LoRA: W' = W + B@A，只训练低秩矩阵 A 和 B")
    print("  2. 参数量减少 99%+，效果接近全参数微调")
    print("  3. 推理时可合并权重，无额外开销")
    print("  4. 常用 rank=8-16，目标模块 q_proj + v_proj")
    print("  5. 实际使用推荐 PEFT 库（Hugging Face 官方）")


if __name__ == "__main__":
    main()
