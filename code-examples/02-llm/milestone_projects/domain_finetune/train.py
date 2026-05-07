"""
LoRA 微调训练脚本 — 领域微调完整流程

知识点：数据加载、LoRA 配置、训练循环、模型保存、
       Unsloth 加速微调

Python 版本：3.11+
依赖：torch>=2.1
可选依赖：transformers>=4.36, peft>=0.7, trl>=0.7, datasets>=2.16
最后验证：2024-12-01

⚠️ 运行环境：
  - Google Colab 免费 T4 GPU（推荐）
  - 本地 NVIDIA GPU（RTX 3060+ / 12GB+）

⚠️ 免费替代方案：
  - Google Colab: 免费 T4 GPU
  - Unsloth: 2x 加速，60% 显存节省
  - LLaMA-Factory: WebUI 零代码微调
"""

from __future__ import annotations

import json
from pathlib import Path


# ============================================================
# 1. 数据准备
# ============================================================

def prepare_training_data(output_path: str = "train_data.json") -> str:
    """准备训练数据（Alpaca 格式）。

    Returns:
        数据文件路径
    """
    # 示例：Python 编程领域微调数据
    training_data = [
        {
            "instruction": "解释 Python 中的装饰器",
            "input": "",
            "output": "装饰器是 Python 的语法糖，用 @decorator 语法将一个函数包装在另一个函数中。"
                     "它本质上是一个接受函数作为参数并返回新函数的高阶函数。"
                     "常见用途：日志记录、权限检查、缓存、重试机制等。",
        },
        {
            "instruction": "写一个 Python 单例模式",
            "input": "",
            "output": "```python\nclass Singleton:\n    _instance = None\n\n"
                     "    def __new__(cls):\n        if cls._instance is None:\n"
                     "            cls._instance = super().__new__(cls)\n"
                     "        return cls._instance\n```",
        },
        {
            "instruction": "解释 async/await 的工作原理",
            "input": "",
            "output": "async/await 是 Python 的协程语法。async def 定义协程函数，"
                     "await 暂停协程执行并等待异步操作完成。事件循环（asyncio）"
                     "管理协程调度，实现单线程并发 I/O。",
        },
        {
            "instruction": "什么是 Python 的 GIL？",
            "input": "",
            "output": "GIL（全局解释器锁）是 CPython 的互斥锁，确保同一时刻只有一个线程执行 Python 字节码。"
                     "这意味着多线程无法利用多核 CPU 进行并行计算。"
                     "解决方案：多进程（multiprocessing）、C 扩展、或使用无 GIL 的实现。",
        },
        {
            "instruction": "如何优化 Python 代码性能？",
            "input": "",
            "output": "1. 使用内置函数和标准库（C 实现，比纯 Python 快）\n"
                     "2. 列表推导式替代 for 循环\n"
                     "3. 使用 NumPy 向量化替代循环\n"
                     "4. 缓存（functools.lru_cache）\n"
                     "5. 异步 I/O（asyncio）处理网络请求\n"
                     "6. 性能分析（cProfile/line_profiler）定位瓶颈",
        },
    ]

    # 保存为 JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)

    print(f"  训练数据已保存: {output_path}")
    print(f"  数据量: {len(training_data)} 条")
    return output_path


# ============================================================
# 2. 训练流程（伪代码 + 说明）
# ============================================================

def show_training_workflow() -> None:
    """展示完整训练流程。"""
    print("\n" + "=" * 60)
    print("2. 完整训练流程（伪代码）")
    print("=" * 60)

    code = '''
    # ===== 方法 1: Unsloth 加速微调（推荐）=====
    from unsloth import FastLanguageModel

    # 加载模型
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2-7B-bnb-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
    )

    # 添加 LoRA
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # 训练
    from trl import SFTTrainer, SFTConfig
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir="./output",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
        ),
    )
    trainer.train()

    # 保存
    model.save_pretrained("./lora_weights")
    model.save_pretrained_gguf("./gguf", tokenizer, quantization_method="q4_k_m")


    # ===== 方法 2: 标准 PEFT 微调 =====
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    # 4-bit 量化加载
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B", quantization_config=bnb_config, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 训练（同上）
    '''
    print(code)


# ============================================================
# 3. 训练参数建议
# ============================================================

def show_training_tips() -> None:
    """展示训练参数建议。"""
    print("\n" + "=" * 60)
    print("3. 训练参数建议")
    print("=" * 60)

    print("""
    ┌──────────────────┬──────────────┬──────────────────────┐
    │ 参数             │ 推荐值       │ 说明                 │
    ├──────────────────┼──────────────┼──────────────────────┤
    │ LoRA rank        │ 8-16         │ 复杂任务用 16-64     │
    │ LoRA alpha       │ 2x rank      │ alpha=2*r 是常见选择 │
    │ Learning rate    │ 1e-4 ~ 3e-4  │ QLoRA 推荐 2e-4      │
    │ Batch size       │ 4-8          │ 配合 gradient_accum  │
    │ Epochs           │ 1-3          │ 数据少时 3 轮        │
    │ Max seq length   │ 1024-2048    │ 根据数据长度调整     │
    │ Warmup ratio     │ 0.03-0.1     │ 前 3-10% 步预热      │
    │ Weight decay     │ 0.01         │ 防止过拟合           │
    │ Optimizer        │ adamw_8bit   │ 节省显存             │
    └──────────────────┴──────────────┴──────────────────────┘

    💡 数据量建议：
    - 最少: 100 条高质量数据（效果有限）
    - 推荐: 1000-5000 条（效果明显）
    - 充足: 10000+ 条（接近最佳效果）
    - 质量 > 数量！100 条高质量 > 10000 条低质量
    """)


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行训练脚本演示。"""
    print("🎯 LoRA 微调训练脚本 — 领域微调完整流程")
    print("=" * 60)

    # 1. 准备数据
    print("\n1. 准备训练数据")
    data_path = prepare_training_data()

    # 2. 展示训练流程
    show_training_workflow()

    # 3. 训练参数建议
    show_training_tips()

    # 清理临时文件
    Path(data_path).unlink(missing_ok=True)

    print("\n" + "=" * 60)
    print("✅ 演示完成！")
    print("\n💡 下一步:")
    print("  1. 准备领域数据（Alpaca/ShareGPT 格式）")
    print("  2. 在 Google Colab 上运行微调")
    print("  3. 导出 GGUF 并用 Ollama 部署")
    print("  4. 用 deploy.py 部署为 API 服务")


if __name__ == "__main__":
    main()
