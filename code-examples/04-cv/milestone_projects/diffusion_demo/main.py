"""
图像生成 Demo — 里程碑项目

项目说明：Stable Diffusion + ControlNet 条件生成服务
功能：文生图、图生图、ControlNet 条件生成、LoRA 管理、生成历史

知识点：Diffusion 模型服务化、生成参数管理、
       ControlNet 集成、LoRA 动态加载、任务队列

Python 版本：3.11+
依赖：numpy>=1.24, pydantic>=2.0（模拟模式）
真实环境依赖：
  pip install diffusers transformers torch accelerate
  pip install fastapi uvicorn controlnet-aux
最后验证：2024-12-01

运行方式（模拟）：
  python main.py
真实运行：
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import json
import time
import uuid
import numpy as np
from dataclasses import dataclass, field
from typing import Any
from enum import Enum


# ============================================================
# 1. 数据模型
# ============================================================

class GenerationMode(Enum):
    """生成模式。"""
    TEXT2IMG = "text2img"
    IMG2IMG = "img2img"
    INPAINTING = "inpainting"
    CONTROLNET = "controlnet"


class ControlNetType(Enum):
    """ControlNet 类型。"""
    CANNY = "canny"
    DEPTH = "depth"
    OPENPOSE = "openpose"
    SCRIBBLE = "scribble"


@dataclass
class GenerationParams:
    """生成参数。"""
    prompt: str = ""
    negative_prompt: str = "blurry, low quality, distorted, watermark"
    width: int = 512
    height: int = 512
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    seed: int | None = None
    num_images: int = 1
    # 图生图参数
    strength: float = 0.75
    # ControlNet 参数
    controlnet_type: ControlNetType | None = None
    controlnet_scale: float = 0.7
    # LoRA
    lora_name: str | None = None
    lora_scale: float = 0.8

    def to_dict(self) -> dict[str, Any]:
        result = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "width": self.width,
            "height": self.height,
            "steps": self.num_inference_steps,
            "cfg_scale": self.guidance_scale,
            "seed": self.seed,
            "num_images": self.num_images,
        }
        if self.controlnet_type:
            result["controlnet"] = {
                "type": self.controlnet_type.value,
                "scale": self.controlnet_scale,
            }
        if self.lora_name:
            result["lora"] = {"name": self.lora_name, "scale": self.lora_scale}
        return result


@dataclass
class GenerationResult:
    """生成结果。"""
    task_id: str
    mode: GenerationMode
    params: GenerationParams
    images: list[np.ndarray] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)
    elapsed_time: float = 0.0
    status: str = "completed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "mode": self.mode.value,
            "status": self.status,
            "num_images": len(self.images),
            "image_sizes": [f"{img.shape[1]}x{img.shape[0]}" for img in self.images],
            "seeds": self.seeds,
            "elapsed_time_ms": round(self.elapsed_time * 1000, 2),
            "params": self.params.to_dict(),
        }


@dataclass
class LoRAInfo:
    """LoRA 信息。"""
    name: str
    filename: str
    description: str
    trigger_word: str
    size_mb: float
    loaded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "filename": self.filename,
            "description": self.description,
            "trigger_word": self.trigger_word,
            "size_mb": self.size_mb,
            "loaded": self.loaded,
        }


# ============================================================
# 2. 模拟生成引擎
# ============================================================

class MockDiffusionEngine:
    """模拟 Diffusion 生成引擎。"""

    PRESET_STYLES = {
        "photorealistic": {
            "prompt_suffix": ", photorealistic, 8k, detailed",
            "negative": "cartoon, anime, painting, drawing",
            "steps": 30,
            "cfg": 7.5,
        },
        "anime": {
            "prompt_suffix": ", anime style, vibrant colors, detailed",
            "negative": "realistic, photo, 3d render",
            "steps": 25,
            "cfg": 8.0,
        },
        "oil_painting": {
            "prompt_suffix": ", oil painting, masterpiece, classical art",
            "negative": "photo, digital, modern",
            "steps": 35,
            "cfg": 9.0,
        },
        "watercolor": {
            "prompt_suffix": ", watercolor painting, soft colors, artistic",
            "negative": "photo, sharp, digital",
            "steps": 25,
            "cfg": 7.0,
        },
    }

    def __init__(self, model_id: str = "stabilityai/stable-diffusion-2-1"):
        self.model_id = model_id
        self.loaded = False
        self.loras: dict[str, LoRAInfo] = {}
        self.active_lora: str | None = None

    def load(self) -> None:
        """加载模型。"""
        time.sleep(0.1)
        self.loaded = True
        # 注册可用 LoRA
        self._register_loras()
        print(f"  ✅ Diffusion 引擎加载: {self.model_id}")

    def _register_loras(self) -> None:
        """注册可用 LoRA。"""
        self.loras = {
            "anime_style": LoRAInfo(
                name="anime_style", filename="anime_style_v2.safetensors",
                description="动漫风格 LoRA", trigger_word="anime style",
                size_mb=18.5,
            ),
            "detail_enhancer": LoRAInfo(
                name="detail_enhancer", filename="detail_enhancer.safetensors",
                description="细节增强 LoRA", trigger_word="detailed",
                size_mb=12.3,
            ),
            "portrait": LoRAInfo(
                name="portrait", filename="portrait_v3.safetensors",
                description="人像优化 LoRA", trigger_word="portrait photo",
                size_mb=24.1,
            ),
        }

    def load_lora(self, name: str, scale: float = 0.8) -> bool:
        """加载 LoRA。"""
        if name in self.loras:
            self.loras[name].loaded = True
            self.active_lora = name
            print(f"  🔌 加载 LoRA: {name} (scale={scale})")
            return True
        return False

    def unload_lora(self) -> None:
        """卸载当前 LoRA。"""
        if self.active_lora:
            self.loras[self.active_lora].loaded = False
            print(f"  🔌 卸载 LoRA: {self.active_lora}")
            self.active_lora = None

    def text2img(self, params: GenerationParams) -> GenerationResult:
        """文生图。"""
        return self._generate(GenerationMode.TEXT2IMG, params)

    def img2img(self, params: GenerationParams,
                init_image: np.ndarray) -> GenerationResult:
        """图生图。"""
        return self._generate(GenerationMode.IMG2IMG, params)

    def controlnet_generate(self, params: GenerationParams,
                            control_image: np.ndarray) -> GenerationResult:
        """ControlNet 条件生成。"""
        return self._generate(GenerationMode.CONTROLNET, params)

    def _generate(self, mode: GenerationMode,
                  params: GenerationParams) -> GenerationResult:
        """统一生成逻辑。"""
        task_id = str(uuid.uuid4())[:8]
        start = time.time()

        images = []
        seeds = []

        for i in range(params.num_images):
            seed = params.seed if params.seed is not None else np.random.randint(0, 2**32)
            if params.seed is not None:
                seed += i  # 多图时递增种子
            rng = np.random.RandomState(seed)

            image = rng.randint(0, 256, (params.height, params.width, 3)).astype(np.uint8)
            images.append(image)
            seeds.append(seed)

        elapsed = time.time() - start + np.random.uniform(2, 10)

        return GenerationResult(
            task_id=task_id,
            mode=mode,
            params=params,
            images=images,
            seeds=seeds,
            elapsed_time=elapsed,
        )


# ============================================================
# 3. 生成历史管理
# ============================================================

class GenerationHistory:
    """生成历史管理。"""

    def __init__(self, max_history: int = 100) -> None:
        self.history: list[dict[str, Any]] = []
        self.max_history = max_history

    def add(self, result: GenerationResult) -> None:
        """添加生成记录。"""
        record = {
            "task_id": result.task_id,
            "mode": result.mode.value,
            "prompt": result.params.prompt[:50],
            "size": f"{result.params.width}x{result.params.height}",
            "seeds": result.seeds,
            "elapsed_ms": round(result.elapsed_time * 1000),
            "timestamp": time.time(),
        }
        self.history.append(record)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_recent(self, n: int = 10) -> list[dict[str, Any]]:
        """获取最近 N 条记录。"""
        return self.history[-n:]

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息。"""
        if not self.history:
            return {"total": 0}

        modes = {}
        total_time = 0.0
        for record in self.history:
            mode = record["mode"]
            modes[mode] = modes.get(mode, 0) + 1
            total_time += record["elapsed_ms"]

        return {
            "total_generations": len(self.history),
            "mode_distribution": modes,
            "avg_time_ms": round(total_time / len(self.history)),
        }


# ============================================================
# 4. Diffusion Demo 应用
# ============================================================

class DiffusionDemoApp:
    """图像生成 Demo 应用。"""

    def __init__(self) -> None:
        self.engine = MockDiffusionEngine()
        self.history = GenerationHistory()
        self.start_time = time.time()
        print(f"\n  🚀 Diffusion Demo 初始化")

    def startup(self) -> None:
        """启动服务。"""
        self.engine.load()
        print(f"  ✅ 服务就绪")

    def text2img(self, prompt: str, **kwargs: Any) -> GenerationResult:
        """POST /generate/text2img — 文生图。"""
        params = GenerationParams(prompt=prompt, **kwargs)
        result = self.engine.text2img(params)
        self.history.add(result)
        return result

    def img2img(self, prompt: str, init_image: np.ndarray,
                strength: float = 0.75, **kwargs: Any) -> GenerationResult:
        """POST /generate/img2img — 图生图。"""
        params = GenerationParams(prompt=prompt, strength=strength, **kwargs)
        result = self.engine.img2img(params, init_image)
        self.history.add(result)
        return result

    def controlnet(self, prompt: str, control_image: np.ndarray,
                   control_type: ControlNetType = ControlNetType.CANNY,
                   control_scale: float = 0.7,
                   **kwargs: Any) -> GenerationResult:
        """POST /generate/controlnet — ControlNet 生成。"""
        params = GenerationParams(
            prompt=prompt,
            controlnet_type=control_type,
            controlnet_scale=control_scale,
            **kwargs,
        )
        result = self.engine.controlnet_generate(params, control_image)
        self.history.add(result)
        return result

    def apply_style(self, prompt: str, style: str,
                    **kwargs: Any) -> GenerationResult:
        """POST /generate/styled — 风格化生成。"""
        preset = self.engine.PRESET_STYLES.get(style, {})
        full_prompt = prompt + preset.get("prompt_suffix", "")
        neg_prompt = preset.get("negative", "")
        steps = preset.get("steps", 30)
        cfg = preset.get("cfg", 7.5)

        params = GenerationParams(
            prompt=full_prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=steps,
            guidance_scale=cfg,
            **kwargs,
        )
        result = self.engine.text2img(params)
        self.history.add(result)
        return result

    def manage_lora(self, action: str, name: str = "",
                    scale: float = 0.8) -> dict[str, Any]:
        """LoRA 管理。"""
        if action == "load":
            success = self.engine.load_lora(name, scale)
            return {"action": "load", "name": name, "success": success}
        elif action == "unload":
            self.engine.unload_lora()
            return {"action": "unload", "success": True}
        elif action == "list":
            return {
                "loras": {k: v.to_dict() for k, v in self.engine.loras.items()},
                "active": self.engine.active_lora,
            }
        return {"error": f"未知操作: {action}"}

    def get_styles(self) -> dict[str, Any]:
        """获取可用风格预设。"""
        return {
            name: {
                "prompt_suffix": info["prompt_suffix"],
                "steps": info["steps"],
                "cfg": info["cfg"],
            }
            for name, info in self.engine.PRESET_STYLES.items()
        }

    def get_stats(self) -> dict[str, Any]:
        """获取服务统计。"""
        gen_stats = self.history.get_stats()
        return {
            "uptime_seconds": round(time.time() - self.start_time, 1),
            "model": self.engine.model_id,
            "model_loaded": self.engine.loaded,
            "active_lora": self.engine.active_lora,
            **gen_stats,
        }


# ============================================================
# 5. 演示函数
# ============================================================

def demo_startup() -> DiffusionDemoApp:
    """演示服务启动。"""
    print("\n" + "=" * 60)
    print("1. 服务启动")
    print("=" * 60)

    app = DiffusionDemoApp()
    app.startup()
    return app


def demo_text2img(app: DiffusionDemoApp) -> None:
    """演示文生图。"""
    print("\n" + "=" * 60)
    print("2. 文生图（Text-to-Image）")
    print("=" * 60)

    result = app.text2img(
        prompt="a beautiful sunset over the ocean, golden light",
        width=512, height=512,
        num_inference_steps=30,
        guidance_scale=7.5,
        seed=42,
    )

    print(f"\n  生成结果:")
    result_dict = result.to_dict()
    for key in ["task_id", "mode", "num_images", "seeds", "elapsed_time_ms"]:
        print(f"    {key}: {result_dict[key]}")


def demo_img2img(app: DiffusionDemoApp) -> None:
    """演示图生图。"""
    print("\n" + "=" * 60)
    print("3. 图生图（Image-to-Image）")
    print("=" * 60)

    init_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

    for strength in [0.3, 0.5, 0.75]:
        result = app.img2img(
            prompt="transform into watercolor painting",
            init_image=init_image,
            strength=strength,
        )
        print(f"  strength={strength}: {result.elapsed_time*1000:.0f}ms, "
              f"seed={result.seeds[0]}")


def demo_controlnet(app: DiffusionDemoApp) -> None:
    """演示 ControlNet 生成。"""
    print("\n" + "=" * 60)
    print("4. ControlNet 条件生成")
    print("=" * 60)

    control_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

    for ct in [ControlNetType.CANNY, ControlNetType.DEPTH, ControlNetType.OPENPOSE]:
        result = app.controlnet(
            prompt="a beautiful house in the countryside",
            control_image=control_image,
            control_type=ct,
            control_scale=0.7,
        )
        print(f"  {ct.value}: {result.elapsed_time*1000:.0f}ms, "
              f"task_id={result.task_id}")


def demo_style_presets(app: DiffusionDemoApp) -> None:
    """演示风格预设。"""
    print("\n" + "=" * 60)
    print("5. 风格预设")
    print("=" * 60)

    styles = app.get_styles()
    print(f"\n  可用风格:")
    for name, info in styles.items():
        print(f"    {name}: steps={info['steps']}, cfg={info['cfg']}")

    # 同一 prompt 不同风格
    prompt = "a cat sitting on a windowsill"
    print(f"\n  Prompt: '{prompt}'")

    for style in ["photorealistic", "anime", "oil_painting", "watercolor"]:
        result = app.apply_style(prompt, style, seed=42)
        print(f"    {style}: {result.elapsed_time*1000:.0f}ms")


def demo_lora_management(app: DiffusionDemoApp) -> None:
    """演示 LoRA 管理。"""
    print("\n" + "=" * 60)
    print("6. LoRA 管理")
    print("=" * 60)

    # 列出可用 LoRA
    lora_info = app.manage_lora("list")
    print(f"\n  可用 LoRA:")
    for name, info in lora_info["loras"].items():
        print(f"    {name}: {info['description']} ({info['size_mb']}MB)")

    # 加载 LoRA
    app.manage_lora("load", "anime_style", scale=0.8)

    # 使用 LoRA 生成
    result = app.text2img(
        prompt="a girl in anime style, detailed",
        lora_name="anime_style",
        lora_scale=0.8,
    )
    print(f"  LoRA 生成: {result.elapsed_time*1000:.0f}ms")

    # 卸载 LoRA
    app.manage_lora("unload")


def demo_generation_history(app: DiffusionDemoApp) -> None:
    """演示生成历史。"""
    print("\n" + "=" * 60)
    print("7. 生成历史与统计")
    print("=" * 60)

    # 查看最近记录
    recent = app.history.get_recent(5)
    print(f"\n  最近 {len(recent)} 条生成记录:")
    for record in recent:
        print(f"    [{record['task_id']}] {record['mode']}: "
              f"'{record['prompt']}' ({record['elapsed_ms']}ms)")

    # 统计信息
    stats = app.get_stats()
    print(f"\n  服务统计:")
    for key, value in stats.items():
        print(f"    {key}: {value}")


def demo_api_routes() -> None:
    """展示 API 路由。"""
    print("\n" + "=" * 60)
    print("8. API 路由")
    print("=" * 60)

    routes = [
        ("POST", "/generate/text2img", "文生图"),
        ("POST", "/generate/img2img", "图生图"),
        ("POST", "/generate/controlnet", "ControlNet 生成"),
        ("POST", "/generate/styled", "风格化生成"),
        ("GET", "/styles", "获取风格预设"),
        ("POST", "/lora/load", "加载 LoRA"),
        ("POST", "/lora/unload", "卸载 LoRA"),
        ("GET", "/lora/list", "列出 LoRA"),
        ("GET", "/history", "生成历史"),
        ("GET", "/stats", "服务统计"),
        ("GET", "/health", "健康检查"),
    ]

    for method, path, desc in routes:
        print(f"  {method:<6} {path:<30} — {desc}")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行图像生成 Demo。"""
    print("🐍 图像生成 Demo — 里程碑项目")
    print("=" * 60)

    app = demo_startup()
    demo_text2img(app)
    demo_img2img(app)
    demo_controlnet(app)
    demo_style_presets(app)
    demo_lora_management(app)
    demo_generation_history(app)
    demo_api_routes()

    print("\n" + "=" * 60)
    print("✅ 里程碑项目演示完成！")
    print("\n💡 项目要点:")
    print("  1. 支持文生图/图生图/ControlNet 三种生成模式")
    print("  2. 风格预设简化用户操作（photorealistic/anime/oil_painting）")
    print("  3. LoRA 动态加载/卸载，灵活切换风格")
    print("  4. 生成历史记录和统计分析")
    print("  5. seed 控制确保结果可复现")
    print("  6. 生产部署: FastAPI + Redis(队列) + GPU Worker")


if __name__ == "__main__":
    main()
