---
title: "模块 4：计算机视觉"
module: "cv"
description: "图像基础、目标检测、图像生成、多模态和语义分割"
---

# 模块 4：计算机视觉

> 🔭 **可选进阶模块** — 本模块覆盖计算机视觉核心技术，适合 6-8 个月全栈 AI 工程师路线。
>
> **前置依赖**：模块 0（前提准备）、模块 1（AI/ML 基础理论）
>
> **建议学习时间**：4-6 周

## 知识点索引

### 图像基础（OpenCV）

| 序号 | 知识点 | 难度 | 代码示例 |
|------|--------|------|---------|
| 01 | [OpenCV 基础](/4-cv/01-opencv-basics) | ⭐ | [01_image_basics.py](https://github.com/your-repo/tree/main/code-examples/04-cv/opencv/01_image_basics.py) |
| 02 | [图像处理](/4-cv/02-image-processing) | ⭐⭐ | [02_image_processing.py](https://github.com/your-repo/tree/main/code-examples/04-cv/opencv/02_image_processing.py) |
| 03 | [颜色空间](/4-cv/03-color-spaces) | ⭐⭐ | [01_image_basics.py](https://github.com/your-repo/tree/main/code-examples/04-cv/opencv/01_image_basics.py) |
| 04 | [图像变换](/4-cv/04-image-transforms) | ⭐⭐ | [01_image_basics.py](https://github.com/your-repo/tree/main/code-examples/04-cv/opencv/01_image_basics.py) |
| 05 | [视频处理](/4-cv/05-video-processing) | ⭐⭐ | [03_video_processing.py](https://github.com/your-repo/tree/main/code-examples/04-cv/opencv/03_video_processing.py) |

### 目标检测（YOLO）

| 序号 | 知识点 | 难度 | 代码示例 |
|------|--------|------|---------|
| 06 | [YOLO 目标检测](/4-cv/06-yolo-detection) | ⭐⭐ | [01_detection.py](https://github.com/your-repo/tree/main/code-examples/04-cv/yolo/01_detection.py) |
| 07 | [模型训练](/4-cv/07-model-training) | ⭐⭐⭐ | [02_custom_training.py](https://github.com/your-repo/tree/main/code-examples/04-cv/yolo/02_custom_training.py) |
| 08 | [模型微调](/4-cv/08-model-finetuning) | ⭐⭐⭐ | [02_custom_training.py](https://github.com/your-repo/tree/main/code-examples/04-cv/yolo/02_custom_training.py) |
| 09 | [模型评估](/4-cv/09-model-evaluation) | ⭐⭐⭐ | [01_detection.py](https://github.com/your-repo/tree/main/code-examples/04-cv/yolo/01_detection.py) |
| 10 | [模型导出](/4-cv/10-model-export) | ⭐⭐⭐ | [03_model_export.py](https://github.com/your-repo/tree/main/code-examples/04-cv/yolo/03_model_export.py) |

### 图像生成（Diffusion）

| 序号 | 知识点 | 难度 | 代码示例 |
|------|--------|------|---------|
| 11 | [Diffusion Model 原理](/4-cv/11-diffusion-model) | ⭐⭐⭐⭐ | [01_diffusers_basics.py](https://github.com/your-repo/tree/main/code-examples/04-cv/diffusion/01_diffusers_basics.py) |
| 12 | [Stable Diffusion](/4-cv/12-stable-diffusion) | ⭐⭐⭐⭐ | [02_stable_diffusion.py](https://github.com/your-repo/tree/main/code-examples/04-cv/diffusion/02_stable_diffusion.py) |
| 13 | [Diffusers 库](/4-cv/13-diffusers-library) | ⭐⭐⭐ | [03_controlnet.py](https://github.com/your-repo/tree/main/code-examples/04-cv/diffusion/03_controlnet.py) |

### 多模态

| 序号 | 知识点 | 难度 | 代码示例 |
|------|--------|------|---------|
| 14 | [LLaVA 多模态](/4-cv/14-llava-multimodal) | ⭐⭐⭐⭐ | [01_llava.py](https://github.com/your-repo/tree/main/code-examples/04-cv/multimodal/01_llava.py) |
| 15 | [视觉-语言模型](/4-cv/15-vision-language) | ⭐⭐⭐⭐ | [02_vision_language.py](https://github.com/your-repo/tree/main/code-examples/04-cv/multimodal/02_vision_language.py) |

### 语义分割

| 序号 | 知识点 | 难度 | 代码示例 |
|------|--------|------|---------|
| 16 | [语义分割](/4-cv/16-semantic-segmentation) | ⭐⭐⭐ | [01_detection.py](https://github.com/your-repo/tree/main/code-examples/04-cv/yolo/01_detection.py) |
| 17 | [Segment Anything (SAM)](/4-cv/17-sam) | ⭐⭐⭐⭐ | [01_detection.py](https://github.com/your-repo/tree/main/code-examples/04-cv/yolo/01_detection.py) |

## 里程碑项目

| 项目 | 说明 | 代码 |
|------|------|------|
| 🎯 YOLO 实时检测 API | 自定义训练→模型导出→FastAPI 服务 | [yolo_api/](https://github.com/your-repo/tree/main/code-examples/04-cv/milestone_projects/yolo_api/) |
| 🖼️ 多模态应用 | LLaVA/Qwen-VL 图文理解 + API 服务 | [multimodal_app/](https://github.com/your-repo/tree/main/code-examples/04-cv/milestone_projects/multimodal_app/) |
| 🎨 Diffusion 图像生成 | Stable Diffusion + ControlNet 条件生成 | [diffusion_demo/](https://github.com/your-repo/tree/main/code-examples/04-cv/milestone_projects/diffusion_demo/) |

## 辅助资料

- [面试指南](/4-cv/interview) — YOLO 架构演进、Diffusion 原理、多模态模型等高频题
- [速查卡片](/4-cv/cheatsheet) — 核心概念和常用 API 速查

## 推荐学习资源

| 资源 | 说明 | 链接 |
|------|------|------|
| 📖 OpenCV 官方教程 | 图像处理基础 | [docs.opencv.org](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html) |
| 🎯 Ultralytics 文档 | YOLO 训练和部署 | [docs.ultralytics.com](https://docs.ultralytics.com/) |
| 🤗 Diffusers 文档 | 扩散模型库 | [huggingface.co/docs/diffusers](https://huggingface.co/docs/diffusers/) |
| 📝 Lil'Log Diffusion | Diffusion 原理详解 | [lilianweng.github.io](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) |
| 🎓 CS231n | 斯坦福 CV 课程 | [cs231n.stanford.edu](http://cs231n.stanford.edu/) |
