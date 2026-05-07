---
title: "模块 4 速查卡片"
module: "cv"
description: "计算机视觉核心概念和常用 API 速查"
---

# 模块 4 速查卡片

## OpenCV 速查

### 图像读写

```python
import cv2
import numpy as np

img = cv2.imread("image.jpg")              # BGR 格式
gray = cv2.imread("image.jpg", 0)          # 灰度
cv2.imwrite("output.jpg", img)             # 保存
cv2.imwrite("out.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
```

### 颜色空间转换

```python
rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
lab   = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
```

### 图像处理

```python
# 滤波
blur     = cv2.GaussianBlur(img, (5, 5), 0)
median   = cv2.medianBlur(img, 5)
bilateral = cv2.bilateralFilter(img, 9, 75, 75)

# 二值化
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
_, otsu   = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 边缘检测
edges = cv2.Canny(gray, 50, 150)

# 形态学
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
```

### 图像变换

```python
# 缩放
resized = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)

# 旋转
M = cv2.getRotationMatrix2D((w//2, h//2), angle=45, scale=1.0)
rotated = cv2.warpAffine(img, M, (w, h))

# 翻转
flipped = cv2.flip(img, 1)  # 1=水平, 0=垂直, -1=两者

# 透视变换
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped = cv2.warpPerspective(img, M, (w, h))
```

### HSV 颜色范围

| 颜色 | H 范围 | S 范围 | V 范围 |
|------|--------|--------|--------|
| 红色 | 0-10, 170-179 | 100-255 | 100-255 |
| 绿色 | 35-85 | 100-255 | 100-255 |
| 蓝色 | 100-130 | 100-255 | 100-255 |
| 黄色 | 26-34 | 100-255 | 100-255 |

## YOLO 速查

### 基础使用

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")           # 加载模型
results = model("image.jpg")          # 推理
results = model("video.mp4", stream=True)  # 视频推理

# 训练
model.train(data="data.yaml", epochs=100, imgsz=640)

# 验证
metrics = model.val(data="data.yaml")

# 导出
model.export(format="onnx", imgsz=640)
```

### 模型规格

| 模型 | 参数量 | mAP50-95 | 速度 | 适用 |
|------|--------|----------|------|------|
| yolov8n | 3.2M | 37.3 | 1.2ms | 边缘/实时 |
| yolov8s | 11.2M | 44.9 | 1.9ms | 平衡 |
| yolov8m | 25.9M | 50.2 | 4.3ms | 通用 |
| yolov8l | 43.7M | 52.9 | 6.2ms | 高精度 |
| yolov8x | 68.2M | 53.9 | 9.8ms | 最高精度 |

### 导出格式

| 格式 | 命令 | 目标平台 |
|------|------|---------|
| ONNX | `format="onnx"` | 通用 |
| TensorRT | `format="engine"` | NVIDIA GPU |
| CoreML | `format="coreml"` | Apple |
| TFLite | `format="tflite"` | 移动端 |
| OpenVINO | `format="openvino"` | Intel |

### 评估指标

| 指标 | 含义 | 公式 |
|------|------|------|
| IoU | 交并比 | 交集/并集 |
| Precision | 精确率 | TP/(TP+FP) |
| Recall | 召回率 | TP/(TP+FN) |
| mAP50 | VOC 标准 | IoU=0.5 下的 mAP |
| mAP50-95 | COCO 标准 | 10 个 IoU 阈值平均 |

## Diffusion 速查

### Diffusers Pipeline

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

# 文生图
image = pipe("a cat", guidance_scale=7.5, num_inference_steps=30).images[0]

# 内存优化
pipe.enable_attention_slicing()
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| guidance_scale | 7.5 | CFG 引导强度（5-15） |
| num_inference_steps | 50 | 去噪步数（20-50） |
| strength | 0.75 | 图生图改变程度（0-1） |
| negative_prompt | "" | 负向提示词 |

### SD 架构组件

| 组件 | 作用 | 参数量 |
|------|------|--------|
| CLIP Text Encoder | 文本→嵌入 | ~400M |
| UNet | 潜空间去噪 | ~860M |
| VAE | 像素↔潜空间 | ~80M |

## 多模态速查

### VLM 对比

| 模型 | 开源 | 中文 | 部署 |
|------|------|------|------|
| GPT-4o | ❌ | ⭐⭐⭐⭐ | API |
| Claude 3 | ❌ | ⭐⭐⭐⭐ | API |
| Qwen-VL | ✅ | ⭐⭐⭐⭐⭐ | 本地/API |
| LLaVA | ✅ | ⭐⭐⭐ | 本地 |
| InternVL2 | ✅ | ⭐⭐⭐⭐⭐ | 本地 |

### 分割类型

| 类型 | 区分实例 | 背景 | 代表模型 |
|------|---------|------|---------|
| 语义分割 | ❌ | ✅ | DeepLabV3+ |
| 实例分割 | ✅ | ❌ | Mask R-CNN |
| 全景分割 | ✅ | ✅ | Panoptic FPN |

## 常用命令

```bash
# 安装依赖
pip install opencv-python>=4.8
pip install ultralytics>=8.0
pip install diffusers transformers accelerate
pip install segment-anything
pip install onnxruntime-gpu

# YOLO 命令行
yolo detect train data=data.yaml model=yolov8n.pt epochs=100
yolo detect predict model=best.pt source=image.jpg
yolo detect val model=best.pt data=data.yaml
yolo export model=best.pt format=onnx
```
