---
title: "OpenCV 基础"
module: "cv"
difficulty: "beginner"
interviewFrequency: "medium"
tags:
  - "OpenCV"
  - "图像读取"
  - "图像显示"
  - "图像保存"
  - "图像属性"
  - "计算机视觉"
codeExample: "04-cv/opencv/"
relatedEntries:
  - "/4-cv/02-image-processing"
  - "/4-cv/03-color-spaces"
  - "/4-cv/04-image-transforms"
prerequisites:
  - "/0-prerequisites/numpy-basics"
estimatedTime: "45min"
toolReferences:
  - name: "Cursor"
    usage: "辅助编写 OpenCV 代码，快速调试图像处理逻辑"
    link: "/7-ai-tools/7.1-efficiency/ai-coding"
  - name: "ChatGPT"
    usage: "解释 OpenCV API 参数和图像处理原理"
    link: "/7-ai-tools/7.1-efficiency/ai-chat"
  - name: "Perplexity"
    usage: "搜索 OpenCV 最新版本特性和最佳实践"
    link: "/7-ai-tools/7.1-efficiency/ai-search"
---

# OpenCV 基础

## 概念说明

**OpenCV**（Open Source Computer Vision Library）是最流行的开源计算机视觉库，提供了 2500+ 个优化算法，覆盖图像处理、视频分析、目标检测、机器学习等领域。Python 版本通过 `cv2` 模块使用。

### 为什么学习 OpenCV？

- **行业标准**：几乎所有 CV 项目都会用到 OpenCV 做预处理
- **性能优秀**：底层 C++ 实现，Python 接口方便调用
- **生态丰富**：与 NumPy、PyTorch、TensorFlow 无缝集成
- **面试必备**：CV 岗位面试必问图像基础操作

### OpenCV 在 AI 工程中的位置

```mermaid
graph LR
    A[原始图像/视频] --> B[OpenCV 预处理]
    B --> C[数据增强]
    C --> D[模型推理<br/>YOLO/SAM/SD]
    D --> E[后处理<br/>NMS/可视化]
    E --> F[输出结果]
    
    style B fill:#e1f5fe
    style D fill:#fff3e0
```

## 核心原理

### 1. 图像的本质——NumPy 数组

在 OpenCV 中，图像本质上是一个 NumPy 多维数组：

| 图像类型 | 数组形状 | 通道含义 | 值范围 |
|----------|----------|----------|--------|
| 灰度图 | (H, W) | 亮度 | 0-255 |
| 彩色图（BGR） | (H, W, 3) | Blue, Green, Red | 0-255 |
| 带透明通道 | (H, W, 4) | B, G, R, Alpha | 0-255 |

> ⚠️ **注意**：OpenCV 默认使用 **BGR** 通道顺序，而非 RGB。这是历史原因导致的，与 Matplotlib、PIL 等库不同。

### 2. 图像读取 — `cv2.imread()`

```python
import cv2
import numpy as np

# 读取彩色图像（默认 BGR）
img = cv2.imread("image.jpg")              # 返回 (H, W, 3) ndarray
img_gray = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)  # 灰度读取
img_unchanged = cv2.imread("image.jpg", cv2.IMREAD_UNCHANGED)  # 保留 Alpha 通道

# 检查是否读取成功
if img is None:
    raise FileNotFoundError("图像文件不存在或格式不支持")
```

读取标志（flags）：

| 标志 | 值 | 说明 |
|------|-----|------|
| `IMREAD_COLOR` | 1 | 彩色图（默认），忽略 Alpha |
| `IMREAD_GRAYSCALE` | 0 | 灰度图 |
| `IMREAD_UNCHANGED` | -1 | 保留所有通道（含 Alpha） |

### 3. 图像属性

```python
print(f"形状: {img.shape}")      # (高度, 宽度, 通道数)
print(f"尺寸: {img.size}")       # 总像素数 = H × W × C
print(f"数据类型: {img.dtype}")  # uint8（0-255）
print(f"高度: {img.shape[0]}, 宽度: {img.shape[1]}")
```

### 4. 图像显示 — `cv2.imshow()` 与替代方案

```python
# 方式 1：OpenCV 窗口（需要 GUI 环境）
cv2.imshow("Window Title", img)
cv2.waitKey(0)          # 等待按键，0 表示无限等待
cv2.destroyAllWindows()

# 方式 2：Matplotlib（推荐，适合 Jupyter）
import matplotlib.pyplot as plt
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR → RGB
plt.imshow(img_rgb)
plt.axis("off")
plt.show()
```

### 5. 图像保存 — `cv2.imwrite()`

```python
# 保存图像
cv2.imwrite("output.jpg", img)                          # JPEG
cv2.imwrite("output.png", img)                          # PNG（无损）
cv2.imwrite("output.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])  # 指定质量
```

### 6. ROI（感兴趣区域）与像素操作

```python
# NumPy 切片提取 ROI
roi = img[100:300, 200:400]   # [y1:y2, x1:x2]

# 单像素操作
pixel = img[100, 200]         # 获取 (100, 200) 处的 BGR 值
img[100, 200] = [255, 0, 0]  # 设置为蓝色（BGR）

# 通道分离与合并
b, g, r = cv2.split(img)
merged = cv2.merge([b, g, r])
```

## 代码示例

> 💻 完整可运行代码：[code-examples/04-cv/opencv/01_image_basics.py](https://github.com/skyhe58/guide-ai/tree/main/code-examples/04-cv/opencv/01_image_basics.py)
> 🐍 Python 版本：3.11+
> 📦 依赖：numpy（模拟模式）、opencv-python>=4.8（完整模式）

## 实战要点

**生产环境注意事项：**
- **BGR vs RGB**：与 PyTorch/TensorFlow 交互时必须转换通道顺序
- **内存管理**：大图像（4K/8K）注意内存占用，及时释放
- **路径编码**：中文路径可能导致 `imread` 返回 None，使用 `cv2.imdecode` 替代
- **数据类型**：深度学习模型通常需要 float32 归一化到 [0, 1]

**常见陷阱：**
- `imread` 失败不会抛异常，只返回 None——必须检查
- 图像坐标是 (y, x) 而非 (x, y)
- `imshow` 在无 GUI 的服务器上会崩溃——用 Matplotlib 或保存文件

## 常见面试题

### Q1: OpenCV 读取图像的通道顺序是什么？为什么？

**难度**：⭐ | **频率**：🔥🔥🔥

**答题思路**：说明 BGR 顺序 → 历史原因 → 转换方法

**标准答案**：OpenCV 默认使用 BGR 通道顺序，这是因为早期摄像头硬件和 Windows BMP 格式使用 BGR 排列，OpenCV 为了兼容保留了这个约定。与 PIL、Matplotlib 的 RGB 顺序不同。转换方法：`cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`。在与深度学习框架交互时必须注意通道转换。

**深入追问**：
- 如何高效地进行通道转换？（`img[:, :, ::-1]` NumPy 切片比 cvtColor 更快）
- 不转换通道会导致什么问题？（颜色偏差，红蓝互换）

## 推荐工具

> 📌 以下工具可帮助你更高效地学习和实践本知识点，详见 [模块 7：AI 使用与实践](/7-ai-tools/)

| 工具 | 用途 | 详情 |
|------|------|------|
| Cursor | 辅助编写 OpenCV 代码 | [AI 编程辅助](/7-ai-tools/7.1-efficiency/ai-coding) |
| ChatGPT | 解释 OpenCV API 和参数 | [AI 对话助手](/7-ai-tools/7.1-efficiency/ai-chat) |
| Perplexity | 搜索 OpenCV 最佳实践 | [AI 搜索](/7-ai-tools/7.1-efficiency/ai-search) |

## 参考资料

- [OpenCV 官方文档](https://docs.opencv.org/4.x/)
- [OpenCV-Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Learn OpenCV — learnopencv.com](https://learnopencv.com/)
- [NumPy 与 OpenCV 图像操作](https://numpy.org/doc/stable/reference/arrays.ndarray.html)
