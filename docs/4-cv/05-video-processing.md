---
title: "视频处理"
module: "cv"
difficulty: "intermediate"
interviewFrequency: "medium"
tags:
  - "视频处理"
  - "帧提取"
  - "实时处理"
  - "摄像头"
  - "VideoCapture"
codeExample: "04-cv/opencv/"
relatedEntries:
  - "/4-cv/01-opencv-basics"
  - "/4-cv/06-yolo-detection"
prerequisites:
  - "/4-cv/01-opencv-basics"
  - "/4-cv/02-image-processing"
estimatedTime: "45min"
toolReferences:
  - name: "Cursor"
    usage: "辅助编写视频处理管道代码"
    link: "/7-ai-tools/7.1-efficiency/ai-coding"
  - name: "ChatGPT"
    usage: "解释视频编解码和帧率控制"
    link: "/7-ai-tools/7.1-efficiency/ai-chat"
  - name: "Perplexity"
    usage: "搜索视频处理优化方案"
    link: "/7-ai-tools/7.1-efficiency/ai-search"
---

# 视频处理

## 概念说明

视频本质上是连续的图像帧序列。OpenCV 提供了 `VideoCapture` 和 `VideoWriter` 类来读取和写入视频，支持文件视频和实时摄像头流。视频处理是实时目标检测、视频监控等应用的基础。

### 视频处理流程

```mermaid
graph LR
    A[视频源<br/>文件/摄像头/RTSP] --> B[逐帧读取<br/>VideoCapture]
    B --> C[帧处理<br/>检测/分割/增强]
    C --> D[结果输出<br/>显示/保存/推流]
    
    style B fill:#e1f5fe
    style C fill:#fff3e0
```

## 核心原理

### 1. 视频读取 — `cv2.VideoCapture`

```python
import cv2

# 从文件读取
cap = cv2.VideoCapture("video.mp4")

# 从摄像头读取（0 = 默认摄像头）
cap = cv2.VideoCapture(0)

# 从 RTSP 流读取
cap = cv2.VideoCapture("rtsp://admin:password@192.168.1.100:554/stream")

# 检查是否打开成功
if not cap.isOpened():
    raise RuntimeError("无法打开视频源")
```

### 2. 视频属性

```python
# 获取视频属性
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps if fps > 0 else 0

print(f"分辨率: {width}x{height}")
print(f"帧率: {fps} FPS")
print(f"总帧数: {frame_count}")
print(f"时长: {duration:.1f} 秒")
```

**常用属性常量：**

| 属性 | 常量 | 说明 |
|------|------|------|
| 宽度 | `CAP_PROP_FRAME_WIDTH` | 帧宽度（像素） |
| 高度 | `CAP_PROP_FRAME_HEIGHT` | 帧高度（像素） |
| 帧率 | `CAP_PROP_FPS` | 每秒帧数 |
| 总帧数 | `CAP_PROP_FRAME_COUNT` | 视频总帧数 |
| 当前帧 | `CAP_PROP_POS_FRAMES` | 当前帧位置 |
| 编解码器 | `CAP_PROP_FOURCC` | 视频编解码器 |

### 3. 逐帧处理

```python
while True:
    ret, frame = cap.read()
    if not ret:
        break  # 视频结束或读取失败
    
    # 处理帧（例如：灰度转换 + 边缘检测）
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # 显示结果
    cv2.imshow("Edges", edges)
    
    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 4. 视频写入 — `cv2.VideoWriter`

```python
# 定义编解码器和输出
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 编码
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 处理帧
    processed = cv2.GaussianBlur(frame, (5, 5), 0)
    
    # 写入输出视频
    out.write(processed)

out.release()
```

**常用编解码器：**

| FourCC | 格式 | 说明 |
|--------|------|------|
| `mp4v` | MP4 | 通用 MP4 编码 |
| `XVID` | AVI | 常用 AVI 编码 |
| `MJPG` | AVI | Motion JPEG，质量高 |
| `H264` | MP4 | H.264 编码，压缩率高 |

### 5. 帧跳跃与定位

```python
# 跳转到指定帧
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)  # 跳到第 100 帧

# 跳转到指定时间（毫秒）
cap.set(cv2.CAP_PROP_POS_MSEC, 5000)  # 跳到第 5 秒

# 每隔 N 帧处理一次（降低计算量）
frame_interval = 5
frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1
    if frame_id % frame_interval != 0:
        continue
    # 处理帧...
```

### 6. 实时处理性能优化

```mermaid
graph TB
    subgraph 优化策略
        A["降低分辨率<br/>resize 到 640x480"]
        B["跳帧处理<br/>每 N 帧处理一次"]
        C["多线程读取<br/>读取和处理分离"]
        D["GPU 加速<br/>CUDA / OpenCL"]
        E["模型优化<br/>TensorRT / ONNX"]
    end
```

```python
import threading
import queue

class VideoStream:
    """多线程视频读取，避免 I/O 阻塞。"""
    
    def __init__(self, src=0, queue_size=128):
        self.cap = cv2.VideoCapture(src)
        self.queue = queue.Queue(maxsize=queue_size)
        self.stopped = False
    
    def start(self):
        thread = threading.Thread(target=self._read, daemon=True)
        thread.start()
        return self
    
    def _read(self):
        while not self.stopped:
            if not self.queue.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    break
                self.queue.put(frame)
    
    def read(self):
        return self.queue.get()
    
    def stop(self):
        self.stopped = True
        self.cap.release()
```

## 代码示例

> 💻 完整可运行代码：[code-examples/04-cv/opencv/03_video_processing.py](https://github.com/your-repo/tree/main/code-examples/04-cv/opencv/03_video_processing.py)
> 🐍 Python 版本：3.11+
> 📦 依赖：numpy（模拟模式）、opencv-python>=4.8（完整模式）

## 实战要点

**生产环境视频处理：**
- **帧率控制**：实时检测时，如果模型推理慢于帧率，需要跳帧或异步处理
- **内存管理**：长时间运行注意内存泄漏，及时释放帧数据
- **错误恢复**：网络摄像头可能断连，需要重连机制
- **多路视频**：多摄像头场景用多线程/多进程并行处理

**常见陷阱：**
- 忘记调用 `cap.release()` 导致资源泄漏
- `waitKey(1)` 的返回值需要 `& 0xFF` 处理
- 视频写入的分辨率必须与帧尺寸一致

## 常见面试题

### Q1: 如何优化实时视频处理的性能？

**难度**：⭐⭐⭐ | **频率**：🔥🔥

**答题思路**：多个维度的优化策略 → 具体方法

**标准答案**：(1) 降低输入分辨率，resize 到模型需要的尺寸；(2) 跳帧处理，每 N 帧处理一次；(3) 多线程分离读取和处理，避免 I/O 阻塞；(4) 模型优化，使用 TensorRT/ONNX Runtime 加速推理；(5) 批处理，累积多帧一起推理；(6) GPU 加速，使用 CUDA 版 OpenCV。

**深入追问**：
- 多线程和多进程在视频处理中如何选择？（I/O 密集用多线程，CPU 密集用多进程）
- 如何处理网络摄像头的延迟和丢帧？（只处理最新帧，丢弃队列中的旧帧）

## 推荐工具

> 📌 以下工具可帮助你更高效地学习和实践本知识点，详见 [模块 7：AI 使用与实践](/7-ai-tools/)

| 工具 | 用途 | 详情 |
|------|------|------|
| Cursor | 辅助编写视频处理代码 | [AI 编程辅助](/7-ai-tools/7.1-efficiency/ai-coding) |
| ChatGPT | 解释视频编解码原理 | [AI 对话助手](/7-ai-tools/7.1-efficiency/ai-chat) |
| Perplexity | 搜索视频处理优化方案 | [AI 搜索](/7-ai-tools/7.1-efficiency/ai-search) |

## 参考资料

- [OpenCV 视频处理教程](https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html)
- [OpenCV VideoCapture API](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html)
- [FFmpeg — 视频编解码工具](https://ffmpeg.org/)
- [GStreamer — 多媒体框架](https://gstreamer.freedesktop.org/)
