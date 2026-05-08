"""
视频处理模拟 — 帧读取/处理/写入/实时摄像头

知识点：视频读取（VideoCapture）、帧处理循环、视频写入（VideoWriter）、
       编解码器（FourCC）、帧率控制、实时摄像头处理、
       视频属性获取、多视频流处理

Python 版本：3.11+
依赖：numpy>=1.24（模拟模式）
真实环境依赖：opencv-python>=4.8（pip install opencv-python）
最后验证：2024-12-01

真实库安装：
  pip install opencv-python          # 包含视频 I/O 支持
  pip install opencv-contrib-python  # 扩展版
"""

from __future__ import annotations

import time
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import numpy as np

# ============================================================
# 1. 视频属性与编解码器
# ============================================================

class FourCC:
    """视频编解码器 FourCC 代码。

    真实 OpenCV：
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    """
    MP4V = "mp4v"    # MPEG-4 编码（.mp4）
    XVID = "XVID"    # Xvid 编码（.avi）
    MJPG = "MJPG"    # Motion JPEG（.avi）
    H264 = "H264"    # H.264 编码（.mp4，需要额外编解码器）
    AVC1 = "avc1"    # H.264 的另一种标识

    @staticmethod
    def info() -> dict[str, str]:
        """编解码器说明。"""
        return {
            "mp4v": "MPEG-4 Part 2 — 兼容性好，文件较大",
            "XVID": "Xvid MPEG-4 — 开源，适合 .avi",
            "MJPG": "Motion JPEG — 每帧独立压缩，编辑友好",
            "H264": "H.264/AVC — 压缩率高，需要 ffmpeg 支持",
            "avc1": "H.264 macOS 版本标识",
        }


@dataclass
class VideoProperties:
    """视频属性。"""
    width: int = 640
    height: int = 480
    fps: float = 30.0
    frame_count: int = 300
    fourcc: str = "mp4v"
    duration: float = 0.0

    def __post_init__(self) -> None:
        if self.fps > 0:
            self.duration = self.frame_count / self.fps

    def summary(self) -> str:
        """属性摘要。"""
        return (f"  分辨率: {self.width}x{self.height} | FPS: {self.fps} | "
                f"帧数: {self.frame_count} | 时长: {self.duration:.1f}s | "
                f"编码: {self.fourcc}")


# ============================================================
# 2. 模拟 VideoCapture
# ============================================================

class MockVideoCapture:
    """模拟 OpenCV VideoCapture。

    真实 OpenCV：
        cap = cv2.VideoCapture("video.mp4")       # 打开视频文件
        cap = cv2.VideoCapture(0)                  # 打开默认摄像头
        cap = cv2.VideoCapture("rtsp://...")        # 打开网络流
    """

    # 属性 ID 常量
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_COUNT = 7
    CAP_PROP_POS_FRAMES = 1
    CAP_PROP_POS_MSEC = 0
    CAP_PROP_FOURCC = 6

    def __init__(self, source: str | int = 0):
        """初始化视频捕获。

        Args:
            source: 视频文件路径或摄像头索引（0=默认摄像头）
        """
        self.source = source
        self._is_opened = True
        self._current_frame = 0

        # 根据来源设置属性
        if isinstance(source, int):
            self._props = VideoProperties(
                width=640, height=480, fps=30.0,
                frame_count=-1, fourcc="MJPG",
            )
            self._source_type = "camera"
            print(f"  📹 打开摄像头: 设备 {source}")
        else:
            self._props = VideoProperties(
                width=1920, height=1080, fps=30.0,
                frame_count=900, fourcc="mp4v",
            )
            self._source_type = "file"
            print(f"  🎬 打开视频: {source}")

        print(f"  {self._props.summary()}")

    def isOpened(self) -> bool:
        """检查是否成功打开。"""
        return self._is_opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        """读取一帧。

        返回 (success, frame)。
        真实 OpenCV：
            ret, frame = cap.read()
        """
        if not self._is_opened:
            return False, None

        # 文件模式：检查是否到末尾
        if self._source_type == "file" and self._current_frame >= self._props.frame_count:
            return False, None

        # 生成模拟帧
        frame = np.random.randint(
            0, 256,
            (self._props.height, self._props.width, 3),
            dtype=np.uint8,
        )
        self._current_frame += 1
        return True, frame

    def get(self, prop_id: int) -> float:
        """获取视频属性。

        真实 OpenCV：
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            fps = cap.get(cv2.CAP_PROP_FPS)
        """
        prop_map = {
            self.CAP_PROP_FRAME_WIDTH: float(self._props.width),
            self.CAP_PROP_FRAME_HEIGHT: float(self._props.height),
            self.CAP_PROP_FPS: self._props.fps,
            self.CAP_PROP_FRAME_COUNT: float(self._props.frame_count),
            self.CAP_PROP_POS_FRAMES: float(self._current_frame),
        }
        return prop_map.get(prop_id, 0.0)

    def set(self, prop_id: int, value: float) -> bool:
        """设置视频属性（如跳转到指定帧）。

        真实 OpenCV：
            cap.set(cv2.CAP_PROP_POS_FRAMES, 100)  # 跳到第 100 帧
        """
        if prop_id == self.CAP_PROP_POS_FRAMES:
            self._current_frame = int(value)
            print(f"  ⏩ 跳转到第 {int(value)} 帧")
            return True
        return False

    def release(self) -> None:
        """释放资源。"""
        self._is_opened = False
        print(f"  🔒 释放视频资源: {self.source}")

    def __enter__(self) -> MockVideoCapture:
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()


# ============================================================
# 3. 模拟 VideoWriter
# ============================================================

class MockVideoWriter:
    """模拟 OpenCV VideoWriter。

    真实 OpenCV：
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter('output.mp4', fourcc, 30.0, (640, 480))
        writer.write(frame)
        writer.release()
    """

    def __init__(self, filename: str, fourcc: str, fps: float,
                 frame_size: tuple[int, int]):
        self.filename = filename
        self.fourcc = fourcc
        self.fps = fps
        self.frame_size = frame_size
        self._frame_count = 0
        self._is_opened = True
        print(f"  📝 创建视频写入器: {filename}")
        print(f"     编码={fourcc}, FPS={fps}, 尺寸={frame_size[0]}x{frame_size[1]}")

    def isOpened(self) -> bool:
        return self._is_opened

    def write(self, frame: np.ndarray) -> None:
        """写入一帧。"""
        if not self._is_opened:
            return
        self._frame_count += 1

    def release(self) -> None:
        """释放资源并完成写入。"""
        self._is_opened = False
        duration = self._frame_count / self.fps if self.fps > 0 else 0
        print(f"  💾 视频写入完成: {self.filename}")
        print(f"     帧数={self._frame_count}, 时长={duration:.1f}s")

    def __enter__(self) -> MockVideoWriter:
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()


# ============================================================
# 4. 帧处理器
# ============================================================

class FrameProcessor:
    """视频帧处理工具集。"""

    @staticmethod
    def to_grayscale(frame: np.ndarray) -> np.ndarray:
        """帧转灰度。"""
        return np.mean(frame, axis=2).astype(np.uint8)

    @staticmethod
    def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
        """缩放帧（简化实现）。"""
        # 简化：直接裁剪/填充到目标尺寸
        result = np.zeros((height, width, 3), dtype=np.uint8)
        h = min(height, frame.shape[0])
        w = min(width, frame.shape[1])
        result[:h, :w] = frame[:h, :w]
        return result

    @staticmethod
    def add_timestamp(frame: np.ndarray, frame_num: int, fps: float) -> np.ndarray:
        """在帧上添加时间戳（模拟）。

        真实 OpenCV：
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
        """
        timestamp = frame_num / fps if fps > 0 else 0
        # 模拟在左上角写入时间戳（实际是修改像素）
        result = frame.copy()
        result[5:15, 5:100] = [0, 255, 0]  # 绿色条表示时间戳区域
        return result

    @staticmethod
    def detect_motion(prev_frame: np.ndarray, curr_frame: np.ndarray,
                      threshold: int = 30) -> tuple[bool, float]:
        """简单运动检测（帧差法）。

        真实 OpenCV：
            diff = cv2.absdiff(prev_gray, curr_gray)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            motion_ratio = np.sum(thresh > 0) / thresh.size
        """
        # 转灰度
        prev_gray = np.mean(prev_frame, axis=2)
        curr_gray = np.mean(curr_frame, axis=2)

        # 帧差
        diff = np.abs(prev_gray.astype(np.float64) - curr_gray.astype(np.float64))
        motion_pixels = np.sum(diff > threshold)
        motion_ratio = motion_pixels / diff.size

        has_motion = motion_ratio > 0.05  # 超过 5% 像素变化视为有运动
        return has_motion, motion_ratio

    @staticmethod
    def extract_keyframes(frame_count: int, interval: int = 30) -> list[int]:
        """提取关键帧索引（每隔 N 帧取一帧）。"""
        keyframes = list(range(0, frame_count, interval))
        print(f"  🔑 关键帧提取: 每 {interval} 帧取 1 帧, 共 {len(keyframes)} 帧")
        return keyframes


# ============================================================
# 5. 视频处理管道
# ============================================================

class VideoPipeline:
    """视频处理管道 — 串联多个处理步骤。"""

    def __init__(self) -> None:
        self.steps: list[tuple[str, Any]] = []

    def add_step(self, name: str, processor: Any) -> VideoPipeline:
        """添加处理步骤。"""
        self.steps.append((name, processor))
        return self

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """处理单帧。"""
        result = frame
        for name, proc in self.steps:
            if callable(proc):
                result = proc(result)
        return result

    def process_video(self, cap: MockVideoCapture, writer: MockVideoWriter,
                      max_frames: int = 100) -> dict[str, Any]:
        """处理整个视频。"""
        stats = {"processed": 0, "skipped": 0, "elapsed": 0.0}
        start_time = time.time()

        for i in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break

            processed = self.process_frame(frame)
            writer.write(processed)
            stats["processed"] += 1

        stats["elapsed"] = time.time() - start_time
        return stats


# ============================================================
# 6. 帧生成器（内存友好）
# ============================================================

def frame_generator(cap: MockVideoCapture,
                    max_frames: int = -1) -> Generator[tuple[int, np.ndarray], None, None]:
    """帧生成器 — 逐帧读取，节省内存。

    使用方式：
        for frame_num, frame in frame_generator(cap):
            process(frame)
    """
    frame_num = 0
    while True:
        if 0 < max_frames <= frame_num:
            break
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        yield frame_num, frame
        frame_num += 1


# ============================================================
# 7. 多视频流管理
# ============================================================

class MultiStreamManager:
    """多视频流管理器（如多摄像头同步处理）。"""

    def __init__(self) -> None:
        self.streams: dict[str, MockVideoCapture] = {}

    def add_stream(self, name: str, source: str | int) -> None:
        """添加视频流。"""
        cap = MockVideoCapture(source)
        if cap.isOpened():
            self.streams[name] = cap
            print(f"  ✅ 添加流: {name}")

    def read_all(self) -> dict[str, np.ndarray | None]:
        """同步读取所有流的当前帧。"""
        frames: dict[str, np.ndarray | None] = {}
        for name, cap in self.streams.items():
            ret, frame = cap.read()
            frames[name] = frame if ret else None
        return frames

    def release_all(self) -> None:
        """释放所有流。"""
        for name, cap in self.streams.items():
            cap.release()
        self.streams.clear()
        print(f"  🔒 所有视频流已释放")


# ============================================================
# 8. 演示函数
# ============================================================

def demo_video_read() -> None:
    """演示视频读取。"""
    print("\n" + "=" * 60)
    print("1. 视频读取与属性获取")
    print("=" * 60)

    cap = MockVideoCapture("sample_video.mp4")

    # 获取属性
    width = int(cap.get(MockVideoCapture.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(MockVideoCapture.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(MockVideoCapture.CAP_PROP_FPS)
    total = int(cap.get(MockVideoCapture.CAP_PROP_FRAME_COUNT))

    print(f"\n  视频属性: {width}x{height} @ {fps}fps, 共 {total} 帧")

    # 读取前 5 帧
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            print(f"  帧 {i}: shape={frame.shape}, mean={frame.mean():.1f}")

    # 跳转到指定帧
    cap.set(MockVideoCapture.CAP_PROP_POS_FRAMES, 100)
    ret, frame = cap.read()
    print(f"  跳转后读取: ret={ret}")

    cap.release()


def demo_video_write() -> None:
    """演示视频写入。"""
    print("\n" + "=" * 60)
    print("2. 视频写入")
    print("=" * 60)

    # 编解码器信息
    print("\n  编解码器选择:")
    for codec, desc in FourCC.info().items():
        print(f"    {codec}: {desc}")

    # 创建写入器
    writer = MockVideoWriter("output.mp4", FourCC.MP4V, 30.0, (640, 480))

    # 写入模拟帧
    for i in range(90):  # 3 秒视频
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        writer.write(frame)

    writer.release()


def demo_frame_processing() -> None:
    """演示帧处理。"""
    print("\n" + "=" * 60)
    print("3. 帧处理操作")
    print("=" * 60)

    processor = FrameProcessor()

    # 创建模拟帧
    frame1 = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    frame2 = frame1.copy()
    frame2[100:200, 100:200] = 255  # 模拟运动区域

    # 灰度转换
    gray = processor.to_grayscale(frame1)
    print(f"  灰度帧: {gray.shape}")

    # 缩放
    resized = processor.resize_frame(frame1, 320, 240)
    print(f"  缩放帧: {resized.shape}")

    # 添加时间戳
    stamped = processor.add_timestamp(frame1, frame_num=150, fps=30.0)
    print(f"  时间戳帧: {stamped.shape}")

    # 运动检测
    has_motion, ratio = processor.detect_motion(frame1, frame2)
    print(f"  运动检测: 有运动={has_motion}, 变化比例={ratio:.4f}")

    # 关键帧提取
    keyframes = processor.extract_keyframes(900, interval=30)


def demo_video_pipeline() -> None:
    """演示视频处理管道。"""
    print("\n" + "=" * 60)
    print("4. 视频处理管道")
    print("=" * 60)

    processor = FrameProcessor()

    # 构建处理管道
    pipeline = VideoPipeline()
    pipeline.add_step("resize", lambda f: processor.resize_frame(f, 320, 240))
    pipeline.add_step("grayscale_to_3ch",
                      lambda f: np.stack([processor.to_grayscale(f)] * 3, axis=2))

    print(f"  管道步骤: {[name for name, _ in pipeline.steps]}")

    # 处理视频
    cap = MockVideoCapture("input.mp4")
    writer = MockVideoWriter("processed.mp4", FourCC.MP4V, 30.0, (320, 240))

    stats = pipeline.process_video(cap, writer, max_frames=30)
    print(f"  处理统计: {stats}")

    cap.release()
    writer.release()


def demo_camera_simulation() -> None:
    """演示摄像头实时处理（模拟）。"""
    print("\n" + "=" * 60)
    print("5. 摄像头实时处理（模拟）")
    print("=" * 60)

    print("\n  真实 OpenCV 摄像头处理循环:")
    print("  ```python")
    print("  cap = cv2.VideoCapture(0)")
    print("  while True:")
    print("      ret, frame = cap.read()")
    print("      if not ret: break")
    print("      # 处理帧...")
    print("      cv2.imshow('Camera', frame)")
    print("      if cv2.waitKey(1) & 0xFF == ord('q'): break")
    print("  cap.release()")
    print("  cv2.destroyAllWindows()")
    print("  ```")

    # 模拟处理几帧
    cap = MockVideoCapture(0)  # 摄像头
    processor = FrameProcessor()
    prev_frame = None

    for i in range(5):
        ret, frame = cap.read()
        if not ret:
            break

        if prev_frame is not None:
            has_motion, ratio = processor.detect_motion(prev_frame, frame)
            status = "🔴 运动" if has_motion else "🟢 静止"
            print(f"  帧 {i}: {status} (变化={ratio:.4f})")

        prev_frame = frame

    cap.release()


def demo_multi_stream() -> None:
    """演示多视频流处理。"""
    print("\n" + "=" * 60)
    print("6. 多视频流管理")
    print("=" * 60)

    manager = MultiStreamManager()
    manager.add_stream("前置摄像头", 0)
    manager.add_stream("后置摄像头", 1)

    # 同步读取
    frames = manager.read_all()
    for name, frame in frames.items():
        if frame is not None:
            print(f"  {name}: shape={frame.shape}")

    manager.release_all()


def demo_frame_generator() -> None:
    """演示帧生成器。"""
    print("\n" + "=" * 60)
    print("7. 帧生成器（内存友好）")
    print("=" * 60)

    cap = MockVideoCapture("long_video.mp4")

    frame_count = 0
    for num, frame in frame_generator(cap, max_frames=10):
        frame_count += 1

    print(f"  生成器读取: {frame_count} 帧")
    print(f"  💡 生成器逐帧处理，不会一次性加载整个视频到内存")

    cap.release()


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行所有视频处理演示。"""
    print("🐍 视频处理模拟 — 帧读取/处理/写入")
    print("=" * 60)

    demo_video_read()
    demo_video_write()
    demo_frame_processing()
    demo_video_pipeline()
    demo_camera_simulation()
    demo_multi_stream()
    demo_frame_generator()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. VideoCapture 支持文件、摄像头、网络流三种来源")
    print("  2. 视频处理核心循环: read() → process() → write()")
    print("  3. 编解码器选择: mp4v（通用）、H264（高压缩）、MJPG（编辑友好）")
    print("  4. 实时处理注意帧率控制: cv2.waitKey(1)")
    print("  5. 大视频用生成器逐帧处理，避免内存溢出")
    print("  6. 运动检测基础: 帧差法 → 阈值 → 轮廓检测")


if __name__ == "__main__":
    main()
