"""
图像基础操作模拟 — 图像创建/属性/通道操作/ROI

知识点：图像表示（NumPy 数组）、像素操作、通道分离与合并、
       ROI（感兴趣区域）、图像属性、颜色空间基础、
       图像拼接、图像翻转与旋转

Python 版本：3.11+
依赖：numpy>=1.24（模拟模式）
真实环境依赖：opencv-python>=4.8（pip install opencv-python）
最后验证：2024-12-01

真实库安装：
  pip install opencv-python          # 基础版
  pip install opencv-contrib-python  # 扩展版（含 SIFT/SURF 等）
  pip install opencv-python-headless # 无 GUI 版（服务器环境）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# ============================================================
# 1. 模拟图像类 — 用 NumPy 数组表示图像
# ============================================================

@dataclass
class MockImage:
    """模拟 OpenCV 图像对象。

    OpenCV 中图像本质是 NumPy ndarray：
    - 灰度图: shape = (height, width)
    - 彩色图: shape = (height, width, 3)  # BGR 顺序
    - 带透明通道: shape = (height, width, 4)  # BGRA
    """
    data: np.ndarray
    name: str = "unnamed"

    @property
    def height(self) -> int:
        """图像高度（行数）。"""
        return self.data.shape[0]

    @property
    def width(self) -> int:
        """图像宽度（列数）。"""
        return self.data.shape[1]

    @property
    def channels(self) -> int:
        """通道数。"""
        return self.data.shape[2] if len(self.data.shape) == 3 else 1

    @property
    def dtype(self) -> np.dtype:
        """数据类型（通常为 uint8）。"""
        return self.data.dtype

    @property
    def size(self) -> int:
        """总像素数（height * width * channels）。"""
        return self.data.size

    def info(self) -> dict[str, Any]:
        """获取图像属性信息。"""
        return {
            "名称": self.name,
            "尺寸": f"{self.width}x{self.height}",
            "通道数": self.channels,
            "数据类型": str(self.dtype),
            "形状": self.data.shape,
            "总像素": self.size,
            "内存占用": f"{self.data.nbytes / 1024:.1f} KB",
        }


# ============================================================
# 2. 模拟 OpenCV 图像 I/O
# ============================================================

class MockCV2:
    """模拟 OpenCV (cv2) 核心图像操作。

    真实 OpenCV 使用示例：
        import cv2
        img = cv2.imread("image.jpg")          # 读取图像
        cv2.imshow("window", img)               # 显示图像
        cv2.imwrite("output.jpg", img)          # 保存图像
    """

    # 模拟 imread 标志
    IMREAD_COLOR = 1        # 彩色模式（默认）
    IMREAD_GRAYSCALE = 0    # 灰度模式
    IMREAD_UNCHANGED = -1   # 包含 Alpha 通道

    # 模拟颜色转换代码
    COLOR_BGR2GRAY = 6
    COLOR_BGR2RGB = 4
    COLOR_RGB2BGR = 5
    COLOR_BGR2HSV = 40
    COLOR_HSV2BGR = 54

    @staticmethod
    def imread(filepath: str, flags: int = 1) -> MockImage:
        """模拟读取图像。

        真实 OpenCV：
            img = cv2.imread("photo.jpg")
            img_gray = cv2.imread("photo.jpg", cv2.IMREAD_GRAYSCALE)
        """
        print(f"  📖 模拟读取图像: {filepath} (flags={flags})")

        # 根据标志创建不同类型的模拟图像
        if flags == MockCV2.IMREAD_GRAYSCALE:
            data = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        elif flags == MockCV2.IMREAD_UNCHANGED:
            data = np.random.randint(0, 256, (480, 640, 4), dtype=np.uint8)
        else:
            data = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        return MockImage(data=data, name=filepath)

    @staticmethod
    def imwrite(filepath: str, image: MockImage) -> bool:
        """模拟保存图像。

        真实 OpenCV：
            cv2.imwrite("output.jpg", img)
            cv2.imwrite("output.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        """
        print(f"  💾 模拟保存图像: {filepath} ({image.width}x{image.height})")
        return True

    @staticmethod
    def cvtColor(image: MockImage, code: int) -> MockImage:
        """模拟颜色空间转换。

        真实 OpenCV：
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        """
        conversions = {
            MockCV2.COLOR_BGR2GRAY: "BGR→灰度",
            MockCV2.COLOR_BGR2RGB: "BGR→RGB",
            MockCV2.COLOR_RGB2BGR: "RGB→BGR",
            MockCV2.COLOR_BGR2HSV: "BGR→HSV",
            MockCV2.COLOR_HSV2BGR: "HSV→BGR",
        }
        conv_name = conversions.get(code, f"未知转换({code})")
        print(f"  🎨 颜色转换: {conv_name}")

        if code == MockCV2.COLOR_BGR2GRAY:
            # 灰度转换：加权平均 B*0.114 + G*0.587 + R*0.299
            gray_data = np.mean(image.data, axis=2).astype(np.uint8)
            return MockImage(data=gray_data, name=f"{image.name}_gray")
        else:
            # 其他转换保持相同形状
            return MockImage(data=image.data.copy(), name=f"{image.name}_{conv_name}")

    @staticmethod
    def resize(image: MockImage, dsize: tuple[int, int]) -> MockImage:
        """模拟图像缩放。

        真实 OpenCV：
            resized = cv2.resize(img, (320, 240))
            resized = cv2.resize(img, None, fx=0.5, fy=0.5)
        """
        w, h = dsize
        print(f"  📐 缩放: {image.width}x{image.height} → {w}x{h}")

        if image.channels > 1:
            new_data = np.random.randint(0, 256, (h, w, image.channels), dtype=np.uint8)
        else:
            new_data = np.random.randint(0, 256, (h, w), dtype=np.uint8)

        return MockImage(data=new_data, name=f"{image.name}_resized")

    @staticmethod
    def flip(image: MockImage, flip_code: int) -> MockImage:
        """模拟图像翻转。

        flip_code: 0=垂直翻转, 1=水平翻转, -1=同时翻转
        真实 OpenCV：
            flipped = cv2.flip(img, 1)  # 水平翻转
        """
        flip_names = {0: "垂直", 1: "水平", -1: "垂直+水平"}
        print(f"  🔄 翻转: {flip_names.get(flip_code, '未知')}")
        return MockImage(data=np.flip(image.data, axis=flip_code if flip_code >= 0 else None).copy(),
                         name=f"{image.name}_flipped")


# ============================================================
# 3. 通道操作
# ============================================================

class ChannelOps:
    """图像通道操作工具类。"""

    @staticmethod
    def split(image: MockImage) -> list[MockImage]:
        """分离通道。

        真实 OpenCV：
            b, g, r = cv2.split(img)
        """
        if image.channels == 1:
            print("  ⚠️ 灰度图只有一个通道")
            return [image]

        channels = []
        channel_names = ["Blue", "Green", "Red"] if image.channels == 3 else ["B", "G", "R", "A"]
        for i in range(image.channels):
            ch = MockImage(
                data=image.data[:, :, i].copy(),
                name=f"{image.name}_{channel_names[i]}",
            )
            channels.append(ch)
        print(f"  🔀 通道分离: {image.channels} 个通道 → {[c.name for c in channels]}")
        return channels

    @staticmethod
    def merge(channels: list[MockImage]) -> MockImage:
        """合并通道。

        真实 OpenCV：
            merged = cv2.merge([b, g, r])
        """
        stacked = np.stack([ch.data for ch in channels], axis=2)
        print(f"  🔗 通道合并: {len(channels)} 个通道 → 多通道图像")
        return MockImage(data=stacked, name="merged")

    @staticmethod
    def swap_channels(image: MockImage, order: tuple[int, ...] = (2, 1, 0)) -> MockImage:
        """交换通道顺序（如 BGR → RGB）。"""
        if image.channels == 1:
            return image
        new_data = image.data[:, :, list(order)].copy()
        print(f"  🔄 通道交换: {order}")
        return MockImage(data=new_data, name=f"{image.name}_swapped")


# ============================================================
# 4. ROI（感兴趣区域）操作
# ============================================================

class ROIOperator:
    """ROI（Region of Interest）操作。"""

    @staticmethod
    def extract(image: MockImage, x: int, y: int, w: int, h: int) -> MockImage:
        """提取 ROI 区域。

        真实 OpenCV：
            roi = img[y:y+h, x:x+w]  # NumPy 切片
        """
        # 边界检查
        x = max(0, min(x, image.width - 1))
        y = max(0, min(y, image.height - 1))
        w = min(w, image.width - x)
        h = min(h, image.height - y)

        if image.channels > 1:
            roi_data = image.data[y:y+h, x:x+w, :].copy()
        else:
            roi_data = image.data[y:y+h, x:x+w].copy()

        print(f"  ✂️ 提取 ROI: ({x},{y}) 大小 {w}x{h}")
        return MockImage(data=roi_data, name=f"{image.name}_roi")

    @staticmethod
    def paste(target: MockImage, roi: MockImage, x: int, y: int) -> MockImage:
        """将 ROI 粘贴到目标图像。

        真实 OpenCV：
            img[y:y+h, x:x+w] = roi  # 直接赋值
        """
        result = MockImage(data=target.data.copy(), name=f"{target.name}_pasted")
        h, w = roi.height, roi.width

        # 边界裁剪
        paste_h = min(h, target.height - y)
        paste_w = min(w, target.width - x)

        if target.channels > 1 and roi.channels > 1:
            result.data[y:y+paste_h, x:x+paste_w, :] = roi.data[:paste_h, :paste_w, :]
        else:
            result.data[y:y+paste_h, x:x+paste_w] = roi.data[:paste_h, :paste_w]

        print(f"  📋 粘贴 ROI: 位置 ({x},{y}) 大小 {paste_w}x{paste_h}")
        return result

    @staticmethod
    def draw_rectangle(image: MockImage, x: int, y: int, w: int, h: int,
                       color: tuple[int, ...] = (0, 255, 0)) -> MockImage:
        """在图像上绘制矩形框。

        真实 OpenCV：
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        """
        result = MockImage(data=image.data.copy(), name=f"{image.name}_rect")
        # 简化绘制：设置边框像素
        if image.channels >= 3:
            result.data[y, x:x+w, :3] = color[:3]          # 上边
            result.data[y+h-1, x:x+w, :3] = color[:3]      # 下边
            result.data[y:y+h, x, :3] = color[:3]           # 左边
            result.data[y:y+h, x+w-1, :3] = color[:3]       # 右边
        print(f"  🟩 绘制矩形: ({x},{y}) {w}x{h} 颜色={color}")
        return result


# ============================================================
# 5. 图像创建工具
# ============================================================

class ImageFactory:
    """图像创建工厂。"""

    @staticmethod
    def create_blank(width: int, height: int, channels: int = 3,
                     color: tuple[int, ...] = (0, 0, 0)) -> MockImage:
        """创建纯色图像。

        真实 OpenCV + NumPy：
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            blank[:] = (255, 0, 0)  # 蓝色（BGR）
        """
        if channels == 1:
            data = np.full((height, width), color[0], dtype=np.uint8)
        else:
            data = np.full((height, width, channels), color[:channels], dtype=np.uint8)
        print(f"  🖼️ 创建空白图像: {width}x{height}x{channels} 颜色={color}")
        return MockImage(data=data, name="blank")

    @staticmethod
    def create_gradient(width: int, height: int, direction: str = "horizontal") -> MockImage:
        """创建渐变图像。"""
        if direction == "horizontal":
            gradient = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
        else:
            gradient = np.tile(
                np.linspace(0, 255, height, dtype=np.uint8).reshape(-1, 1),
                (1, width),
            )
        # 转为 3 通道
        data = np.stack([gradient, gradient, gradient], axis=2)
        print(f"  🌈 创建渐变图像: {width}x{height} 方向={direction}")
        return MockImage(data=data, name=f"gradient_{direction}")

    @staticmethod
    def create_checkerboard(width: int, height: int, block_size: int = 32) -> MockImage:
        """创建棋盘格图像（常用于相机标定）。"""
        board = np.zeros((height, width), dtype=np.uint8)
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                if ((y // block_size) + (x // block_size)) % 2 == 0:
                    board[y:y+block_size, x:x+block_size] = 255
        data = np.stack([board, board, board], axis=2)
        print(f"  ♟️ 创建棋盘格: {width}x{height} 块大小={block_size}")
        return MockImage(data=data, name="checkerboard")

    @staticmethod
    def create_noise(width: int, height: int, noise_type: str = "gaussian") -> MockImage:
        """创建噪声图像。"""
        if noise_type == "gaussian":
            data = np.random.normal(128, 50, (height, width, 3)).clip(0, 255).astype(np.uint8)
        elif noise_type == "salt_pepper":
            data = np.full((height, width, 3), 128, dtype=np.uint8)
            # 椒盐噪声
            salt = np.random.random((height, width)) > 0.95
            pepper = np.random.random((height, width)) < 0.05
            data[salt] = 255
            data[pepper] = 0
        else:
            data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        print(f"  📡 创建噪声图像: {width}x{height} 类型={noise_type}")
        return MockImage(data=data, name=f"noise_{noise_type}")


# ============================================================
# 6. 演示函数
# ============================================================

def demo_image_creation() -> None:
    """演示图像创建。"""
    print("\n" + "=" * 60)
    print("1. 图像创建与属性")
    print("=" * 60)

    factory = ImageFactory()

    # 创建各种图像
    blank = factory.create_blank(640, 480, color=(255, 128, 0))
    gradient = factory.create_gradient(640, 480, "horizontal")
    checker = factory.create_checkerboard(640, 480, block_size=64)
    noise = factory.create_noise(640, 480, "gaussian")

    # 显示属性
    for img in [blank, gradient, checker, noise]:
        info = img.info()
        print(f"\n  📊 {info['名称']}: {info['尺寸']} | "
              f"{info['通道数']}通道 | {info['数据类型']} | {info['内存占用']}")


def demo_io_operations() -> None:
    """演示图像读写操作。"""
    print("\n" + "=" * 60)
    print("2. 图像读写操作")
    print("=" * 60)

    cv2 = MockCV2()

    # 读取图像（不同模式）
    img_color = cv2.imread("photo.jpg", MockCV2.IMREAD_COLOR)
    img_gray = cv2.imread("photo.jpg", MockCV2.IMREAD_GRAYSCALE)
    img_alpha = cv2.imread("logo.png", MockCV2.IMREAD_UNCHANGED)

    print(f"\n  彩色图: {img_color.data.shape}")
    print(f"  灰度图: {img_gray.data.shape}")
    print(f"  带Alpha: {img_alpha.data.shape}")

    # 保存图像
    cv2.imwrite("output_color.jpg", img_color)
    cv2.imwrite("output_gray.png", img_gray)

    # 颜色转换
    gray = cv2.cvtColor(img_color, MockCV2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(img_color, MockCV2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img_color, MockCV2.COLOR_BGR2HSV)

    print(f"\n  灰度转换后: {gray.data.shape}")
    print(f"  RGB 转换后: {rgb.data.shape}")
    print(f"  HSV 转换后: {hsv.data.shape}")


def demo_channel_operations() -> None:
    """演示通道操作。"""
    print("\n" + "=" * 60)
    print("3. 通道分离与合并")
    print("=" * 60)

    cv2 = MockCV2()
    img = cv2.imread("photo.jpg")
    ops = ChannelOps()

    # 分离通道
    channels = ops.split(img)
    for ch in channels:
        print(f"    通道 {ch.name}: shape={ch.data.shape}")

    # 合并通道
    merged = ops.merge(channels)
    print(f"  合并后: shape={merged.data.shape}")

    # 交换通道（BGR → RGB）
    swapped = ops.swap_channels(img, (2, 1, 0))
    print(f"  交换后: shape={swapped.data.shape}")

    print("\n  💡 OpenCV 默认使用 BGR 顺序，matplotlib 使用 RGB 顺序")
    print("  💡 显示时需要转换: plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))")


def demo_roi_operations() -> None:
    """演示 ROI 操作。"""
    print("\n" + "=" * 60)
    print("4. ROI（感兴趣区域）操作")
    print("=" * 60)

    cv2 = MockCV2()
    img = cv2.imread("photo.jpg")
    roi_op = ROIOperator()

    # 提取 ROI
    roi = roi_op.extract(img, x=100, y=50, w=200, h=150)
    print(f"  ROI 尺寸: {roi.width}x{roi.height}")

    # 粘贴 ROI 到另一位置
    result = roi_op.paste(img, roi, x=300, y=200)
    print(f"  粘贴后图像: {result.width}x{result.height}")

    # 绘制矩形标注
    annotated = roi_op.draw_rectangle(img, x=100, y=50, w=200, h=150, color=(0, 255, 0))
    print(f"  标注后图像: {annotated.width}x{annotated.height}")


def demo_pixel_operations() -> None:
    """演示像素级操作。"""
    print("\n" + "=" * 60)
    print("5. 像素级操作")
    print("=" * 60)

    cv2 = MockCV2()
    img = cv2.imread("photo.jpg")

    # 访问单个像素
    pixel = img.data[100, 200]  # (B, G, R)
    print(f"  像素 (200, 100) 的 BGR 值: {pixel}")

    # 修改像素
    img.data[100, 200] = [255, 0, 0]  # 设为蓝色
    print(f"  修改后像素值: {img.data[100, 200]}")

    # 批量像素操作（NumPy 向量化）
    # 将图像亮度提升 50
    brightened = np.clip(img.data.astype(np.int16) + 50, 0, 255).astype(np.uint8)
    print(f"  亮度提升后均值: {brightened.mean():.1f}")

    # 图像混合（加权叠加）
    img2_data = np.random.randint(0, 256, img.data.shape, dtype=np.uint8)
    alpha = 0.7
    blended = np.clip(
        img.data.astype(np.float32) * alpha + img2_data.astype(np.float32) * (1 - alpha),
        0, 255,
    ).astype(np.uint8)
    print(f"  混合图像 (alpha={alpha}): shape={blended.shape}")

    print("\n  💡 像素操作尽量用 NumPy 向量化，避免 Python for 循环（慢 100x+）")
    print("  💡 真实 OpenCV: cv2.addWeighted(img1, 0.7, img2, 0.3, 0)")


def demo_resize_flip() -> None:
    """演示缩放与翻转。"""
    print("\n" + "=" * 60)
    print("6. 图像缩放与翻转")
    print("=" * 60)

    cv2 = MockCV2()
    img = cv2.imread("photo.jpg")

    # 缩放
    small = cv2.resize(img, (320, 240))
    large = cv2.resize(img, (1280, 960))
    print(f"  缩小: {small.width}x{small.height}")
    print(f"  放大: {large.width}x{large.height}")

    # 翻转
    h_flip = cv2.flip(img, 1)   # 水平翻转
    v_flip = cv2.flip(img, 0)   # 垂直翻转
    hv_flip = cv2.flip(img, -1) # 同时翻转

    print(f"\n  💡 缩放插值方法:")
    print(f"    cv2.INTER_NEAREST — 最近邻（最快，质量低）")
    print(f"    cv2.INTER_LINEAR  — 双线性（默认，平衡）")
    print(f"    cv2.INTER_CUBIC   — 双三次（放大推荐）")
    print(f"    cv2.INTER_AREA    — 区域（缩小推荐）")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行所有图像基础操作演示。"""
    print("🐍 图像基础操作模拟 — OpenCV 核心概念")
    print("=" * 60)

    demo_image_creation()
    demo_io_operations()
    demo_channel_operations()
    demo_roi_operations()
    demo_pixel_operations()
    demo_resize_flip()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. OpenCV 图像本质是 NumPy ndarray，通道顺序为 BGR")
    print("  2. 像素操作优先使用 NumPy 向量化，避免 Python 循环")
    print("  3. ROI 通过 NumPy 切片实现: roi = img[y:y+h, x:x+w]")
    print("  4. 通道分离/合并: cv2.split() / cv2.merge()")
    print("  5. 颜色转换: cv2.cvtColor(img, cv2.COLOR_BGR2RGB)")
    print("  6. 缩放时注意选择合适的插值方法")


if __name__ == "__main__":
    main()
