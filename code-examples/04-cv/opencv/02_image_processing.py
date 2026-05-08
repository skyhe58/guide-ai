"""
图像处理模拟 — 灰度转换/二值化/滤波/边缘检测/形态学操作

知识点：灰度转换原理、阈值二值化（全局/自适应/Otsu）、
       空间滤波（均值/高斯/中值/双边）、边缘检测（Sobel/Canny）、
       形态学操作（腐蚀/膨胀/开运算/闭运算）、直方图均衡化

Python 版本：3.11+
依赖：numpy>=1.24（模拟模式）
真实环境依赖：opencv-python>=4.8（pip install opencv-python）
最后验证：2024-12-01

真实库安装：
  pip install opencv-python
  pip install matplotlib  # 用于可视化
"""

from __future__ import annotations

import numpy as np

# ============================================================
# 1. 灰度转换
# ============================================================

class GrayscaleConverter:
    """灰度转换工具。

    灰度转换原理：
    - 加权平均法（ITU-R BT.601）: Gray = 0.299*R + 0.587*G + 0.114*B
    - 最大值法: Gray = max(R, G, B)
    - 平均值法: Gray = (R + G + B) / 3
    """

    @staticmethod
    def weighted_average(image: np.ndarray) -> np.ndarray:
        """加权平均法（标准方法）。

        真实 OpenCV：
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        """
        # OpenCV BGR 顺序：B=0, G=1, R=2
        gray = (0.114 * image[:, :, 0] +
                0.587 * image[:, :, 1] +
                0.299 * image[:, :, 2]).astype(np.uint8)
        print(f"  🔲 加权平均灰度转换: {image.shape} → {gray.shape}")
        return gray

    @staticmethod
    def max_value(image: np.ndarray) -> np.ndarray:
        """最大值法。"""
        gray = np.max(image, axis=2).astype(np.uint8)
        print(f"  🔲 最大值灰度转换: {image.shape} → {gray.shape}")
        return gray

    @staticmethod
    def average(image: np.ndarray) -> np.ndarray:
        """平均值法。"""
        gray = np.mean(image, axis=2).astype(np.uint8)
        print(f"  🔲 平均值灰度转换: {image.shape} → {gray.shape}")
        return gray


# ============================================================
# 2. 阈值二值化
# ============================================================

class ThresholdProcessor:
    """阈值二值化处理器。

    二值化将灰度图转为只有 0 和 255 的黑白图。
    """

    # 阈值类型常量
    THRESH_BINARY = 0       # 大于阈值为白，否则为黑
    THRESH_BINARY_INV = 1   # 大于阈值为黑，否则为白
    THRESH_TRUNC = 2        # 大于阈值截断为阈值
    THRESH_TOZERO = 3       # 小于阈值设为 0
    THRESH_OTSU = 8         # Otsu 自动阈值

    @staticmethod
    def global_threshold(gray: np.ndarray, thresh: int = 127,
                         max_val: int = 255) -> tuple[int, np.ndarray]:
        """全局阈值二值化。

        真实 OpenCV：
            ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        """
        binary = np.where(gray > thresh, max_val, 0).astype(np.uint8)
        print(f"  ⬛ 全局阈值: thresh={thresh}, 白色像素占比={np.mean(binary > 0):.2%}")
        return thresh, binary

    @staticmethod
    def otsu_threshold(gray: np.ndarray, max_val: int = 255) -> tuple[int, np.ndarray]:
        """Otsu 自动阈值（大津法）。

        原理：遍历所有可能阈值，找到使类间方差最大的阈值。
        真实 OpenCV：
            ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        """
        # 计算直方图
        hist = np.histogram(gray.flatten(), bins=256, range=(0, 256))[0]
        hist = hist.astype(np.float64) / hist.sum()

        best_thresh = 0
        best_variance = 0.0

        for t in range(256):
            # 前景和背景概率
            w0 = hist[:t].sum()
            w1 = hist[t:].sum()
            if w0 == 0 or w1 == 0:
                continue

            # 前景和背景均值
            mu0 = np.sum(np.arange(t) * hist[:t]) / w0
            mu1 = np.sum(np.arange(t, 256) * hist[t:]) / w1

            # 类间方差
            variance = w0 * w1 * (mu0 - mu1) ** 2
            if variance > best_variance:
                best_variance = variance
                best_thresh = t

        binary = np.where(gray > best_thresh, max_val, 0).astype(np.uint8)
        print(f"  ⬛ Otsu 阈值: 自动计算 thresh={best_thresh}")
        return best_thresh, binary

    @staticmethod
    def adaptive_threshold(gray: np.ndarray, max_val: int = 255,
                           block_size: int = 11, c: int = 2) -> np.ndarray:
        """自适应阈值（局部阈值）。

        对每个像素，根据其邻域计算局部阈值。
        真实 OpenCV：
            binary = cv2.adaptiveThreshold(gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        """
        # 简化实现：使用均值滤波近似局部均值
        from scipy.ndimage import uniform_filter  # type: ignore
        try:
            local_mean = uniform_filter(gray.astype(np.float64), size=block_size)
        except ImportError:
            # 如果没有 scipy，用简单的滑动窗口近似
            kernel = np.ones((block_size, block_size)) / (block_size * block_size)
            local_mean = _convolve2d(gray.astype(np.float64), kernel)

        binary = np.where(gray > local_mean - c, max_val, 0).astype(np.uint8)
        print(f"  ⬛ 自适应阈值: block_size={block_size}, C={c}")
        return binary


# ============================================================
# 3. 空间滤波（卷积操作）
# ============================================================

def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """简单的 2D 卷积实现。

    真实 OpenCV：
        result = cv2.filter2D(img, -1, kernel)
    """
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2

    # 填充边界
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode='reflect')
    output = np.zeros_like(image, dtype=np.float64)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)

    return output


class SpatialFilter:
    """空间滤波器集合。"""

    @staticmethod
    def mean_filter(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
        """均值滤波（平滑/模糊）。

        真实 OpenCV：
            blurred = cv2.blur(img, (3, 3))
        """
        kernel = np.ones((ksize, ksize), dtype=np.float64) / (ksize * ksize)
        result = _convolve2d(gray, kernel).clip(0, 255).astype(np.uint8)
        print(f"  🌫️ 均值滤波: kernel={ksize}x{ksize}")
        return result

    @staticmethod
    def gaussian_filter(gray: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
        """高斯滤波。

        真实 OpenCV：
            blurred = cv2.GaussianBlur(img, (5, 5), 1.0)
        """
        # 生成高斯核
        ax = np.arange(-ksize // 2 + 1, ksize // 2 + 1)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()

        result = _convolve2d(gray, kernel).clip(0, 255).astype(np.uint8)
        print(f"  🌫️ 高斯滤波: kernel={ksize}x{ksize}, sigma={sigma}")
        return result

    @staticmethod
    def median_filter(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
        """中值滤波（去椒盐噪声效果好）。

        真实 OpenCV：
            denoised = cv2.medianBlur(img, 5)
        """
        ph = ksize // 2
        padded = np.pad(gray, ph, mode='reflect')
        result = np.zeros_like(gray)

        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                region = padded[i:i+ksize, j:j+ksize]
                result[i, j] = np.median(region)

        print(f"  🌫️ 中值滤波: kernel={ksize}x{ksize}")
        return result.astype(np.uint8)

    @staticmethod
    def bilateral_info() -> dict[str, str]:
        """双边滤波说明（计算量大，仅提供说明）。

        真实 OpenCV：
            smoothed = cv2.bilateralFilter(img, 9, 75, 75)
        """
        info = {
            "原理": "同时考虑空间距离和像素值差异",
            "优势": "保边去噪 — 平滑区域模糊，边缘保留",
            "参数": "d=滤波直径, sigmaColor=颜色空间sigma, sigmaSpace=坐标空间sigma",
            "适用": "人脸美颜、医学图像去噪",
        }
        print(f"  🌫️ 双边滤波（说明）: 保边去噪滤波器")
        return info


# ============================================================
# 4. 边缘检测
# ============================================================

class EdgeDetector:
    """边缘检测器。"""

    @staticmethod
    def sobel(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sobel 边缘检测。

        真实 OpenCV：
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        """
        # Sobel 算子
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

        grad_x = _convolve2d(gray.astype(np.float64), kernel_x)
        grad_y = _convolve2d(gray.astype(np.float64), kernel_y)

        # 梯度幅值
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)

        print(f"  🔍 Sobel 边缘检测: 梯度范围 [{magnitude.min()}, {magnitude.max()}]")
        return grad_x.astype(np.int16), grad_y.astype(np.int16), magnitude

    @staticmethod
    def canny(gray: np.ndarray, low_thresh: int = 50,
              high_thresh: int = 150) -> np.ndarray:
        """Canny 边缘检测（简化版）。

        Canny 步骤：
        1. 高斯模糊去噪
        2. 计算梯度幅值和方向（Sobel）
        3. 非极大值抑制（NMS）
        4. 双阈值检测 + 边缘连接

        真实 OpenCV：
            edges = cv2.Canny(gray, 50, 150)
        """
        # 步骤 1: 高斯模糊
        blurred = SpatialFilter.gaussian_filter(gray, ksize=5, sigma=1.4)

        # 步骤 2: Sobel 梯度
        _, _, magnitude = EdgeDetector.sobel(blurred)

        # 步骤 3+4: 简化的双阈值
        edges = np.zeros_like(magnitude)
        strong = magnitude > high_thresh
        weak = (magnitude >= low_thresh) & (magnitude <= high_thresh)
        edges[strong] = 255

        # 简化的边缘连接：弱边缘如果邻接强边缘则保留
        for i in range(1, edges.shape[0] - 1):
            for j in range(1, edges.shape[1] - 1):
                if weak[i, j]:
                    if np.any(edges[i-1:i+2, j-1:j+2] == 255):
                        edges[i, j] = 255

        print(f"  🔍 Canny 边缘检测: low={low_thresh}, high={high_thresh}, "
              f"边缘像素={np.sum(edges > 0)}")
        return edges

    @staticmethod
    def laplacian(gray: np.ndarray) -> np.ndarray:
        """Laplacian 边缘检测。

        真实 OpenCV：
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        """
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        result = _convolve2d(gray.astype(np.float64), kernel)
        result = np.abs(result)
        result = (result / result.max() * 255).clip(0, 255).astype(np.uint8)
        print(f"  🔍 Laplacian 边缘检测: 边缘像素={np.sum(result > 50)}")
        return result


# ============================================================
# 5. 形态学操作
# ============================================================

class MorphologyProcessor:
    """形态学操作处理器。

    形态学操作基于结构元素（kernel）对二值图像进行处理。
    """

    @staticmethod
    def get_kernel(shape: str = "rect", size: int = 3) -> np.ndarray:
        """获取结构元素。

        真实 OpenCV：
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        """
        if shape == "rect":
            return np.ones((size, size), dtype=np.uint8)
        elif shape == "cross":
            k = np.zeros((size, size), dtype=np.uint8)
            k[size // 2, :] = 1
            k[:, size // 2] = 1
            return k
        elif shape == "ellipse":
            k = np.zeros((size, size), dtype=np.uint8)
            center = size // 2
            for i in range(size):
                for j in range(size):
                    if (i - center)**2 + (j - center)**2 <= center**2:
                        k[i, j] = 1
            return k
        return np.ones((size, size), dtype=np.uint8)

    @staticmethod
    def erode(binary: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
        """腐蚀操作 — 缩小白色区域。

        真实 OpenCV：
            eroded = cv2.erode(binary, kernel, iterations=1)
        """
        result = binary.copy()
        kh, kw = kernel.shape
        ph, pw = kh // 2, kw // 2

        for _ in range(iterations):
            padded = np.pad(result, ((ph, ph), (pw, pw)), mode='constant', constant_values=0)
            temp = np.zeros_like(result)
            for i in range(result.shape[0]):
                for j in range(result.shape[1]):
                    region = padded[i:i+kh, j:j+kw]
                    # 腐蚀：所有核覆盖位置都为白色时才保留
                    if np.all(region[kernel == 1] == 255):
                        temp[i, j] = 255
            result = temp

        print(f"  🔽 腐蚀: iterations={iterations}, 白色像素 {np.sum(binary > 0)} → {np.sum(result > 0)}")
        return result

    @staticmethod
    def dilate(binary: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
        """膨胀操作 — 扩大白色区域。

        真实 OpenCV：
            dilated = cv2.dilate(binary, kernel, iterations=1)
        """
        result = binary.copy()
        kh, kw = kernel.shape
        ph, pw = kh // 2, kw // 2

        for _ in range(iterations):
            padded = np.pad(result, ((ph, ph), (pw, pw)), mode='constant', constant_values=0)
            temp = np.zeros_like(result)
            for i in range(result.shape[0]):
                for j in range(result.shape[1]):
                    region = padded[i:i+kh, j:j+kw]
                    # 膨胀：核覆盖位置有任一白色即保留
                    if np.any(region[kernel == 1] == 255):
                        temp[i, j] = 255
            result = temp

        print(f"  🔼 膨胀: iterations={iterations}, 白色像素 {np.sum(binary > 0)} → {np.sum(result > 0)}")
        return result

    @staticmethod
    def opening(binary: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """开运算 = 先腐蚀后膨胀（去除小白点/噪声）。

        真实 OpenCV：
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        """
        eroded = MorphologyProcessor.erode(binary, kernel)
        result = MorphologyProcessor.dilate(eroded, kernel)
        print(f"  🔄 开运算: 去除小白点噪声")
        return result

    @staticmethod
    def closing(binary: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """闭运算 = 先膨胀后腐蚀（填充小黑洞）。

        真实 OpenCV：
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        """
        dilated = MorphologyProcessor.dilate(binary, kernel)
        result = MorphologyProcessor.erode(dilated, kernel)
        print(f"  🔄 闭运算: 填充小黑洞")
        return result


# ============================================================
# 6. 直方图操作
# ============================================================

class HistogramProcessor:
    """直方图处理。"""

    @staticmethod
    def calc_histogram(gray: np.ndarray) -> np.ndarray:
        """计算灰度直方图。

        真实 OpenCV：
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        """
        hist = np.histogram(gray.flatten(), bins=256, range=(0, 256))[0]
        print(f"  📊 直方图: 最亮={np.argmax(hist)}, 均值={gray.mean():.1f}")
        return hist

    @staticmethod
    def equalize(gray: np.ndarray) -> np.ndarray:
        """直方图均衡化 — 增强对比度。

        真实 OpenCV：
            equalized = cv2.equalizeHist(gray)
        """
        hist = np.histogram(gray.flatten(), bins=256, range=(0, 256))[0]
        cdf = hist.cumsum()
        cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
        cdf_normalized = cdf_normalized.astype(np.uint8)

        equalized = cdf_normalized[gray]
        print(f"  📊 直方图均衡化: 标准差 {gray.std():.1f} → {equalized.std():.1f}")
        return equalized


# ============================================================
# 7. 演示函数
# ============================================================

def demo_grayscale() -> None:
    """演示灰度转换。"""
    print("\n" + "=" * 60)
    print("1. 灰度转换方法对比")
    print("=" * 60)

    # 创建模拟彩色图像
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    converter = GrayscaleConverter()

    gray_weighted = converter.weighted_average(img)
    gray_max = converter.max_value(img)
    gray_avg = converter.average(img)

    print(f"\n  加权平均: mean={gray_weighted.mean():.1f}")
    print(f"  最大值法: mean={gray_max.mean():.1f}")
    print(f"  平均值法: mean={gray_avg.mean():.1f}")
    print("\n  💡 加权平均法最接近人眼感知，是 OpenCV 默认方法")


def demo_threshold() -> None:
    """演示阈值二值化。"""
    print("\n" + "=" * 60)
    print("2. 阈值二值化")
    print("=" * 60)

    gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    proc = ThresholdProcessor()

    # 全局阈值
    thresh, binary = proc.global_threshold(gray, thresh=127)

    # Otsu 自动阈值
    otsu_thresh, otsu_binary = proc.otsu_threshold(gray)

    print(f"\n  全局阈值={thresh}, Otsu 阈值={otsu_thresh}")
    print("  💡 Otsu 适合双峰直方图，自动找到最佳分割阈值")


def demo_filters() -> None:
    """演示空间滤波。"""
    print("\n" + "=" * 60)
    print("3. 空间滤波对比")
    print("=" * 60)

    # 创建带噪声的图像
    gray = np.random.randint(80, 180, (50, 50), dtype=np.uint8)
    filt = SpatialFilter()

    mean_result = filt.mean_filter(gray, ksize=3)
    gauss_result = filt.gaussian_filter(gray, ksize=5, sigma=1.0)
    median_result = filt.median_filter(gray, ksize=3)
    bilateral_info = filt.bilateral_info()

    print(f"\n  滤波效果对比（标准差越小越平滑）:")
    print(f"    原图: std={gray.std():.2f}")
    print(f"    均值: std={mean_result.std():.2f}")
    print(f"    高斯: std={gauss_result.std():.2f}")
    print(f"    中值: std={median_result.std():.2f}")
    print(f"    双边: {bilateral_info['优势']}")


def demo_edge_detection() -> None:
    """演示边缘检测。"""
    print("\n" + "=" * 60)
    print("4. 边缘检测")
    print("=" * 60)

    # 创建有明显边缘的图像
    gray = np.zeros((50, 50), dtype=np.uint8)
    gray[10:40, 10:40] = 200  # 中间白色方块

    detector = EdgeDetector()

    _, _, sobel_mag = detector.sobel(gray)
    canny_edges = detector.canny(gray, low_thresh=30, high_thresh=100)
    laplacian = detector.laplacian(gray)

    print(f"\n  💡 Canny 是最常用的边缘检测算法")
    print(f"  💡 调参技巧: high_thresh = 2~3 × low_thresh")


def demo_morphology() -> None:
    """演示形态学操作。"""
    print("\n" + "=" * 60)
    print("5. 形态学操作")
    print("=" * 60)

    # 创建带噪声的二值图像
    binary = np.zeros((50, 50), dtype=np.uint8)
    binary[15:35, 15:35] = 255  # 白色方块
    # 添加噪声点
    noise_points = np.random.random((50, 50)) > 0.95
    binary[noise_points] = 255

    morph = MorphologyProcessor()
    kernel = morph.get_kernel("rect", 3)

    eroded = morph.erode(binary, kernel)
    dilated = morph.dilate(binary, kernel)
    opened = morph.opening(binary, kernel)
    closed = morph.closing(binary, kernel)

    print(f"\n  💡 开运算去噪声（小白点），闭运算填空洞（小黑点）")
    print(f"  💡 形态学梯度 = 膨胀 - 腐蚀 → 提取边缘轮廓")


def demo_histogram() -> None:
    """演示直方图操作。"""
    print("\n" + "=" * 60)
    print("6. 直方图均衡化")
    print("=" * 60)

    # 创建低对比度图像
    gray = np.random.randint(80, 180, (100, 100), dtype=np.uint8)
    proc = HistogramProcessor()

    hist = proc.calc_histogram(gray)
    equalized = proc.equalize(gray)

    print(f"\n  原图范围: [{gray.min()}, {gray.max()}]")
    print(f"  均衡后范围: [{equalized.min()}, {equalized.max()}]")
    print(f"  💡 均衡化扩展动态范围，增强暗部/亮部细节")
    print(f"  💡 CLAHE（自适应均衡化）效果更好: cv2.createCLAHE()")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行所有图像处理演示。"""
    print("🐍 图像处理模拟 — 滤波/边缘检测/形态学")
    print("=" * 60)

    demo_grayscale()
    demo_threshold()
    demo_filters()
    demo_edge_detection()
    demo_morphology()
    demo_histogram()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. 灰度转换用加权平均法（cv2.cvtColor）")
    print("  2. 二值化首选 Otsu 自动阈值，光照不均用自适应阈值")
    print("  3. 去噪：高斯滤波（通用）、中值滤波（椒盐）、双边滤波（保边）")
    print("  4. 边缘检测首选 Canny，注意双阈值调参")
    print("  5. 形态学：开运算去噪、闭运算填洞、梯度提取边缘")
    print("  6. 直方图均衡化增强对比度，CLAHE 效果更自然")


if __name__ == "__main__":
    main()
