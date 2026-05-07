"""
YOLO 目标检测模拟 — 模型加载/推理/NMS/结果解析

知识点：YOLO 架构原理（YOLOv8/YOLOv11）、Ultralytics 框架、
       目标检测推理流程、非极大值抑制（NMS）、
       检测结果解析、置信度过滤、类别映射

Python 版本：3.11+
依赖：numpy>=1.24（模拟模式）
真实环境依赖：ultralytics>=8.0（pip install ultralytics）
最后验证：2024-12-01

真实库安装：
  pip install ultralytics          # YOLOv8/YOLOv11
  pip install torch torchvision    # PyTorch 后端
  pip install onnxruntime          # ONNX 推理（可选）
"""

from __future__ import annotations

import json
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Any


# ============================================================
# 1. COCO 类别定义
# ============================================================

COCO_CLASSES: dict[int, str] = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    4: "airplane", 5: "bus", 6: "train", 7: "truck",
    8: "boat", 9: "traffic light", 10: "fire hydrant",
    11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse",
    18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
    22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella",
    26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle",
    56: "chair", 57: "couch", 58: "potted plant",
    59: "bed", 60: "dining table", 62: "tv", 63: "laptop",
    64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone",
}


# ============================================================
# 2. 检测结果数据结构
# ============================================================

@dataclass
class BoundingBox:
    """边界框（Bounding Box）。"""
    x1: float    # 左上角 x
    y1: float    # 左上角 y
    x2: float    # 右下角 x
    y2: float    # 右下角 y

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_xywh(self) -> tuple[float, float, float, float]:
        """转为 (x_center, y_center, width, height) 格式。"""
        cx, cy = self.center
        return (cx, cy, self.width, self.height)

    def iou(self, other: BoundingBox) -> float:
        """计算与另一个框的 IoU（交并比）。"""
        # 交集
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        union_area = self.area + other.area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0


@dataclass
class Detection:
    """单个检测结果。"""
    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: str

    def summary(self) -> str:
        cx, cy = self.bbox.center
        return (f"{self.class_name} ({self.confidence:.2f}) "
                f"@ [{self.bbox.x1:.0f},{self.bbox.y1:.0f},"
                f"{self.bbox.x2:.0f},{self.bbox.y2:.0f}]")


@dataclass
class DetectionResult:
    """完整检测结果。"""
    detections: list[Detection] = field(default_factory=list)
    image_size: tuple[int, int] = (640, 640)
    inference_time: float = 0.0
    model_name: str = ""

    @property
    def count(self) -> int:
        return len(self.detections)

    def filter_by_confidence(self, threshold: float = 0.5) -> list[Detection]:
        """按置信度过滤。"""
        return [d for d in self.detections if d.confidence >= threshold]

    def filter_by_class(self, class_name: str) -> list[Detection]:
        """按类别过滤。"""
        return [d for d in self.detections if d.class_name == class_name]

    def summary(self) -> str:
        lines = [f"模型: {self.model_name} | 图像: {self.image_size} | "
                 f"推理: {self.inference_time:.1f}ms | 检测数: {self.count}"]
        # 按类别统计
        class_counts: dict[str, int] = {}
        for d in self.detections:
            class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1
        for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {cls}: {cnt}")
        return "\n".join(lines)


# ============================================================
# 3. 非极大值抑制（NMS）
# ============================================================

class NMSProcessor:
    """非极大值抑制 — 去除重叠检测框。

    NMS 原理：
    1. 按置信度排序所有检测框
    2. 选择置信度最高的框
    3. 计算其与剩余框的 IoU
    4. 移除 IoU > 阈值的框（认为是同一目标的重复检测）
    5. 重复 2-4 直到处理完所有框
    """

    @staticmethod
    def apply(detections: list[Detection],
              iou_threshold: float = 0.45) -> list[Detection]:
        """执行 NMS。

        真实 Ultralytics 内部已集成 NMS：
            results = model(image, conf=0.25, iou=0.45)
        """
        if not detections:
            return []

        # 按置信度降序排序
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: list[Detection] = []

        while sorted_dets:
            # 保留置信度最高的
            best = sorted_dets.pop(0)
            kept.append(best)

            # 移除与 best 重叠过多的框
            remaining = []
            for det in sorted_dets:
                if det.class_id != best.class_id:
                    # 不同类别不做 NMS
                    remaining.append(det)
                elif best.bbox.iou(det.bbox) < iou_threshold:
                    remaining.append(det)
                # else: IoU >= threshold，移除（重复检测）

            sorted_dets = remaining

        print(f"  🔧 NMS: {len(detections)} 个框 → {len(kept)} 个框 "
              f"(IoU阈值={iou_threshold})")
        return kept


# ============================================================
# 4. 模拟 YOLO 模型
# ============================================================

class MockYOLO:
    """模拟 Ultralytics YOLO 模型。

    真实 Ultralytics 使用：
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")           # 加载预训练模型
        results = model("image.jpg")          # 推理
        results = model("image.jpg", conf=0.5, iou=0.45)
    """

    # 模型变体
    VARIANTS = {
        "yolov8n": {"params": "3.2M", "flops": "8.7G", "map50": 37.3, "speed": "1.2ms"},
        "yolov8s": {"params": "11.2M", "flops": "28.6G", "map50": 44.9, "speed": "2.1ms"},
        "yolov8m": {"params": "25.9M", "flops": "78.9G", "map50": 50.2, "speed": "4.7ms"},
        "yolov8l": {"params": "43.7M", "flops": "165.2G", "map50": 52.9, "speed": "7.3ms"},
        "yolov8x": {"params": "68.2M", "flops": "257.8G", "map50": 53.9, "speed": "11.5ms"},
        "yolo11n": {"params": "2.6M", "flops": "6.5G", "map50": 39.5, "speed": "1.5ms"},
        "yolo11s": {"params": "9.4M", "flops": "21.5G", "map50": 47.0, "speed": "2.5ms"},
        "yolo11m": {"params": "20.1M", "flops": "68.0G", "map50": 51.5, "speed": "4.7ms"},
    }

    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model_path = model_path
        self.model_name = model_path.replace(".pt", "")
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.nms = NMSProcessor()

        variant = self.VARIANTS.get(self.model_name, {})
        print(f"  🤖 加载模型: {model_path}")
        if variant:
            print(f"     参数量={variant['params']}, FLOPs={variant['flops']}, "
                  f"mAP50={variant['map50']}")

    def predict(self, image: np.ndarray | str,
                conf: float = 0.25,
                iou: float = 0.45) -> DetectionResult:
        """执行目标检测推理。

        真实 Ultralytics：
            results = model.predict("image.jpg", conf=0.25, iou=0.45)
            for r in results:
                boxes = r.boxes.xyxy      # 边界框坐标
                confs = r.boxes.conf      # 置信度
                classes = r.boxes.cls     # 类别 ID
        """
        start_time = time.time()

        # 确定图像尺寸
        if isinstance(image, str):
            img_size = (640, 640)
            print(f"  📷 推理图像: {image}")
        else:
            img_size = (image.shape[1], image.shape[0])

        # 模拟生成检测结果
        raw_detections = self._simulate_detections(img_size)

        # 置信度过滤
        filtered = [d for d in raw_detections if d.confidence >= conf]

        # NMS
        final = self.nms.apply(filtered, iou_threshold=iou)

        inference_time = (time.time() - start_time) * 1000 + np.random.uniform(5, 20)

        result = DetectionResult(
            detections=final,
            image_size=img_size,
            inference_time=inference_time,
            model_name=self.model_name,
        )
        return result

    def _simulate_detections(self, img_size: tuple[int, int]) -> list[Detection]:
        """模拟生成检测结果。"""
        w, h = img_size
        num_detections = np.random.randint(3, 10)
        detections = []

        common_classes = [0, 2, 5, 15, 16, 56, 62, 63]  # 常见类别

        for _ in range(num_detections):
            class_id = np.random.choice(common_classes)
            class_name = COCO_CLASSES.get(class_id, "unknown")

            # 随机生成边界框
            cx = np.random.uniform(0.1 * w, 0.9 * w)
            cy = np.random.uniform(0.1 * h, 0.9 * h)
            bw = np.random.uniform(0.05 * w, 0.3 * w)
            bh = np.random.uniform(0.05 * h, 0.3 * h)

            bbox = BoundingBox(
                x1=max(0, cx - bw / 2),
                y1=max(0, cy - bh / 2),
                x2=min(w, cx + bw / 2),
                y2=min(h, cy + bh / 2),
            )

            det = Detection(
                bbox=bbox,
                confidence=np.random.uniform(0.3, 0.95),
                class_id=class_id,
                class_name=class_name,
            )
            detections.append(det)

        return detections

    @staticmethod
    def list_models() -> None:
        """列出可用模型变体。"""
        print("\n  YOLO 模型变体对比:")
        print(f"  {'模型':<12} {'参数量':<10} {'FLOPs':<10} {'mAP50':<8} {'速度':<8}")
        print("  " + "-" * 48)
        for name, info in MockYOLO.VARIANTS.items():
            print(f"  {name:<12} {info['params']:<10} {info['flops']:<10} "
                  f"{info['map50']:<8} {info['speed']:<8}")


# ============================================================
# 5. 检测结果可视化（模拟）
# ============================================================

class DetectionVisualizer:
    """检测结果可视化工具。"""

    # 类别颜色映射
    COLORS = {
        "person": (0, 255, 0),
        "car": (255, 0, 0),
        "bus": (0, 0, 255),
        "cat": (255, 255, 0),
        "dog": (0, 255, 255),
    }

    @staticmethod
    def draw_results(image_shape: tuple[int, ...],
                     result: DetectionResult) -> np.ndarray:
        """在图像上绘制检测结果（模拟）。

        真实 OpenCV 绘制：
            for det in detections:
                cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
                cv2.putText(img, label, (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        """
        canvas = np.zeros(image_shape, dtype=np.uint8)

        for det in result.detections:
            color = DetectionVisualizer.COLORS.get(det.class_name, (128, 128, 128))
            x1, y1 = int(det.bbox.x1), int(det.bbox.y1)
            x2, y2 = int(det.bbox.x2), int(det.bbox.y2)

            # 模拟绘制边框
            x1 = max(0, min(x1, canvas.shape[1] - 1))
            y1 = max(0, min(y1, canvas.shape[0] - 1))
            x2 = max(0, min(x2, canvas.shape[1] - 1))
            y2 = max(0, min(y2, canvas.shape[0] - 1))

            if x2 > x1 and y2 > y1:
                canvas[y1, x1:x2] = color
                canvas[y2, x1:x2] = color
                canvas[y1:y2, x1] = color
                canvas[y1:y2, x2] = color

        print(f"  🎨 绘制 {result.count} 个检测框")
        return canvas

    @staticmethod
    def print_results(result: DetectionResult) -> None:
        """打印检测结果。"""
        print(f"\n  📊 检测结果:")
        print(f"  {result.summary()}")
        print(f"\n  详细检测:")
        for i, det in enumerate(result.detections):
            print(f"    [{i}] {det.summary()}")


# ============================================================
# 6. 演示函数
# ============================================================

def demo_model_loading() -> None:
    """演示模型加载与变体对比。"""
    print("\n" + "=" * 60)
    print("1. YOLO 模型加载与变体")
    print("=" * 60)

    MockYOLO.list_models()

    print("\n  💡 选择建议:")
    print("    n (nano): 边缘设备、实时检测")
    print("    s (small): 移动端、轻量服务")
    print("    m (medium): 服务器、精度与速度平衡")
    print("    l/x (large/xlarge): 高精度场景")


def demo_detection() -> None:
    """演示目标检测推理。"""
    print("\n" + "=" * 60)
    print("2. 目标检测推理")
    print("=" * 60)

    model = MockYOLO("yolov8n.pt")

    # 单张图像推理
    result = model.predict("street_scene.jpg", conf=0.3, iou=0.45)
    DetectionVisualizer.print_results(result)

    # 按类别过滤
    persons = result.filter_by_class("person")
    cars = result.filter_by_class("car")
    print(f"\n  行人: {len(persons)} 个, 车辆: {len(cars)} 个")


def demo_nms() -> None:
    """演示 NMS 过程。"""
    print("\n" + "=" * 60)
    print("3. 非极大值抑制（NMS）")
    print("=" * 60)

    # 创建重叠检测框
    detections = [
        Detection(BoundingBox(100, 100, 300, 300), 0.9, 0, "person"),
        Detection(BoundingBox(110, 105, 310, 305), 0.85, 0, "person"),  # 重叠
        Detection(BoundingBox(115, 110, 315, 310), 0.7, 0, "person"),   # 重叠
        Detection(BoundingBox(400, 200, 550, 400), 0.8, 2, "car"),
        Detection(BoundingBox(405, 205, 555, 405), 0.6, 2, "car"),      # 重叠
    ]

    print(f"  NMS 前: {len(detections)} 个框")
    for d in detections:
        print(f"    {d.summary()}")

    # 计算 IoU 示例
    iou_val = detections[0].bbox.iou(detections[1].bbox)
    print(f"\n  框 0 与框 1 的 IoU: {iou_val:.4f}")

    # 执行 NMS
    nms = NMSProcessor()
    kept = nms.apply(detections, iou_threshold=0.45)

    print(f"\n  NMS 后: {len(kept)} 个框")
    for d in kept:
        print(f"    {d.summary()}")


def demo_batch_inference() -> None:
    """演示批量推理。"""
    print("\n" + "=" * 60)
    print("4. 批量推理")
    print("=" * 60)

    model = MockYOLO("yolov8s.pt")

    images = ["image_001.jpg", "image_002.jpg", "image_003.jpg"]
    total_time = 0.0

    for img_path in images:
        result = model.predict(img_path, conf=0.3)
        total_time += result.inference_time
        print(f"  {img_path}: {result.count} 个目标, "
              f"{result.inference_time:.1f}ms")

    avg_time = total_time / len(images)
    print(f"\n  平均推理时间: {avg_time:.1f}ms/张")
    print(f"  💡 批量推理可用 model.predict([img1, img2, ...]) 一次传入")


def demo_iou_calculation() -> None:
    """演示 IoU 计算。"""
    print("\n" + "=" * 60)
    print("5. IoU（交并比）计算")
    print("=" * 60)

    box_a = BoundingBox(0, 0, 100, 100)
    test_boxes = [
        ("完全重叠", BoundingBox(0, 0, 100, 100)),
        ("50% 重叠", BoundingBox(50, 0, 150, 100)),
        ("25% 重叠", BoundingBox(50, 50, 150, 150)),
        ("不重叠", BoundingBox(200, 200, 300, 300)),
        ("包含关系", BoundingBox(25, 25, 75, 75)),
    ]

    print(f"  基准框 A: [{box_a.x1},{box_a.y1},{box_a.x2},{box_a.y2}]")
    for name, box_b in test_boxes:
        iou = box_a.iou(box_b)
        print(f"  {name}: IoU = {iou:.4f}")

    print(f"\n  💡 IoU 阈值:")
    print(f"    0.5  — PASCAL VOC 标准 (mAP@50)")
    print(f"    0.75 — 严格匹配 (mAP@75)")
    print(f"    0.5:0.95 — COCO 标准 (mAP@50:95)")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行所有 YOLO 检测演示。"""
    print("🐍 YOLO 目标检测模拟 — 推理/NMS/结果解析")
    print("=" * 60)

    demo_model_loading()
    demo_detection()
    demo_nms()
    demo_batch_inference()
    demo_iou_calculation()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. YOLO 是单阶段检测器，速度快适合实时场景")
    print("  2. 模型选择: n/s 适合边缘设备，m/l/x 适合服务器")
    print("  3. NMS 去除重叠框: IoU > 阈值的低置信度框被移除")
    print("  4. IoU 是评估检测精度的核心指标")
    print("  5. 推理参数: conf（置信度阈值）和 iou（NMS 阈值）")
    print("  6. Ultralytics 框架: model = YOLO('yolov8n.pt'); results = model(img)")


if __name__ == "__main__":
    main()
