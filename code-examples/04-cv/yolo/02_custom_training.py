"""
YOLO 自定义训练模拟 — 数据集准备/训练配置/训练循环/评估

知识点：YOLO 数据格式（标注文件）、数据集目录结构、
       训练超参数配置、数据增强策略、训练循环模拟、
       损失函数（分类/定位/置信度）、学习率调度

Python 版本：3.11+
依赖：numpy>=1.24（模拟模式）
真实环境依赖：ultralytics>=8.0（pip install ultralytics）
最后验证：2024-12-01

真实库安装：
  pip install ultralytics
  pip install torch torchvision
  pip install albumentations    # 高级数据增强（可选）
  pip install roboflow          # 数据集管理（可选）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ============================================================
# 1. 数据集格式与结构
# ============================================================

@dataclass
class YOLOAnnotation:
    """YOLO 标注格式。

    YOLO 标注文件格式（每行一个目标）：
    <class_id> <x_center> <y_center> <width> <height>
    所有坐标归一化到 [0, 1]
    """
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_line(self) -> str:
        """转为 YOLO 标注行。"""
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"

    @staticmethod
    def from_line(line: str) -> YOLOAnnotation:
        """从标注行解析。"""
        parts = line.strip().split()
        return YOLOAnnotation(
            class_id=int(parts[0]),
            x_center=float(parts[1]),
            y_center=float(parts[2]),
            width=float(parts[3]),
            height=float(parts[4]),
        )

    @staticmethod
    def from_xyxy(class_id: int, x1: float, y1: float, x2: float, y2: float,
                  img_w: int, img_h: int) -> YOLOAnnotation:
        """从 xyxy 像素坐标转换为 YOLO 归一化格式。"""
        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h
        return YOLOAnnotation(class_id, x_center, y_center, width, height)


@dataclass
class DatasetConfig:
    """YOLO 数据集配置（data.yaml）。

    真实 data.yaml 示例：
        path: /datasets/my_dataset
        train: images/train
        val: images/val
        test: images/test
        names:
          0: cat
          1: dog
          2: bird
    """
    path: str
    train: str = "images/train"
    val: str = "images/val"
    test: str = "images/test"
    names: dict[int, str] = field(default_factory=dict)
    nc: int = 0  # 类别数

    def __post_init__(self) -> None:
        self.nc = len(self.names)

    def to_yaml(self) -> str:
        """生成 YAML 配置内容。"""
        lines = [
            f"path: {self.path}",
            f"train: {self.train}",
            f"val: {self.val}",
            f"test: {self.test}",
            f"nc: {self.nc}",
            "names:",
        ]
        for idx, name in sorted(self.names.items()):
            lines.append(f"  {idx}: {name}")
        return "\n".join(lines)


# ============================================================
# 2. 数据集准备工具
# ============================================================

class DatasetPreparer:
    """数据集准备工具。"""

    @staticmethod
    def create_directory_structure(base_path: str) -> dict[str, str]:
        """创建 YOLO 数据集目录结构。

        标准目录结构：
        dataset/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        ├── labels/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── data.yaml
        """
        dirs = {
            "images_train": f"{base_path}/images/train",
            "images_val": f"{base_path}/images/val",
            "images_test": f"{base_path}/images/test",
            "labels_train": f"{base_path}/labels/train",
            "labels_val": f"{base_path}/labels/val",
            "labels_test": f"{base_path}/labels/test",
        }
        print(f"  📁 数据集目录结构:")
        for name, path in dirs.items():
            print(f"    {path}/")
        return dirs

    @staticmethod
    def split_dataset(total_images: int,
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.2,
                      test_ratio: float = 0.1) -> dict[str, int]:
        """划分数据集。"""
        train_count = int(total_images * train_ratio)
        val_count = int(total_images * val_ratio)
        test_count = total_images - train_count - val_count

        split = {
            "train": train_count,
            "val": val_count,
            "test": test_count,
        }
        print(f"  📊 数据集划分: 总计 {total_images} 张")
        for name, count in split.items():
            ratio = count / total_images * 100
            print(f"    {name}: {count} 张 ({ratio:.0f}%)")
        return split

    @staticmethod
    def generate_mock_annotations(num_images: int, num_classes: int,
                                  max_objects: int = 5) -> list[list[YOLOAnnotation]]:
        """生成模拟标注数据。"""
        all_annotations = []
        for _ in range(num_images):
            num_objects = np.random.randint(1, max_objects + 1)
            annotations = []
            for _ in range(num_objects):
                ann = YOLOAnnotation(
                    class_id=np.random.randint(0, num_classes),
                    x_center=np.random.uniform(0.1, 0.9),
                    y_center=np.random.uniform(0.1, 0.9),
                    width=np.random.uniform(0.05, 0.4),
                    height=np.random.uniform(0.05, 0.4),
                )
                annotations.append(ann)
            all_annotations.append(annotations)

        total_objects = sum(len(a) for a in all_annotations)
        print(f"  📝 生成标注: {num_images} 张图像, {total_objects} 个目标")
        return all_annotations

    @staticmethod
    def validate_annotations(annotations: list[list[YOLOAnnotation]],
                             num_classes: int) -> dict[str, Any]:
        """验证标注数据质量。"""
        issues: list[str] = []
        total_objects = 0
        class_counts: dict[int, int] = {}

        for img_idx, anns in enumerate(annotations):
            for ann in anns:
                total_objects += 1
                class_counts[ann.class_id] = class_counts.get(ann.class_id, 0) + 1

                # 检查坐标范围
                if not (0 <= ann.x_center <= 1 and 0 <= ann.y_center <= 1):
                    issues.append(f"图像 {img_idx}: 中心坐标越界")
                if not (0 < ann.width <= 1 and 0 < ann.height <= 1):
                    issues.append(f"图像 {img_idx}: 宽高异常")
                if ann.class_id >= num_classes:
                    issues.append(f"图像 {img_idx}: 类别 ID {ann.class_id} 超出范围")

        result = {
            "总图像数": len(annotations),
            "总目标数": total_objects,
            "类别分布": class_counts,
            "问题数": len(issues),
            "问题列表": issues[:5],  # 只显示前 5 个
        }
        print(f"  ✅ 标注验证: {total_objects} 个目标, {len(issues)} 个问题")
        return result


# ============================================================
# 3. 数据增强
# ============================================================

class DataAugmentation:
    """数据增强策略。

    真实 Ultralytics 内置增强（训练时自动应用）：
        - Mosaic: 4 张图拼接
        - MixUp: 两张图混合
        - HSV 调整: 色调/饱和度/明度
        - 翻转/旋转/缩放
    """

    @staticmethod
    def mosaic(images: list[np.ndarray], size: int = 640) -> np.ndarray:
        """Mosaic 数据增强（4 张图拼接）。

        将 4 张图像拼接成一张，增加小目标检测能力。
        """
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        half = size // 2

        for i, img in enumerate(images[:4]):
            # 简化：直接填充到对应象限
            h, w = min(half, img.shape[0]), min(half, img.shape[1])
            row = (i // 2) * half
            col = (i % 2) * half
            canvas[row:row+h, col:col+w] = img[:h, :w]

        print(f"  🧩 Mosaic 增强: 4 张图 → {size}x{size}")
        return canvas

    @staticmethod
    def mixup(img1: np.ndarray, img2: np.ndarray,
              alpha: float = 0.5) -> np.ndarray:
        """MixUp 数据增强（两张图混合）。"""
        mixed = (img1.astype(np.float32) * alpha +
                 img2.astype(np.float32) * (1 - alpha))
        print(f"  🔀 MixUp 增强: alpha={alpha}")
        return mixed.clip(0, 255).astype(np.uint8)

    @staticmethod
    def hsv_augment(image: np.ndarray,
                    h_gain: float = 0.015,
                    s_gain: float = 0.7,
                    v_gain: float = 0.4) -> np.ndarray:
        """HSV 颜色空间增强。"""
        # 简化：直接在 RGB 空间做亮度/对比度调整
        gains = np.array([1 + np.random.uniform(-v_gain, v_gain)] * 3)
        result = (image.astype(np.float32) * gains).clip(0, 255).astype(np.uint8)
        print(f"  🎨 HSV 增强: h_gain={h_gain}, s_gain={s_gain}, v_gain={v_gain}")
        return result

    @staticmethod
    def random_flip(image: np.ndarray, p: float = 0.5) -> tuple[np.ndarray, bool]:
        """随机水平翻转。"""
        flipped = np.random.random() < p
        if flipped:
            image = np.fliplr(image).copy()
        print(f"  🔄 随机翻转: {'是' if flipped else '否'}")
        return image, flipped

    @staticmethod
    def list_augmentations() -> dict[str, str]:
        """列出所有增强策略。"""
        return {
            "Mosaic": "4 张图拼接，增强小目标检测",
            "MixUp": "两张图混合，增加泛化能力",
            "HSV": "色调/饱和度/明度随机调整",
            "翻转": "水平/垂直翻转",
            "缩放": "随机缩放 0.5~1.5x",
            "旋转": "随机旋转 ±10°",
            "裁剪": "随机裁剪",
            "Copy-Paste": "从其他图像复制目标粘贴",
        }


# ============================================================
# 4. 训练配置
# ============================================================

@dataclass
class TrainingConfig:
    """训练超参数配置。

    真实 Ultralytics 训练：
        model.train(
            data="data.yaml",
            epochs=100,
            imgsz=640,
            batch=16,
            lr0=0.01,
            ...
        )
    """
    # 基础参数
    model: str = "yolov8n.pt"
    data: str = "data.yaml"
    epochs: int = 100
    batch: int = 16
    imgsz: int = 640

    # 学习率
    lr0: float = 0.01        # 初始学习率
    lrf: float = 0.01        # 最终学习率（lr0 * lrf）
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8

    # 数据增强
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0     # 旋转角度
    translate: float = 0.1
    scale: float = 0.5
    fliplr: float = 0.5      # 水平翻转概率
    mosaic: float = 1.0      # Mosaic 概率
    mixup: float = 0.0       # MixUp 概率

    # 其他
    patience: int = 50        # 早停耐心值
    save_period: int = -1     # 保存间隔（-1=只保存最优）
    device: str = "0"         # GPU 设备

    def summary(self) -> str:
        """配置摘要。"""
        return (f"模型={self.model}, epochs={self.epochs}, batch={self.batch}, "
                f"imgsz={self.imgsz}, lr0={self.lr0}")


# ============================================================
# 5. 模拟训练循环
# ============================================================

@dataclass
class EpochMetrics:
    """单个 epoch 的指标。"""
    epoch: int
    box_loss: float      # 定位损失
    cls_loss: float      # 分类损失
    dfl_loss: float      # 分布焦点损失
    precision: float
    recall: float
    map50: float         # mAP@50
    map50_95: float      # mAP@50:95
    lr: float

    def summary(self) -> str:
        return (f"Epoch {self.epoch:3d} | "
                f"box={self.box_loss:.4f} cls={self.cls_loss:.4f} dfl={self.dfl_loss:.4f} | "
                f"P={self.precision:.3f} R={self.recall:.3f} "
                f"mAP50={self.map50:.3f} mAP50-95={self.map50_95:.3f} | "
                f"lr={self.lr:.6f}")


class MockTrainer:
    """模拟 YOLO 训练过程。"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.history: list[EpochMetrics] = []
        self.best_map50 = 0.0
        self.best_epoch = 0

    def train(self, num_epochs: int | None = None, verbose: bool = True) -> list[EpochMetrics]:
        """模拟训练循环。

        真实 Ultralytics：
            results = model.train(data="data.yaml", epochs=100)
        """
        epochs = num_epochs or self.config.epochs
        print(f"\n  🏋️ 开始训练: {self.config.summary()}")
        print(f"  {'Epoch':>7} | {'box_loss':>9} {'cls_loss':>9} {'dfl_loss':>9} | "
              f"{'P':>6} {'R':>6} {'mAP50':>7} {'mAP50-95':>9} | {'lr':>10}")
        print("  " + "-" * 85)

        for epoch in range(1, epochs + 1):
            metrics = self._simulate_epoch(epoch, epochs)
            self.history.append(metrics)

            if metrics.map50 > self.best_map50:
                self.best_map50 = metrics.map50
                self.best_epoch = epoch

            if verbose and (epoch <= 3 or epoch % max(1, epochs // 5) == 0 or epoch == epochs):
                print(f"  {metrics.summary()}")

        print(f"\n  🏆 最佳: Epoch {self.best_epoch}, mAP50={self.best_map50:.3f}")
        return self.history

    def _simulate_epoch(self, epoch: int, total_epochs: int) -> EpochMetrics:
        """模拟单个 epoch 的训练指标。"""
        progress = epoch / total_epochs

        # 损失逐渐下降（加噪声）
        box_loss = 1.5 * (1 - progress * 0.7) + np.random.normal(0, 0.05)
        cls_loss = 2.0 * (1 - progress * 0.75) + np.random.normal(0, 0.08)
        dfl_loss = 1.2 * (1 - progress * 0.6) + np.random.normal(0, 0.03)

        # 精度逐渐上升
        precision = min(0.95, 0.3 + progress * 0.6 + np.random.normal(0, 0.02))
        recall = min(0.90, 0.25 + progress * 0.55 + np.random.normal(0, 0.02))
        map50 = min(0.92, 0.2 + progress * 0.65 + np.random.normal(0, 0.02))
        map50_95 = min(0.75, 0.1 + progress * 0.55 + np.random.normal(0, 0.02))

        # 学习率调度（余弦退火）
        lr = self.config.lr0 * (1 + np.cos(np.pi * progress)) / 2

        return EpochMetrics(
            epoch=epoch,
            box_loss=max(0.01, box_loss),
            cls_loss=max(0.01, cls_loss),
            dfl_loss=max(0.01, dfl_loss),
            precision=max(0, precision),
            recall=max(0, recall),
            map50=max(0, map50),
            map50_95=max(0, map50_95),
            lr=lr,
        )


# ============================================================
# 6. 演示函数
# ============================================================

def demo_dataset_format() -> None:
    """演示数据集格式。"""
    print("\n" + "=" * 60)
    print("1. YOLO 数据集格式")
    print("=" * 60)

    # 标注格式
    ann = YOLOAnnotation(class_id=0, x_center=0.5, y_center=0.5, width=0.3, height=0.4)
    print(f"  标注行: {ann.to_line()}")

    # 从像素坐标转换
    ann2 = YOLOAnnotation.from_xyxy(1, 100, 200, 300, 400, img_w=640, img_h=480)
    print(f"  像素→YOLO: {ann2.to_line()}")

    # 数据集配置
    config = DatasetConfig(
        path="/datasets/my_project",
        names={0: "cat", 1: "dog", 2: "bird"},
    )
    print(f"\n  data.yaml 内容:")
    for line in config.to_yaml().split("\n"):
        print(f"    {line}")


def demo_dataset_preparation() -> None:
    """演示数据集准备。"""
    print("\n" + "=" * 60)
    print("2. 数据集准备")
    print("=" * 60)

    preparer = DatasetPreparer()

    # 创建目录结构
    preparer.create_directory_structure("/datasets/my_project")

    # 划分数据集
    preparer.split_dataset(1000)

    # 生成模拟标注
    annotations = preparer.generate_mock_annotations(100, num_classes=3)

    # 验证标注
    validation = preparer.validate_annotations(annotations, num_classes=3)
    print(f"  类别分布: {validation['类别分布']}")


def demo_augmentation() -> None:
    """演示数据增强。"""
    print("\n" + "=" * 60)
    print("3. 数据增强策略")
    print("=" * 60)

    aug = DataAugmentation()

    # 列出所有增强
    print("  可用增强策略:")
    for name, desc in aug.list_augmentations().items():
        print(f"    {name}: {desc}")

    # 演示增强操作
    img = np.random.randint(0, 256, (320, 320, 3), dtype=np.uint8)
    images = [np.random.randint(0, 256, (320, 320, 3), dtype=np.uint8) for _ in range(4)]

    print()
    mosaic = aug.mosaic(images, size=640)
    mixup = aug.mixup(images[0], images[1], alpha=0.5)
    hsv = aug.hsv_augment(img)
    flipped, was_flipped = aug.random_flip(img)


def demo_training() -> None:
    """演示训练过程。"""
    print("\n" + "=" * 60)
    print("4. 模拟训练过程")
    print("=" * 60)

    config = TrainingConfig(
        model="yolov8n.pt",
        epochs=20,
        batch=16,
        imgsz=640,
        lr0=0.01,
    )

    trainer = MockTrainer(config)
    history = trainer.train(num_epochs=20)

    # 训练曲线摘要
    print(f"\n  训练曲线摘要:")
    print(f"    初始 mAP50: {history[0].map50:.3f}")
    print(f"    最终 mAP50: {history[-1].map50:.3f}")
    print(f"    最佳 mAP50: {trainer.best_map50:.3f} (Epoch {trainer.best_epoch})")


def demo_training_tips() -> None:
    """训练技巧总结。"""
    print("\n" + "=" * 60)
    print("5. 训练技巧")
    print("=" * 60)

    tips = {
        "数据质量": "标注准确性 > 数据数量，优先清洗标注",
        "预训练模型": "始终从预训练权重开始，不要从零训练",
        "学习率": "初始 lr=0.01，使用余弦退火调度",
        "Batch Size": "GPU 显存允许的最大值，至少 8",
        "图像尺寸": "640 是默认值，小目标可用 1280",
        "数据增强": "Mosaic + HSV + 翻转是标配",
        "早停": "patience=50，防止过拟合",
        "冻结层": "小数据集可冻结 backbone 前几层",
    }

    for tip, desc in tips.items():
        print(f"  💡 {tip}: {desc}")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行所有自定义训练演示。"""
    print("🐍 YOLO 自定义训练模拟 — 数据集/增强/训练")
    print("=" * 60)

    demo_dataset_format()
    demo_dataset_preparation()
    demo_augmentation()
    demo_training()
    demo_training_tips()

    print("\n" + "=" * 60)
    print("✅ 所有演示完成！")
    print("\n💡 关键要点:")
    print("  1. YOLO 标注格式: <class_id> <x_center> <y_center> <w> <h>（归一化）")
    print("  2. 数据集划分: train 70% / val 20% / test 10%")
    print("  3. 数据增强: Mosaic + MixUp + HSV + 翻转")
    print("  4. 训练命令: model.train(data='data.yaml', epochs=100)")
    print("  5. 关注 mAP50 和 mAP50-95 指标")
    print("  6. 小数据集用预训练 + 冻结层 + 强增强")


if __name__ == "__main__":
    main()
