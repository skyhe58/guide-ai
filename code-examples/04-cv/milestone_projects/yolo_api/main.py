"""
YOLO 实时检测 FastAPI 服务 — 里程碑项目

项目说明：自定义数据集训练 → 模型导出 → FastAPI 服务
功能：图像上传检测、批量检测、模型管理、健康检查

知识点：YOLO 推理服务化、FastAPI 文件上传、异步处理、
       模型热加载、检测结果 JSON 序列化、性能监控

Python 版本：3.11+
依赖：numpy>=1.24, pydantic>=2.0（模拟模式）
真实环境依赖：
  pip install ultralytics fastapi uvicorn python-multipart Pillow
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
from dataclasses import dataclass
from typing import Any

import numpy as np

# ============================================================
# 1. 数据模型（Pydantic 风格）
# ============================================================

@dataclass
class BoundingBox:
    """检测边界框。"""
    x1: float
    y1: float
    x2: float
    y2: float
    width: float = 0.0
    height: float = 0.0

    def __post_init__(self) -> None:
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1

    def to_dict(self) -> dict[str, float]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2,
                "width": self.width, "height": self.height}


@dataclass
class DetectionItem:
    """单个检测结果。"""
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox.to_dict(),
        }


@dataclass
class DetectionResponse:
    """检测 API 响应。"""
    request_id: str
    model: str
    image_size: tuple[int, int]
    detections: list[DetectionItem]
    inference_time_ms: float
    total_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model": self.model,
            "image_size": {"width": self.image_size[0], "height": self.image_size[1]},
            "num_detections": len(self.detections),
            "detections": [d.to_dict() for d in self.detections],
            "inference_time_ms": round(self.inference_time_ms, 2),
            "total_time_ms": round(self.total_time_ms, 2),
        }


@dataclass
class ModelInfo:
    """模型信息。"""
    name: str
    version: str
    classes: list[str]
    input_size: int
    loaded: bool = False
    load_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "num_classes": len(self.classes),
            "classes": self.classes,
            "input_size": self.input_size,
            "loaded": self.loaded,
        }


@dataclass
class HealthStatus:
    """健康检查状态。"""
    status: str = "healthy"
    model_loaded: bool = False
    uptime_seconds: float = 0.0
    total_requests: int = 0
    avg_inference_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "model_loaded": self.model_loaded,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "total_requests": self.total_requests,
            "avg_inference_ms": round(self.avg_inference_ms, 2),
        }


# ============================================================
# 2. 模拟 YOLO 推理引擎
# ============================================================

class YOLOInferenceEngine:
    """YOLO 推理引擎。

    真实实现：
        from ultralytics import YOLO
        model = YOLO("best.pt")
        results = model(image, conf=0.25, iou=0.45)
    """

    MOCK_CLASSES = ["cat", "dog", "person", "car", "bicycle",
                    "chair", "table", "bottle", "phone", "laptop"]

    def __init__(self, model_path: str = "best.pt", device: str = "cuda:0"):
        self.model_path = model_path
        self.device = device
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.input_size = 640
        self.loaded = False
        self.classes = self.MOCK_CLASSES

    def load(self) -> float:
        """加载模型。"""
        start = time.time()
        time.sleep(0.1)  # 模拟加载延迟
        self.loaded = True
        load_time = time.time() - start + np.random.uniform(0.5, 2.0)
        print(f"  ✅ 模型加载完成: {self.model_path} ({load_time:.1f}s)")
        return load_time

    def predict(self, image: np.ndarray,
                conf: float | None = None,
                iou: float | None = None) -> tuple[list[DetectionItem], float]:
        """执行推理。"""
        if not self.loaded:
            raise RuntimeError("模型未加载")

        conf = conf or self.conf_threshold
        iou = iou or self.iou_threshold

        start = time.time()
        h, w = image.shape[:2]

        # 模拟检测结果
        num_dets = np.random.randint(1, 8)
        detections = []

        for _ in range(num_dets):
            class_id = np.random.randint(0, len(self.classes))
            confidence = np.random.uniform(conf, 0.98)

            cx = np.random.uniform(0.1 * w, 0.9 * w)
            cy = np.random.uniform(0.1 * h, 0.9 * h)
            bw = np.random.uniform(0.05 * w, 0.3 * w)
            bh = np.random.uniform(0.05 * h, 0.3 * h)

            det = DetectionItem(
                class_id=class_id,
                class_name=self.classes[class_id],
                confidence=confidence,
                bbox=BoundingBox(
                    x1=max(0, cx - bw / 2),
                    y1=max(0, cy - bh / 2),
                    x2=min(w, cx + bw / 2),
                    y2=min(h, cy + bh / 2),
                ),
            )
            detections.append(det)

        inference_time = (time.time() - start) * 1000 + np.random.uniform(5, 25)
        return detections, inference_time

    def get_info(self) -> ModelInfo:
        """获取模型信息。"""
        return ModelInfo(
            name=self.model_path,
            version="yolov8n-custom",
            classes=self.classes,
            input_size=self.input_size,
            loaded=self.loaded,
        )


# ============================================================
# 3. 模拟 FastAPI 应用
# ============================================================

class MockFastAPIApp:
    """模拟 FastAPI 应用。

    真实 FastAPI 实现：
        from fastapi import FastAPI, UploadFile, File
        from PIL import Image
        import io

        app = FastAPI(title="YOLO Detection API")
        model = YOLO("best.pt")

        @app.post("/detect")
        async def detect(file: UploadFile = File(...)):
            image = Image.open(io.BytesIO(await file.read()))
            results = model(image)
            return format_results(results)
    """

    def __init__(self, title: str = "YOLO Detection API"):
        self.title = title
        self.engine = YOLOInferenceEngine()
        self.start_time = time.time()
        self.request_count = 0
        self.total_inference_time = 0.0

        print(f"\n  🚀 初始化 {title}")

    def startup(self) -> None:
        """应用启动事件。"""
        print(f"\n  📡 启动 {self.title}")
        load_time = self.engine.load()
        print(f"  ✅ 服务就绪: http://localhost:8000")
        print(f"  📖 API 文档: http://localhost:8000/docs")

    def detect(self, image: np.ndarray,
               conf: float = 0.25,
               iou: float = 0.45) -> DetectionResponse:
        """POST /detect — 单张图像检测。"""
        request_id = str(uuid.uuid4())[:8]
        total_start = time.time()

        detections, inference_time = self.engine.predict(image, conf=conf, iou=iou)

        total_time = (time.time() - total_start) * 1000 + inference_time

        self.request_count += 1
        self.total_inference_time += inference_time

        return DetectionResponse(
            request_id=request_id,
            model=self.engine.model_path,
            image_size=(image.shape[1], image.shape[0]),
            detections=detections,
            inference_time_ms=inference_time,
            total_time_ms=total_time,
        )

    def batch_detect(self, images: list[np.ndarray],
                     conf: float = 0.25) -> list[DetectionResponse]:
        """POST /detect/batch — 批量检测。"""
        results = []
        for img in images:
            result = self.detect(img, conf=conf)
            results.append(result)
        return results

    def health(self) -> HealthStatus:
        """GET /health — 健康检查。"""
        uptime = time.time() - self.start_time
        avg_inference = (self.total_inference_time / self.request_count
                         if self.request_count > 0 else 0)
        return HealthStatus(
            status="healthy" if self.engine.loaded else "degraded",
            model_loaded=self.engine.loaded,
            uptime_seconds=uptime,
            total_requests=self.request_count,
            avg_inference_ms=avg_inference,
        )

    def model_info(self) -> ModelInfo:
        """GET /model/info — 模型信息。"""
        return self.engine.get_info()

    def update_config(self, conf: float | None = None,
                      iou: float | None = None) -> dict[str, float]:
        """PUT /model/config — 更新推理配置。"""
        if conf is not None:
            self.engine.conf_threshold = conf
        if iou is not None:
            self.engine.iou_threshold = iou
        return {
            "conf_threshold": self.engine.conf_threshold,
            "iou_threshold": self.engine.iou_threshold,
        }


# ============================================================
# 4. API 路由文档
# ============================================================

def print_api_docs() -> None:
    """打印 API 文档。"""
    print("\n" + "=" * 60)
    print("API 路由文档")
    print("=" * 60)

    routes = [
        ("POST", "/detect", "单张图像检测",
         "上传图像文件，返回检测结果"),
        ("POST", "/detect/batch", "批量图像检测",
         "上传多张图像，返回批量结果"),
        ("POST", "/detect/url", "URL 图像检测",
         "提供图像 URL，下载后检测"),
        ("GET", "/health", "健康检查",
         "返回服务状态和统计信息"),
        ("GET", "/model/info", "模型信息",
         "返回当前模型的详细信息"),
        ("PUT", "/model/config", "更新配置",
         "动态调整置信度和 IoU 阈值"),
        ("POST", "/model/reload", "重新加载模型",
         "热更新模型权重"),
    ]

    for method, path, name, desc in routes:
        print(f"  {method:<6} {path:<20} — {name}")
        print(f"         {desc}")


# ============================================================
# 5. 真实 FastAPI 代码模板
# ============================================================

FASTAPI_TEMPLATE = '''
# ============================================================
# 真实 FastAPI 实现代码（需要安装依赖）
# ============================================================
"""
pip install ultralytics fastapi uvicorn python-multipart Pillow

运行: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from ultralytics import YOLO
import io
import time

app = FastAPI(title="YOLO Detection API", version="1.0.0")

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局模型实例
model = None

@app.on_event("startup")
async def load_model():
    global model
    model = YOLO("best.pt")
    print("模型加载完成")

@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conf: float = 0.25,
    iou: float = 0.45,
):
    """上传图像进行目标检测。"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "请上传图像文件")

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    start = time.time()
    results = model(image, conf=conf, iou=iou)
    inference_time = (time.time() - start) * 1000

    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class_id": int(box.cls[0]),
                "class_name": model.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": {
                    "x1": float(box.xyxy[0][0]),
                    "y1": float(box.xyxy[0][1]),
                    "x2": float(box.xyxy[0][2]),
                    "y2": float(box.xyxy[0][3]),
                },
            })

    return {
        "num_detections": len(detections),
        "detections": detections,
        "inference_time_ms": round(inference_time, 2),
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}
'''


# ============================================================
# 6. 演示函数
# ============================================================

def demo_api_startup() -> MockFastAPIApp:
    """演示 API 启动。"""
    print("\n" + "=" * 60)
    print("1. API 服务启动")
    print("=" * 60)

    app = MockFastAPIApp("YOLO Detection API v1.0")
    app.startup()
    return app


def demo_single_detection(app: MockFastAPIApp) -> None:
    """演示单张检测。"""
    print("\n" + "=" * 60)
    print("2. 单张图像检测")
    print("=" * 60)

    image = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
    result = app.detect(image, conf=0.3)

    print(f"\n  响应 JSON:")
    response_json = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
    # 只打印前 500 字符
    print(f"  {response_json[:500]}...")


def demo_batch_detection(app: MockFastAPIApp) -> None:
    """演示批量检测。"""
    print("\n" + "=" * 60)
    print("3. 批量图像检测")
    print("=" * 60)

    images = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
              for _ in range(5)]

    results = app.batch_detect(images, conf=0.3)

    total_dets = sum(len(r.detections) for r in results)
    total_time = sum(r.total_time_ms for r in results)
    print(f"\n  批量结果: {len(results)} 张图像, {total_dets} 个检测, "
          f"总耗时={total_time:.1f}ms")


def demo_health_check(app: MockFastAPIApp) -> None:
    """演示健康检查。"""
    print("\n" + "=" * 60)
    print("4. 健康检查与监控")
    print("=" * 60)

    health = app.health()
    print(f"\n  健康状态:")
    for key, value in health.to_dict().items():
        print(f"    {key}: {value}")


def demo_model_management(app: MockFastAPIApp) -> None:
    """演示模型管理。"""
    print("\n" + "=" * 60)
    print("5. 模型管理")
    print("=" * 60)

    # 模型信息
    info = app.model_info()
    print(f"\n  模型信息:")
    for key, value in info.to_dict().items():
        if key != "classes":
            print(f"    {key}: {value}")
    print(f"    classes: {info.classes[:5]}... (共 {len(info.classes)} 类)")

    # 更新配置
    new_config = app.update_config(conf=0.5, iou=0.5)
    print(f"\n  更新配置: {new_config}")


def demo_api_routes() -> None:
    """演示 API 路由。"""
    print_api_docs()


def demo_real_code() -> None:
    """展示真实代码模板。"""
    print("\n" + "=" * 60)
    print("7. 真实 FastAPI 代码模板")
    print("=" * 60)
    print(FASTAPI_TEMPLATE[:800])
    print("  ... (完整代码见源文件)")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """运行 YOLO 检测 API 演示。"""
    print("🐍 YOLO 实时检测 FastAPI 服务 — 里程碑项目")
    print("=" * 60)

    app = demo_api_startup()
    demo_single_detection(app)
    demo_batch_detection(app)
    demo_health_check(app)
    demo_model_management(app)
    demo_api_routes()
    demo_real_code()

    print("\n" + "=" * 60)
    print("✅ 里程碑项目演示完成！")
    print("\n💡 项目要点:")
    print("  1. FastAPI + YOLO = 高性能检测 API 服务")
    print("  2. 支持单张/批量/URL 三种检测方式")
    print("  3. 健康检查 + 性能监控保障服务稳定性")
    print("  4. 模型热加载支持不停机更新")
    print("  5. CORS 中间件支持前端跨域调用")
    print("  6. 生产部署: uvicorn + gunicorn + nginx")


if __name__ == "__main__":
    main()
