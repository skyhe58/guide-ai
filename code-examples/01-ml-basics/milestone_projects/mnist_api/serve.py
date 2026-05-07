"""
MNIST FastAPI 服务 — 加载模型提供推理 API

Python 版本：3.11+
依赖：fastapi>=0.109, uvicorn>=0.27, torch>=2.1, numpy>=1.26
最后验证：2024-12-01

运行方式：
  1. 先训练模型：python train.py
  2. 启动服务：python serve.py
  3. 访问文档：http://localhost:8000/docs
"""
from __future__ import annotations
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from train import MNISTNet

app = FastAPI(title="MNIST 手写数字识别 API", version="1.0.0")

# 加载模型
model = MNISTNet()
MODEL_PATH = "/tmp/mnist_model.pth"
try:
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    print(f"✅ 模型已加载: {MODEL_PATH}")
except FileNotFoundError:
    print(f"⚠️ 模型文件不存在: {MODEL_PATH}")
    print("💡 请先运行 python train.py 训练模型")


class PredictRequest(BaseModel):
    """预测请求：28x28 灰度图像的像素值（0-255）。"""
    pixels: list[list[float]] = Field(description="28x28 像素矩阵")


class PredictResponse(BaseModel):
    digit: int
    confidence: float
    probabilities: list[float]


@app.get("/", summary="健康检查")
async def health():
    return {"status": "ok", "model": "MNIST CNN"}


@app.post("/predict", response_model=PredictResponse, summary="识别手写数字")
async def predict(req: PredictRequest):
    """输入 28x28 像素矩阵，返回识别结果。"""
    try:
        pixels = np.array(req.pixels, dtype=np.float32)
        if pixels.shape != (28, 28):
            raise ValueError(f"期望 28x28，实际 {pixels.shape}")
        # 归一化
        tensor = torch.FloatTensor(pixels).unsqueeze(0).unsqueeze(0)
        tensor = (tensor / 255.0 - 0.1307) / 0.3081
    except Exception as e:
        raise HTTPException(400, f"输入格式错误: {e}")

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        digit = probs.argmax().item()
        confidence = probs[digit].item()

    return PredictResponse(
        digit=digit,
        confidence=round(confidence, 4),
        probabilities=[round(p, 4) for p in probs.tolist()],
    )


@app.post("/predict/random", summary="随机生成测试图像并识别")
async def predict_random():
    """生成随机 MNIST 测试图像并识别（用于快速测试）。"""
    from torchvision import datasets, transforms
    test_data = datasets.MNIST("/tmp/mnist", train=False, download=True,
                               transform=transforms.ToTensor())
    idx = np.random.randint(len(test_data))
    image, label = test_data[idx]

    tensor = (image - 0.1307) / 0.3081
    with torch.no_grad():
        output = model(tensor.unsqueeze(0))
        probs = torch.softmax(output, dim=1)[0]
        digit = probs.argmax().item()

    return {
        "true_label": label,
        "predicted": digit,
        "correct": digit == label,
        "confidence": round(probs[digit].item(), 4),
    }


if __name__ == "__main__":
    import uvicorn
    print("🚀 启动 MNIST API: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
