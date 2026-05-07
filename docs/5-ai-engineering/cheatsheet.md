---
title: "模块 5 速查卡片"
module: "ai-engineering"
---

# AI 工程化速查卡片

> 核心概念、常用命令、关键数字速查。

## 📋 MLOps 速查

### 训练流水线阶段
```
数据准备 → 模型训练 → 模型评估 → 模型注册
  DVC       MLflow      质量门禁    Model Registry
```

### MLflow 常用命令
```bash
# 启动 MLflow UI
mlflow ui --port 5000

# Docker 启动 MLflow Server
docker run -p 5000:5000 ghcr.io/mlflow/mlflow:latest mlflow server --host 0.0.0.0

# 记录实验
mlflow.log_params({"lr": 0.001, "epochs": 10})
mlflow.log_metrics({"accuracy": 0.95, "loss": 0.12})
mlflow.pytorch.log_model(model, "model")

# 注册模型
mlflow.register_model("runs:/{run_id}/model", "my-model")
```

### MLOps 成熟度
| 级别 | 特征 | 关键工具 |
|------|------|----------|
| **L0** | 手动 Notebook | Jupyter |
| **L1** | 自动化训练 | MLflow + Airflow |
| **L2** | CI/CD 自动化 | GitHub Actions + K8s |

---

## 🚀 模型服务化速查

### vLLM 部署
```bash
# Docker 启动 vLLM
docker run --gpus all -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2-7B-Instruct \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9

# Python API
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen2-7B-Instruct")
output = llm.generate(["你好"], SamplingParams(temperature=0.7))
```

### TGI 部署
```bash
# Docker 启动 TGI
docker run --gpus all -p 8080:80 \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id Qwen/Qwen2-7B-Instruct \
  --max-input-length 2048 \
  --max-total-tokens 4096
```

### OpenAI 兼容调用
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="Qwen/Qwen2-7B-Instruct",
    messages=[{"role": "user", "content": "你好"}],
)
```

### 推理框架对比
| 框架 | 核心优势 | 适用场景 |
|------|----------|----------|
| **vLLM** | PagedAttention 高吞吐 | 高并发推理 |
| **TGI** | HF 生态、Docker 一键 | 快速部署 |
| **Ollama** | 本地运行、简单 | 开发测试 |
| **TensorRT-LLM** | NVIDIA 优化、极致性能 | 生产环境 |

---

## 🎮 GPU 速查

### 显存需求估算
```
推理显存 ≈ 参数量(B) × 2 (FP16) + KV Cache
训练显存 ≈ 参数量(B) × 2 × 4 (权重+梯度+优化器) + 激活值

示例（FP16）：
  7B  推理 ≈ 14GB + KV    训练 ≈ 100GB+
  13B 推理 ≈ 26GB + KV    训练 ≈ 200GB+
  70B 推理 ≈ 140GB + KV   训练 ≈ 1TB+
```

### GPU 选型速查
| GPU | 显存 | FP16 TFLOPS | 适用 |
|-----|------|-------------|------|
| RTX 4090 | 24GB | 82.6 | 推理/LoRA |
| A100-40G | 40GB | 312 | 训练/推理 |
| A100-80G | 80GB | 312 | 大模型训练 |
| H100-80G | 80GB | 989 | 大规模训练 |

### 显存优化技术
```
混合精度 (AMP)     → 显存减半，速度提升 1.5-2x
梯度检查点          → 激活值显存减少 60%，速度降低 30%
梯度累积            → 小 batch 模拟大 batch
DeepSpeed ZeRO-1   → 分片优化器状态
DeepSpeed ZeRO-2   → + 分片梯度
DeepSpeed ZeRO-3   → + 分片模型参数
Flash Attention     → 注意力显存 O(N²) → O(N)
```

### 混合精度代码
```python
# PyTorch AMP
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast(dtype=torch.float16):  # 或 torch.bfloat16
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 📊 数据工程速查

### 数据清洗流水线
```
格式标准化 → 去重 → 质量过滤 → 敏感信息脱敏 → 内容过滤
```

### 去重方法
| 方法 | 原理 | 适用 |
|------|------|------|
| **Hash 去重** | MD5/SHA256 精确匹配 | 完全相同的文本 |
| **MinHash LSH** | 局部敏感哈希 | 近似重复文本 |
| **SimHash** | 相似度哈希 | 大规模去重 |

### 合成数据方法
| 方法 | 特点 | 适用 |
|------|------|------|
| **Self-Instruct** | 从种子扩展 | 快速扩充数据量 |
| **Evol-Instruct** | 指令进化 | 高质量复杂数据 |
| **领域 QA 生成** | 基于文档生成 | 领域定制数据 |

---

## 💰 成本优化速查

### API 定价参考（2024）
| 模型 | 输入 ($/1M tokens) | 输出 ($/1M tokens) |
|------|--------------------|--------------------|
| GPT-4o | $2.5 | $10 |
| GPT-4o-mini | $0.15 | $0.6 |
| Claude 3.5 Sonnet | $3 | $15 |
| Qwen2-7B (自部署) | GPU 固定成本 | GPU 固定成本 |

### 成本优化优先级
```
1. 缓存（最高 ROI）
2. 模型选择（简单任务用小模型）
3. Prompt 优化（减少 Token）
4. 批处理（合并请求）
5. 自部署（高流量时）
```

### Fallback 降级链
```
GPT-4o → GPT-4o-mini → 本地 Qwen2-7B → 缓存响应 → 友好提示
```

---

## 📈 监控速查

### Prometheus 指标
```python
from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter("llm_requests_total", "请求总数", ["model", "status"])
REQUEST_LATENCY = Histogram("llm_request_duration_seconds", "请求延迟", ["model"])
GPU_UTIL = Gauge("gpu_utilization_percent", "GPU 利用率", ["gpu_id"])
```

### 常用 PromQL
```promql
# QPS
rate(llm_requests_total[5m])

# P99 延迟
histogram_quantile(0.99, rate(llm_request_duration_seconds_bucket[5m]))

# 错误率
rate(llm_requests_total{status="error"}[5m]) / rate(llm_requests_total[5m])
```

### 告警分级
| 级别 | 响应时间 | 通知方式 | 示例 |
|------|----------|----------|------|
| **P0** | 5 分钟 | 电话+短信+IM | 服务不可用 |
| **P1** | 1 小时 | IM+邮件 | P99 延迟 > 10s |
| **P2** | 工作日 | 邮件 | GPU 利用率 > 90% |

### Docker Compose 监控栈
```bash
# 启动 Prometheus + Grafana
docker compose -f docker-compose.monitoring.yml up -d

# 访问
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
```

---

## 🔑 关键数字记忆

| 数字 | 含义 |
|------|------|
| **2 bytes** | FP16 每参数显存 |
| **14 GB** | 7B 模型 FP16 权重 |
| **4x** | 训练显存 ≈ 4x 推理显存 |
| **60-80%** | 传统 KV Cache 浪费比例 |
| **2-5x** | Continuous Batching 吞吐提升 |
| **30%** | 梯度检查点额外训练时间 |
| **0.95+** | 语义缓存相似度阈值 |
| **< 1%** | 生产环境错误率目标 |
