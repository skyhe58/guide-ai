---
title: "GPU 环境配置指南"
description: "CUDA 安装、PyTorch GPU 版本、Docker GPU 支持和常见问题排查"
---

# 🖥️ GPU 环境配置指南

> 本指南帮助你配置 AI 开发所需的 GPU 环境，包括 CUDA 安装、PyTorch GPU 版本选择、Docker GPU 支持和常见问题排查。

---

## 环境要求

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| GPU | NVIDIA GTX 1060 6GB | RTX 4090 24GB / A100 40GB |
| CUDA | 11.8+ | 12.1+ |
| 驱动 | 525.60+ | 535.86+ |
| 系统 | Ubuntu 20.04 / Windows 10 | Ubuntu 22.04 |
| 内存 | 16GB | 32GB+ |

---

## 1. NVIDIA 驱动安装

### Ubuntu

```bash
# 检查 GPU 型号
lspci | grep -i nvidia

# 方法 1：使用 ubuntu-drivers（推荐）
sudo ubuntu-drivers autoinstall
sudo reboot

# 方法 2：手动安装指定版本
sudo apt install nvidia-driver-535
sudo reboot

# 验证安装
nvidia-smi
```

### Windows

1. 访问 [NVIDIA 驱动下载页](https://www.nvidia.com/Download/index.aspx)
2. 选择 GPU 型号，下载最新 Game Ready 或 Studio 驱动
3. 运行安装程序，选择"自定义安装" → "执行清洁安装"
4. 重启后在命令行运行 `nvidia-smi` 验证

---

## 2. CUDA Toolkit 安装

### Ubuntu

```bash
# 方法 1：使用 conda 安装（推荐，自动管理版本）
conda install -c nvidia cuda-toolkit=12.1

# 方法 2：系统级安装
# 访问 https://developer.nvidia.com/cuda-downloads 下载
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

# 添加环境变量
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证
nvcc --version
```

### CUDA 版本与驱动对应关系

| CUDA 版本 | 最低驱动版本 | 推荐场景 |
|:---------:|:----------:|----------|
| 11.8 | 520.61 | 兼容性最好，大部分库支持 |
| 12.1 | 530.30 | 推荐，PyTorch 2.x 默认 |
| 12.4 | 550.54 | 最新特性 |

---

## 3. PyTorch GPU 版本安装

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# PyTorch + CUDA 12.1（推荐）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# PyTorch + CUDA 11.8（兼容旧 GPU）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 使用 conda 安装
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 验证 GPU 可用

```python
import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda}")
print(f"GPU 数量: {torch.cuda.device_count()}")
print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
print(f"GPU 显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

# 简单测试
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()
z = x @ y
print(f"GPU 计算测试通过: {z.shape}")
```

---

## 4. Docker GPU 支持

### 安装 NVIDIA Container Toolkit

```bash
# Ubuntu
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 验证 Docker GPU

```bash
# 测试 GPU 在 Docker 中可用
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Docker Compose GPU 配置

```yaml
# docker-compose.yml 中启用 GPU
services:
  vllm:
    image: vllm/vllm-openai:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all  # 或指定数量: 1
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```

---

## 5. 云 GPU 租赁

| 平台 | 特点 | 适用场景 | 参考价格 |
|------|------|----------|----------|
| AutoDL | 国内平台，价格低 | 训练/微调 | RTX 4090: ~2 元/时 |
| Lambda Cloud | 海外平台，A100 可用 | 大模型训练 | A100: ~1.1 美元/时 |
| Google Colab | 免费 T4 GPU | 学习/实验 | 免费（有限制） |
| Vast.ai | 社区 GPU 市场 | 灵活选择 | 价格浮动 |

### AutoDL 快速上手

```bash
# 1. 注册 AutoDL 账号
# 2. 选择 GPU 实例（推荐 RTX 4090 24GB）
# 3. 选择镜像（推荐 PyTorch 2.x + CUDA 12.1）
# 4. SSH 连接
ssh -p <端口> root@<地址>

# 5. 数据传输
scp -P <端口> local_file root@<地址>:/root/
```

---

## 6. 常见问题排查

### Q1: `torch.cuda.is_available()` 返回 False

```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 CUDA 版本
nvcc --version

# 检查 PyTorch CUDA 版本是否匹配
python -c "import torch; print(torch.version.cuda)"

# 常见原因：
# 1. 安装了 CPU 版本的 PyTorch → 重新安装 GPU 版本
# 2. CUDA 版本不匹配 → 安装对应版本的 PyTorch
# 3. 驱动版本过低 → 更新 NVIDIA 驱动
```

### Q2: CUDA out of memory

```python
# 查看 GPU 显存使用
print(torch.cuda.memory_summary())

# 解决方案：
# 1. 减小 batch_size
# 2. 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
with autocast():
    output = model(input)

# 3. 使用梯度检查点
from torch.utils.checkpoint import checkpoint
output = checkpoint(model.layer, input)

# 4. 清理缓存
torch.cuda.empty_cache()
```

### Q3: Docker 中无法访问 GPU

```bash
# 检查 nvidia-container-toolkit 是否安装
nvidia-ctk --version

# 检查 Docker runtime 配置
docker info | grep -i runtime

# 重新配置
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Q4: 多 GPU 训练配置

```python
import torch
import torch.distributed as dist

# 检查可用 GPU
print(f"可用 GPU 数量: {torch.cuda.device_count()}")

# DataParallel（简单，但效率较低）
model = torch.nn.DataParallel(model)

# DistributedDataParallel（推荐）
# 启动命令: torchrun --nproc_per_node=2 train.py
dist.init_process_group(backend='nccl')
model = torch.nn.parallel.DistributedDataParallel(model)
```

---

## GPU 选型建议

| GPU | 显存 | 适用场景 | 参考价格 |
|-----|:----:|----------|----------|
| RTX 4060 Ti | 16GB | 学习/小模型推理 | ~3000 元 |
| RTX 4090 | 24GB | 微调 7B 模型/推理 | ~14000 元 |
| A100 | 40/80GB | 训练/微调大模型 | 云租赁 |
| H100 | 80GB | 大规模训练 | 云租赁 |

> 💡 建议：学习阶段用 Google Colab 免费 T4，微调阶段租 AutoDL RTX 4090，生产部署用 A100。
