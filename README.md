# TTA Segmentation with Generative Model Assistance

基于 PyTorch Lightning + Hydra 的 Test-Time Adaptation (TTA) 分割项目。

## 项目概述

本项目使用生成式模型 (Stable Diffusion 3) 辅助判别式模型 (SegFormer) 进行测试时自适应。通过在测试阶段利用 SD3 的流匹配损失来更新模型，提升在损坏图像上的分割性能。

## 主要特性

- **PyTorch Lightning**: 清晰的训练逻辑和模块化设计
- **Hydra**: 灵活的配置管理系统
- **FSDP**: 双 GPU 模型并行
- **W&B**: 完整的实验记录
- **滑动窗口**: 沿长边滑动，适应不同图像方向

## 环境要求

```bash
# Python 3.9+
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning hydra-core omegaconf
pip install transformers diffusers accelerate
pip install torchmetrics wandb einops
pip install pillow numpy tqdm
```

## 数据准备

### 目录结构

```
data/
├── ADE20K_val-c/
│   ├── gaussian_noise/
│   │   └── 5/
│   │       └── validation/
│   │           └── *.jpg
│   ├── fog/
│   │   └── 5/
│   │       └── validation/
│   │           └── *.jpg
│   └── ... (其他损坏类型)
└── annotations/
    └── validation/
        └── *.png
```

### 下载数据

```bash
# 创建数据目录
mkdir -p data && cd data

# 下载 ADE20K-C (使用提供的链接)
gdown --fuzzy https://drive.google.com/file/d/1vTYoksyYHdpARqDZxu1LRJny9__tf8xT/view
tar -xzvf ADE20K_val-c.tar.gz

# Annotations 需要从 ADE20K 原始数据集获取
# 链接原始 ADE20K 数据集
ln -s /path/to/ADEChallengeData2016/annotations annotations
```

## 使用方法

### 基本用法

```bash
# 使用两块 GPU 运行完整 TTA
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_tta.py
```

### 配置覆盖

```bash
# 只运行部分损坏类型
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_tta.py \
    data.corruptions=[gaussian_noise,fog,snow]

# 只运行判别模型（用于对比实验）
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_tta.py \
    tta.forward_mode=discriminative_only

# 调整学习率
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_tta.py \
    tta.learning_rate=0.0001

# 连续测试（不重置模型）
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_tta.py \
    tta.continual=true

# 自定义实验名称
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_tta.py \
    experiment_name="my_experiment"
```

### 配置文件

主要配置文件:

| 文件 | 说明 |
|------|------|
| `configs/config.yaml` | 主配置入口 |
| `configs/model/combined.yaml` | 模型组合配置 |
| `configs/model/discriminative/segformer_b5.yaml` | SegFormer 配置 |
| `configs/model/generative/sd3.yaml` | SD3 配置 |
| `configs/data/ade20k_c.yaml` | 数据集配置 |
| `configs/trainer/default.yaml` | Trainer 配置 |

## 项目结构

```
seg_new/
├── configs/                    # Hydra 配置
│   ├── config.yaml
│   ├── model/
│   ├── data/
│   └── trainer/
├── src/
│   ├── models/
│   │   ├── discriminative.py  # SegFormer 封装
│   │   ├── generative/
│   │   │   └── sd3.py         # SD3 + 滑动窗口
│   │   └── combined.py        # 组合模型
│   ├── tta/
│   │   └── module.py          # Lightning TTA Module
│   ├── data/
│   │   ├── dataset.py         # ADE20K-C Dataset
│   │   └── datamodule.py      # Lightning DataModule
│   └── utils/
│       ├── categories.py      # 类别定义
│       └── metrics.py         # mIoU 等指标
├── scripts/
│   └── run_tta.py             # 程序入口
└── README.md
```

## 核心算法

### Top-K 类别选取

对于每个像素，选取 logits 中的 top-k 类别进行处理。当唯一类别数超过阈值时，优先保留所有 top-1 类别，随机采样其余类别。

### 加权 Loss 计算

使用 softmax 概率对预测的速度场进行加权：

```python
# pred_velocity: (B*K, C, H, W) -> (B, K, C, H, W)
weighted_pred = einsum('bkhw,bkchw->bchw', topk_probs, pred_velocity)
loss = mse_loss(weighted_pred, target)
```

### 滑动窗口

- 图像短边缩放到 512
- 窗口大小 512x512
- **沿长边滑动**（非固定水平方向）
- 每个窗口独立计算 loss 后累加

## W&B 日志

实验会记录以下指标：

- `task_{id}/loss`: 每步 loss
- `task_{id}/mIoU`: 每步累计 mIoU
- `final/{task_name}_mIoU`: 每个任务的最终 mIoU
- `final/average_mIoU`: 所有任务的平均 mIoU

## 常见问题

### 显存不足

1. 确保使用 `batch_size=1`
2. 确保使用 `bf16-mixed` 精度
3. 减少 `classes_threshold` 值

### 模型下载问题

使用镜像站点:

```bash
HF_ENDPOINT=https://hf-mirror.com python scripts/run_tta.py
```

### FSDP 相关问题

确保：
1. 使用 `CUDA_VISIBLE_DEVICES=0,1` 指定两块 GPU
2. 数据和模型在相同设备上

## 引用

如果您使用了本项目，请引用:

```bibtex
@article{dusa2024,
  title={DUSA: Decoupled Unsupervised Semantic Adaptation},
  author={...},
  journal={...},
  year={2024}
}
```
