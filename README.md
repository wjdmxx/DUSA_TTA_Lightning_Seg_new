# DUSA TTA Lightning

Test-Time Adaptation using Diffusion-Guided Unsupervised Adaptation (DUSA) for semantic segmentation.

## Overview

This project implements Test-Time Adaptation (TTA) for semantic segmentation using diffusion models (Stable Diffusion 3) to guide the adaptation of discriminative models (SegFormer). The core idea is to use the denoising loss from the generative model as a supervisory signal during test time.

### Key Features

- **Lightning Framework**: Built on PyTorch Lightning for clean, modular code
- **Hydra Configuration**: Flexible configuration system with YAML files
- **W&B Logging**: Comprehensive experiment tracking with Weights & Biases
- **Multi-GPU Support**: Easy scaling from single GPU to multi-GPU setups
- **Sliding Window**: Memory-efficient processing for high-resolution images

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
dusa_tta_lightning/
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # Main configuration
│   ├── model/                 # Model configurations
│   ├── data/                  # Dataset configurations
│   └── trainer/               # Trainer configurations
├── src/
│   ├── models/
│   │   ├── discriminative.py  # SegFormer wrapper
│   │   ├── generative/        # SD3 components
│   │   └── combined.py        # Combined model
│   ├── data/
│   │   ├── datamodule.py      # Lightning DataModule
│   │   └── ade20k_c.py        # ADE20K-C dataset
│   ├── tta/
│   │   └── module.py          # TTA Lightning Module
│   ├── utils/
│   │   ├── categories.py      # Class definitions
│   │   ├── slide_inference.py # Sliding window
│   │   └── device_utils.py    # Multi-GPU utilities
│   └── callbacks/
│       └── tta_callbacks.py   # Custom callbacks
└── scripts/
    └── run_tta.py             # Main entry point
```

## Usage

### Default Usage (Dual GPU - Recommended)

By default, the project is configured to run on **2 GPUs** to handle memory constraints. The models are distributed as follows:

| Component | Device | Memory Usage (approx.) |
|-----------|--------|------------------------|
| SegFormer (Discriminative) | `cuda:0` | ~2GB |
| VAE (SD3 Encoder) | `cuda:0` | ~1GB |
| SD3 Transformer | `cuda:1` | ~12GB |

**Run with:**
```bash
# Linux/Mac
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_tta.py data.data_root=/path/to/ade20k

# Windows (PowerShell)
$env:CUDA_VISIBLE_DEVICES="0,1"; python scripts/run_tta.py data.data_root=/path/to/ade20k

# Windows (CMD)
set CUDA_VISIBLE_DEVICES=0,1 && python scripts/run_tta.py data.data_root=/path/to/ade20k
```

### Single GPU Mode (High VRAM Required)

If you have a single GPU with ≥24GB VRAM, you can use single GPU mode:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_tta.py \
    data.data_root=/path/to/ade20k \
    model/generative=sd3
```

### Custom GPU Assignment

You can customize which GPUs to use:

```bash
# Use GPU 2 and 3 instead of 0 and 1
CUDA_VISIBLE_DEVICES=2,3 python scripts/run_tta.py data.data_root=/path/to/ade20k

# Or override device config directly
python scripts/run_tta.py \
    data.data_root=/path/to/ade20k \
    model.discriminative.device=cuda:0 \
    model.generative.vae_device=cuda:0 \
    model.generative.transformer_device_config.device=cuda:1
```

### Configuration Override Examples

```bash
# Change learning rate
python scripts/run_tta.py tta.learning_rate=5e-5

# Only update norm layers
python scripts/run_tta.py model.update_norm_only=true

# Specific corruption types
python scripts/run_tta.py 'data.corruption_types=[gaussian_noise,shot_noise]'

# Debug mode
python scripts/run_tta.py trainer.fast_dev_run=true
```

## Algorithm

DUSA uses the Flow Matching loss from SD3 to guide TTA:

$$\mathcal{L} = \left\| \sum_{k=1}^{K} p_k \cdot \hat{v}_k - (\epsilon - z_0) \right\|^2$$

Where:
- $z_0$ = VAE encoded latent
- $\epsilon$ = Sampled noise
- $\hat{v}_k$ = Predicted velocity for class $k$
- $p_k$ = Softmax probability for class $k$ from segmentation model

## Data Preparation

### ADE20K-C Dataset

Expected directory structure:
```
data_root/
├── ADE20K_val-c/
│   ├── gaussian_noise/
│   │   └── 5/
│   │       └── validation/
│   │           └── *.jpg
│   ├── shot_noise/
│   │   └── 5/
│   │       └── validation/
│   │           └── *.jpg
│   └── ... (other corruptions)
└── annotations/
    └── validation/
        └── *.png
```

## Configuration Reference

### Model Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.update_discriminative` | `true` | Update discriminative model |
| `model.update_generative` | `true` | Update generative model |
| `model.update_norm_only` | `false` | Only update norm layers |

### SD3 Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.generative.timestep` | `0.25` | Flow matching timestep |
| `model.generative.topk` | `1` | Number of top classes |
| `model.generative.classes_threshold` | `20` | Max unique classes |
| `model.generative.crop_size` | `[512, 512]` | Sliding window size |
| `model.generative.slide_stride` | `[0, 171]` | Sliding stride |

### TTA Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tta.learning_rate` | `1e-4` | Learning rate |
| `tta.optimizer_type` | `adamw` | Optimizer type |
| `tta.use_amp` | `true` | Use mixed precision |

## License

MIT License
