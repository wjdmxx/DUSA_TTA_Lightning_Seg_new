# TTA Segmentation with Generative Models

Test-Time Adaptation for semantic segmentation using Stable Diffusion 3 as auxiliary generative model.

## Overview

This project implements Test-Time Adaptation (TTA) for semantic segmentation by combining:
- **Discriminative Model**: Segformer (from HuggingFace Transformers)
- **Generative Model**: Stable Diffusion 3 (SD3) for providing TTA loss signal

The approach uses the generative model to compute a diffusion loss based on segmentation predictions, which helps adapt the model to distribution shifts at test time.

## Features

- PyTorch Lightning + Hydra for clean, configurable training
- Model parallelism: Split large SD3 transformer across multiple GPUs
- Sliding window processing for handling variable-sized images
- W&B logging for experiment tracking
- Support for ADE20K-C corruption benchmark
- bf16 mixed precision training

## Requirements

```bash
pip install torch torchvision
pip install pytorch-lightning
pip install hydra-core omegaconf
pip install transformers diffusers accelerate
pip install wandb
pip install einops
pip install pillow numpy
pip install torchmetrics
```

## Project Structure

```
seg_new/
├── configs/
│   ├── config.yaml              # Main configuration
│   ├── model/
│   │   ├── combined.yaml        # Combined model config
│   │   ├── discriminative/
│   │   │   └── segformer_mit-b5.yaml
│   │   └── generative/
│   │       └── sd3.yaml
│   ├── data/
│   │   └── ade20k_c.yaml
│   └── trainer/
│       └── default.yaml
├── src/
│   ├── models/
│   │   ├── discriminative.py    # Segformer wrapper
│   │   ├── generative/
│   │   │   └── sd3.py           # SD3 with sliding window
│   │   └── combined.py          # Combined model
│   ├── tta/
│   │   └── module.py            # Lightning module
│   ├── data/
│   │   ├── ade20k.py            # Dataset classes
│   │   └── datamodule.py        # Lightning DataModule
│   └── utils/
│       └── categories.py        # Class definitions
├── scripts/
│   └── run_tta.py               # Main entry point
└── README.md
```

## Usage

### Basic Usage

Run TTA with default configuration:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_tta.py
```

### Custom Experiment Name

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_tta.py experiment.name=my_experiment
```

### Baseline (Discriminative Only)

Run without the generative model for comparison:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_tta.py tta.forward_mode=discriminative_only
```

### Custom Data Path

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_tta.py data.data_root=/path/to/ade20k_c
```

### Specific Corruptions

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_tta.py \
    'data.corruptions=[gaussian_noise,shot_noise,impulse_noise]'
```

### Continual Adaptation

Run TTA without resetting between tasks:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_tta.py tta.continual=true
```

## Data Preparation

### ADE20K-C Dataset

The dataset should be organized as:

```
data/ade20k_c/
├── images/
│   └── validation/
│       ├── gaussian_noise_5/
│       │   ├── ADE_val_00000001.jpg
│       │   └── ...
│       ├── shot_noise_5/
│       │   └── ...
│       └── ...
└── annotations/
    └── validation/
        ├── ADE_val_00000001.png
        └── ...
```

## Model Parallelism

This project uses model parallelism to run on two GPUs:

- **GPU 0 (cuda:0)**:
  - Segformer (discriminative model)
  - VAE encoder
  - Text encoders
  - First half of SD3 transformer blocks

- **GPU 1 (cuda:1)**:
  - Second half of SD3 transformer blocks
  - Output layers

This allows running the full model which would not fit on a single GPU.

## Key Configuration Options

### TTA Settings (`tta.forward_mode`)
- `"tta"`: Full TTA with generative model
- `"discriminative_only"`: Baseline without generative loss

### Model Updates (`model.update`)
- `discriminative: true/false`: Update Segformer during TTA
- `generative: true/false`: Update SD3 during TTA
- `update_norm_only: true/false`: Only update normalization layers

### Loss Settings (`model.loss`)
- `topk`: Number of top classes to consider per pixel
- `classes_threshold`: Maximum unique classes per window
- `temperature`: Softmax temperature for class probabilities

## Metrics

- **mIoU**: Mean Intersection over Union (primary metric)
- Per-class IoU logged to W&B
- Loss values for TTA monitoring

## W&B Logging

Results are logged to Weights & Biases:

1. Set your W&B entity in the config or via environment:
   ```bash
   export WANDB_ENTITY=your_entity
   ```

2. Or specify in config:
   ```yaml
   logging:
     wandb:
       entity: your_entity
   ```

## Algorithm Overview

1. **Image Preprocessing**: Resize input so short edge = 512
2. **Discriminative Forward**: Get segmentation logits (at 1/4 resolution)
3. **Sliding Window**: Process image in 512x512 windows along the long edge
4. **TopK Selection**: Select most confident classes per window
5. **Diffusion Loss**:
   - Encode window to latent space
   - Add noise at random timestep
   - Predict velocity conditioned on class embeddings
   - Compute MSE loss weighted by class probabilities
6. **Backprop**: Update both discriminative and generative models

## Citation

If you use this code, please cite the relevant papers for:
- Segformer
- Stable Diffusion 3
- Test-Time Adaptation methods

## License

This project is for research purposes. Please check licenses for:
- HuggingFace Transformers
- Diffusers
- Stable Diffusion 3
