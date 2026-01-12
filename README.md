# SD3-TTA: Test-Time Adaptation with Stable Diffusion 3

A lightweight, modular framework for Test-Time Adaptation (TTA) of semantic segmentation models using Stable Diffusion 3 as a generative prior.

## Features

- **Hydra Configuration**: Flexible, hierarchical configuration system
- **Multi-GPU Support**: Automatic model distribution using `accelerate dispatch_model`
- **W&B Logging**: Comprehensive experiment tracking with Weights & Biases
- **bf16 Mixed Precision**: Efficient training with bfloat16
- **Modular Design**: Easy to extend and customize

## Installation

```bash
# Clone the repository
git clone <repo_url>
cd sd3_tta

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

The project expects the ADE20K-C dataset in the following structure:

```
data/
├── ADE20K_val-c/
│   ├── gaussian_noise/
│   │   └── 5/
│   │       └── validation/
│   │           └── *.jpg
│   ├── shot_noise/
│   │   └── ...
│   └── ... (15 corruption types)
└── annotations/
    └── validation/
        └── *.png
```

## Usage

### Basic Usage

```bash
# Run full TTA with default settings
python -m src.main

# Run baseline (discriminative only, no TTA)
python -m src.main experiment=baseline
```

### Multi-GPU Usage

```bash
# Use 4 GPUs for SD3 transformer distribution
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.main

# Use 2 GPUs
CUDA_VISIBLE_DEVICES=0,1 python -m src.main
```

### Custom Configuration

```bash
# Change experiment ID (for separate W&B runs)
python -m src.main experiment.id=2

# Change forward mode
python -m src.main tta.forward_mode=discriminative_only

# Disable gradient checkpointing
python -m src.main model.generative.transformer.gradient_checkpointing=false

# Change learning rate
python -m src.main tta.optimizer.lr=0.0001

# Run with continual adaptation (no reset between corruptions)
python -m src.main tta.continual=true

# Change data path
python -m src.main data.root=/path/to/data
```

### Configuration Files

Configuration is managed via Hydra with the following structure:

```
configs/
├── config.yaml                    # Main config entry
├── model/
│   ├── discriminative/
│   │   └── segformer_b5.yaml     # SegFormer B5 config
│   └── generative/
│       └── sd3.yaml              # SD3 config
├── data/
│   └── ade20k_c.yaml             # Dataset config
├── tta/
│   └── default.yaml              # TTA parameters
├── logging/
│   └── wandb.yaml                # W&B config
└── experiment/
    ├── full_tta.yaml             # Full TTA experiment
    └── baseline.yaml             # Baseline experiment
```

## Key Configuration Options

### TTA Settings (`configs/tta/default.yaml`)

| Option | Values | Description |
|--------|--------|-------------|
| `forward_mode` | `tta`, `discriminative_only` | TTA mode or baseline |
| `continual` | `true`, `false` | Reset model between corruptions |
| `batch_size` | int | Batch size (1 recommended) |
| `optimizer.lr` | float | Learning rate |

### SD3 Settings (`configs/model/generative/sd3.yaml`)

| Option | Values | Description |
|--------|--------|-------------|
| `transformer.device_map` | `balanced`, `auto` | GPU distribution strategy |
| `transformer.gradient_checkpointing` | `true`, `false` | Memory optimization |
| `sliding_window.size` | int | Window size (512) |
| `sliding_window.stride` | int | Stride between windows |

### Experiment Settings (`configs/config.yaml`)

| Option | Values | Description |
|--------|--------|-------------|
| `experiment.name` | string | Experiment name for logging |
| `experiment.id` | int | Experiment ID (for W&B) |
| `device.precision` | `bf16`, `fp16`, `fp32` | Computation precision |

## Project Structure

```
sd3_tta/
├── configs/                  # Hydra configuration files
├── src/
│   ├── main.py              # Entry point
│   ├── models/
│   │   ├── discriminative.py    # SegFormer wrapper
│   │   ├── combined.py          # Combined model
│   │   ├── device_utils.py      # Multi-GPU utilities
│   │   └── generative/
│   │       ├── sd3.py           # SD3 model
│   │       ├── sliding_window.py # Sliding window processor
│   │       └── text_embeddings.py # Embedding cache
│   ├── data/
│   │   ├── dataset.py           # ADE20K-C dataset
│   │   └── transforms.py        # Data transforms
│   ├── tta/
│   │   ├── runner.py            # TTA orchestration
│   │   ├── loss.py              # Loss computation
│   │   └── topk.py              # TopK selection
│   ├── metrics/
│   │   └── segmentation.py      # mIoU metrics
│   └── utils/
│       ├── categories.py        # ADE20K categories
│       └── logging.py           # W&B logging
├── embeddings_cache/        # Cached text embeddings
└── outputs/                 # Experiment outputs
```

## How It Works

1. **Image Preprocessing**: Images are resized so the short edge is 512 pixels
2. **Discriminative Model**: SegFormer produces segmentation logits (4x downsampled)
3. **Sliding Window**: For generative model, images are processed in sliding windows along the long edge
4. **Generative Loss**: SD3 computes a flow-matching loss using probability-weighted predictions
5. **Gradient Update**: Loss backpropagates through both models
6. **Metrics**: mIoU computed using torchmetrics

## Metrics

The framework tracks:
- **mIoU** (mean Intersection over Union) - primary metric
- Accuracy
- F1 Score
- Precision
- Recall

All metrics are logged to W&B and printed to console.

## Citation

If you use this code, please cite:

```bibtex
@article{sd3_tta,
  title={Test-Time Adaptation with Stable Diffusion 3},
  year={2024}
}
```

## License

MIT License
