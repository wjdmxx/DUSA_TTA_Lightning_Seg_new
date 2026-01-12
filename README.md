# TTA Segmentation (Refactored)

This project implements Test-Time Adaptation (TTA) for Semantic Segmentation using a Generative Model (Stable Diffusion 3) as a guide.
The codebase has been refactored to use pure PyTorch and Hydra for configuration, ensuring flexibility and research-friendly experimentation.

## Features
- **Models**: NVIDIA Segformer (Discriminative) + Stable Diffusion 3 Medium (Generative).
- **Architecture**: Pure PyTorch with Hydra Configs.
- **TTA Strategy**: Minimizing Generative Reconstruction Loss via Sliding Window.
- **Efficient**: SD3 Transformer dispatched across GPUs (Model Parallelism). bf16 mixed precision.

## Installation
1. Install dependencies:
   ```bash
   pip install torch torchvision transformers diffusers accelerate hydra-core wandb
   ```

## Configuration
The project uses Hydra. Configs are in `configs/`.
- `config.yaml`: Main config.
- `model/`: Model params.
- `dataset/`: Dataset paths.

## Usage

### 1. Run Standard TTA
```bash
python main.py
```
This runs with default settings: TTA enabled, default dataset paths.

### 2. Run Source Only (No TTA)
```bash
python main.py tta.forward_mode="source_only"
```

### 3. Change Dataset Corruption
```bash
python main.py dataset.corruption="shot_noise" dataset.severity=5
```

### 4. Adjust TTA Hyperparameters
```bash
python main.py tta.lr=1e-4 tta.steps=5
```

### 5. Multi-GPU Dispatch
The system automatically detects GPUs and tries to split the SD3 Transformer across them using `accelerate`.
Ensure you have set `CUDA_VISIBLE_DEVICES` if checking specific splits.
```bash
CUDA_VISIBLE_DEVICES=0,1 python main.py
```

## Structure
- `src/dataset.py`: ADE20K-C loading.
- `src/models.py`: Segformer & SD3 Wrappers.
- `src/utils.py`: Sliding window loss logic.
- `main.py`: Main execution loop.
