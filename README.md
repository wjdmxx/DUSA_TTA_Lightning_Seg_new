# Lightning TTA (SegFormer + SD3)

Test-time adaptation for ADE20K-C using a discriminative SegFormer and a generative SD3 loss in Lightning + Hydra.

## Data
- Expected layout: `/mnt/bit/liyuanxi/projects/DUSA_flow/segmentation/data/`
  - Images: `ADE20K_val-c/{corruption}/{severity}/{split}/*.jpg`
  - Masks: `annotations/{split}/*.png`
- Configurable via `configs/data/ade20k_c.yaml`.

## Devices & Precision
- Run with `CUDA_VISIBLE_DEVICES=0,1`.
- SegFormer + VAE + text encoders stay on `cuda:0`; SD3 transformer blocks are split half on `cuda:0`, half on `cuda:1` using `accelerate.dispatch_model`.
- Mixed precision: bf16 (`precision: bf16-mixed`).

## Sliding Window (SD3)
- Images are resized to short-edge 512 before dataloading.
- SD3 runs on window size 512 along the long edge (stride configurable, default 256). Logits are cropped the same way, then downsampled to latent resolution before loss.
- Top-k class selection and MSE-on-velocity loss match the legacy logic; text embeddings are precomputed once and reused.

## Forward Modes
- `model.forward_mode=joint`: run SegFormer + SD3 loss (default).
- `model.forward_mode=discriminative_only`: skip SD3, backprop a zero loss for metrics-only comparisons.

## Run
```bash
export CUDA_VISIBLE_DEVICES=0,1
python -m pip install -r requirements.txt  # if needed
python script/run_tta.py
```
- Configure corruptions/severities and logging in `configs/`.
- Each corruption/severity task builds a fresh LightningModule and Trainer so `max_epochs` applies per task.

## Logging
- W&B enabled by default; set `logging.enable_wandb=false` to disable.
- Run names combine `experiment_name` + `corruption` + `severity` to avoid collisions.

## Key Files
- Hydra entry: `script/run_tta.py`
- Lightning module: `src/tta/module.py`
- Models: `src/models/discriminative.py`, `src/models/generative/sd3.py`, `src/models/combined.py`
- Data: `src/data/datamodule.py`
- Configs: `configs/`
