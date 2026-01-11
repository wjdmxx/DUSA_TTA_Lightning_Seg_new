#!/usr/bin/env python
"""
DUSA TTA Runner Script

Main entry point for running Test-Time Adaptation with diffusion guidance.
Uses Hydra for configuration management and W&B for logging.

Usage:
    python scripts/run_tta.py
    python scripts/run_tta.py data.data_root=/path/to/ade20k
    python scripts/run_tta.py model/generative=sd3_multi_gpu  # Multi-GPU mode
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import torch
import wandb

from src.models.combined import CombinedModel
from src.data.datamodule import TTADataModule
from src.tta.module import DUSATTAModule, TTARunner
from src.callbacks.tta_callbacks import TTAProgressCallback


def setup_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    pl.seed_everything(seed, workers=True)


def build_model(cfg: DictConfig) -> CombinedModel:
    """Build the combined model from configuration."""
    # Extract discriminative config
    discriminative_config = OmegaConf.to_container(cfg.model.discriminative, resolve=True)
    
    # Extract generative config
    generative_config = OmegaConf.to_container(cfg.model.generative, resolve=True)
    
    # Build combined model
    model = CombinedModel(
        discriminative_config=discriminative_config,
        generative_config=generative_config,
        update_discriminative=cfg.model.update_discriminative,
        update_generative=cfg.model.update_generative,
        update_norm_only=cfg.model.update_norm_only,
    )
    
    return model


def build_datamodule(cfg: DictConfig) -> TTADataModule:
    """Build the data module from configuration."""
    data_cfg = cfg.data
    
    return TTADataModule(
        data_root=data_cfg.data_root,
        corruption_types=list(data_cfg.corruption_types),
        severity=data_cfg.severity,
        target_short_side=data_cfg.target_short_side,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
    )


def build_tta_module(model: CombinedModel, cfg: DictConfig) -> DUSATTAModule:
    """Build the TTA Lightning module."""
    tta_cfg = cfg.tta
    
    return DUSATTAModule(
        model=model,
        learning_rate=tta_cfg.learning_rate,
        optimizer_type=tta_cfg.optimizer_type,
        optimizer_betas=tuple(tta_cfg.optimizer_betas),
        weight_decay=tta_cfg.weight_decay,
        use_amp=tta_cfg.use_amp,
    )


def build_wandb_logger(cfg: DictConfig) -> WandbLogger:
    """Build W&B logger from configuration."""
    wandb_cfg = cfg.wandb
    
    # Generate run name if not provided
    run_name = wandb_cfg.name
    if run_name is None:
        run_name = f"{cfg.experiment.name}_{wandb_cfg.tags[0]}"
    
    return WandbLogger(
        project=wandb_cfg.project,
        entity=wandb_cfg.entity,
        name=run_name,
        tags=list(wandb_cfg.tags),
        offline=wandb_cfg.offline,
        config=OmegaConf.to_container(cfg, resolve=True),
    )


def build_trainer_config(cfg: DictConfig) -> dict:
    """Build trainer configuration dict."""
    trainer_cfg = cfg.trainer
    
    return {
        "accelerator": trainer_cfg.accelerator,
        "devices": trainer_cfg.devices,
        "strategy": trainer_cfg.strategy,
        "precision": trainer_cfg.precision,
        "enable_progress_bar": trainer_cfg.enable_progress_bar,
        "enable_model_summary": trainer_cfg.enable_model_summary,
        "enable_checkpointing": trainer_cfg.enable_checkpointing,
        "deterministic": trainer_cfg.deterministic,
        "fast_dev_run": trainer_cfg.fast_dev_run,
        "limit_test_batches": trainer_cfg.limit_test_batches,
        "log_every_n_steps": trainer_cfg.log_every_n_steps,
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for TTA."""
    # Print configuration
    print("=" * 60)
    print("DUSA TTA Configuration")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)
    
    # Setup seed
    setup_seed(cfg.experiment.seed)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DUSA TTA")
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Build components
    print("\nBuilding model...")
    model = build_model(cfg)
    
    # Save initial state for reset between tasks
    model.save_initial_state()
    
    print("\nBuilding data module...")
    datamodule = build_datamodule(cfg)
    
    print("\nBuilding TTA module...")
    tta_module = build_tta_module(model, cfg)
    
    # Build W&B logger
    print("\nInitializing W&B logger...")
    wandb_logger = build_wandb_logger(cfg)
    
    # Build trainer config
    trainer_config = build_trainer_config(cfg)
    
    # Create TTA runner
    print("\nStarting TTA...")
    runner = TTARunner(
        module=tta_module,
        datamodule=datamodule,
        trainer_config=trainer_config,
        wandb_logger=wandb_logger,
    )
    
    # Run TTA across all corruption types
    results = runner.run()
    
    # Print final results
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    
    for corruption_type, metrics in results.items():
        if corruption_type == "aggregate":
            continue
        print(f"{corruption_type:20s}: mIoU = {metrics['miou']:.4f}, Acc = {metrics['accuracy']:.4f}")
    
    print("-" * 60)
    print(f"{'Mean mIoU':20s}: {results['aggregate']['mean_miou']:.4f}")
    print("=" * 60)
    
    # Log final results to W&B
    wandb.log({"final/mean_miou": results["aggregate"]["mean_miou"]})
    
    # Finish W&B run
    wandb.finish()
    
    print("\nTTA completed successfully!")


if __name__ == "__main__":
    main()
