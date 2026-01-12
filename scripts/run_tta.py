#!/usr/bin/env python
"""Main entry point for TTA experiments.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python scripts/run_tta.py
    
    # Override config values:
    CUDA_VISIBLE_DEVICES=0,1 python scripts/run_tta.py tta.forward_mode=discriminative_only
    CUDA_VISIBLE_DEVICES=0,1 python scripts/run_tta.py data.corruptions=[gaussian_noise,fog]
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from copy import deepcopy

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import FSDPStrategy
from pytorch_lightning.callbacks import TQDMProgressBar

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.discriminative import DiscriminativeModel
from src.models.generative.sd3 import SD3GenerativeModel
from src.models.combined import CombinedModel
from src.tta.module import TTAModule
from src.data.datamodule import ADE20KCDataModule


def build_discriminative_model(cfg: DictConfig) -> DiscriminativeModel:
    """Build discriminative model from config."""
    return DiscriminativeModel(
        model_name=cfg.model.discriminative.model_name,
        num_classes=cfg.model.discriminative.num_classes,
    )


def build_generative_model(cfg: DictConfig) -> Optional[SD3GenerativeModel]:
    """Build generative model from config if TTA mode."""
    if cfg.tta.forward_mode == "discriminative_only":
        return None
    
    gen_cfg = cfg.model.generative
    return SD3GenerativeModel(
        model_path=gen_cfg.model_path,
        window_size=gen_cfg.sliding_window.window_size,
        stride=gen_cfg.sliding_window.stride,
        timestep_range=tuple(gen_cfg.timestep_range),
        topk=gen_cfg.topk,
        temperature=gen_cfg.temperature,
        classes_threshold=gen_cfg.classes_threshold,
        prompt_template=gen_cfg.prompt,
        class_names=gen_cfg.class_names,
        embedding_cache_dir=gen_cfg.get('embedding_cache_dir', './embedding_cache'),
    )


def build_combined_model(cfg: DictConfig) -> CombinedModel:
    """Build combined model from config."""
    discriminative = build_discriminative_model(cfg)
    generative = build_generative_model(cfg)
    
    model = CombinedModel(
        discriminative=discriminative,
        generative=generative,
    )
    
    # Configure gradients
    model.configure_grad(
        update_discriminative=cfg.tta.get('update_discriminative', True),
        update_generative=cfg.tta.get('update_generative', True),
    )
    
    # Ensure uniform dtype for FSDP compatibility
    # This must be called before trainer.fit() to avoid FSDP dtype mismatch errors
    model.ensure_uniform_dtype(torch.float32)
    
    # Save initial state for reset between tasks
    model.save_initial_state()
    
    return model


def build_datamodule(cfg: DictConfig) -> ADE20KCDataModule:
    """Build data module from config."""
    data_cfg = cfg.data
    return ADE20KCDataModule(
        data_root=data_cfg.data_root,
        corruptions=list(data_cfg.corruptions),
        severity=data_cfg.severity,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        shuffle=data_cfg.shuffle,
        short_edge_size=data_cfg.get('short_edge_size', 512),
    )


def build_trainer(
    cfg: DictConfig,
    logger: WandbLogger,
    task_id: int,
) -> pl.Trainer:
    """Build trainer for a single TTA task.
    
    A new trainer is created for each task so that max_epochs works correctly.
    """
    trainer_cfg = cfg.trainer
    
    # Configure FSDP strategy
    strategy = FSDPStrategy(
        # Let Lightning handle auto-wrapping
        auto_wrap_policy=None,
        # Use bf16 mixed precision handled by trainer
    )
    
    # Progress bar callback
    callbacks = [
        TQDMProgressBar(refresh_rate=1),
    ]
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=trainer_cfg.devices,
        strategy=strategy,
        precision=trainer_cfg.precision,
        max_epochs=cfg.tta.max_epochs,
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=logger,
        callbacks=callbacks,
        # Disable validation
        limit_val_batches=0,
        num_sanity_val_steps=0,
    )
    
    return trainer


def run_tta_tasks(cfg: DictConfig) -> Dict[str, float]:
    """Run TTA on all corruption tasks.
    
    Args:
        cfg: Hydra config
        
    Returns:
        Dictionary mapping task names to mIoU values
    """
    # Initialize W&B logger
    wandb_cfg = cfg.logging.wandb
    logger = WandbLogger(
        project=wandb_cfg.project,
        name=cfg.experiment_name,
        entity=wandb_cfg.get('entity', None),
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    
    # Build model
    print("\n" + "="*60)
    print("Building models...")
    print("="*60 + "\n")
    
    combined_model = build_combined_model(cfg)
    
    # Build data module
    datamodule = build_datamodule(cfg)
    corruptions = datamodule.get_all_corruptions()
    
    print(f"\nWill run TTA on {len(corruptions)} corruption types:")
    for i, c in enumerate(corruptions):
        print(f"  [{i}] {c}")
    print()
    
    # Track results
    results: Dict[str, float] = {}
    all_mious: List[float] = []
    
    # Run TTA for each corruption task
    for task_id, corruption in enumerate(corruptions):
        print(f"\n{'='*60}")
        print(f"Task [{task_id}/{len(corruptions)}]: {corruption}")
        print(f"{'='*60}\n")
        
        # Reset model if not continual
        if not cfg.tta.continual and task_id > 0:
            print("Resetting model to initial state...")
            combined_model.reset_to_initial()
        
        # Set current corruption
        datamodule.set_corruption(corruption)
        
        # Create TTA module for this task
        tta_module = TTAModule(
            model=combined_model,
            num_classes=cfg.model.discriminative.num_classes,
            ignore_index=255,
            learning_rate=cfg.tta.learning_rate,
            weight_decay=cfg.tta.get('weight_decay', 0.0),
            optimizer_type=cfg.tta.get('optimizer_type', 'adamw'),
            betas=tuple(cfg.tta.get('betas', [0.9, 0.999])),
            forward_mode=cfg.tta.forward_mode,
            task_id=task_id,
            task_name=datamodule.get_task_name(),
        )
        
        # Build trainer for this task
        trainer = build_trainer(cfg, logger, task_id)
        
        # Run TTA
        trainer.fit(tta_module, datamodule)
        
        # Get results
        final_metrics = tta_module.get_final_metrics()
        task_miou = final_metrics['mIoU']
        
        task_name = datamodule.get_task_name()
        results[task_name] = task_miou
        all_mious.append(task_miou)
        
        # Log to W&B
        logger.log_metrics({
            f'final/{task_name}_mIoU': task_miou,
            'final/task_id': task_id,
        })
        
        print(f"\nTask {task_name} mIoU: {task_miou:.4f}")
    
    # Compute and log average
    avg_miou = sum(all_mious) / len(all_mious) if all_mious else 0.0
    results['average'] = avg_miou
    
    logger.log_metrics({'final/average_mIoU': avg_miou})
    
    # Print summary
    print("\n" + "="*60)
    print("TTA Results Summary")
    print("="*60)
    print(f"\n{'Task':<40} {'mIoU':>10}")
    print("-"*52)
    for task_name, miou in results.items():
        if task_name != 'average':
            print(f"{task_name:<40} {miou:>10.4f}")
    print("-"*52)
    print(f"{'Average':<40} {avg_miou:>10.4f}")
    print("="*60 + "\n")
    
    # Finish W&B run
    logger.finalize("success")
    
    return results


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for TTA experiments."""
    # Print config
    print("\n" + "="*60)
    print("Configuration")
    print("="*60)
    print(OmegaConf.to_yaml(cfg))
    print("="*60 + "\n")
    
    # Set seed for reproducibility
    if cfg.get('seed', None) is not None:
        pl.seed_everything(cfg.seed, workers=True)
    
    # Run TTA
    results = run_tta_tasks(cfg)
    
    print("\nTTA completed successfully!")


if __name__ == "__main__":
    main()
