"""Main script for running TTA on segmentation tasks.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python scripts/run_tta.py
    CUDA_VISIBLE_DEVICES=0,1 python scripts/run_tta.py experiment.name=my_exp
    CUDA_VISIBLE_DEVICES=0,1 python scripts/run_tta.py tta.forward_mode=discriminative_only
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar
import wandb
import torch

from src.tta.module import TTAModule, create_tta_module
from src.data.datamodule import TTADataModule, SingleTaskDataModule, get_task_names
from src.data.ade20k import ADE20KCorruptedDataset


def setup_wandb(cfg: DictConfig) -> WandbLogger:
    """Setup Weights & Biases logger.

    Args:
        cfg: Configuration

    Returns:
        WandbLogger instance
    """
    wandb_cfg = cfg.logging.wandb

    logger = WandbLogger(
        project=wandb_cfg.project,
        entity=wandb_cfg.entity,
        name=wandb_cfg.name,
        tags=list(wandb_cfg.tags) if wandb_cfg.tags else None,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    return logger


def create_trainer(cfg: DictConfig, logger: WandbLogger, task_name: str) -> pl.Trainer:
    """Create PyTorch Lightning Trainer.

    Args:
        cfg: Configuration
        logger: WandB logger
        task_name: Name of current task

    Returns:
        Trainer instance
    """
    trainer_cfg = cfg.trainer

    # Custom progress bar
    progress_bar = TQDMProgressBar(refresh_rate=10)

    trainer = pl.Trainer(
        accelerator=trainer_cfg.accelerator,
        # Don't specify devices - we handle device placement manually
        devices=1,  # Use single GPU from Lightning's perspective
        strategy="auto",
        precision=trainer_cfg.precision,
        max_epochs=cfg.tta.num_epochs,
        gradient_clip_val=trainer_cfg.gradient_clip_val,
        accumulate_grad_batches=trainer_cfg.accumulate_grad_batches,
        log_every_n_steps=trainer_cfg.log_every_n_steps,
        enable_progress_bar=trainer_cfg.enable_progress_bar,
        enable_model_summary=False,  # We handle this manually
        enable_checkpointing=False,  # No checkpointing for TTA
        logger=logger,
        callbacks=[progress_bar],
        deterministic=trainer_cfg.deterministic,
        benchmark=trainer_cfg.benchmark,
    )

    return trainer


def run_single_task(
    cfg: DictConfig,
    task_idx: int,
    task_name: str,
    datamodule: SingleTaskDataModule,
    logger: WandbLogger,
    initial_state: dict = None
):
    """Run TTA on a single task.

    Args:
        cfg: Configuration
        task_idx: Task index
        task_name: Task name
        datamodule: DataModule for this task
        logger: WandB logger
        initial_state: Initial model state (for reset between tasks)

    Returns:
        Final mIoU for this task
    """
    print(f"\n{'='*60}")
    print(f"Task {task_idx}: {task_name}")
    print(f"{'='*60}\n")

    # Create TTA module
    tta_module = create_tta_module(
        model_cfg=cfg.model,
        tta_cfg=cfg.tta,
        logging_cfg=cfg.logging,
        data_cfg=cfg.data,
        task_name=task_name,
        task_idx=task_idx
    )

    # Load initial state if provided (for continual=False)
    if initial_state is not None and not cfg.tta.continual:
        tta_module.model.load_state_dict(initial_state, strict=False)

    # Create trainer for this task
    trainer = create_trainer(cfg, logger, task_name)

    # Run TTA
    trainer.fit(tta_module, datamodule)

    # Get final metrics
    final_miou = tta_module.train_miou.compute().item() * 100

    # Log task completion
    if wandb.run is not None:
        wandb.log({
            f"task_{task_idx}_final_mIoU": final_miou,
            "task_completed": task_idx + 1,
        })

    print(f"\nTask {task_idx} ({task_name}) completed - mIoU: {final_miou:.2f}%\n")

    # Return state for next task (if continual)
    return final_miou, tta_module.model.state_dict()


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main entry point for TTA.

    Args:
        cfg: Hydra configuration
    """
    # Print configuration
    print(OmegaConf.to_yaml(cfg))

    # Set seed
    pl.seed_everything(cfg.experiment.seed, workers=True)

    # Setup W&B
    logger = setup_wandb(cfg)

    # Get task list
    task_names = get_task_names(cfg.data)
    num_tasks = len(task_names)

    print(f"\nRunning TTA on {num_tasks} tasks:")
    for i, name in enumerate(task_names):
        print(f"  {i}: {name}")
    print()

    # Create initial model to get initial state
    print("Creating initial model...")
    initial_module = create_tta_module(
        model_cfg=cfg.model,
        tta_cfg=cfg.tta,
        logging_cfg=cfg.logging,
        data_cfg=cfg.data,
        task_name="init",
        task_idx=-1
    )
    initial_state = initial_module.model.get_initial_state()
    del initial_module
    torch.cuda.empty_cache()

    # Run TTA on each task
    all_miou = []
    current_state = initial_state

    for task_idx, task_name in enumerate(task_names):
        # Create dataset for this task
        corruption = task_name.rsplit("_s", 1)[0]
        severity = int(task_name.rsplit("_s", 1)[1])

        dataset = ADE20KCorruptedDataset(
            data_root=cfg.data.data_root,
            corruption=corruption,
            severity=severity,
            target_short_edge=cfg.data.preprocessing.target_short_edge,
            reduce_zero_label=cfg.data.reduce_zero_label,
        )

        datamodule = SingleTaskDataModule(
            dataset=dataset,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            shuffle=cfg.data.shuffle,
            pin_memory=cfg.data.pin_memory,
        )

        # Run TTA on this task
        miou, current_state = run_single_task(
            cfg=cfg,
            task_idx=task_idx,
            task_name=task_name,
            datamodule=datamodule,
            logger=logger,
            initial_state=initial_state if not cfg.tta.continual else current_state
        )

        all_miou.append(miou)

        # Clear cache between tasks
        torch.cuda.empty_cache()

    # Print summary
    print("\n" + "="*60)
    print("TTA Complete - Summary")
    print("="*60)
    print(f"\nmIoU per task:")
    for name, miou in zip(task_names, all_miou):
        print(f"  {name}: {miou:.2f}%")
    print(f"\nAverage mIoU: {sum(all_miou)/len(all_miou):.2f}%")

    # Log final summary to W&B
    if wandb.run is not None:
        wandb.log({
            "final_avg_mIoU": sum(all_miou) / len(all_miou),
            "total_tasks": num_tasks,
        })

        # Log summary table
        table_data = [[name, miou] for name, miou in zip(task_names, all_miou)]
        table = wandb.Table(data=table_data, columns=["Task", "mIoU"])
        wandb.log({"task_summary": table})

        wandb.finish()


if __name__ == "__main__":
    main()
