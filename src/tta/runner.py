"""TTA Runner - Main orchestration for Test-Time Adaptation."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset import ADE20KCorruptedDataset, create_dataloader
from ..data.cityscapes_c_dataset import CityscapesCCorruptedDataset
from ..data.acdc_dataset import ACDCDataset
from ..metrics.segmentation import SegmentationMetrics
from ..models.combined import CombinedModel
from ..utils.categories import ADE20K_CORRUPTIONS
from ..utils.logging import WandBLogger

logger = logging.getLogger(__name__)


class TTARunner:
    """Test-Time Adaptation Runner.

    Orchestrates the TTA process across multiple corruption types,
    handling model updates, metrics computation, and logging.
    """

    def __init__(self, config: DictConfig):
        """Initialize TTA runner.

        Args:
            config: Hydra configuration
        """
        self.config = config

        # Extract key configurations
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.precision = config.get("device", {}).get("precision", "bf16")
        self.continual = config.get("tta", {}).get("continual", False)
        self.batch_size = config.get("tta", {}).get("batch_size", 1)
        self.forward_mode = config.get("tta", {}).get("forward_mode", "tta")
        
        # Gradient accumulation settings
        self.accumulation_steps = config.get("tta", {}).get("accumulation_steps", 1)

        # Task settings (corruptions for -C datasets, conditions for ACDC)
        self.dataset_name = config.get("data", {}).get("dataset", "ADE20K-C")
        self.tasks = list(config.get("data", {}).get(
            "corruptions", list(ADE20K_CORRUPTIONS)
        ))
        self.severity = config.get("data", {}).get("severity", None)

        # Data settings
        self.data_root = config.get("data", {}).get("root", "./data")
        self.num_workers = config.get("tta", {}).get("dataloader", {}).get(
            "num_workers", 4
        )
        self.shuffle = config.get("tta", {}).get("dataloader", {}).get("shuffle", True)

        # Initialize components
        self.model: Optional[CombinedModel] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.metrics: Optional[SegmentationMetrics] = None
        self.wandb_logger: Optional[WandBLogger] = None

        # Experiment tracking
        self.experiment_name = config.get("experiment", {}).get("name", "sd3_tta")
        self.experiment_id = config.get("experiment", {}).get("id", 1)

        # Global step counter
        self.global_step = 0
        
        # Accumulation step counter (resets after each optimizer step)
        self.accumulation_count = 0

        logger.info(f"TTA Runner initialized: mode={self.forward_mode}, "
                    f"continual={self.continual}, precision={self.precision}, "
                    f"accumulation_steps={self.accumulation_steps}")

    def setup(self) -> None:
        """Setup all components."""
        logger.info("Setting up TTA Runner...")

        # Initialize model
        self.model = CombinedModel(self.config)
        self.model.setup(self.device)
        self.model.eval()  # TTA uses eval mode

        # Configure gradients
        if self.forward_mode == "tta":
            self.model.config_tta_grad()
            self._setup_optimizer()

        # Initialize metrics
        num_classes = self.config.get("data", {}).get("num_classes", 150)
        self.metrics = SegmentationMetrics(
            num_classes=num_classes,
            ignore_index=255,
            device=self.device,
        )

        # Initialize W&B logger
        wandb_enabled = self.config.get("logging", {}).get("wandb", {}).get(
            "enabled", True
        )
        self.wandb_logger = WandBLogger(self.config, enabled=wandb_enabled)

        logger.info("TTA Runner setup complete")

    def _setup_optimizer(self) -> None:
        """Setup optimizer for TTA.

        Supports separate learning rates for discriminative and generative models.
        Config keys (under tta.optimizer):
            lr: Global default learning rate (fallback)
            lr_discriminative: Learning rate for discriminative model (optional)
            lr_generative: Learning rate for generative model (optional)
        If lr_discriminative / lr_generative are not set, they fall back to lr.
        """
        opt_config = self.config.get("tta", {}).get("optimizer", {})
        opt_type = opt_config.get("type", "AdamW")
        lr_default = opt_config.get("lr", 6e-5)
        lr_disc = opt_config.get("lr_discriminative", lr_default)
        lr_gen = opt_config.get("lr_generative", lr_default)
        weight_decay = opt_config.get("weight_decay", 0.0)
        betas = tuple(opt_config.get("betas", [0.9, 0.999]))

        # Get trainable parameters grouped by model component
        param_groups_dict = self.model.get_trainable_params()

        # Build optimizer parameter groups
        optimizer_param_groups = []
        total_params = 0

        disc_params = param_groups_dict.get("discriminative", [])
        if disc_params:
            optimizer_param_groups.append({
                "params": disc_params,
                "lr": lr_disc,
                "name": "discriminative",
            })
            total_params += len(disc_params)
            logger.info(f"  Discriminative: {len(disc_params)} params, lr={lr_disc}")

        gen_params = param_groups_dict.get("generative", [])
        if gen_params:
            optimizer_param_groups.append({
                "params": gen_params,
                "lr": lr_gen,
                "name": "generative",
            })
            total_params += len(gen_params)
            logger.info(f"  Generative: {len(gen_params)} params, lr={lr_gen}")

        if total_params == 0:
            logger.warning("No trainable parameters found!")
            self.optimizer = None
            return

        # Create optimizer with parameter groups
        if opt_type.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                optimizer_param_groups,
                betas=betas,
                weight_decay=weight_decay,
            )
        elif opt_type.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                optimizer_param_groups,
                betas=betas,
            )
        elif opt_type.lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                optimizer_param_groups,
                momentum=opt_config.get("momentum", 0.9),
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")

        logger.info(f"Optimizer: {opt_type}, total_params={total_params}")

    def run(self) -> Dict[str, float]:
        """Run TTA on all corruptions.

        Returns:
            Dictionary of average metrics across all corruptions
        """
        all_results = {}

        for task_name in self.tasks:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing task: {task_name}")
            logger.info(f"{'='*50}")

            # Reset model if not continual
            if not self.continual:
                self.model.reset()
                if self.optimizer is not None:
                    self._setup_optimizer()  # Rebuild optimizer with fresh params

            # Run TTA on this task
            result = self.run_task(task_name)
            all_results[task_name] = result

            # Log to W&B
            if self.wandb_logger is not None:
                self.wandb_logger.log_task_metrics(
                    result,
                    task_name=task_name,
                    step=self.global_step,
                )

        # Compute and log summary
        summary = self._compute_summary(all_results)

        # Prepare mIoU data
        miou_values = [all_results[t]["mIoU"] for t in self.tasks]
        avg_miou = sum(miou_values) / len(miou_values) if miou_values else 0.0
        all_names = list(self.tasks) + ["Avg"]
        all_values = miou_values + [avg_miou]

        # Display with rich table (max 8 columns per row)
        self._print_results_table(all_names, all_values)

        # Plain one-line for copying (space-separated)
        copy_line = " ".join(f"{v:.4f}" for v in all_values)
        print(copy_line)

        if self.wandb_logger is not None:
            self.wandb_logger.log_summary(summary)
            self.wandb_logger.finish()

        return summary

    def run_task(self, task_name: str) -> Dict[str, float]:
        """Run TTA on a single task (corruption or condition).

        Args:
            task_name: Corruption type or condition name

        Returns:
            Dictionary of metrics for this task
        """
        # Create dataset based on config
        dataset = self._create_dataset(task_name)

        dataloader = create_dataloader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

        # Reset metrics
        self.metrics.reset()
        
        # Reset gradient accumulation counter for this task
        self.accumulation_count = 0
        if self.optimizer is not None:
            self.optimizer.zero_grad()

        # Process samples
        pbar = tqdm(dataloader, desc=f"TTA-{task_name}")
        for batch in pbar:
            metrics = self._tta_step(batch)
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({"mIoU": f"{metrics['mIoU']:.4f}"})

            # Log to W&B periodically
            log_interval = self.config.get("logging", {}).get("wandb", {}).get(
                "log_interval", 10
            )
            if self.global_step % log_interval == 0:
                if self.wandb_logger is not None:
                    self.wandb_logger.log(
                        metrics,
                        step=self.global_step,
                        prefix=f"{task_name}",
                    )

        # Compute final metrics
        final_metrics = self.metrics.compute()
        logger.info(f"Task {task_name} complete: mIoU={final_metrics['mIoU']:.4f}")

        return final_metrics

    def _tta_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Execute a single TTA step.

        Args:
            batch: Batch dictionary from dataloader

        Returns:
            Dictionary of current metrics
        """
        images = batch["image"].to(self.device)
        masks = batch["mask"].to(self.device)

        # Determine dtype for autocast
        if self.precision == "bf16":
            dtype = torch.bfloat16
        elif self.precision == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        # Forward pass with autocast
        with torch.cuda.amp.autocast(dtype=dtype):
            logits, loss = self.model(images)

        # Backward pass if in TTA mode with loss
        if loss is not None and self.optimizer is not None:
            # Scale loss by accumulation steps for gradient averaging
            scaled_loss = loss / self.accumulation_steps
            scaled_loss.backward()
            
            # Increment accumulation counter
            self.accumulation_count += 1
            
            # Only step optimizer when we've accumulated enough gradients
            if self.accumulation_count >= self.accumulation_steps:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.accumulation_count = 0

        # Resize logits to mask size for evaluation
        logits_resized = F.interpolate(
            logits.float(),
            size=(masks.shape[1], masks.shape[2]),
            mode="bilinear",
            align_corners=False,
        )

        # Update metrics
        self.metrics.update(logits_resized.detach(), masks)

        # Return current metrics
        return self.metrics.compute()

    def _create_dataset(self, task_name: str):
        """Create the appropriate dataset based on config.

        Args:
            task_name: Corruption type (ADE20K-C, Cityscapes-C) or
                       condition name (ACDC)

        Returns:
            Dataset instance
        """
        if self.dataset_name == "ADE20K-C":
            return ADE20KCorruptedDataset(
                data_root=self.data_root,
                corruption=task_name,
                severity=self.severity,
            )
        elif self.dataset_name == "Cityscapes-C":
            return CityscapesCCorruptedDataset(
                data_root=self.data_root,
                corruption=task_name,
                severity=self.severity,
            )
        elif self.dataset_name == "ACDC":
            return ACDCDataset(
                data_root=self.data_root,
                condition=task_name,
            )
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def _print_results_table(
        self,
        names: List[str],
        values: List[float],
        max_cols: int = 8,
    ) -> None:
        """Print mIoU results as a rich table, splitting rows if needed.

        Args:
            names: List of task names (including 'Avg')
            values: List of mIoU values (including average)
            max_cols: Maximum number of columns per table row
        """
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()

            # Split into chunks of max_cols
            for chunk_start in range(0, len(names), max_cols):
                chunk_end = min(chunk_start + max_cols, len(names))
                chunk_names = names[chunk_start:chunk_end]
                chunk_values = values[chunk_start:chunk_end]

                table = Table(
                    title="mIoU Results" if chunk_start == 0 else None,
                    show_header=True,
                    header_style="bold magenta",
                )

                # Add columns
                for name in chunk_names:
                    if name == "Avg":
                        table.add_column(name, justify="center", style="bold cyan")
                    else:
                        table.add_column(name, justify="center")

                # Add value row
                row_cells = []
                for i, val in enumerate(chunk_values):
                    cell = f"{val:.4f}"
                    if chunk_names[i] == "Avg":
                        cell = f"[bold cyan]{cell}[/bold cyan]"
                    row_cells.append(cell)
                table.add_row(*row_cells)

                console.print(table)

        except ImportError:
            # Fallback: plain text table
            for chunk_start in range(0, len(names), max_cols):
                chunk_end = min(chunk_start + max_cols, len(names))
                chunk_names = names[chunk_start:chunk_end]
                chunk_values = values[chunk_start:chunk_end]
                header = " | ".join(f"{n:>14s}" for n in chunk_names)
                vals = " | ".join(f"{v:>14.4f}" for v in chunk_values)
                print(header)
                print(vals)


    def _compute_summary(
        self,
        all_results: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """Compute summary statistics across all tasks.

        Args:
            all_results: Dictionary mapping corruption to metrics

        Returns:
            Summary statistics
        """
        summary = {}

        # Average each metric across corruptions
        metric_names = list(next(iter(all_results.values())).keys())
        for metric in metric_names:
            values = [results[metric] for results in all_results.values()]
            summary[f"mean_{metric}"] = sum(values) / len(values)

        # Add per-corruption mIoU
        for corruption, results in all_results.items():
            summary[f"mIoU_{corruption}"] = results["mIoU"]

        return summary


def run_tta(config: DictConfig) -> Dict[str, float]:
    """Main entry point for running TTA.

    Args:
        config: Hydra configuration

    Returns:
        Summary metrics
    """
    runner = TTARunner(config)
    runner.setup()
    return runner.run()
