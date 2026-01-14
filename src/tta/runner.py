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

        # Corruption settings
        self.corruptions = config.get("data", {}).get(
            "corruptions", list(ADE20K_CORRUPTIONS)
        )
        self.severity = config.get("data", {}).get("severity", 5)

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
        """Setup optimizer for TTA."""
        opt_config = self.config.get("tta", {}).get("optimizer", {})
        opt_type = opt_config.get("type", "AdamW")
        lr = opt_config.get("lr", 6e-5)
        weight_decay = opt_config.get("weight_decay", 0.0)
        betas = tuple(opt_config.get("betas", [0.9, 0.999]))

        # Get trainable parameters
        params = self.model.get_trainable_params()

        if len(params) == 0:
            logger.warning("No trainable parameters found!")
            self.optimizer = None
            return

        # Create optimizer
        if opt_type.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                params,
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
            )
        elif opt_type.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                params,
                lr=lr,
                betas=betas,
            )
        elif opt_type.lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                params,
                lr=lr,
                momentum=opt_config.get("momentum", 0.9),
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")

        logger.info(f"Optimizer: {opt_type}, lr={lr}, params={len(params)}")

    def run(self) -> Dict[str, float]:
        """Run TTA on all corruptions.

        Returns:
            Dictionary of average metrics across all corruptions
        """
        all_results = {}

        for corruption in self.corruptions:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing corruption: {corruption}")
            logger.info(f"{'='*50}")

            # Reset model if not continual
            if not self.continual:
                self.model.reset()
                if self.optimizer is not None:
                    self._setup_optimizer()  # Rebuild optimizer with fresh params

            # Run TTA on this corruption
            result = self.run_task(corruption)
            all_results[corruption] = result

            # Log to W&B
            if self.wandb_logger is not None:
                self.wandb_logger.log_task_metrics(
                    result,
                    task_name=corruption,
                    step=self.global_step,
                )

        # Compute and log summary
        summary = self._compute_summary(all_results)
        logger.info(f"\n{'='*50}")
        logger.info("Final Results Summary:")
        for metric, value in summary.items():
            logger.info(f"  {metric}: {value:.4f}")

        if self.wandb_logger is not None:
            self.wandb_logger.log_summary(summary)
            self.wandb_logger.finish()

        return summary

    def run_task(self, corruption: str) -> Dict[str, float]:
        """Run TTA on a single corruption type.

        Args:
            corruption: Corruption type name

        Returns:
            Dictionary of metrics for this task
        """
        # Create dataset and dataloader
        dataset = ADE20KCorruptedDataset(
            data_root=self.data_root,
            corruption=corruption,
            severity=self.severity,
            short_edge_size=512,
        )

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
        pbar = tqdm(dataloader, desc=f"TTA-{corruption}")
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
                        prefix=f"{corruption}",
                    )

        # Compute final metrics
        final_metrics = self.metrics.compute()
        logger.info(f"Task {corruption} complete: mIoU={final_metrics['mIoU']:.4f}")

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
