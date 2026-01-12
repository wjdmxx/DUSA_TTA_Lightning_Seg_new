"""Lightning Module for Test-Time Adaptation (TTA).

This module implements the TTA training loop using PyTorch Lightning,
with support for FSDP and W&B logging.
"""

from typing import Optional, Dict, Any, List, Tuple
from copy import deepcopy

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics import MeanMetric

from ..models.combined import CombinedModel
from ..utils.metrics import SegmentationMetrics


class TTAModule(pl.LightningModule):
    """Lightning Module for Test-Time Adaptation.
    
    Uses training_step with models in eval mode for TTA updates.
    Supports both full TTA and discriminative-only baseline modes.
    """
    
    def __init__(
        self,
        model: CombinedModel,
        num_classes: int = 150,
        ignore_index: int = 255,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.0,
        optimizer_type: str = "adamw",
        betas: Tuple[float, float] = (0.9, 0.999),
        forward_mode: str = "tta",
        task_id: int = 0,
        task_name: str = "unknown",
    ):
        """Initialize TTA module.
        
        Args:
            model: Combined model for TTA
            num_classes: Number of segmentation classes
            ignore_index: Label value to ignore in metrics
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            optimizer_type: Type of optimizer ('adam', 'adamw')
            betas: Beta coefficients for Adam
            forward_mode: 'tta' for full TTA, 'discriminative_only' for baseline
            task_id: Current task index (for logging)
            task_name: Current task name (for logging)
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        self.model = model
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type
        self.betas = betas
        self.forward_mode = forward_mode
        self.task_id = task_id
        self.task_name = task_name
        
        # Metrics
        self.train_metrics = SegmentationMetrics(
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        
        # Running loss tracker
        self.loss_tracker = MeanMetric()
        
        # Online mIoU for progress bar
        self._running_miou = 0.0
        self._sample_count = 0
    
    def on_train_start(self) -> None:
        """Called at the beginning of training.
        
        Sets all models to eval mode for TTA.
        """
        # IMPORTANT: Set models to eval mode for TTA
        self.model.set_eval_mode()
        
        # Reset metrics
        self.train_metrics.reset()
        self.loss_tracker.reset()
        self._running_miou = 0.0
        self._sample_count = 0
        
        # Log task info
        self.print(f"\n{'='*60}")
        self.print(f"Starting TTA for task [{self.task_id}]: {self.task_name}")
        self.print(f"Forward mode: {self.forward_mode}")
        self.print(f"{'='*60}\n")
    
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> STEP_OUTPUT:
        """Execute one TTA step.
        
        Args:
            batch: Dictionary containing 'image', 'mask', etc.
            batch_idx: Batch index
            
        Returns:
            Loss tensor or None for discriminative-only mode
        """
        images = batch['image']  # (B, 3, H, W) in [0, 255]
        masks = batch['mask']    # (B, H, W) with class labels
        
        # Forward pass
        logits, loss = self.model(images, forward_mode=self.forward_mode)
        
        # Compute predictions for metrics
        # Upsample logits to mask size for mIoU computation
        with torch.no_grad():
            upsampled_logits = F.interpolate(
                logits,
                size=masks.shape[1:],
                mode='bilinear',
                align_corners=False
            )
            predictions = upsampled_logits.argmax(dim=1)
            
            # Update metrics
            self.train_metrics.update(predictions, masks)
            
            # Compute online mIoU
            metrics = self.train_metrics.compute()
            current_miou = metrics['mIoU'].item()
            self._running_miou = current_miou
            self._sample_count += 1
        
        # Log metrics
        if loss is not None:
            self.loss_tracker.update(loss.detach())
            self.log(
                f'task_{self.task_id}/loss',
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
        
        self.log(
            f'task_{self.task_id}/mIoU',
            current_miou,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        
        # Return loss for backward (None if discriminative_only)
        if loss is not None:
            return loss
        else:
            # Return dummy zero loss that doesn't affect model
            return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        # Compute final metrics
        metrics = self.train_metrics.compute()
        final_miou = metrics['mIoU'].item()
        
        # Log epoch-level metrics
        self.log(f'task_{self.task_id}/epoch_mIoU', final_miou)
        
        if self.loss_tracker.compute() is not None:
            avg_loss = self.loss_tracker.compute().item()
            self.log(f'task_{self.task_id}/epoch_loss', avg_loss)
        
        # Print summary
        self.print(f"\n{'='*60}")
        self.print(f"Task [{self.task_id}] {self.task_name} completed")
        self.print(f"Final mIoU: {final_miou:.4f}")
        self.print(f"{'='*60}\n")
    
    def configure_optimizers(self):
        """Configure optimizer for TTA.
        
        Returns:
            Optimizer instance
        """
        # Get trainable parameters
        trainable_params = self.model.get_trainable_params()
        
        if len(trainable_params) == 0:
            # No trainable params, return dummy optimizer
            self.print("Warning: No trainable parameters found!")
            return torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=0)
        
        # Create optimizer
        if self.optimizer_type.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.learning_rate,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(
                trainable_params,
                lr=self.learning_rate,
                betas=self.betas,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")
        
        return optimizer
    
    def get_final_metrics(self) -> Dict[str, float]:
        """Get final metrics after TTA.
        
        Returns:
            Dictionary of metric name -> value
        """
        metrics = self.train_metrics.compute()
        return {
            'mIoU': metrics['mIoU'].item(),
            'accuracy': metrics['accuracy'].item(),
        }
    
    def update_task_info(self, task_id: int, task_name: str) -> None:
        """Update task information for a new task.
        
        Args:
            task_id: New task index
            task_name: New task name
        """
        self.task_id = task_id
        self.task_name = task_name
        
        # Reset metrics for new task
        self.train_metrics.reset()
        self.loss_tracker.reset()
        self._running_miou = 0.0
        self._sample_count = 0


def create_tta_module(
    model: CombinedModel,
    config: dict,
    task_id: int = 0,
    task_name: str = "unknown",
) -> TTAModule:
    """Factory function to create TTA module from config.
    
    Args:
        model: Combined model
        config: Configuration dictionary
        task_id: Task index
        task_name: Task name
        
    Returns:
        Configured TTAModule
    """
    return TTAModule(
        model=model,
        num_classes=config.get('num_classes', 150),
        ignore_index=config.get('ignore_index', 255),
        learning_rate=config.get('learning_rate', 1e-5),
        weight_decay=config.get('weight_decay', 0.0),
        optimizer_type=config.get('optimizer_type', 'adamw'),
        betas=tuple(config.get('betas', [0.9, 0.999])),
        forward_mode=config.get('forward_mode', 'tta'),
        task_id=task_id,
        task_name=task_name,
    )
