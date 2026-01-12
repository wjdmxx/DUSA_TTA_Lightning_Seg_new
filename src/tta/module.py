"""TTA Lightning Module for test-time adaptation."""

from typing import Dict, Any, Optional, List
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassJaccardIndex
from omegaconf import DictConfig
import wandb

from ..models.combined import CombinedModel


class TTAModule(pl.LightningModule):
    """Lightning Module for Test-Time Adaptation.

    This module:
    - Uses training_step but sets models to eval mode (no dropout, etc.)
    - Computes TTA loss and backpropagates
    - Tracks and logs mIoU metrics
    - Logs to W&B
    """

    def __init__(
        self,
        model_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        tta_cfg: DictConfig,
        logging_cfg: DictConfig,
        num_classes: int = 150,
        ignore_index: int = 255,
        task_name: str = "task_0",
        task_idx: int = 0
    ):
        """Initialize TTA Module.

        Args:
            model_cfg: Model configuration
            optimizer_cfg: Optimizer configuration
            tta_cfg: TTA configuration
            logging_cfg: Logging configuration
            num_classes: Number of segmentation classes
            ignore_index: Index to ignore in evaluation
            task_name: Name of current task for logging
            task_idx: Index of current task for logging
        """
        super().__init__()
        self.save_hyperparameters()

        self.model_cfg = model_cfg
        self.optimizer_cfg = optimizer_cfg
        self.tta_cfg = tta_cfg
        self.logging_cfg = logging_cfg
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.task_name = task_name
        self.task_idx = task_idx

        # Build combined model
        self.model = CombinedModel(
            discriminative_cfg=model_cfg.discriminative,
            generative_cfg=model_cfg.generative,
            loss_cfg=model_cfg.loss,
            update_cfg=model_cfg.update,
            forward_mode=tta_cfg.forward_mode
        )

        # Store initial state for reset
        self.initial_state = None

        # Metrics
        self.train_miou = MulticlassJaccardIndex(
            num_classes=num_classes,
            ignore_index=ignore_index,
            average='macro'
        )
        self.train_loss_avg = MeanMetric()

        # Per-class IoU for detailed logging
        self.train_iou_per_class = MulticlassJaccardIndex(
            num_classes=num_classes,
            ignore_index=ignore_index,
            average=None
        )

        # Automatic optimization is enabled by default
        self.automatic_optimization = True

    def setup(self, stage: str):
        """Setup called at the beginning of fit/test."""
        if stage == "fit":
            # Configure gradients for TTA
            self.model.configure_tta_grad()

            # Store initial state for reset
            self.initial_state = self.model.get_initial_state()

            # Set models to eval mode (but gradients still flow)
            self.model.set_eval_mode()

    def on_train_start(self):
        """Called at the start of training."""
        # Ensure models are in eval mode
        self.model.set_eval_mode()

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Optional[torch.Tensor]:
        """Training step for TTA.

        Note: We use training_step but models are in eval mode.
        This allows gradient computation while avoiding dropout, etc.

        Args:
            batch: Dictionary containing:
                - "image": Image tensor of shape (B, C, H, W)
                - "label": Ground truth labels of shape (B, H, W)
            batch_idx: Batch index

        Returns:
            Loss tensor for backpropagation (or None if discriminative_only mode)
        """
        images = batch["image"]
        labels = batch["label"]

        # Ensure models stay in eval mode
        self.model.set_eval_mode()

        # Forward pass
        result = self.model(images, return_predictions=True)
        loss = result["loss"]
        predictions = result["predictions"]

        # Update metrics
        # Move predictions and labels to same device
        predictions = predictions.to(labels.device)
        self.train_miou.update(predictions, labels)
        self.train_iou_per_class.update(predictions, labels)

        if loss is not None:
            self.train_loss_avg.update(loss)

            # Log loss
            self.log(
                f"{self.task_name}/loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True
            )

        # Log mIoU
        current_miou = self.train_miou.compute()
        self.log(
            f"{self.task_name}/mIoU",
            current_miou * 100,  # Convert to percentage
            on_step=True,
            on_epoch=True,
            prog_bar=True
        )

        return loss

    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        # Compute final metrics
        final_miou = self.train_miou.compute()
        iou_per_class = self.train_iou_per_class.compute()

        # Log to W&B
        if wandb.run is not None:
            # Log summary metrics
            wandb.log({
                f"{self.task_name}/final_mIoU": final_miou.item() * 100,
                f"task_{self.task_idx}/mIoU": final_miou.item() * 100,
            })

            # Log per-class IoU
            for i, iou in enumerate(iou_per_class):
                if not torch.isnan(iou):
                    wandb.log({
                        f"{self.task_name}/class_{i}_IoU": iou.item() * 100
                    })

        # Log average loss if available
        if self.train_loss_avg.compute() > 0:
            avg_loss = self.train_loss_avg.compute()
            self.log(f"{self.task_name}/avg_loss", avg_loss)

        # Reset metrics for next epoch
        self.train_miou.reset()
        self.train_iou_per_class.reset()
        self.train_loss_avg.reset()

    def configure_optimizers(self):
        """Configure optimizer for TTA."""
        params = self.model.get_trainable_params()

        if len(params) == 0:
            # No trainable params (discriminative_only mode with no updates)
            return None

        optimizer_type = self.optimizer_cfg.type.lower()

        if optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                params,
                lr=self.optimizer_cfg.lr,
                weight_decay=self.optimizer_cfg.weight_decay
            )
        elif optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                params,
                lr=self.optimizer_cfg.lr,
                weight_decay=self.optimizer_cfg.get("weight_decay", 0)
            )
        elif optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                params,
                lr=self.optimizer_cfg.lr,
                momentum=self.optimizer_cfg.get("momentum", 0.9),
                weight_decay=self.optimizer_cfg.get("weight_decay", 0)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        return optimizer

    def reset_to_initial(self):
        """Reset model to initial state (before TTA)."""
        if self.initial_state is not None:
            self.model.reset_to_initial_state(self.initial_state)
            # Re-configure gradients
            self.model.configure_tta_grad()
            # Set back to eval mode
            self.model.set_eval_mode()

    def get_predictions(self, images: torch.Tensor) -> torch.Tensor:
        """Get predictions without computing loss.

        Args:
            images: Input images

        Returns:
            Predictions tensor
        """
        return self.model.get_predictions(images)


def create_tta_module(
    model_cfg: DictConfig,
    tta_cfg: DictConfig,
    logging_cfg: DictConfig,
    data_cfg: DictConfig,
    task_name: str = "task_0",
    task_idx: int = 0
) -> TTAModule:
    """Factory function to create TTA module.

    Args:
        model_cfg: Model configuration
        tta_cfg: TTA configuration
        logging_cfg: Logging configuration
        data_cfg: Data configuration (for num_classes)
        task_name: Name of current task
        task_idx: Index of current task

    Returns:
        TTAModule instance
    """
    return TTAModule(
        model_cfg=model_cfg,
        optimizer_cfg=model_cfg.optimizer,
        tta_cfg=tta_cfg,
        logging_cfg=logging_cfg,
        num_classes=data_cfg.num_classes,
        ignore_index=data_cfg.ignore_index,
        task_name=task_name,
        task_idx=task_idx
    )
