"""
DUSA TTA Lightning Module.

Implements Test-Time Adaptation using diffusion-guided loss.
Uses manual optimization for gradient control with sliding window processing.
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy
from tqdm import tqdm

from src.models.combined import CombinedModel


class DUSATTAModule(pl.LightningModule):
    """
    Lightning Module for DUSA Test-Time Adaptation.
    
    Uses manual optimization to handle:
    - Gradient accumulation across sliding windows
    - Separate gradient paths for discriminative and generative models
    
    Models remain in eval mode throughout TTA.
    """
    
    def __init__(
        self,
        model: CombinedModel,
        learning_rate: float = 1e-4,
        optimizer_type: str = "adamw",
        optimizer_betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        use_amp: bool = True,
    ):
        """
        Args:
            model: Combined DUSA model (discriminative + generative)
            learning_rate: Learning rate for optimizer
            optimizer_type: Type of optimizer ("adamw", "sgd")
            optimizer_betas: Beta parameters for Adam-style optimizers
            weight_decay: Weight decay for optimizer
            use_amp: Whether to use automatic mixed precision
        """
        super().__init__()
        
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.optimizer_betas = optimizer_betas
        self.weight_decay = weight_decay
        self.use_amp = use_amp
        
        # Manual optimization for gradient control
        self.automatic_optimization = False
        
        # Metrics for evaluation
        self._setup_metrics()
        
        # Running statistics
        self.running_loss = 0.0
        self.num_samples = 0
        
    def _setup_metrics(self) -> None:
        """Setup torchmetrics for evaluation."""
        num_classes = self.model.num_classes
        
        # Main metrics
        self.metrics = MetricCollection({
            "miou": MulticlassJaccardIndex(
                num_classes=num_classes,
                average="macro",
                ignore_index=255,  # Ignore void class
            ),
            "accuracy": MulticlassAccuracy(
                num_classes=num_classes,
                average="micro",
                ignore_index=255,
            ),
        })
        
        # Per-class IoU for detailed logging
        self.per_class_iou = MulticlassJaccardIndex(
            num_classes=num_classes,
            average=None,
            ignore_index=255,
        )
    
    def configure_optimizers(self):
        """Configure optimizer for TTA."""
        params = self.model.get_trainable_params()
        
        if len(params) == 0:
            # No trainable parameters - return dummy optimizer
            return None
        
        if self.optimizer_type.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                params,
                lr=self.learning_rate,
                betas=self.optimizer_betas,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD(
                params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")
        
        return optimizer
    
    def on_test_start(self) -> None:
        """Called at the beginning of testing."""
        # Ensure model is in eval mode
        self.model.eval()
        
        # Configure TTA parameters
        self.model.configure_tta_params()
        
        # Reset metrics
        self.metrics.reset()
        self.per_class_iou.reset()
        
        # Reset running statistics
        self.running_loss = 0.0
        self.num_samples = 0
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """
        Single TTA step.
        
        1. Forward through discriminative model (get logits)
        2. Compute DUSA loss using generative model
        3. Backward and update parameters
        4. Compute predictions and update metrics
        """
        optimizer = self.optimizers()
        
        images = batch["image"]  # (B, 3, H, W) in [0, 255]
        labels = batch["label"]  # (B, H, W)
        original_sizes = batch["original_size"]  # List of (H, W)
        
        # Get device
        device = images.device
        
        # Zero gradients
        if optimizer is not None:
            optimizer.zero_grad()
        
        # 1. Forward through discriminative model
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            logits, preproc_size = self.model(images)
        
        # 2. Compute DUSA loss and backward
        # Returns (final_loss_tensor, accumulated_loss_value)
        final_loss, accumulated_loss_value = self.model.compute_tta_loss(images, logits)
        
        # Backward for the last window's loss
        if final_loss is not None:
            if self.use_amp:
                self.manual_backward(final_loss)
            else:
                final_loss.backward()
        
        loss_value = accumulated_loss_value
        
        # 3. Optimizer step
        if optimizer is not None:
            optimizer.step()
        
        # 4. Get predictions for metrics (no grad needed)
        with torch.no_grad():
            # Upsample logits to label size for evaluation
            label_size = labels.shape[-2:]
            upsampled_logits = F.interpolate(
                logits.float(),
                size=label_size,
                mode="bilinear",
                align_corners=False,
            )
            preds = upsampled_logits.argmax(dim=1)  # (B, H, W)
            
            # Update metrics
            self.metrics.update(preds, labels)
            self.per_class_iou.update(preds, labels)
        
        # Update running statistics
        self.running_loss += loss_value
        self.num_samples += 1
        
        # Log loss
        self.log("loss", loss_value, prog_bar=True, logger=True)
        
        return {"loss": loss_value}
    
    def on_test_epoch_end(self) -> None:
        """Called at the end of test epoch."""
        # Compute final metrics
        metrics = self.metrics.compute()
        per_class_iou = self.per_class_iou.compute()
        
        # Log main metrics
        avg_loss = self.running_loss / max(self.num_samples, 1)
        self.log("avg_loss", avg_loss, logger=True)
        self.log("miou", metrics["miou"], prog_bar=True, logger=True)
        self.log("accuracy", metrics["accuracy"], logger=True)
        
        # Log per-class IoU
        from src.utils.categories import ADE20K_CATEGORIES
        for i, iou in enumerate(per_class_iou):
            if i < len(ADE20K_CATEGORIES):
                self.log(f"iou/{ADE20K_CATEGORIES[i]}", iou, logger=True)
    
    def reset_for_new_task(self) -> None:
        """
        Reset model and metrics for a new corruption task.
        Call this between different corruption types.
        """
        # Reset model to initial state
        self.model.reset_model()
        
        # Reset metrics
        self.metrics.reset()
        self.per_class_iou.reset()
        
        # Reset running statistics
        self.running_loss = 0.0
        self.num_samples = 0


class TTARunner:
    """
    High-level runner for TTA across multiple corruption tasks.
    
    Handles:
    - Iterating over corruption types
    - Creating trainer per task
    - Logging aggregate results
    - Progress tracking
    
    For each task:
    1. Reset model to initial state
    2. Create new trainer
    3. Run TTA on the corruption dataset
    4. Record metrics
    """
    
    def __init__(
        self,
        module: DUSATTAModule,
        datamodule: "TTADataModule",
        trainer_config: Dict[str, Any],
        wandb_logger: Optional[Any] = None,
        show_inner_progress: bool = True,
    ):
        """
        Args:
            module: DUSA TTA module
            datamodule: Data module with corruption tasks
            trainer_config: Configuration for Lightning Trainer
            wandb_logger: Optional W&B logger
            show_inner_progress: Whether to show progress bar for each task
        """
        self.module = module
        self.datamodule = datamodule
        self.trainer_config = trainer_config
        self.wandb_logger = wandb_logger
        self.show_inner_progress = show_inner_progress
        
        # Results storage
        self.results: Dict[str, Dict[str, float]] = {}
    
    def run(self) -> Dict[str, Any]:
        """
        Run TTA across all corruption tasks.
        
        Returns:
            Dictionary of results per corruption type and aggregate metrics
        """
        corruption_types = self.datamodule.get_all_corruption_types()
        
        all_mious = []
        
        # Progress bar for corruption types
        pbar = tqdm(corruption_types, desc="TTA Progress", unit="task", position=0)
        
        for corruption_type in pbar:
            pbar.set_description(f"TTA: {corruption_type}")
            
            # Reset model for new task
            self.module.model.reset_model()
            
            # Reset module state
            self.module.metrics.reset()
            self.module.per_class_iou.reset()
            self.module.running_loss = 0.0
            self.module.num_samples = 0
            
            # Get dataloader for this corruption
            dataloader = self.datamodule.get_test_dataloader(corruption_type)
            
            # Create fresh trainer for this task
            trainer = self._create_trainer(corruption_type)
            
            # Run TTA with inner progress bar
            trainer.test(self.module, dataloaders=dataloader, verbose=False)
            
            # Collect results
            miou = self.module.metrics.compute()["miou"].item()
            accuracy = self.module.metrics.compute()["accuracy"].item()
            avg_loss = self.module.running_loss / max(self.module.num_samples, 1)
            
            self.results[corruption_type] = {
                "miou": miou,
                "accuracy": accuracy,
                "avg_loss": avg_loss,
            }
            
            all_mious.append(miou)
            
            # Update progress bar
            pbar.set_postfix({
                "mIoU": f"{miou:.4f}", 
                "avg_mIoU": f"{sum(all_mious)/len(all_mious):.4f}"
            })
            
            # Log to W&B if available
            if self.wandb_logger is not None:
                self.wandb_logger.experiment.log({
                    f"{corruption_type}/miou": miou,
                    f"{corruption_type}/accuracy": accuracy,
                    f"{corruption_type}/avg_loss": avg_loss,
                })
        
        # Compute aggregate metrics
        avg_miou = sum(all_mious) / len(all_mious)
        self.results["aggregate"] = {
            "mean_miou": avg_miou,
            "mious": all_mious,
        }
        
        # Log aggregate to W&B
        if self.wandb_logger is not None:
            self.wandb_logger.experiment.log({
                "aggregate/mean_miou": avg_miou,
            })
        
        print(f"\n{'='*50}")
        print(f"TTA Complete - Mean mIoU: {avg_miou:.4f}")
        print(f"{'='*50}")
        
        return self.results
    
    def _create_trainer(self, task_name: str) -> pl.Trainer:
        """Create a trainer for a specific task."""
        from src.callbacks.tta_callbacks import TTAInnerProgressCallback
        
        config = self.trainer_config.copy()
        
        # Add logger if provided
        if self.wandb_logger is not None:
            config["logger"] = self.wandb_logger
        
        # Disable default progress bar (we use our own)
        config["enable_progress_bar"] = False
        
        # Add callbacks
        callbacks = config.get("callbacks", []) or []
        if self.show_inner_progress:
            callbacks.append(TTAInnerProgressCallback(task_name=task_name))
        config["callbacks"] = callbacks
        
        return pl.Trainer(**config)
