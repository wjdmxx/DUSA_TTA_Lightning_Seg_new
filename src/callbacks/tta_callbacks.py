"""
TTA callbacks for Lightning.
"""

from typing import Any, Dict, Optional
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, TQDMProgressBar
from tqdm import tqdm


class TTAProgressCallback(Callback):
    """
    Custom progress callback for TTA.
    Shows minimal but essential information during TTA.
    
    Displays:
    - Current batch / total batches
    - Current loss
    - Running mIoU
    """
    
    def __init__(self, show_per_batch: bool = True):
        super().__init__()
        self.pbar = None
        self.current_miou = 0.0
        self.show_per_batch = show_per_batch
        self.total_batches = 0
        
    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Initialize progress bar at test start."""
        # Get total number of batches
        if trainer.test_dataloaders is not None:
            try:
                self.total_batches = len(trainer.test_dataloaders)
            except:
                self.total_batches = None
        else:
            self.total_batches = None
        
        if self.show_per_batch:
            self.pbar = tqdm(
                total=self.total_batches,
                desc="Processing",
                unit="batch",
                dynamic_ncols=True,
                leave=False,
            )
    
    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Update progress bar after each batch."""
        if self.pbar is not None:
            # Get current metrics
            loss = outputs.get("loss", 0.0) if isinstance(outputs, dict) else 0.0
            
            # Compute running mIoU
            if hasattr(pl_module, "metrics"):
                try:
                    metrics = pl_module.metrics.compute()
                    self.current_miou = metrics["miou"].item()
                except:
                    pass
            
            # Update progress bar
            self.pbar.update(1)
            self.pbar.set_postfix({
                "loss": f"{loss:.4f}",
                "mIoU": f"{self.current_miou:.4f}",
            })
    
    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Close progress bar at test end."""
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


class TTAInnerProgressCallback(Callback):
    """
    Inner progress callback for tracking individual samples within a task.
    Used when TTARunner manages the outer task loop.
    """
    
    def __init__(self, task_name: str = ""):
        super().__init__()
        self.pbar = None
        self.task_name = task_name
        self.current_miou = 0.0
        
    def set_task_name(self, task_name: str) -> None:
        """Update the task name for display."""
        self.task_name = task_name
        
    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Initialize progress bar at test start."""
        try:
            total = len(trainer.test_dataloaders)
        except:
            total = None
            
        self.pbar = tqdm(
            total=total,
            desc=f"  {self.task_name}",
            unit="img",
            dynamic_ncols=True,
            leave=False,
            position=1,
        )
    
    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Update progress bar after each batch."""
        if self.pbar is not None:
            loss = outputs.get("loss", 0.0) if isinstance(outputs, dict) else 0.0
            
            if hasattr(pl_module, "metrics"):
                try:
                    metrics = pl_module.metrics.compute()
                    self.current_miou = metrics["miou"].item()
                except:
                    pass
            
            self.pbar.update(1)
            self.pbar.set_postfix({
                "loss": f"{loss:.4f}",
                "mIoU": f"{self.current_miou:.4f}",
            })
    
    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Close progress bar at test end."""
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


class ModelResetCallback(Callback):
    """
    Callback to reset model between tasks.
    Not typically used with TTARunner, but available for custom workflows.
    """
    
    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Ensure model is reset at the start of each test."""
        if hasattr(pl_module, "model") and hasattr(pl_module.model, "reset_model"):
            try:
                pl_module.model.reset_model()
            except RuntimeError:
                # Initial state not saved yet
                pass
