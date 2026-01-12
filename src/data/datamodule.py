"""Lightning DataModule for ADE20K-C dataset."""

from pathlib import Path
from typing import Optional, List

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import ADE20KCDataset, get_corruption_types


class ADE20KCDataModule(pl.LightningDataModule):
    """Lightning DataModule for ADE20K-C test-time adaptation.
    
    This module manages data loading for TTA experiments, handling
    multiple corruption types and severity levels.
    """
    
    def __init__(
        self,
        data_root: str = "data",
        corruption: Optional[str] = None,
        corruptions: Optional[List[str]] = None,
        severity: int = 5,
        batch_size: int = 1,
        num_workers: int = 2,
        shuffle: bool = True,
        short_edge_size: int = 512,
        pin_memory: bool = True,
    ):
        """Initialize the DataModule.
        
        Args:
            data_root: Root directory containing data
            corruption: Single corruption type to use (mutually exclusive with corruptions)
            corruptions: List of corruption types to use
            severity: Corruption severity level (1-5)
            batch_size: Batch size for dataloaders (should be 1 for TTA)
            num_workers: Number of dataloader workers
            shuffle: Whether to shuffle data
            short_edge_size: Target size for short edge scaling
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.data_root = Path(data_root)
        self.severity = severity
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.short_edge_size = short_edge_size
        self.pin_memory = pin_memory
        
        # Handle corruption(s) argument
        if corruption is not None and corruptions is not None:
            raise ValueError("Specify either 'corruption' or 'corruptions', not both")
        
        if corruption is not None:
            self.corruptions = [corruption]
        elif corruptions is not None:
            self.corruptions = corruptions
        else:
            # Default: all 15 corruption types
            self.corruptions = get_corruption_types()
        
        # Dataset instances (created in setup)
        self._current_corruption: Optional[str] = None
        self._dataset: Optional[ADE20KCDataset] = None
    
    @property
    def current_corruption(self) -> Optional[str]:
        """Get the currently active corruption type."""
        return self._current_corruption
    
    def get_all_corruptions(self) -> List[str]:
        """Get list of all corruption types to evaluate."""
        return self.corruptions
    
    def set_corruption(self, corruption: str) -> None:
        """Set the current corruption type and create dataset.
        
        Args:
            corruption: Corruption type to use
        """
        if corruption not in self.corruptions:
            raise ValueError(f"Unknown corruption: {corruption}. Available: {self.corruptions}")
        
        self._current_corruption = corruption
        self._dataset = ADE20KCDataset(
            data_root=str(self.data_root),
            corruption=corruption,
            severity=self.severity,
            short_edge_size=self.short_edge_size,
        )
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets.
        
        Note: For TTA, we use training dataloader even though we're "testing".
        This is because we want to enable gradient updates.
        
        Args:
            stage: Stage ('fit', 'validate', 'test', 'predict')
        """
        # If no corruption is set, use the first one
        if self._current_corruption is None and len(self.corruptions) > 0:
            self.set_corruption(self.corruptions[0])
    
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader (used for TTA).
        
        Returns:
            DataLoader for TTA training
        """
        if self._dataset is None:
            raise RuntimeError(
                "Dataset not initialized. Call set_corruption() first."
            )
        
        return DataLoader(
            self._dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader (same as train for TTA)."""
        return self.train_dataloader()
    
    def test_dataloader(self) -> DataLoader:
        """Get test dataloader (same as train for TTA)."""
        return self.train_dataloader()
    
    def get_dataset_size(self) -> int:
        """Get current dataset size.
        
        Returns:
            Number of samples in current dataset
        """
        if self._dataset is None:
            return 0
        return len(self._dataset)
    
    def get_task_name(self) -> str:
        """Get current task name for logging.
        
        Returns:
            Task name string like "gaussian_noise_5"
        """
        if self._current_corruption is None:
            return "unknown"
        return f"{self._current_corruption}_{self.severity}"
