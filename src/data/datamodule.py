"""
Lightning DataModule for TTA.
Manages data loading for multiple corruption tasks.
"""

from typing import Dict, List, Optional, Any
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from .ade20k_c import ADE20KCorruptedDataset, CORRUPTION_TYPES, DEFAULT_SEVERITY
from .transforms import TTATransform


class TTADataModule(pl.LightningDataModule):
    """
    Lightning DataModule for TTA evaluation.
    
    Manages multiple corruption tasks and provides dataloaders for each.
    """
    
    def __init__(
        self,
        data_root: str,
        corruption_types: Optional[List[str]] = None,
        severity: int = DEFAULT_SEVERITY,
        target_short_side: int = 512,
        batch_size: int = 1,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        """
        Args:
            data_root: Root directory containing dataset
            corruption_types: List of corruption types to evaluate
            severity: Corruption severity level (1-5)
            target_short_side: Target short side for image resizing
            batch_size: Batch size (default 1 for TTA)
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        super().__init__()
        
        self.data_root = data_root
        self.corruption_types = corruption_types or CORRUPTION_TYPES
        self.severity = severity
        self.target_short_side = target_short_side
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Transform
        self.transform = TTATransform(target_short_side=target_short_side)
        
        # Current task
        self._current_corruption = None
        self._current_dataset = None
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup is handled per-task in get_test_dataloader."""
        pass
    
    def get_test_dataloader(self, corruption_type: str) -> DataLoader:
        """
        Get DataLoader for a specific corruption task.
        
        Args:
            corruption_type: Type of corruption
            
        Returns:
            DataLoader for the specified corruption
        """
        dataset = ADE20KCorruptedDataset(
            data_root=self.data_root,
            corruption_type=corruption_type,
            severity=self.severity,
            transform=self.transform,
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
    
    def get_all_corruption_types(self) -> List[str]:
        """Get list of all corruption types to evaluate."""
        return self.corruption_types.copy()
    
    def test_dataloader(self) -> DataLoader:
        """
        Required by Lightning but not used directly.
        Use get_test_dataloader(corruption_type) instead.
        """
        if self._current_corruption is None:
            self._current_corruption = self.corruption_types[0]
        return self.get_test_dataloader(self._current_corruption)
    
    def set_current_corruption(self, corruption_type: str) -> None:
        """Set the current corruption type for test_dataloader()."""
        self._current_corruption = corruption_type


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for TTA.
    
    Args:
        batch: List of sample dicts
        
    Returns:
        Batched dict
    """
    import torch
    
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "original_size": [item["original_size"] for item in batch],
        "img_path": [item["img_path"] for item in batch],
    }
