"""PyTorch Lightning DataModule for TTA."""

from typing import Optional, List, Dict, Any
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig

from .ade20k import ADE20KCorruptedDataset, ADE20KDataset


class TTADataModule(pl.LightningDataModule):
    """DataModule for TTA that creates separate dataloaders for each corruption task.

    This module manages loading data for multiple corruption types and severities,
    creating separate tasks for TTA evaluation.
    """

    def __init__(self, cfg: DictConfig):
        """Initialize DataModule.

        Args:
            cfg: Data configuration containing:
                - data_root: Root directory of dataset
                - corruptions: List of corruption types
                - severity_levels: List of severity levels
                - batch_size: Batch size (should be 1)
                - num_workers: Number of data loading workers
                - shuffle: Whether to shuffle data
        """
        super().__init__()
        self.cfg = cfg
        self.data_root = cfg.data_root
        self.split = cfg.get("split", "validation")
        self.corruptions = list(cfg.corruptions)
        self.severity_levels = list(cfg.severity_levels)
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.shuffle = cfg.shuffle
        self.pin_memory = cfg.get("pin_memory", True)
        self.target_short_edge = cfg.preprocessing.target_short_edge
        self.reduce_zero_label = cfg.get("reduce_zero_label", True)

        # Build task list
        self.tasks = self._build_task_list()

    def _build_task_list(self) -> List[Dict[str, Any]]:
        """Build list of tasks (corruption + severity combinations)."""
        tasks = []
        for corruption in self.corruptions:
            for severity in self.severity_levels:
                tasks.append({
                    "corruption": corruption,
                    "severity": severity,
                    "name": f"{corruption}_s{severity}"
                })
        return tasks

    def get_task_dataloader(self, task_idx: int) -> DataLoader:
        """Get dataloader for a specific task.

        Args:
            task_idx: Index of the task

        Returns:
            DataLoader for the specified task
        """
        task = self.tasks[task_idx]
        dataset = ADE20KCorruptedDataset(
            data_root=self.data_root,
            corruption=task["corruption"],
            severity=task["severity"],
            split=self.split,
            target_short_edge=self.target_short_edge,
            reduce_zero_label=self.reduce_zero_label,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def get_task_name(self, task_idx: int) -> str:
        """Get name of a specific task."""
        return self.tasks[task_idx]["name"]

    def get_num_tasks(self) -> int:
        """Get total number of tasks."""
        return len(self.tasks)

    def train_dataloader(self) -> DataLoader:
        """Return train dataloader (first task by default).

        Note: For TTA, we typically iterate through tasks manually,
        but this provides a default for Lightning Trainer.
        """
        return self.get_task_dataloader(0)


class SingleTaskDataModule(pl.LightningDataModule):
    """DataModule for a single TTA task.

    This is used internally when running TTA on a specific task.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        num_workers: int = 4,
        shuffle: bool = True,
        pin_memory: bool = True
    ):
        """Initialize single task DataModule.

        Args:
            dataset: Dataset instance
            batch_size: Batch size
            num_workers: Number of workers
            shuffle: Whether to shuffle
            pin_memory: Whether to pin memory
        """
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )


def create_task_datamodules(cfg: DictConfig) -> List[SingleTaskDataModule]:
    """Create list of DataModules, one per task.

    Args:
        cfg: Data configuration

    Returns:
        List of SingleTaskDataModule instances
    """
    data_root = cfg.data_root
    split = cfg.get("split", "validation")
    corruptions = list(cfg.corruptions)
    severity_levels = list(cfg.severity_levels)
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    shuffle = cfg.shuffle
    pin_memory = cfg.get("pin_memory", True)
    target_short_edge = cfg.preprocessing.target_short_edge
    reduce_zero_label = cfg.get("reduce_zero_label", True)

    datamodules = []

    for corruption in corruptions:
        for severity in severity_levels:
            dataset = ADE20KCorruptedDataset(
                data_root=data_root,
                corruption=corruption,
                severity=severity,
                split=split,
                target_short_edge=target_short_edge,
                reduce_zero_label=reduce_zero_label,
            )

            dm = SingleTaskDataModule(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
                pin_memory=pin_memory,
            )

            datamodules.append(dm)

    return datamodules


def get_task_names(cfg: DictConfig) -> List[str]:
    """Get list of task names.

    Args:
        cfg: Data configuration

    Returns:
        List of task names
    """
    names = []
    for corruption in cfg.corruptions:
        for severity in cfg.severity_levels:
            names.append(f"{corruption}_s{severity}")
    return names
