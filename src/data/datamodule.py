import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF


class ADE20KCorruptionDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        corruption: str,
        severity: int,
        split: str = "val",
        short_edge: int = 512,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.corruption = corruption
        self.severity = severity
        self.split = split
        self.short_edge = short_edge
        self.image_dir = self.data_root / "ADE20K_val-c" / corruption / str(severity) / split
        self.annotation_dir = self.data_root / "annotations" / split
        self.image_paths = sorted([p for p in self.image_dir.glob("*.*") if p.is_file()])

    def __len__(self):
        return len(self.image_paths)

    def _resize_pair(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        w, h = image.size
        short = min(h, w)
        scale = self.short_edge / short
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        image = image.resize((new_w, new_h), resample=Image.BILINEAR)
        mask = mask.resize((new_w, new_h), resample=Image.NEAREST)
        return image, mask

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.annotation_dir / image_path.name
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        image, mask = self._resize_pair(image, mask)
        image_tensor = TF.to_tensor(image)
        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.int64))
        return {"image": image_tensor, "mask": mask_tensor}


class ADE20KCorruptionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        corruption: str,
        severity: int,
        split: str = "val",
        short_edge: int = 512,
        batch_size: int = 1,
        num_workers: int = 4,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.corruption = corruption
        self.severity = severity
        self.split = split
        self.short_edge = short_edge
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.dataset: Optional[ADE20KCorruptionDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = ADE20KCorruptionDataset(
            data_root=self.data_root,
            corruption=self.corruption,
            severity=self.severity,
            split=self.split,
            short_edge=self.short_edge,
        )

    def train_dataloader(self):
        assert self.dataset is not None
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        assert self.dataset is not None
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
