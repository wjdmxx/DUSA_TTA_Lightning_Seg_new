"""ADE20K and ADE20K-C dataset implementations."""

import os
from typing import Optional, Callable, List, Tuple, Dict
from pathlib import Path

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch.nn.functional as F


class ADE20KCorruptedDataset(Dataset):
    """ADE20K-C (Corrupted) Dataset for TTA evaluation.

    This dataset loads corrupted images from ADE20K-C benchmark.
    Images are resized so that the short edge equals target_short_edge.

    Directory structure expected:
    data_root/
        ADE20K_val-c/
            {corruption}/
                {severity}/
                    validation/
                        ADE_val_00000001.jpg
                        ...
        annotations/
            validation/
                ADE_val_00000001.png
                ...
    """

    def __init__(
        self,
        data_root: str,
        corruption: str,
        severity: int = 5,
        split: str = "validation",
        target_short_edge: int = 512,
        reduce_zero_label: bool = True,
        transform: Optional[Callable] = None,
    ):
        """Initialize ADE20K-C dataset.

        Args:
            data_root: Root directory of the dataset
            corruption: Corruption type (e.g., "gaussian_noise")
            severity: Severity level (1-5)
            split: Data split ("validation")
            target_short_edge: Target size for short edge resize
            reduce_zero_label: Whether to reduce zero label (ADE20K specific)
            transform: Optional transform to apply
        """
        super().__init__()
        self.data_root = Path(data_root)
        self.corruption = corruption
        self.severity = severity
        self.split = split
        self.target_short_edge = target_short_edge
        self.reduce_zero_label = reduce_zero_label
        self.transform = transform

        # Build file paths
        # Image directory: data_root / ADE20K_val-c / corruption / severity / split
        self.img_dir = self.data_root / "ADE20K_val-c" / corruption / str(severity) / split
        # Annotation directory: data_root / annotations / split
        self.ann_dir = self.data_root / "annotations" / split

        # Get all image files
        self.img_files = sorted(list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.png")))

        if len(self.img_files) == 0:
            raise RuntimeError(f"No images found in {self.img_dir}")

        # Verify annotation directory exists
        if not self.ann_dir.exists():
            raise RuntimeError(f"Annotation directory not found: {self.ann_dir}")

        print(f"Loaded {len(self.img_files)} images for {corruption} severity {severity}")

    def __len__(self) -> int:
        return len(self.img_files)

    def _get_annotation_path(self, img_path: Path) -> Path:
        """Get the corresponding annotation path for an image.

        ADE20K annotation files have the same stem as image files but with .png extension.

        Args:
            img_path: Path to the image file

        Returns:
            Path to the corresponding annotation file
        """
        # Get the image filename stem (without extension)
        img_stem = img_path.stem

        # Annotation file is in ann_dir with .png extension
        ann_path = self.ann_dir / f"{img_stem}.png"

        return ann_path

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load image
        img_path = self.img_files[idx]
        image = Image.open(img_path).convert("RGB")

        # Load annotation
        ann_path = self._get_annotation_path(img_path)
        if not ann_path.exists():
            raise FileNotFoundError(f"Annotation not found: {ann_path} for image {img_path}")
        label = Image.open(ann_path)

        # Convert to tensors
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()  # (C, H, W)
        label = torch.from_numpy(np.array(label)).long()  # (H, W)

        # Resize image (short edge to target)
        image, label = self._resize_short_edge(image, label)

        # Handle ADE20K label offset
        if self.reduce_zero_label:
            # ADE20K labels are 1-150, convert to 0-149
            # 0 in annotation means unlabeled, map to ignore_index (255)
            label = label - 1
            label[label == -1] = 255

        # Apply transform if any
        if self.transform is not None:
            image, label = self.transform(image, label)

        return {
            "image": image,
            "label": label,
            "img_path": str(img_path),
        }

    def _resize_short_edge(
        self,
        image: torch.Tensor,
        label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resize image and label so short edge equals target_short_edge.

        Handles both horizontal and vertical images correctly.

        Args:
            image: Image tensor of shape (C, H, W)
            label: Label tensor of shape (H, W)

        Returns:
            Resized image and label tensors
        """
        _, h, w = image.shape

        # Determine scale factor
        if h <= w:
            # Height is shorter
            scale = self.target_short_edge / h
        else:
            # Width is shorter
            scale = self.target_short_edge / w

        new_h = int(h * scale)
        new_w = int(w * scale)

        # Resize image (bilinear interpolation)
        image = image.unsqueeze(0)  # Add batch dim
        image = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
        image = image.squeeze(0)

        # Resize label (nearest neighbor to preserve class labels)
        label = label.unsqueeze(0).unsqueeze(0).float()  # Add batch and channel dims
        label = F.interpolate(label, size=(new_h, new_w), mode='nearest')
        label = label.squeeze(0).squeeze(0).long()

        return image, label


class ADE20KDataset(Dataset):
    """Standard ADE20K validation dataset (without corruption).

    Directory structure expected:
    data_root/
        images/
            validation/
                ADE_val_00000001.jpg
                ...
        annotations/
            validation/
                ADE_val_00000001.png
                ...
    """

    def __init__(
        self,
        data_root: str,
        split: str = "validation",
        target_short_edge: int = 512,
        reduce_zero_label: bool = True,
        transform: Optional[Callable] = None,
    ):
        """Initialize ADE20K dataset.

        Args:
            data_root: Root directory of the dataset
            split: Data split ("validation" or "training")
            target_short_edge: Target size for short edge resize
            reduce_zero_label: Whether to reduce zero label (ADE20K specific)
            transform: Optional transform to apply
        """
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.target_short_edge = target_short_edge
        self.reduce_zero_label = reduce_zero_label
        self.transform = transform

        # Build file lists
        self.img_dir = self.data_root / "images" / split
        self.ann_dir = self.data_root / "annotations" / split

        # Get all image files
        self.img_files = sorted(list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.png")))

        if len(self.img_files) == 0:
            raise RuntimeError(f"No images found in {self.img_dir}")

        print(f"Loaded {len(self.img_files)} images from ADE20K {split} set")

    def __len__(self) -> int:
        return len(self.img_files)

    def _get_annotation_path(self, img_path: Path) -> Path:
        """Get the corresponding annotation path for an image."""
        img_stem = img_path.stem
        ann_path = self.ann_dir / f"{img_stem}.png"
        return ann_path

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load image
        img_path = self.img_files[idx]
        image = Image.open(img_path).convert("RGB")

        # Load annotation
        ann_path = self._get_annotation_path(img_path)
        if not ann_path.exists():
            raise FileNotFoundError(f"Annotation not found: {ann_path} for image {img_path}")
        label = Image.open(ann_path)

        # Convert to tensors
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()  # (C, H, W)
        label = torch.from_numpy(np.array(label)).long()  # (H, W)

        # Resize image (short edge to target)
        image, label = self._resize_short_edge(image, label)

        # Handle ADE20K label offset
        if self.reduce_zero_label:
            label = label - 1
            label[label == -1] = 255

        # Apply transform if any
        if self.transform is not None:
            image, label = self.transform(image, label)

        return {
            "image": image,
            "label": label,
            "img_path": str(img_path),
        }

    def _resize_short_edge(
        self,
        image: torch.Tensor,
        label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resize image and label so short edge equals target_short_edge."""
        _, h, w = image.shape

        if h <= w:
            scale = self.target_short_edge / h
        else:
            scale = self.target_short_edge / w

        new_h = int(h * scale)
        new_w = int(w * scale)

        image = image.unsqueeze(0)
        image = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
        image = image.squeeze(0)

        label = label.unsqueeze(0).unsqueeze(0).float()
        label = F.interpolate(label, size=(new_h, new_w), mode='nearest')
        label = label.squeeze(0).squeeze(0).long()

        return image, label
