"""ADE20K-C dataset for Test-Time Adaptation."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from ..utils.categories import ADE20K_CORRUPTIONS, ADE20K_NUM_CLASSES

logger = logging.getLogger(__name__)


class ADE20KCorruptedDataset(Dataset):
    """ADE20K-C dataset with corruption types and severity levels.

    Dataset structure:
        data_root/
            ADE20K_val-c/
                {corruption}/
                    {severity}/
                        validation/
                            *.jpg
            annotations/
                validation/
                    *.png
    """

    # Available corruption types
    CORRUPTIONS = ADE20K_CORRUPTIONS

    def __init__(
        self,
        data_root: str,
        corruption: str,
        severity: int = 5,
    ):
        """Initialize the dataset.

        Args:
            data_root: Root directory containing the dataset
            corruption: Corruption type (e.g., 'gaussian_noise', 'fog')
            severity: Corruption severity level (1-5)
        """
        self.data_root = Path(data_root)
        self.corruption = corruption
        self.severity = severity

        # Validate corruption type
        if corruption not in self.CORRUPTIONS:
            raise ValueError(
                f"Unknown corruption: {corruption}. "
                f"Available: {self.CORRUPTIONS}"
            )

        # Validate severity
        if not 1 <= severity <= 5:
            raise ValueError(f"Severity must be between 1 and 5, got {severity}")

        # Setup paths
        self.image_dir = (
            self.data_root / "ADE20K_val-c" / corruption / str(severity) / "validation"
        )
        self.annotation_dir = self.data_root / "annotations" / "validation"

        # Verify directories exist
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.annotation_dir.exists():
            raise FileNotFoundError(f"Annotation directory not found: {self.annotation_dir}")

        # Collect image files
        self.image_files = self._collect_image_files()
        logger.info(
            f"Loaded ADE20K-C dataset: corruption={corruption}, severity={severity}, "
            f"num_samples={len(self.image_files)}"
        )

    def _collect_image_files(self) -> List[Tuple[Path, Path]]:
        """Collect pairs of image and annotation files.

        Returns:
            List of (image_path, annotation_path) tuples
        """
        files = []
        # Get all jpg images from the corruption directory
        for img_path in sorted(self.image_dir.glob("*.jpg")):
            # Annotation file has same name but .png extension
            ann_name = img_path.stem + ".png"
            ann_path = self.annotation_dir / ann_name

            if ann_path.exists():
                files.append((img_path, ann_path))
            else:
                logger.warning(f"Annotation not found for {img_path.name}")

        return files

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample.

        Returns original-size image and mask without any resize.
        Resize is handled by each model internally.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - image: [C, H, W] tensor, values in [0, 1], original size
                - mask: [H, W] tensor, class indices, original size
                - meta: Metadata dict with filename, original_size, etc.
        """
        img_path, ann_path = self.image_files[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        original_size = image.size  # (W, H)

        # Load annotation
        mask = Image.open(ann_path)

        # Convert to tensors (no resize)
        image_tensor = torch.from_numpy(
            (np.array(image) / 255.0).astype("float32")
        ).permute(2, 0, 1)  # HWC -> CHW, [0,255] -> [0,1]

        mask_tensor = torch.from_numpy(np.array(mask).astype("int64"))

        # ADE20K mask values: 0 = background/ignore, 1-150 = classes
        # Convert to 0-indexed: subtract 1, set background to ignore_index (255)
        mask_tensor = mask_tensor - 1
        mask_tensor[mask_tensor == -1] = 255  # Background/ignore

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "meta": {
                "filename": img_path.name,
                "original_size": original_size,  # (W, H)
                "corruption": self.corruption,
                "severity": self.severity,
            },
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for the dataset.

    Args:
        dataset: Dataset to wrap
        batch_size: Batch size (default: 1 for TTA)
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory

    Returns:
        DataLoader instance
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
