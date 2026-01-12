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
        short_edge_size: int = 512,
        return_original_size: bool = True,
    ):
        """Initialize the dataset.

        Args:
            data_root: Root directory containing the dataset
            corruption: Corruption type (e.g., 'gaussian_noise', 'fog')
            severity: Corruption severity level (1-5)
            short_edge_size: Target size for short edge resize
            return_original_size: Whether to return original image size
        """
        self.data_root = Path(data_root)
        self.corruption = corruption
        self.severity = severity
        self.short_edge_size = short_edge_size
        self.return_original_size = return_original_size

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

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - image: [C, H, W] tensor, values in [0, 1]
                - mask: [H, W] tensor, class indices
                - meta: Metadata dict with filename, original_size, etc.
        """
        img_path, ann_path = self.image_files[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        original_size = image.size  # (W, H)

        # Load annotation
        # ADE20K annotations are 0-indexed, with 0 being the first class
        # We need to handle the ignore regions (if any)
        mask = Image.open(ann_path)

        # Convert to tensors
        image_tensor = torch.from_numpy(
            # HWC -> CHW, uint8 -> float32, [0, 255] -> [0, 1]
            (np.array(Image.open(img_path).convert("RGB")) / 255.0).astype("float32")
        ).permute(2, 0, 1)

        mask_tensor = torch.from_numpy(np.array(mask).astype("int64"))

        # ADE20K mask values: 0 = background/ignore, 1-150 = classes
        # Convert to 0-indexed: subtract 1, set background to ignore_index (255)
        # Note: Some implementations keep 0 as first class and use 255 for ignore
        # We follow the convention where mask values 1-150 map to classes 0-149
        mask_tensor = mask_tensor - 1
        mask_tensor[mask_tensor == -1] = 255  # Background/ignore

        # Resize short edge to target size
        image_tensor, mask_tensor = self._resize_short_edge(image_tensor, mask_tensor)

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

    def _resize_short_edge(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resize image and mask by scaling short edge.

        Args:
            image: [C, H, W] tensor
            mask: [H, W] tensor

        Returns:
            Tuple of resized (image, mask)
        """
        C, H, W = image.shape

        # Calculate new dimensions
        if H < W:
            # Height is short edge
            new_h = self.short_edge_size
            new_w = int(W * self.short_edge_size / H)
        else:
            # Width is short edge
            new_w = self.short_edge_size
            new_h = int(H * self.short_edge_size / W)

        # Resize image with bilinear interpolation
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Resize mask with nearest neighbor
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=(new_h, new_w),
            mode="nearest",
        ).squeeze(0).squeeze(0).long()

        return image, mask


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
