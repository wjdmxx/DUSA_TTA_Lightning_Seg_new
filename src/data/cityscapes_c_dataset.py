"""Cityscapes-C dataset for Test-Time Adaptation.

Directory structure:
    data_root/
        Cityscapes-C/
            {corruption}/
                {severity}/
                    val/
                        {city}/
                            *.png
        gtFine/
            val/
                {city}/
                    *_gtFine_labelIds.png
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from ..utils.categories import (
    CITYSCAPES_CORRUPTIONS,
    CITYSCAPES_NUM_CLASSES,
    create_cityscapes_label_mapping,
)

logger = logging.getLogger(__name__)


class CityscapesCCorruptedDataset(Dataset):
    """Cityscapes-C dataset with corruption types and severity levels.

    GT masks use labelIds which are mapped to trainIds (0-18) at load time.
    """

    CORRUPTIONS = CITYSCAPES_CORRUPTIONS

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

        # Build the label mapping lookup table (labelId -> trainId)
        self.label_mapping = create_cityscapes_label_mapping()

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
        # Images: {data_root}/Cityscapes-C/{corruption}/{severity}/val/{city}/*.png
        self.image_dir = (
            self.data_root / "Cityscapes-C" / corruption / str(severity) / "val"
        )
        # GT: {data_root}/gtFine/val/{city}/*_gtFine_labelIds.png
        self.gt_dir = self.data_root / "gtFine" / "val"

        # Verify directories exist
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.gt_dir.exists():
            raise FileNotFoundError(f"GT directory not found: {self.gt_dir}")

        # Collect image files
        self.image_files = self._collect_image_files()
        logger.info(
            f"Loaded Cityscapes-C dataset: corruption={corruption}, severity={severity}, "
            f"num_samples={len(self.image_files)}"
        )

    def _collect_image_files(self) -> List[Tuple[Path, Path]]:
        """Collect pairs of image and GT files.

        Returns:
            List of (image_path, gt_path) tuples
        """
        files = []

        # Iterate over city directories
        for city_dir in sorted(self.image_dir.iterdir()):
            if not city_dir.is_dir():
                continue

            city = city_dir.name
            gt_city_dir = self.gt_dir / city

            if not gt_city_dir.exists():
                logger.warning(f"GT directory not found for city: {city}")
                continue

            for img_path in sorted(city_dir.glob("*.png")):
                # Image filename: {city}_{seq}_{frame}_leftImg8bit.png
                # GT filename: {city}_{seq}_{frame}_gtFine_labelIds.png
                base_name = img_path.stem.replace("_leftImg8bit", "")
                gt_name = f"{base_name}_gtFine_labelIds.png"
                gt_path = gt_city_dir / gt_name

                if gt_path.exists():
                    files.append((img_path, gt_path))
                else:
                    logger.warning(f"GT not found for {img_path.name}")

        return files

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample with labelId -> trainId mapping applied to GT.

        Returns:
            Dictionary containing:
                - image: [C, H, W] tensor, values in [0, 1], original size
                - mask: [H, W] tensor, trainIds (0-18), ignore=255
                - meta: Metadata dict
        """
        img_path, gt_path = self.image_files[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        original_size = image.size  # (W, H)

        # Load GT mask (labelIds)
        mask = np.array(Image.open(gt_path))

        # Apply labelId -> trainId mapping
        mask = self.label_mapping[mask].astype(np.int64)

        # Convert to tensors
        image_tensor = torch.from_numpy(
            (np.array(image) / 255.0).astype("float32")
        ).permute(2, 0, 1)  # HWC -> CHW

        mask_tensor = torch.from_numpy(mask)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "meta": {
                "filename": img_path.name,
                "original_size": original_size,
                "corruption": self.corruption,
                "severity": self.severity,
            },
        }
