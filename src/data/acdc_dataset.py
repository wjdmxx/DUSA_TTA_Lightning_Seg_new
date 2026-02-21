"""ACDC dataset for Test-Time Adaptation.

Directory structure:
    data_root/
        rgb_anon/
            {condition}/
                val/
                    {sequence}/
                        *.png
        gt/
            {condition}/
                val/
                    {sequence}/
                        *_gt_labelTrainIds.png

ACDC uses the Cityscapes 19-class label set.
GT files are already in trainIds format (0-18, 255=ignore).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from ..utils.categories import ACDC_CONDITIONS, ACDC_NUM_CLASSES

logger = logging.getLogger(__name__)


class ACDCDataset(Dataset):
    """ACDC dataset with adverse conditions.

    Conditions: fog, night, rain, snow.
    GT is already in trainIds (same 19 classes as Cityscapes).
    """

    CONDITIONS = ACDC_CONDITIONS

    def __init__(
        self,
        data_root: str,
        condition: str,
    ):
        """Initialize the dataset.

        Args:
            data_root: Root directory containing the dataset
            condition: Adverse condition (fog, night, rain, snow)
        """
        self.data_root = Path(data_root)
        self.condition = condition

        # Validate condition
        if condition not in self.CONDITIONS:
            raise ValueError(
                f"Unknown condition: {condition}. "
                f"Available: {self.CONDITIONS}"
            )

        # Setup paths
        self.image_dir = self.data_root / "rgb_anon" / condition / "train"
        self.gt_dir = self.data_root / "gt" / condition / "train"

        # Verify directories exist
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.gt_dir.exists():
            raise FileNotFoundError(f"GT directory not found: {self.gt_dir}")

        # Collect image files
        self.image_files = self._collect_image_files()
        logger.info(
            f"Loaded ACDC dataset: condition={condition}, "
            f"num_samples={len(self.image_files)}"
        )

    def _collect_image_files(self) -> List[Tuple[Path, Path]]:
        """Collect pairs of image and GT files.

        Returns:
            List of (image_path, gt_path) tuples
        """
        files = []

        # Iterate over sequence directories
        for seq_dir in sorted(self.image_dir.iterdir()):
            if not seq_dir.is_dir():
                continue

            sequence = seq_dir.name
            gt_seq_dir = self.gt_dir / sequence

            if not gt_seq_dir.exists():
                logger.warning(f"GT directory not found for sequence: {sequence}")
                continue

            for img_path in sorted(seq_dir.glob("*.png")):
                # Image: {ref_id}_rgb_anon.png
                # GT:    {ref_id}_gt_labelTrainIds.png
                base_name = img_path.stem.replace("_rgb_anon", "")
                gt_name = f"{base_name}_gt_labelTrainIds.png"
                gt_path = gt_seq_dir / gt_name

                if gt_path.exists():
                    files.append((img_path, gt_path))
                else:
                    logger.warning(f"GT not found for {img_path.name}")

        return files

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample.

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

        # Load GT mask (already trainIds)
        mask = np.array(Image.open(gt_path)).astype(np.int64)

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
                "condition": self.condition,
            },
        }
