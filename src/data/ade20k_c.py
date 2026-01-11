"""
ADE20K-C (corrupted) dataset for TTA evaluation.
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import os

from PIL import Image
import torch
from torch.utils.data import Dataset

from .transforms import TTATransform


# 15 corruption types used in ADE20K-C
CORRUPTION_TYPES = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]

# Default severity level
DEFAULT_SEVERITY = 5


class ADE20KCorruptedDataset(Dataset):
    """
    ADE20K-C dataset with corrupted images for TTA evaluation.
    
    Directory structure expected:
    data_root/
        ADE20K_val-c/
            {corruption_type}/
                {severity}/
                    validation/
                        *.jpg
        annotations/
            validation/
                *.png
    """
    
    def __init__(
        self,
        data_root: str,
        corruption_type: str,
        severity: int = DEFAULT_SEVERITY,
        transform: Optional[TTATransform] = None,
        ann_dir: Optional[str] = None,
    ):
        """
        Args:
            data_root: Root directory containing ADE20K_val-c and annotations
            corruption_type: Type of corruption (e.g., "gaussian_noise")
            severity: Corruption severity level (1-5)
            transform: Transform to apply to images and labels
            ann_dir: Optional custom annotation directory
        """
        self.data_root = Path(data_root)
        self.corruption_type = corruption_type
        self.severity = severity
        self.transform = transform or TTATransform()
        
        # Image directory
        self.img_dir = self.data_root / "ADE20K_val-c" / corruption_type / str(severity) / "validation"
        
        # Annotation directory
        if ann_dir is not None:
            self.ann_dir = Path(ann_dir)
        else:
            self.ann_dir = self.data_root / "annotations" / "validation"
        
        # Get image list
        self.images = self._get_image_list()
        
        if len(self.images) == 0:
            raise RuntimeError(
                f"No images found in {self.img_dir}. "
                f"Please check the directory structure."
            )
    
    def _get_image_list(self) -> List[Tuple[Path, Path]]:
        """
        Get list of (image_path, annotation_path) tuples.
        
        Returns:
            List of path tuples
        """
        images = []
        
        if not self.img_dir.exists():
            return images
        
        for img_path in sorted(self.img_dir.glob("*.jpg")):
            # Annotation filename: same name but .png extension
            ann_name = img_path.stem + ".png"
            ann_path = self.ann_dir / ann_name
            
            if ann_path.exists():
                images.append((img_path, ann_path))
        
        return images
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample.
        
        Returns:
            Dict with keys:
                - image: (3, H, W) tensor in [0, 255]
                - label: (H, W) tensor with class indices
                - original_size: (H, W) tuple
                - img_path: str path to image
        """
        img_path, ann_path = self.images[idx]
        
        # Load image (RGB)
        image = Image.open(img_path).convert("RGB")
        
        # Load annotation
        label = Image.open(ann_path)
        
        # Apply transform
        image_tensor, label_tensor, original_size = self.transform(image, label)
        
        return {
            "image": image_tensor,
            "label": label_tensor,
            "original_size": original_size,
            "img_path": str(img_path),
        }


class ADE20KCorruptedTaskDataset(Dataset):
    """
    Wrapper that creates a dataset for a specific corruption task.
    Useful for iterating over multiple corruption types.
    """
    
    def __init__(
        self,
        data_root: str,
        corruption_types: Optional[List[str]] = None,
        severity: int = DEFAULT_SEVERITY,
        transform: Optional[TTATransform] = None,
    ):
        """
        Args:
            data_root: Root directory
            corruption_types: List of corruption types (default: all 15)
            severity: Corruption severity level
            transform: Transform to apply
        """
        self.data_root = data_root
        self.corruption_types = corruption_types or CORRUPTION_TYPES
        self.severity = severity
        self.transform = transform
        
        self._current_dataset = None
        self._current_corruption = None
    
    def get_dataset(self, corruption_type: str) -> ADE20KCorruptedDataset:
        """
        Get dataset for a specific corruption type.
        
        Args:
            corruption_type: Type of corruption
            
        Returns:
            ADE20KCorruptedDataset instance
        """
        return ADE20KCorruptedDataset(
            data_root=self.data_root,
            corruption_type=corruption_type,
            severity=self.severity,
            transform=self.transform,
        )
    
    def get_all_corruption_types(self) -> List[str]:
        """Get list of all corruption types."""
        return self.corruption_types.copy()
