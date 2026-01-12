"""ADE20K-C Dataset for Test-Time Adaptation."""

import os
from pathlib import Path
from typing import Optional, Callable, Dict, Any, Tuple, List

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class ADE20KCDataset(Dataset):
    """ADE20K-C dataset for semantic segmentation with corruption.
    
    This dataset loads corrupted images from ADE20K-C and their corresponding
    annotations from the original ADE20K dataset.
    
    Directory structure expected:
        data_root/
        ├── ADE20K_val-c/
        │   └── {corruption}/
        │       └── {severity}/
        │           └── validation/
        │               └── *.jpg
        └── annotations/
            └── validation/
                └── *.png
    """
    
    def __init__(
        self,
        data_root: str,
        corruption: str,
        severity: int = 5,
        short_edge_size: int = 512,
        transform: Optional[Callable] = None,
    ):
        """Initialize ADE20K-C dataset.
        
        Args:
            data_root: Root directory containing data
            corruption: Type of corruption (e.g., 'gaussian_noise', 'fog')
            severity: Corruption severity level (1-5)
            short_edge_size: Target size for short edge scaling
            transform: Optional additional transforms
        """
        super().__init__()
        self.data_root = Path(data_root)
        self.corruption = corruption
        self.severity = severity
        self.short_edge_size = short_edge_size
        self.transform = transform
        
        # Build paths
        self.image_dir = self.data_root / "ADE20K_val-c" / corruption / str(severity) / "validation"
        self.annotation_dir = self.data_root / "annotations" / "validation"
        
        # Collect image files
        self.image_files = self._collect_files()
        
        if len(self.image_files) == 0:
            raise RuntimeError(
                f"No images found in {self.image_dir}. "
                f"Please check path and corruption type: {corruption}"
            )
    
    def _collect_files(self) -> List[Tuple[Path, Path]]:
        """Collect pairs of image and annotation files.
        
        Returns:
            List of (image_path, annotation_path) tuples
        """
        files = []
        
        if not self.image_dir.exists():
            return files
        
        for img_file in sorted(self.image_dir.iterdir()):
            if img_file.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                # Annotation file has same name but .png extension
                ann_name = img_file.stem + '.png'
                ann_file = self.annotation_dir / ann_name
                
                if ann_file.exists():
                    files.append((img_file, ann_file))
                else:
                    # Some annotations might have different naming
                    # Try without any suffix modifications
                    print(f"Warning: Annotation not found for {img_file.name}")
        
        return files
    
    def _resize_short_edge(
        self,
        image: Image.Image,
        annotation: Optional[Image.Image] = None
    ) -> Tuple[Image.Image, Optional[Image.Image]]:
        """Resize image so that short edge equals target size.
        
        Args:
            image: PIL Image to resize
            annotation: Optional annotation image to resize
            
        Returns:
            Tuple of (resized_image, resized_annotation)
        """
        w, h = image.size
        
        # Determine scale factor based on short edge
        if h <= w:
            # Height is short edge
            new_h = self.short_edge_size
            new_w = int(w * self.short_edge_size / h)
        else:
            # Width is short edge
            new_w = self.short_edge_size
            new_h = int(h * self.short_edge_size / w)
        
        # Resize image with bilinear interpolation
        resized_image = image.resize((new_w, new_h), Image.BILINEAR)
        
        # Resize annotation with nearest neighbor to preserve labels
        resized_annotation = None
        if annotation is not None:
            resized_annotation = annotation.resize((new_w, new_h), Image.NEAREST)
        
        return resized_image, resized_annotation
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - image: Tensor of shape (3, H, W) in [0, 255]
                - mask: Tensor of shape (H, W) with class labels
                - filename: Original filename
                - original_size: Original image size (H, W)
        """
        img_path, ann_path = self.image_files[idx]
        
        # Load image and annotation
        image = Image.open(img_path).convert('RGB')
        annotation = Image.open(ann_path)
        
        original_size = (image.height, image.width)
        
        # Resize short edge to target size
        image, annotation = self._resize_short_edge(image, annotation)
        
        # Convert to tensors
        # Image: (H, W, 3) uint8 -> (3, H, W) float [0, 255]
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
        
        # Annotation: (H, W) with values 0-150 (0 is background/void in some versions)
        # ADE20K uses 1-150 for classes, 0 for void
        # We convert to 0-149 for classes, 255 for void/ignore
        annotation_np = np.array(annotation).astype(np.int64)
        # Shift labels: 0 (void) -> 255, 1-150 -> 0-149
        annotation_np = annotation_np - 1
        annotation_np[annotation_np == -1] = 255  # void/ignore
        mask_tensor = torch.from_numpy(annotation_np).long()
        
        sample = {
            'image': image_tensor,
            'mask': mask_tensor,
            'filename': img_path.name,
            'original_size': original_size,
        }
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample


def get_corruption_types() -> List[str]:
    """Get list of all corruption types in ADE20K-C.
    
    Returns:
        List of corruption type names
    """
    return [
        'gaussian_noise',
        'shot_noise',
        'impulse_noise',
        'defocus_blur',
        'glass_blur',
        'motion_blur',
        'zoom_blur',
        'snow',
        'frost',
        'fog',
        'brightness',
        'contrast',
        'elastic_transform',
        'pixelate',
        'jpeg_compression',
    ]
