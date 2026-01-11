"""
Data transforms for TTA.
Minimal preprocessing - normalization is handled by each model.
"""

from typing import Tuple, Optional
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np


class TTATransform:
    """
    Transform for TTA data loading.
    
    Only handles:
    - Conversion to tensor
    - Short side resize (maintaining aspect ratio)
    - Ensuring dimensions are divisible by 4
    
    Normalization is handled by individual models.
    """
    
    def __init__(
        self,
        target_short_side: int = 512,
    ):
        """
        Args:
            target_short_side: Target size for the short side
        """
        self.target_short_side = target_short_side
    
    def __call__(
        self,
        image: Image.Image,
        label: Optional[Image.Image] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[int, int]]:
        """
        Transform image and optional label.
        
        Args:
            image: PIL Image (RGB)
            label: Optional PIL Image with segmentation labels
            
        Returns:
            Tuple of:
                - image_tensor: (3, H, W) in [0, 255], RGB
                - label_tensor: (H, W) with class indices, or None
                - original_size: (H, W) before any transform
        """
        # Get original size
        w, h = image.size
        original_size = (h, w)
        
        # Calculate new size (short side to target, maintain aspect ratio)
        if h < w:
            new_h = self.target_short_side
            new_w = int(w * self.target_short_side / h)
        else:
            new_w = self.target_short_side
            new_h = int(h * self.target_short_side / w)
        
        # Ensure divisible by 4
        new_h = (new_h // 4) * 4
        new_w = (new_w // 4) * 4
        
        # Resize image
        image = image.resize((new_w, new_h), Image.BILINEAR)
        
        # Convert to tensor: (H, W, 3) -> (3, H, W), keep in [0, 255]
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
        
        # Process label if provided
        label_tensor = None
        if label is not None:
            label = label.resize((new_w, new_h), Image.NEAREST)
            label_tensor = torch.from_numpy(np.array(label)).long()
        
        return image_tensor, label_tensor, original_size


class TTATransformWithOriginal:
    """
    Transform that also returns the original-size image tensor.
    Useful for evaluation at original resolution.
    """
    
    def __init__(
        self,
        target_short_side: int = 512,
    ):
        self.target_short_side = target_short_side
        self.base_transform = TTATransform(target_short_side)
    
    def __call__(
        self,
        image: Image.Image,
        label: Optional[Image.Image] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Tuple[int, int]]:
        """
        Transform image and optional label.
        
        Returns:
            Tuple of:
                - image_tensor: Resized (3, H', W') in [0, 255]
                - label_tensor: Resized (H', W') or None
                - original_image_tensor: Original size (3, H, W) in [0, 255]
                - original_size: (H, W)
        """
        # Get original image tensor
        original_image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
        
        # Apply base transform
        image_tensor, label_tensor, original_size = self.base_transform(image, label)
        
        return image_tensor, label_tensor, original_image_tensor, original_size
