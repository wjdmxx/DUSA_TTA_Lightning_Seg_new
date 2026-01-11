"""
Image preprocessor for generative models.
Normalizes images to [-1, 1] range as required by Stable Diffusion.
"""

from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class GenerativePreprocessor(nn.Module):
    """
    Preprocessor for generative models (Stable Diffusion).
    
    Converts images from [0, 255] RGB format to [-1, 1] normalized format
    and resizes to target size.
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        normalize_to_neg_one: bool = True,
    ):
        """
        Args:
            target_size: Target (H, W) size for the output
            normalize_to_neg_one: If True, normalize to [-1, 1], else to [0, 1]
        """
        super().__init__()
        self.target_size = target_size
        self.normalize_to_neg_one = normalize_to_neg_one
    
    def forward(
        self,
        images: torch.Tensor,
        resize: bool = True,
    ) -> torch.Tensor:
        """
        Preprocess images for generative model.
        
        Args:
            images: Input images of shape (B, 3, H, W) in range [0, 255], RGB format
            resize: Whether to resize to target_size
            
        Returns:
            Preprocessed images of shape (B, 3, H', W') in range [-1, 1] or [0, 1]
        """
        # Ensure float
        images = images.float()
        
        # Resize if needed
        if resize:
            _, _, h, w = images.shape
            if (h, w) != self.target_size:
                images = F.interpolate(
                    images,
                    size=self.target_size,
                    mode="bilinear",
                    align_corners=False,
                )
        
        # Normalize to [0, 1]
        images = images / 255.0
        
        # Optionally map to [-1, 1]
        if self.normalize_to_neg_one:
            images = images * 2.0 - 1.0
        
        return images
    
    def denormalize(self, images: torch.Tensor) -> torch.Tensor:
        """
        Denormalize images back to [0, 255] range.
        
        Args:
            images: Normalized images in range [-1, 1] or [0, 1]
            
        Returns:
            Images in range [0, 255]
        """
        if self.normalize_to_neg_one:
            images = (images + 1.0) / 2.0
        
        images = images * 255.0
        return images.clamp(0, 255)
