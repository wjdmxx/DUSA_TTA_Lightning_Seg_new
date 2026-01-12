"""Discriminative model (SegFormer) for semantic segmentation."""

from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)


class DiscriminativeModel(nn.Module):
    """Discriminative model wrapper for SegFormer.
    
    Wraps HuggingFace SegFormer for use in TTA pipeline.
    The model outputs logits at 4x downsampled resolution.
    """
    
    def __init__(
        self,
        model_name: str = "nvidia/segformer-b5-finetuned-ade-640-640",
        num_classes: int = 150,
    ):
        """Initialize SegFormer model.
        
        Args:
            model_name: HuggingFace model name or path
            num_classes: Number of segmentation classes
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load model and processor
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        
        # Get normalization parameters from processor
        self.register_buffer(
            'mean',
            torch.tensor(self.processor.image_mean).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor(self.processor.image_std).view(1, 3, 1, 1)
        )
    
    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for SegFormer.
        
        Applies ImageNet normalization.
        
        Args:
            images: Input tensor of shape (B, 3, H, W) in range [0, 255]
            
        Returns:
            Normalized tensor in range suitable for model
        """
        # Convert to [0, 1]
        images = images / 255.0
        
        # Apply ImageNet normalization
        images = (images - self.mean.to(images.device)) / self.std.to(images.device)
        
        return images
    
    def forward(
        self,
        images: torch.Tensor,
        return_dict: bool = False
    ) -> torch.Tensor:
        """Forward pass through SegFormer.
        
        Args:
            images: Input tensor of shape (B, 3, H, W) in range [0, 255]
            return_dict: If True, return full model output dict
            
        Returns:
            Logits tensor of shape (B, num_classes, H/4, W/4)
        """
        # Preprocess
        preprocessed = self.preprocess(images)
        
        # Forward through model
        outputs = self.model(pixel_values=preprocessed)
        
        if return_dict:
            return outputs
        
        # Return logits (4x downsampled)
        return outputs.logits
    
    def forward_with_upsampling(
        self,
        images: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Forward pass with optional upsampling to target size.
        
        Args:
            images: Input tensor of shape (B, 3, H, W) in range [0, 255]
            target_size: Optional (H, W) to upsample logits to
            
        Returns:
            Logits tensor, optionally upsampled
        """
        logits = self.forward(images)
        
        if target_size is not None:
            logits = F.interpolate(
                logits,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
        
        return logits
    
    def get_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to class predictions.
        
        Args:
            logits: Logits tensor of shape (B, C, H, W)
            
        Returns:
            Predictions tensor of shape (B, H, W) with class indices
        """
        return logits.argmax(dim=1)
    
    def postprocess(
        self,
        logits: torch.Tensor,
        target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Postprocess logits to full resolution predictions.
        
        Args:
            logits: Logits tensor of shape (B, C, H, W)
            target_size: Target size (H, W) for upsampling
            
        Returns:
            Predictions tensor of shape (B, H, W)
        """
        # Upsample logits
        upsampled = F.interpolate(
            logits,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )
        
        # Get predictions
        return self.get_predictions(upsampled)
