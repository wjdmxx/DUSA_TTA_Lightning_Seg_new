"""Discriminative model wrapper using HuggingFace Transformers Segformer."""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from omegaconf import DictConfig


class SegformerModel(nn.Module):
    """Wrapper for HuggingFace Segformer model.

    This class handles:
    - Loading pretrained Segformer model
    - Image preprocessing specific to the discriminative model
    - Forward pass returning logits at 1/4 resolution
    """

    def __init__(self, cfg: DictConfig):
        """Initialize Segformer model.

        Args:
            cfg: Configuration containing model_name, device, num_classes, etc.
        """
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.num_classes = cfg.num_classes
        self.output_stride = cfg.output_stride
        self.target_short_edge = cfg.preprocessing.target_short_edge

        # Load model and processor
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            cfg.model_name,
            num_labels=cfg.num_classes,
            ignore_mismatched_sizes=True
        )
        self.processor = SegformerImageProcessor.from_pretrained(
            cfg.model_name,
            do_resize=False,  # We handle resizing ourselves
            do_rescale=True,
            do_normalize=True
        )

        # Move to device
        self.model.to(self.device)

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for Segformer.

        The input images should already be resized so that short edge = 512.
        This function applies the Segformer-specific normalization.

        Args:
            images: Tensor of shape (B, C, H, W) with values in [0, 255], RGB format

        Returns:
            Preprocessed tensor ready for model input
        """
        # Ensure on correct device
        images = images.to(self.device)

        # Convert to float and scale to [0, 1]
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        elif images.max() > 1.0:
            images = images / 255.0

        # Apply ImageNet normalization (Segformer uses ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        images = (images - mean) / std

        return images

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass through Segformer.

        Args:
            images: Preprocessed tensor of shape (B, C, H, W)

        Returns:
            Logits tensor of shape (B, num_classes, H/4, W/4)
        """
        # Ensure on correct device
        images = images.to(self.device)

        # Forward pass
        outputs = self.model(pixel_values=images)
        logits = outputs.logits  # (B, num_classes, H/4, W/4)

        return logits

    def predict(self, images: torch.Tensor, original_size: Tuple[int, int]) -> torch.Tensor:
        """Get predictions at original resolution.

        Args:
            images: Preprocessed tensor of shape (B, C, H, W)
            original_size: Original (H, W) to upsample predictions to

        Returns:
            Predictions tensor of shape (B, H, W) with class indices
        """
        logits = self.forward(images)

        # Upsample to original size
        upsampled = F.interpolate(
            logits,
            size=original_size,
            mode='bilinear',
            align_corners=False
        )

        # Get class predictions
        predictions = upsampled.argmax(dim=1)

        return predictions

    def get_logits_for_loss(self, images: torch.Tensor) -> torch.Tensor:
        """Get logits at 1/4 resolution for TTA loss computation.

        This returns the raw logits without upsampling, which are used
        by the generative model for computing the TTA loss.

        Args:
            images: Preprocessed tensor of shape (B, C, H, W)

        Returns:
            Logits tensor of shape (B, num_classes, H/4, W/4)
        """
        return self.forward(images)

    def configure_tta_grad(self, update: bool = True, norm_only: bool = False):
        """Configure gradient settings for TTA.

        Args:
            update: Whether to update model parameters
            norm_only: If True, only update normalization layers
        """
        if not update:
            self.model.requires_grad_(False)
            return

        if norm_only:
            # First freeze all
            self.model.requires_grad_(False)
            # Then unfreeze norm layers
            for name, module in self.model.named_modules():
                if 'norm' in name.lower() or isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
                    for param in module.parameters():
                        param.requires_grad = True
        else:
            self.model.requires_grad_(True)


def resize_image_short_edge(image: torch.Tensor, target_short_edge: int) -> torch.Tensor:
    """Resize image so that the short edge equals target_short_edge.

    This handles both horizontal and vertical images correctly.

    Args:
        image: Tensor of shape (B, C, H, W) or (C, H, W)
        target_short_edge: Target size for the short edge

    Returns:
        Resized tensor
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    _, _, h, w = image.shape

    # Determine which edge is shorter
    if h <= w:
        # Height is shorter or equal
        new_h = target_short_edge
        new_w = int(w * target_short_edge / h)
    else:
        # Width is shorter
        new_w = target_short_edge
        new_h = int(h * target_short_edge / w)

    # Resize
    resized = F.interpolate(
        image.float(),
        size=(new_h, new_w),
        mode='bilinear',
        align_corners=False
    )

    if squeeze:
        resized = resized.squeeze(0)

    return resized
