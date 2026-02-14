"""Discriminative model (SegFormer) wrapper for semantic segmentation."""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import torchvision.transforms.functional as TF

logger = logging.getLogger(__name__)


class DiscriminativeModel(nn.Module):
    """SegFormer-based discriminative model for semantic segmentation.

    This class wraps HuggingFace's SegformerForSemanticSegmentation with:
    - Internal short-edge resize (via torchvision)
    - ImageNet normalization preprocessing
    - Logits output at 4x downsampled resolution
    - Optional resizing to original resolution
    """

    # ImageNet normalization values
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

    def __init__(
        self,
        config: DictConfig,
    ):
        """Initialize the discriminative model.

        Args:
            config: Model configuration containing:
                - model_name: HuggingFace model name/path
                - num_classes: Number of output classes (default: 150)
                - short_edge_size: Target size for short edge resize (default: 512)
        """
        super().__init__()

        model_name = config.get("model_name", "nvidia/segformer-b5-finetuned-ade-640-640")
        self.num_classes = config.get("num_classes", 150)
        self.output_stride = config.get("output_stride", 4)
        self.short_edge_size = config.get("short_edge_size", 512)

        logger.info(f"Loading SegFormer model: {model_name}")
        logger.info(f"Short edge resize: {self.short_edge_size}")

        # Load model
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True,
        )

        # Store mean/std as buffers (not parameters)
        self.register_buffer(
            "_mean",
            self.IMAGENET_MEAN.view(1, 3, 1, 1),
        )
        self.register_buffer(
            "_std",
            self.IMAGENET_STD.view(1, 3, 1, 1),
        )

        logger.info(f"SegFormer loaded with {self.num_classes} classes")

    def _resize_short_edge(self, images: torch.Tensor) -> torch.Tensor:
        """Resize images so that the short edge equals self.short_edge_size.

        Uses torchvision.transforms.functional.resize which handles
        short-edge resize when size is a single int.

        Args:
            images: Input tensor [B, C, H, W]

        Returns:
            Resized tensor [B, C, H', W'] maintaining aspect ratio
        """
        return TF.resize(
            images,
            size=self.short_edge_size,  # single int = short edge resize
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias=True,
        )

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet normalization.

        Args:
            images: Input tensor [B, C, H, W] with values in [0, 1]

        Returns:
            Normalized tensor
        """
        return (images - self._mean) / self._std

    def forward(
        self,
        images: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        """Forward pass through the model.

        Internally resizes images (short edge) then normalizes.

        Args:
            images: Input tensor [B, C, H, W] with values in [0, 1]
                    at original resolution (no pre-resize needed)
            return_features: If True, also return intermediate features

        Returns:
            logits: Output logits [B, num_classes, H'/4, W'/4]
                    where H', W' are the resized dimensions
        """
        # Short-edge resize
        images = self._resize_short_edge(images)

        # Normalize images
        normalized = self.preprocess(images)

        # Forward through model
        outputs = self.model(
            pixel_values=normalized,
            output_hidden_states=return_features,
            return_dict=True,
        )

        # Get logits (at 4x downsampled resolution)
        logits = outputs.logits

        if return_features:
            return logits, outputs.hidden_states
        return logits

    def predict(
        self,
        images: torch.Tensor,
        resize_to_original: bool = True,
        original_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Get predictions with optional resizing.

        Args:
            images: Input tensor [B, C, H, W]
            resize_to_original: Whether to resize output to input size
            original_size: Optional (H, W) for target resize

        Returns:
            predictions: Class predictions [B, H, W]
        """
        logits = self.forward(images)

        # Resize if requested
        if resize_to_original:
            if original_size is not None:
                target_size = original_size
            else:
                target_size = (images.shape[2], images.shape[3])

            logits = F.interpolate(
                logits,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        return logits.argmax(dim=1)

    def get_logits_size(self, input_size: Tuple[int, int]) -> Tuple[int, int]:
        """Calculate output logits size for given input size.

        Args:
            input_size: (H, W) of input

        Returns:
            (H_out, W_out) of logits
        """
        H, W = input_size
        return (H // self.output_stride, W // self.output_stride)
