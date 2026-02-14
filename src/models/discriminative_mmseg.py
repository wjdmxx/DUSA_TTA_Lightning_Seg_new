"""MMSegmentation-based discriminative model for semantic segmentation."""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
import torchvision.transforms.functional as TF

logger = logging.getLogger(__name__)


class MMSegDiscriminativeModel(nn.Module):
    """MMSegmentation-based SegFormer B5 model for semantic segmentation.

    This class wraps OpenMMLab's SegFormer implementation with:
    - Internal short-edge resize (via torchvision)
    - BGR color space conversion and normalization
    - Compatible interface with the HuggingFace version
    - Local checkpoint loading
    """

    # MMSeg normalization values (for 0-255 range, but we convert from 0-1)
    # Original MMSeg values: mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
    # Converted to 0-1 range:
    MMSEG_MEAN = torch.tensor([123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0])  # RGB order
    MMSEG_STD = torch.tensor([58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0])

    def __init__(
        self,
        config: DictConfig,
    ):
        """Initialize the MMSeg discriminative model.

        Args:
            config: Model configuration containing:
                - checkpoint: Path to local checkpoint file
                - num_classes: Number of output classes (default: 150)
                - output_stride: Output stride (default: 4)
                - short_edge_size: Target size for short edge resize (default: 512)
        """
        super().__init__()

        self.checkpoint = config.get(
            "checkpoint",
            "pretrained_models/segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth"
        )
        self.num_classes = config.get("num_classes", 150)
        self.output_stride = config.get("output_stride", 4)
        self.short_edge_size = config.get("short_edge_size", 512)

        logger.info(f"Loading MMSeg SegFormer model from: {self.checkpoint}")
        logger.info(f"Short edge resize: {self.short_edge_size}")

        # Build model using mmsegmentation
        self._build_model()

        # Store mean/std as buffers (not parameters)
        # Note: MMSeg uses BGR order internally, we'll handle RGB->BGR conversion
        self.register_buffer(
            "_mean",
            self.MMSEG_MEAN.view(1, 3, 1, 1),
        )
        self.register_buffer(
            "_std",
            self.MMSEG_STD.view(1, 3, 1, 1),
        )

        logger.info(f"MMSeg SegFormer loaded with {self.num_classes} classes")

    def _build_model(self):
        """Build the SegFormer model using mmsegmentation components."""
        try:
            from mmseg.models.backbones import MixVisionTransformer
            from mmseg.models.decode_heads import SegformerHead
        except ImportError:
            raise ImportError(
                "mmsegmentation is required for MMSegDiscriminativeModel. "
                "Install it with: pip install mmsegmentation"
            )

        # Build backbone (MIT-B5)
        self.backbone = MixVisionTransformer(
            in_channels=3,
            embed_dims=64,
            num_stages=4,
            num_layers=[3, 6, 40, 3],
            num_heads=[1, 2, 5, 8],
            patch_sizes=[7, 3, 3, 3],
            sr_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3),
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
        )

        # Build decode head
        self.decode_head = SegformerHead(
            in_channels=[64, 128, 320, 512],
            in_index=[0, 1, 2, 3],
            channels=256,
            dropout_ratio=0.1,
            num_classes=self.num_classes,
            norm_cfg=dict(type='BN', requires_grad=True),  # Use BN instead of SyncBN
            align_corners=False,
        )

        # Load pretrained weights
        self._load_checkpoint()

    def _load_checkpoint(self):
        """Load pretrained weights from checkpoint."""
        import os

        if not os.path.exists(self.checkpoint):
            logger.warning(f"Checkpoint not found: {self.checkpoint}")
            logger.warning("Please download the checkpoint from MMSegmentation model zoo")
            return

        checkpoint = torch.load(self.checkpoint, map_location='cpu')

        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Load backbone weights
        backbone_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('backbone.'):
                new_key = key.replace('backbone.', '')
                backbone_state_dict[new_key] = value

        missing_keys, unexpected_keys = self.backbone.load_state_dict(
            backbone_state_dict, strict=False
        )
        if missing_keys:
            logger.debug(f"Backbone missing keys: {missing_keys}")
        if unexpected_keys:
            logger.debug(f"Backbone unexpected keys: {unexpected_keys}")

        # Load decode head weights
        decode_head_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('decode_head.'):
                new_key = key.replace('decode_head.', '')
                decode_head_state_dict[new_key] = value

        missing_keys, unexpected_keys = self.decode_head.load_state_dict(
            decode_head_state_dict, strict=False
        )
        if missing_keys:
            logger.debug(f"Decode head missing keys: {missing_keys}")
        if unexpected_keys:
            logger.debug(f"Decode head unexpected keys: {unexpected_keys}")

        logger.info("Successfully loaded MMSeg checkpoint")

    def _resize_short_edge(self, images: torch.Tensor) -> torch.Tensor:
        """Resize images so that the short edge equals self.short_edge_size.

        Args:
            images: Input tensor [B, C, H, W]

        Returns:
            Resized tensor [B, C, H', W'] maintaining aspect ratio
        """
        return TF.resize(
            images,
            size=self.short_edge_size,
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias=True,
        )

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Apply MMSeg-style normalization.

        Args:
            images: Input tensor [B, C, H, W] with values in [0, 1], RGB order

        Returns:
            Normalized tensor in RGB order (MMSeg handles internally)
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
            images: Input tensor [B, C, H, W] with values in [0, 1], RGB order
                    at original resolution (no pre-resize needed)
            return_features: If True, also return intermediate features

        Returns:
            logits: Output logits [B, num_classes, H'/4, W'/4]
        """
        # Short-edge resize
        images = self._resize_short_edge(images)

        # Normalize images
        normalized = self.preprocess(images)

        # Forward through backbone
        features = self.backbone(normalized)

        # Forward through decode head
        logits = self.decode_head(features)

        if return_features:
            return logits, features
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
