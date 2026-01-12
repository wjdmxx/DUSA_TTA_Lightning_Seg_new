"""Data transforms for image preprocessing."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class ShortEdgeResize:
    """Resize image by scaling the short edge to a target size.

    This transform handles both landscape and portrait images,
    always scaling the shorter dimension to the specified size
    while maintaining aspect ratio.
    """

    def __init__(self, size: int = 512):
        """Initialize the transform.

        Args:
            size: Target size for the short edge
        """
        self.size = size

    def __call__(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply the transform.

        Args:
            image: Image tensor of shape [C, H, W] or [B, C, H, W]
            mask: Optional mask tensor of shape [H, W] or [B, H, W]

        Returns:
            Tuple of (resized_image, resized_mask)
        """
        # Handle both batched and unbatched inputs
        squeeze_batch = False
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_batch = True
            if mask is not None:
                mask = mask.unsqueeze(0)

        B, C, H, W = image.shape

        # Calculate new dimensions based on short edge
        if H < W:
            # Portrait or square - H is short edge
            new_h = self.size
            new_w = int(W * self.size / H)
        else:
            # Landscape - W is short edge
            new_w = self.size
            new_h = int(H * self.size / W)

        # Resize image with bilinear interpolation
        resized_image = F.interpolate(
            image,
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False,
        )

        # Resize mask with nearest neighbor interpolation
        resized_mask = None
        if mask is not None:
            mask_float = mask.unsqueeze(1).float()
            resized_mask = F.interpolate(
                mask_float,
                size=(new_h, new_w),
                mode='nearest',
            ).squeeze(1).long()

        # Remove batch dimension if it was added
        if squeeze_batch:
            resized_image = resized_image.squeeze(0)
            if resized_mask is not None:
                resized_mask = resized_mask.squeeze(0)

        return resized_image, resized_mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class ImageNetNormalize:
    """Normalize image using ImageNet mean and std."""

    # ImageNet statistics
    MEAN = torch.tensor([0.485, 0.456, 0.406])
    STD = torch.tensor([0.229, 0.224, 0.225])

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet normalization.

        Args:
            image: Image tensor of shape [C, H, W] or [B, C, H, W]
                   Values should be in range [0, 1]

        Returns:
            Normalized image tensor
        """
        squeeze_batch = False
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_batch = True

        # Move mean and std to same device
        mean = self.MEAN.to(image.device).view(1, 3, 1, 1)
        std = self.STD.to(image.device).view(1, 3, 1, 1)

        # Normalize
        normalized = (image - mean) / std

        if squeeze_batch:
            normalized = normalized.squeeze(0)

        return normalized


class DiffusionNormalize:
    """Normalize image to [-1, 1] range for diffusion models."""

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize to [-1, 1] range.

        Args:
            image: Image tensor with values in [0, 1]

        Returns:
            Normalized image tensor with values in [-1, 1]
        """
        return image * 2.0 - 1.0


class DiffusionDenormalize:
    """Denormalize image from [-1, 1] to [0, 1] range."""

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Denormalize from [-1, 1] to [0, 1] range.

        Args:
            image: Image tensor with values in [-1, 1]

        Returns:
            Denormalized image tensor with values in [0, 1]
        """
        return (image + 1.0) / 2.0
