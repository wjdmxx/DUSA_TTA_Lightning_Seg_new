"""2D sliding window processor for images and logits.

Slides in both H and W directions (like a convolution kernel),
ensuring every pixel is covered.
"""

import logging
import math
from typing import List, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_window_positions(dim: int, window_size: int, stride: int) -> List[int]:
    """Compute starting positions for sliding windows along one dimension.

    Ensures every pixel is covered. If the last window would go past the edge,
    it is shifted back so it ends exactly at the edge.

    Examples:
        - dim=600, window=300, stride=300 → [0, 300]
        - dim=601, window=300, stride=300 → [0, 300, 301]
        - dim=300, window=300, stride=300 → [0]
        - dim=512, window=512, stride=512 → [0]
        - dim=683, window=512, stride=512 → [0, 171]

    Args:
        dim: Size of the dimension
        window_size: Size of the sliding window
        stride: Stride between windows

    Returns:
        List of starting positions
    """
    if dim <= window_size:
        return [0]

    positions = []
    pos = 0
    while pos + window_size < dim:
        positions.append(pos)
        pos += stride

    # Last window: shift back to ensure it ends at dim
    positions.append(dim - window_size)

    return positions


class SlidingWindowProcessor:
    """2D sliding window processing for images and logits.

    Slides in both H and W directions with the given stride,
    producing square windows of size (window_size x window_size).
    """

    def __init__(
        self,
        window_size: int = 512,
        stride: int = 512,
    ):
        """Initialize the sliding window processor.

        Args:
            window_size: Size of the sliding window (square)
            stride: Stride between windows (same for H and W)
        """
        self.window_size = window_size
        self.stride = stride

    def slide(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """Apply 2D sliding window to a tensor.

        Args:
            tensor: Input tensor [B, C, H, W]

        Returns:
            Tuple of:
            - windows: [B * num_windows, C, window_size, window_size]
            - positions: List of (y, x) top-left corner for each window
        """
        B, C, H, W = tensor.shape

        h_positions = compute_window_positions(H, self.window_size, self.stride)
        w_positions = compute_window_positions(W, self.window_size, self.stride)

        windows = []
        positions = []
        for y in h_positions:
            for x in w_positions:
                window = tensor[:, :, y:y + self.window_size, x:x + self.window_size]
                windows.append(window)
                positions.append((y, x))

        # [num_windows * B, C, window_size, window_size]
        windows = torch.cat(windows, dim=0)

        return windows, positions

    def slide_pair(
        self,
        images: torch.Tensor,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
        """Apply 2D sliding window to both images and logits.

        logits must have the same spatial dimensions as images
        (caller is responsible for resizing logits to match images first).

        Args:
            images: [B, C_img, H, W] — images at the generative model's resolution
            logits: [B, C_logits, H, W] — logits resized to match images' spatial dims

        Returns:
            Tuple of:
            - image_windows: [B * Nw, C_img, window_size, window_size]
            - logit_windows: [B * Nw, C_logits, window_size, window_size]
            - positions: List of (y, x) for each window
        """
        assert images.shape[2:] == logits.shape[2:], (
            f"Spatial dims must match: images {images.shape[2:]} vs logits {logits.shape[2:]}"
        )

        image_windows, positions = self.slide(images)
        logit_windows, _ = self.slide(logits)

        return image_windows, logit_windows, positions


def downsample_logits_to_latent(
    logits: torch.Tensor,
    latent_size: Tuple[int, int],
) -> torch.Tensor:
    """Downsample logits to match latent space resolution.

    Args:
        logits: [B, C, H, W] logits tensor
        latent_size: (H, W) target size

    Returns:
        Downsampled logits [B, C, latent_H, latent_W]
    """
    return F.interpolate(
        logits,
        size=latent_size,
        mode="bilinear",
        align_corners=False,
    )
