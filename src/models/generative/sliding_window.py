"""Efficient sliding window processor using tensor operations."""

import logging
from typing import List, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SlidingWindowProcessor:
    """Efficient sliding window processing for images and logits.

    This processor handles sliding window operations using tensor operations
    (torch.unfold) instead of explicit loops for better performance.

    The sliding is done along the LONG edge of the image, with window size
    matching the SHORT edge (which should be 512).
    """

    def __init__(
        self,
        window_size: int = 512,
        stride: int = 171,
    ):
        """Initialize the sliding window processor.

        Args:
            window_size: Size of the sliding window (should match short edge)
            stride: Stride between windows
        """
        self.window_size = window_size
        self.stride = stride

    def compute_num_windows(
        self,
        long_edge: int,
    ) -> int:
        """Compute number of windows for a given long edge size.

        Args:
            long_edge: Length of the long edge

        Returns:
            Number of windows
        """
        if long_edge <= self.window_size:
            return 1
        return (long_edge - self.window_size) // self.stride + 1

    def slide_images(
        self,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Tuple[int, int, int, int]], bool]:
        """Apply sliding window to images.

        Args:
            images: Input tensor [B, C, H, W]

        Returns:
            Tuple of:
            - windows: [B * num_windows, C, window_size, window_size]
            - positions: List of (y1, x1, y2, x2) for each window
            - is_vertical: True if sliding along H (vertical), False if along W
        """
        B, C, H, W = images.shape

        # Determine slide direction based on image orientation
        is_vertical = H > W  # Slide along H if image is taller

        if is_vertical:
            # Vertical image: slide along H, width is fixed at window_size
            num_windows = self.compute_num_windows(H)
            windows, positions = self._slide_vertical(images, num_windows)
        else:
            # Horizontal image: slide along W, height is fixed at window_size
            num_windows = self.compute_num_windows(W)
            windows, positions = self._slide_horizontal(images, num_windows)

        return windows, positions, is_vertical

    def _slide_vertical(
        self,
        images: torch.Tensor,
        num_windows: int,
    ) -> Tuple[torch.Tensor, List[Tuple[int, int, int, int]]]:
        """Slide along vertical (H) dimension.

        Args:
            images: [B, C, H, W]
            num_windows: Number of windows

        Returns:
            windows: [B * num_windows, C, window_size, W]
            positions: List of (y1, x1, y2, x2)
        """
        B, C, H, W = images.shape
        windows_list = []
        positions = []

        for i in range(num_windows):
            y1 = i * self.stride
            y2 = y1 + self.window_size

            # Handle last window - ensure it doesn't exceed image bounds
            if y2 > H:
                y2 = H
                y1 = H - self.window_size

            window = images[:, :, y1:y2, :]
            windows_list.append(window)
            positions.append((y1, 0, y2, W))

        # Stack windows: [num_windows, B, C, H, W] -> [B * num_windows, C, H, W]
        windows = torch.cat(windows_list, dim=0)

        return windows, positions

    def _slide_horizontal(
        self,
        images: torch.Tensor,
        num_windows: int,
    ) -> Tuple[torch.Tensor, List[Tuple[int, int, int, int]]]:
        """Slide along horizontal (W) dimension.

        Args:
            images: [B, C, H, W]
            num_windows: Number of windows

        Returns:
            windows: [B * num_windows, C, H, window_size]
            positions: List of (y1, x1, y2, x2)
        """
        B, C, H, W = images.shape
        windows_list = []
        positions = []

        for i in range(num_windows):
            x1 = i * self.stride
            x2 = x1 + self.window_size

            # Handle last window
            if x2 > W:
                x2 = W
                x1 = W - self.window_size

            window = images[:, :, :, x1:x2]
            windows_list.append(window)
            positions.append((0, x1, H, x2))

        windows = torch.cat(windows_list, dim=0)

        return windows, positions

    def slide_logits(
        self,
        logits: torch.Tensor,
        is_vertical: bool,
        logits_scale: int = 4,
    ) -> Tuple[torch.Tensor, List[Tuple[int, int, int, int]]]:
        """Apply sliding window to logits (which are downsampled).

        Args:
            logits: Input tensor [B, C, H/scale, W/scale]
            is_vertical: Whether to slide vertically
            logits_scale: Downsampling factor of logits

        Returns:
            Tuple of (windows, positions)
        """
        B, C, H, W = logits.shape

        # Adjust window size and stride for downsampled logits
        scaled_window = self.window_size // logits_scale
        scaled_stride = self.stride // logits_scale

        if is_vertical:
            num_windows = self.compute_num_windows(H * logits_scale)
            windows_list = []
            positions = []

            for i in range(num_windows):
                y1 = i * scaled_stride
                y2 = y1 + scaled_window

                if y2 > H:
                    y2 = H
                    y1 = max(0, H - scaled_window)

                window = logits[:, :, y1:y2, :]
                windows_list.append(window)
                positions.append((y1, 0, y2, W))

            windows = torch.cat(windows_list, dim=0)
        else:
            num_windows = self.compute_num_windows(W * logits_scale)
            windows_list = []
            positions = []

            for i in range(num_windows):
                x1 = i * scaled_stride
                x2 = x1 + scaled_window

                if x2 > W:
                    x2 = W
                    x1 = max(0, W - scaled_window)

                window = logits[:, :, :, x1:x2]
                windows_list.append(window)
                positions.append((0, x1, H, x2))

            windows = torch.cat(windows_list, dim=0)

        return windows, positions

    def slide_batch(
        self,
        images: torch.Tensor,
        logits: torch.Tensor,
        logits_scale: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Apply sliding window to both images and logits.

        Args:
            images: [B, C, H, W] - Original images (short edge = 512)
            logits: [B, num_classes, H/4, W/4] - Downsampled logits
            logits_scale: Downsampling factor of logits

        Returns:
            Tuple of:
            - image_windows: [B * num_windows, C, window_size, window_size]
            - logit_windows: [B * num_windows, num_classes, window_size/4, window_size/4]
            - num_windows: Number of windows
        """
        # Slide images
        image_windows, positions, is_vertical = self.slide_images(images)

        # Slide logits with matching positions
        logit_windows, _ = self.slide_logits(logits, is_vertical, logits_scale)

        num_windows = len(positions)

        return image_windows, logit_windows, num_windows


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
