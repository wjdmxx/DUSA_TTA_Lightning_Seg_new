"""
Sliding window processor for TTA inference.
Handles both input images and segmentation logits with synchronized sliding.
"""

from typing import Tuple, List
import torch
import torch.nn.functional as F


class SlidingWindowProcessor:
    """
    Sliding window processor for handling large images in TTA.
    
    Synchronously slides over input images and their corresponding logits,
    accounting for the 4x downsampling of segmentation outputs.
    """
    
    def __init__(
        self,
        crop_size: Tuple[int, int] = (512, 512),
        stride: Tuple[int, int] = (0, 171),
        logits_downsample_ratio: int = 4,
    ):
        """
        Args:
            crop_size: (H, W) size of each sliding window crop
            stride: (H_stride, W_stride) step size for sliding. 
                    0 means no sliding in that dimension.
            logits_downsample_ratio: The ratio between input size and logits size (default 4)
        """
        self.crop_size = crop_size
        self.stride = stride
        self.logits_downsample_ratio = logits_downsample_ratio
        
    def compute_grid(self, img_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        Compute the number of grid cells in each dimension.
        
        Args:
            img_size: (H, W) of the input image
            
        Returns:
            (h_grids, w_grids): Number of windows in each dimension
        """
        h_img, w_img = img_size
        h_crop, w_crop = self.crop_size
        h_stride, w_stride = self.stride
        
        # If stride is 0, only one window in that dimension
        h_grids = 1 if h_stride == 0 else max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = 1 if w_stride == 0 else max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        
        return h_grids, w_grids
    
    def get_window_coords(
        self, 
        h_idx: int, 
        w_idx: int, 
        img_size: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Get the coordinates of a specific window.
        
        Args:
            h_idx: Height index of the window
            w_idx: Width index of the window
            img_size: (H, W) of the input image
            
        Returns:
            (y1, y2, x1, x2): Window coordinates
        """
        h_img, w_img = img_size
        h_crop, w_crop = self.crop_size
        h_stride, w_stride = self.stride
        
        y1 = h_idx * h_stride
        x1 = w_idx * w_stride
        y2 = min(y1 + h_crop, h_img)
        x2 = min(x1 + w_crop, w_img)
        
        # Adjust to ensure crop size is maintained at boundaries
        y1 = max(y2 - h_crop, 0)
        x1 = max(x2 - w_crop, 0)
        
        return y1, y2, x1, x2
    
    def slide_image(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract sliding windows from input tensor.
        
        Args:
            inputs: Input tensor of shape (B, C, H, W)
            
        Returns:
            List of cropped tensors, each of shape (B, C, crop_h, crop_w)
        """
        _, _, h_img, w_img = inputs.shape
        h_grids, w_grids = self.compute_grid((h_img, w_img))
        
        windows = []
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1, y2, x1, x2 = self.get_window_coords(h_idx, w_idx, (h_img, w_img))
                crop = inputs[:, :, y1:y2, x1:x2]
                windows.append(crop)
                
        return windows
    
    def slide_image_and_logits(
        self, 
        inputs: torch.Tensor, 
        logits: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Synchronously slide over both inputs and logits.
        
        Args:
            inputs: Input image tensor of shape (B, 3, H, W)
            logits: Logits tensor of shape (B, C, H//4, W//4)
            
        Returns:
            List of (input_crop, logits_crop) tuples
        """
        _, _, h_img, w_img = inputs.shape
        h_grids, w_grids = self.compute_grid((h_img, w_img))
        
        ratio = self.logits_downsample_ratio
        h_crop, w_crop = self.crop_size
        logits_crop_size = (h_crop // ratio, w_crop // ratio)
        
        windows = []
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                # Image coordinates
                y1, y2, x1, x2 = self.get_window_coords(h_idx, w_idx, (h_img, w_img))
                input_crop = inputs[:, :, y1:y2, x1:x2]
                
                # Logits coordinates (scaled by downsample ratio)
                ly1, ly2 = y1 // ratio, y2 // ratio
                lx1, lx2 = x1 // ratio, x2 // ratio
                
                # Ensure logits crop size matches expected size
                ly1 = max(ly2 - logits_crop_size[0], 0)
                lx1 = max(lx2 - logits_crop_size[1], 0)
                
                logits_crop = logits[:, :, ly1:ly2, lx1:lx2]
                
                windows.append((input_crop, logits_crop))
                
        return windows
    
    def get_num_windows(self, img_size: Tuple[int, int]) -> int:
        """
        Get the total number of windows for a given image size.
        
        Args:
            img_size: (H, W) of the input image
            
        Returns:
            Total number of windows
        """
        h_grids, w_grids = self.compute_grid(img_size)
        return h_grids * w_grids


def create_sliding_processor(
    crop_size: Tuple[int, int] = (512, 512),
    stride: Tuple[int, int] = (0, 171),
) -> SlidingWindowProcessor:
    """
    Factory function to create a sliding window processor.
    
    Args:
        crop_size: Size of each window
        stride: Step size for sliding (0 means no sliding in that dimension)
        
    Returns:
        Configured SlidingWindowProcessor instance
    """
    return SlidingWindowProcessor(crop_size=crop_size, stride=stride)
