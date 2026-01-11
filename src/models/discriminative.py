"""
Discriminative segmentation model wrapper.
Uses HuggingFace SegFormer for semantic segmentation.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


class DiscriminativeSegModel(nn.Module):
    """
    Wrapper for discriminative segmentation model (SegFormer).
    
    Handles:
    - Model loading from HuggingFace
    - Image preprocessing (short side to 512, ImageNet normalization)
    - Forward pass returning logits (4x downsampled)
    - TTA parameter configuration (which layers require gradients)
    """
    
    def __init__(
        self,
        model_name: str = "nvidia/segformer-b5-finetuned-ade-512-512",
        num_classes: int = 150,
        target_short_side: int = 512,
        update_norm_only: bool = False,
        device: str = "cuda:0",
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            num_classes: Number of segmentation classes
            target_short_side: Target size for the short side of input images
            update_norm_only: If True, only update normalization layers during TTA
            device: Device to place the model on
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.target_short_side = target_short_side
        self.update_norm_only = update_norm_only
        self.device = torch.device(device)
        
        # Load model and processor
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Set to eval mode
        self.model.eval()
        
        # ImageNet normalization parameters
        self.register_buffer(
            "mean", 
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        
    def preprocess(
        self, 
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess images for the segmentation model.
        
        Args:
            images: Input images tensor of shape (B, 3, H, W) in range [0, 255], RGB format
            
        Returns:
            Tuple of (preprocessed_images, original_size)
            - preprocessed_images: (B, 3, H', W') normalized and resized
            - original_size: (H, W) original image size
        """
        B, C, H, W = images.shape
        original_size = (H, W)
        
        # Resize: scale short side to target_short_side while maintaining aspect ratio
        if H < W:
            new_h = self.target_short_side
            new_w = int(W * self.target_short_side / H)
        else:
            new_w = self.target_short_side
            new_h = int(H * self.target_short_side / W)
        
        # Ensure dimensions are divisible by 4 (for 4x downsampling)
        new_h = (new_h // 4) * 4
        new_w = (new_w // 4) * 4
        
        # Resize
        images = F.interpolate(
            images, 
            size=(new_h, new_w), 
            mode="bilinear", 
            align_corners=False
        )
        
        # Normalize: [0, 255] -> [0, 1] -> ImageNet normalization
        images = images / 255.0
        images = (images - self.mean.to(images.device)) / self.std.to(images.device)
        
        return images, original_size
    
    def forward(
        self, 
        images: torch.Tensor,
        return_original_size: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[int, int]]]:
        """
        Forward pass through the segmentation model.
        
        Args:
            images: Input images tensor of shape (B, 3, H, W) in range [0, 255], RGB format
            return_original_size: If True, also return the preprocessed image size
            
        Returns:
            logits: Segmentation logits of shape (B, num_classes, H//4, W//4)
            (optional) original_size: Preprocessed image size (H', W')
        """
        # Move to device
        images = images.to(self.device)
        
        # Preprocess
        preprocessed, original_size = self.preprocess(images)
        
        # Forward pass
        outputs = self.model(pixel_values=preprocessed)
        logits = outputs.logits  # (B, num_classes, H//4, W//4)
        
        if return_original_size:
            return logits, original_size
        return logits
    
    def get_preprocessed_size(self, images: torch.Tensor) -> Tuple[int, int]:
        """
        Get the size of preprocessed images without running forward pass.
        
        Args:
            images: Input images tensor of shape (B, 3, H, W)
            
        Returns:
            (H', W'): Size after preprocessing
        """
        B, C, H, W = images.shape
        
        if H < W:
            new_h = self.target_short_side
            new_w = int(W * self.target_short_side / H)
        else:
            new_w = self.target_short_side
            new_h = int(H * self.target_short_side / W)
        
        new_h = (new_h // 4) * 4
        new_w = (new_w // 4) * 4
        
        return (new_h, new_w)
    
    def configure_tta_params(self) -> None:
        """
        Configure which parameters require gradients for TTA.
        
        If update_norm_only is True, only normalization layers will have gradients.
        Otherwise, all parameters will have gradients.
        """
        # First, disable all gradients
        self.model.requires_grad_(False)
        
        if self.update_norm_only:
            # Enable gradients only for normalization layers
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
                    module.requires_grad_(True)
        else:
            # Enable all gradients
            self.model.requires_grad_(True)
    
    def reset_model(self, initial_state_dict: Dict[str, torch.Tensor]) -> None:
        """
        Reset model to initial state.
        
        Args:
            initial_state_dict: State dict to reset to
        """
        self.model.load_state_dict(initial_state_dict)
        self.model.eval()
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """
        Get list of parameters that require gradients.
        
        Returns:
            List of trainable parameters
        """
        return [p for p in self.model.parameters() if p.requires_grad]
    
    def postprocess_logits(
        self,
        logits: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Upsample logits to target size for evaluation.
        
        Args:
            logits: Logits tensor of shape (B, C, H, W)
            target_size: Target size (H, W) for upsampling
            
        Returns:
            Upsampled logits of shape (B, C, target_H, target_W)
        """
        return F.interpolate(
            logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
    
    def get_predictions(
        self,
        logits: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Get class predictions from logits.
        
        Args:
            logits: Logits tensor of shape (B, C, H, W)
            target_size: Optional target size for upsampling before argmax
            
        Returns:
            Predictions of shape (B, H, W) with class indices
        """
        if target_size is not None:
            logits = self.postprocess_logits(logits, target_size)
        return logits.argmax(dim=1)
