"""Metrics utilities for semantic segmentation evaluation."""

import torch
from torchmetrics import JaccardIndex, Metric
from torchmetrics.classification import MulticlassAccuracy
from typing import Dict, Optional, Tuple


class SegmentationMetrics:
    """Wrapper class for computing segmentation metrics.
    
    Computes mIoU, per-class IoU, and accuracy using torchmetrics.
    """
    
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        device: Optional[torch.device] = None
    ):
        """Initialize segmentation metrics.
        
        Args:
            num_classes: Number of semantic classes
            ignore_index: Label value to ignore (typically 255 for void)
            device: Device to place metrics on
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Mean IoU (Jaccard Index)
        self.miou = JaccardIndex(
            task='multiclass',
            num_classes=num_classes,
            ignore_index=ignore_index,
            average='macro'
        ).to(self.device)
        
        # Per-class IoU
        self.per_class_iou = JaccardIndex(
            task='multiclass',
            num_classes=num_classes,
            ignore_index=ignore_index,
            average='none'
        ).to(self.device)
        
        # Overall accuracy
        self.accuracy = MulticlassAccuracy(
            num_classes=num_classes,
            ignore_index=ignore_index,
            average='micro'
        ).to(self.device)
    
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metrics with new predictions.
        
        Args:
            preds: Predictions tensor with shape (B, H, W) containing class indices
                   or (B, C, H, W) containing logits/probabilities
            target: Ground truth tensor with shape (B, H, W) containing class indices
        """
        # Convert logits to predictions if needed
        if preds.dim() == 4:
            preds = preds.argmax(dim=1)
        
        # Ensure tensors are on the correct device
        preds = preds.to(self.device)
        target = target.to(self.device)
        
        self.miou.update(preds, target)
        self.per_class_iou.update(preds, target)
        self.accuracy.update(preds, target)
    
    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute all metrics.
        
        Returns:
            Dictionary containing 'mIoU', 'per_class_iou', and 'accuracy'
        """
        return {
            'mIoU': self.miou.compute(),
            'per_class_iou': self.per_class_iou.compute(),
            'accuracy': self.accuracy.compute()
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.miou.reset()
        self.per_class_iou.reset()
        self.accuracy.reset()
    
    def to(self, device: torch.device) -> 'SegmentationMetrics':
        """Move metrics to specified device."""
        self.device = device
        self.miou = self.miou.to(device)
        self.per_class_iou = self.per_class_iou.to(device)
        self.accuracy = self.accuracy.to(device)
        return self


def compute_miou(
    preds: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255
) -> torch.Tensor:
    """Compute mean Intersection over Union (mIoU).
    
    Args:
        preds: Predictions tensor with shape (B, H, W) or (B, C, H, W)
        target: Ground truth tensor with shape (B, H, W)
        num_classes: Number of semantic classes
        ignore_index: Label value to ignore
        
    Returns:
        mIoU value as a scalar tensor
    """
    # Convert logits to predictions if needed
    if preds.dim() == 4:
        preds = preds.argmax(dim=1)
    
    # Ensure same device
    device = preds.device
    target = target.to(device)
    
    # Create valid mask
    valid_mask = target != ignore_index
    
    # Initialize confusion matrix components
    intersection = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)
    
    for cls in range(num_classes):
        pred_cls = (preds == cls) & valid_mask
        target_cls = (target == cls) & valid_mask
        
        intersection[cls] = (pred_cls & target_cls).sum().float()
        union[cls] = (pred_cls | target_cls).sum().float()
    
    # Compute IoU for each class (avoiding division by zero)
    iou = torch.zeros(num_classes, device=device)
    valid_classes = union > 0
    iou[valid_classes] = intersection[valid_classes] / union[valid_classes]
    
    # Mean IoU over valid classes
    if valid_classes.sum() > 0:
        miou = iou[valid_classes].mean()
    else:
        miou = torch.tensor(0.0, device=device)
    
    return miou


def compute_per_class_iou(
    preds: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-class IoU.
    
    Args:
        preds: Predictions tensor with shape (B, H, W) or (B, C, H, W)
        target: Ground truth tensor with shape (B, H, W)
        num_classes: Number of semantic classes
        ignore_index: Label value to ignore
        
    Returns:
        Tuple of (per_class_iou, valid_class_mask)
    """
    # Convert logits to predictions if needed
    if preds.dim() == 4:
        preds = preds.argmax(dim=1)
    
    device = preds.device
    target = target.to(device)
    valid_mask = target != ignore_index
    
    iou = torch.zeros(num_classes, device=device)
    valid_classes = torch.zeros(num_classes, dtype=torch.bool, device=device)
    
    for cls in range(num_classes):
        pred_cls = (preds == cls) & valid_mask
        target_cls = (target == cls) & valid_mask
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union > 0:
            iou[cls] = intersection / union
            valid_classes[cls] = True
    
    return iou, valid_classes
