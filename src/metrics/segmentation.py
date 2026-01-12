"""Segmentation metrics using torchmetrics."""

import logging
from typing import Any, Dict, Optional

import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassJaccardIndex,
    MulticlassPrecision,
    MulticlassRecall,
)

logger = logging.getLogger(__name__)


class SegmentationMetrics:
    """Segmentation metrics wrapper using torchmetrics.

    Primary metric: mIoU (mean Intersection over Union)
    Additional metrics: Accuracy, F1, Precision, Recall
    """

    def __init__(
        self,
        num_classes: int = 150,
        ignore_index: int = 255,
        device: str = "cuda",
    ):
        """Initialize segmentation metrics.

        Args:
            num_classes: Number of classes
            ignore_index: Index to ignore in evaluation
            device: Device for metrics
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device

        # Create metric collection
        self.metrics = MetricCollection(
            {
                "mIoU": MulticlassJaccardIndex(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
                "Accuracy": MulticlassAccuracy(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="micro",
                ),
                "F1": MulticlassF1Score(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
                "Precision": MulticlassPrecision(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
                "Recall": MulticlassRecall(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
            }
        ).to(device)

        # Per-class IoU for detailed analysis
        self.per_class_iou = MulticlassJaccardIndex(
            num_classes=num_classes,
            ignore_index=ignore_index,
            average=None,
        ).to(device)

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Update metrics with new predictions.

        Args:
            logits: [B, C, H, W] logits or [B, H, W] predictions
            targets: [B, H, W] ground truth labels
        """
        # Get predictions if logits are provided
        if logits.dim() == 4:
            preds = logits.argmax(dim=1)
        else:
            preds = logits

        # Flatten for metrics
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)

        # Update metrics
        self.metrics.update(preds_flat, targets_flat)
        self.per_class_iou.update(preds_flat, targets_flat)

    def compute(self) -> Dict[str, float]:
        """Compute all metrics.

        Returns:
            Dictionary of metric names and values
        """
        results = {}
        computed = self.metrics.compute()
        for name, value in computed.items():
            results[name] = value.item()
        return results

    def compute_per_class_iou(self) -> Dict[int, float]:
        """Compute per-class IoU.

        Returns:
            Dictionary mapping class index to IoU
        """
        iou_values = self.per_class_iou.compute()
        return {i: iou.item() for i, iou in enumerate(iou_values)}

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.reset()
        self.per_class_iou.reset()

    def get_primary_metric(self) -> float:
        """Get the primary metric (mIoU).

        Returns:
            mIoU value
        """
        return self.compute()["mIoU"]

    def to(self, device: str) -> "SegmentationMetrics":
        """Move metrics to device.

        Args:
            device: Target device

        Returns:
            Self for chaining
        """
        self.device = device
        self.metrics = self.metrics.to(device)
        self.per_class_iou = self.per_class_iou.to(device)
        return self


def compute_miou_batch(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 150,
    ignore_index: int = 255,
) -> float:
    """Compute mIoU for a single batch.

    Args:
        logits: [B, C, H, W] logits
        targets: [B, H, W] ground truth
        num_classes: Number of classes
        ignore_index: Index to ignore

    Returns:
        mIoU value
    """
    preds = logits.argmax(dim=1)
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    # Create mask for valid pixels
    valid = targets_flat != ignore_index

    # Compute per-class IoU
    ious = []
    for c in range(num_classes):
        pred_c = (preds_flat == c) & valid
        target_c = (targets_flat == c) & valid

        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()

        if union > 0:
            ious.append(intersection / union)

    if len(ious) == 0:
        return 0.0

    return torch.stack(ious).mean().item()
