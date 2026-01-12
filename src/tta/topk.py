"""TopK class selection for TTA."""

from typing import Tuple

import torch
import torch.nn.functional as F


class TopKSelector:
    """Select top-K classes from segmentation logits.

    This class implements the topK selection logic used in TTA,
    which selects the K most likely classes for each pixel position.
    """

    def __init__(
        self,
        topk: int = 1,
        classes_threshold: int = 20,
        temperature: float = 1.0,
    ):
        """Initialize TopK selector.

        Args:
            topk: Number of top classes to select per pixel
            classes_threshold: Maximum number of unique classes
            temperature: Temperature for softmax (default: 1.0)
        """
        self.topk = topk
        self.classes_threshold = classes_threshold
        self.temperature = temperature

    def select(
        self,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select top-K classes from logits.

        Args:
            logits: [B, num_classes, H, W] logits tensor

        Returns:
            Tuple of:
            - topk_probs: [B, K, H, W] softmax probabilities for top-K
            - topk_idx: [B, K, H, W] class indices for top-K
            - unique_classes: [N] unique class indices
        """
        B, C, H, W = logits.shape
        device = logits.device

        # Get top-K values and indices
        topk_logits, topk_idx = torch.topk(logits, self.topk, dim=1)

        # Get unique classes across the batch
        unique_classes = topk_idx.unique()

        # Limit number of unique classes
        if unique_classes.shape[0] > self.classes_threshold:
            unique_classes = self._limit_classes(
                logits, topk_idx, unique_classes
            )

        # Compute softmax probabilities
        topk_probs = F.softmax(topk_logits / self.temperature, dim=1)

        return topk_probs, topk_idx, unique_classes

    def _limit_classes(
        self,
        logits: torch.Tensor,
        topk_idx: torch.Tensor,
        unique_classes: torch.Tensor,
    ) -> torch.Tensor:
        """Limit the number of unique classes.

        Priority:
        1. Always include classes that appear as top-1
        2. Fill remaining with random selection from other classes

        Args:
            logits: [B, num_classes, H, W]
            topk_idx: [B, K, H, W]
            unique_classes: [N] all unique classes

        Returns:
            Limited unique classes [M] where M <= classes_threshold
        """
        device = logits.device

        # Get top-1 classes
        top1_idx = logits.argmax(dim=1)  # [B, H, W]
        top1_unique = top1_idx.unique()

        if top1_unique.shape[0] >= self.classes_threshold:
            # Too many top-1 classes, randomly sample
            perm = torch.randperm(top1_unique.shape[0], device=device)
            return top1_unique[perm[: self.classes_threshold]]

        # Keep all top-1, fill remaining with other classes
        other_mask = ~torch.isin(unique_classes, top1_unique)
        other_classes = unique_classes[other_mask]

        remaining = self.classes_threshold - top1_unique.shape[0]
        if remaining > 0 and other_classes.shape[0] > 0:
            perm = torch.randperm(other_classes.shape[0], device=device)
            selected = other_classes[perm[:remaining]]
            return torch.cat([top1_unique, selected])

        return top1_unique

    def create_class_mapping(
        self,
        unique_classes: torch.Tensor,
        num_total_classes: int,
    ) -> torch.Tensor:
        """Create mapping from class index to position in unique_classes.

        Args:
            unique_classes: [N] unique class indices
            num_total_classes: Total number of classes

        Returns:
            Mapping tensor [num_total_classes] where mapping[class_idx] = position
        """
        device = unique_classes.device
        mapping = torch.zeros(num_total_classes, dtype=torch.long, device=device)
        mapping[unique_classes] = torch.arange(
            unique_classes.shape[0], device=device
        )
        return mapping
