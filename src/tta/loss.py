"""Loss computation for TTA."""

from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange


class TTALoss:
    """TTA Loss computation using MSE with probability weighting.

    This implements the flow matching loss for TTA:
    - Target: velocity = noise - latent
    - Prediction: weighted sum of per-class predictions
    - Loss: MSE(weighted_prediction, target)
    """

    def __init__(
        self,
        reduction: str = "mean",
    ):
        """Initialize TTA loss.

        Args:
            reduction: Loss reduction method ("mean", "sum", "none")
        """
        self.reduction = reduction

    def compute(
        self,
        pred_velocity: torch.Tensor,
        target: torch.Tensor,
        probs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute probability-weighted MSE loss.

        Args:
            pred_velocity: [B*K, C, H, W] per-class velocity predictions
            target: [B, C, H, W] target velocity
            probs: [B, K, H, W] class probabilities
            mask: Optional [B, H, W] mask for valid pixels

        Returns:
            Loss tensor
        """
        B = probs.shape[0]
        K = probs.shape[1]

        # Reshape predictions: [B*K, C, H, W] -> [B, K, C, H, W]
        pred = rearrange(pred_velocity, "(b k) c h w -> b k c h w", b=B, k=K)

        # Compute weighted prediction: [B, K, H, W] x [B, K, C, H, W] -> [B, C, H, W]
        weighted_pred = torch.einsum("b k h w, b k c h w -> b c h w", probs, pred)

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, H, W]
            weighted_pred = weighted_pred * mask
            target = target * mask
            # Adjust reduction for masked loss
            if self.reduction == "mean":
                num_valid = mask.sum()
                if num_valid > 0:
                    return F.mse_loss(weighted_pred, target, reduction="sum") / num_valid
                return torch.tensor(0.0, device=weighted_pred.device)

        return F.mse_loss(weighted_pred, target, reduction=self.reduction)

    def compute_from_gathered(
        self,
        gathered_pred: torch.Tensor,
        target: torch.Tensor,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss from already-gathered predictions.

        Args:
            gathered_pred: [B, K, C, H, W] gathered predictions for topk classes
            target: [B, C, H, W] target velocity
            probs: [B, K, H, W] class probabilities

        Returns:
            Loss tensor
        """
        # Compute weighted prediction
        weighted_pred = torch.einsum(
            "b k h w, b k c h w -> b c h w", probs, gathered_pred
        )

        return F.mse_loss(weighted_pred, target, reduction=self.reduction)


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute softmax entropy of logits.

    Args:
        logits: [B, C, H, W] logits tensor

    Returns:
        Entropy [B, H, W]
    """
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    entropy = -(probs * log_probs).sum(dim=1)
    return entropy


def confidence_mask(
    logits: torch.Tensor,
    threshold: float = 0.9,
) -> torch.Tensor:
    """Create mask for high-confidence predictions.

    Args:
        logits: [B, C, H, W] logits tensor
        threshold: Confidence threshold

    Returns:
        Mask [B, H, W] where 1 = high confidence
    """
    probs = F.softmax(logits, dim=1)
    max_prob = probs.max(dim=1)[0]
    return (max_prob >= threshold).float()
