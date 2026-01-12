"""Combined model that unifies discriminative and generative models for TTA."""

from typing import Dict, Tuple, Optional
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from .discriminative import SegformerModel, resize_image_short_edge
from .generative import SD3GenerativeModel


class CombinedModel(nn.Module):
    """Combined model for Test-Time Adaptation.

    This class combines:
    - Discriminative model (Segformer) for semantic segmentation
    - Generative model (SD3) for providing TTA loss signal

    The forward pass:
    1. Preprocesses input image (resize short edge to 512)
    2. Runs discriminative model to get segmentation logits
    3. Runs generative model with logits to compute TTA loss
    4. Returns loss for backpropagation
    """

    def __init__(
        self,
        discriminative_cfg: DictConfig,
        generative_cfg: DictConfig,
        loss_cfg: DictConfig,
        update_cfg: DictConfig,
        forward_mode: str = "tta"
    ):
        """Initialize combined model.

        Args:
            discriminative_cfg: Config for Segformer model
            generative_cfg: Config for SD3 model
            loss_cfg: Config for loss computation (topk, etc.)
            update_cfg: Config for which models to update
            forward_mode: "tta" for full TTA, "discriminative_only" for baseline
        """
        super().__init__()
        self.forward_mode = forward_mode
        self.update_cfg = update_cfg
        self.loss_cfg = loss_cfg

        # Target short edge for initial resize
        self.target_short_edge = discriminative_cfg.preprocessing.target_short_edge

        # Initialize discriminative model
        print("Initializing discriminative model (Segformer)...")
        self.discriminative = SegformerModel(discriminative_cfg)

        # Initialize generative model only if needed
        if forward_mode == "tta":
            print("Initializing generative model (SD3)...")
            # Create a copy of generative_cfg to avoid modifying the original
            gen_cfg = OmegaConf.to_container(generative_cfg, resolve=True)
            gen_cfg = OmegaConf.create(gen_cfg)
            # Pass loss config to generative model
            gen_cfg.topk = loss_cfg.topk
            gen_cfg.temperature = loss_cfg.temperature
            gen_cfg.classes_threshold = loss_cfg.classes_threshold
            self.generative = SD3GenerativeModel(gen_cfg)
        else:
            self.generative = None
            print("Forward mode is 'discriminative_only', skipping generative model")

    def forward(
        self,
        images: torch.Tensor,
        return_predictions: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through combined model.

        Args:
            images: Input images of shape (B, C, H, W) in [0, 255] RGB format
                   Note: images should already be resized by dataloader (short edge = 512)
            return_predictions: Whether to return segmentation predictions

        Returns:
            Dictionary containing:
            - "loss": TTA loss (or None if discriminative_only mode)
            - "logits": Segmentation logits at 1/4 resolution
            - "predictions": Segmentation predictions at input resolution (if return_predictions=True)
        """
        # Store input size (already resized by dataloader)
        input_size = (images.shape[2], images.shape[3])

        # Preprocess for discriminative model (normalize)
        disc_input = self.discriminative.preprocess(images)

        # Get segmentation logits (at 1/4 resolution)
        logits = self.discriminative(disc_input)

        result = {"logits": logits}

        # Compute TTA loss if in TTA mode
        if self.forward_mode == "tta" and self.generative is not None:
            # Pass images (in original scale [0,255]) and logits to generative model
            loss = self.generative(images, logits)
            result["loss"] = loss
        else:
            # No loss in discriminative_only mode
            result["loss"] = None

        # Get predictions if requested
        if return_predictions:
            # Upsample logits to input image size (which is the same as label size after dataloader resize)
            upsampled_logits = F.interpolate(
                logits,
                size=input_size,
                mode='bilinear',
                align_corners=False
            )
            predictions = upsampled_logits.argmax(dim=1)
            result["predictions"] = predictions

        return result

    def get_predictions(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Get segmentation predictions.

        Args:
            images: Input images of shape (B, C, H, W) in [0, 255] RGB format

        Returns:
            Predictions tensor of shape (B, H, W) with class indices
        """
        with torch.no_grad():
            result = self.forward(images, return_predictions=True)

        return result["predictions"]

    def configure_tta_grad(self):
        """Configure gradient settings for TTA based on update config."""
        # Configure discriminative model
        self.discriminative.configure_tta_grad(
            update=self.update_cfg.discriminative,
            norm_only=self.update_cfg.get("update_norm_only", False)
        )

        # Configure generative model
        if self.generative is not None:
            self.generative.configure_tta_grad(
                update=self.update_cfg.generative
            )

    def reset_to_initial_state(self, initial_state: Dict):
        """Reset model to initial state.

        Args:
            initial_state: State dict from get_initial_state()
        """
        self.load_state_dict(initial_state, strict=False)

    def get_initial_state(self) -> Dict:
        """Get current model state for later reset.

        Returns:
            State dict that can be used with reset_to_initial_state()
        """
        return {k: v.clone() for k, v in self.state_dict().items()}

    def set_eval_mode(self):
        """Set both models to eval mode but keep gradients enabled.

        This is used during TTA: we want eval mode behavior (no dropout, etc.)
        but still want to compute gradients for parameter updates.
        """
        self.discriminative.model.eval()
        if self.generative is not None:
            self.generative.vae.eval()
            self.generative.transformer.eval()

    def get_trainable_params(self):
        """Get trainable parameters for optimizer.

        Returns:
            List of parameters that require gradients
        """
        params = []

        # Discriminative model params
        if self.update_cfg.discriminative:
            if self.update_cfg.get("update_norm_only", False):
                # Only norm layer params
                for name, module in self.discriminative.model.named_modules():
                    if 'norm' in name.lower() or isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
                        params.extend(module.parameters())
            else:
                params.extend(self.discriminative.model.parameters())

        # Generative model params
        if self.generative is not None and self.update_cfg.generative:
            params.extend(self.generative.transformer.parameters())

        return [p for p in params if p.requires_grad]
