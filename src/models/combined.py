"""Combined model that integrates discriminative and generative models."""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from .discriminative import DiscriminativeModel
from .generative import SD3GenerativeModel

logger = logging.getLogger(__name__)


class CombinedModel(nn.Module):
    """Combined discriminative and generative model for TTA.

    This model combines:
    - DiscriminativeModel (SegFormer) for segmentation
    - SD3GenerativeModel for generative loss

    Supports two forward modes:
    - "tta": Full TTA with both models
    - "discriminative_only": Only run discriminative model (for baseline)
    """

    def __init__(self, config: DictConfig):
        """Initialize combined model.

        Args:
            config: Configuration containing:
                - model.discriminative: DiscriminativeModel config
                - model.generative: SD3GenerativeModel config
                - tta.forward_mode: "tta" or "discriminative_only"
        """
        super().__init__()
        self.config = config

        # Forward mode
        self.forward_mode = config.get("tta", {}).get("forward_mode", "tta")
        logger.info(f"Combined model forward mode: {self.forward_mode}")

        # Initialize discriminative model
        disc_config = config.get("model", {}).get("discriminative", {})
        self.discriminative = DiscriminativeModel(disc_config)

        # Initialize generative model (lazy setup)
        self.generative: Optional[SD3GenerativeModel] = None
        if self.forward_mode == "tta":
            gen_config = config.get("model", {}).get("generative", {})
            self.generative = SD3GenerativeModel(gen_config)

        # Store initial state for reset
        self._initial_disc_state: Optional[Dict[str, Any]] = None
        self._initial_gen_state: Optional[Dict[str, Any]] = None

    def setup(self, device: str = "cuda:0") -> None:
        """Setup model components.

        This should be called after __init__ to load heavy components.

        Args:
            device: Default device
        """
        # Move discriminative model
        self.discriminative.to(device)

        # Setup generative model if needed
        if self.generative is not None:
            self.generative.setup(device)

        # Store initial states for reset
        # Use clone().detach().cpu() instead of deepcopy to handle dispatched models
        self._initial_disc_state = {
            name: param.clone().detach().cpu()
            for name, param in self.discriminative.state_dict().items()
        }
        if self.generative is not None:
            self._initial_gen_state = {
                name: param.clone().detach().cpu()
                for name, param in self.generative.transformer.named_parameters()
            }

        logger.info("Combined model setup complete")

    def forward(
        self,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the model.

        Args:
            images: Input tensor [B, C, H, W] with values in [0, 1]

        Returns:
            Tuple of:
            - logits: [B, num_classes, H/4, W/4] segmentation logits
            - loss: Generative loss (None if forward_mode is "discriminative_only")
        """
        # Get segmentation logits
        logits = self.discriminative(images)

        # Compute generative loss if in TTA mode
        loss = None
        if self.forward_mode == "tta" and self.generative is not None:
            normed_logits = F.normalize(logits, p=2, dim=1)
            loss = self.generative(images, normed_logits)

        return logits, loss

    def predict(
        self,
        images: torch.Tensor,
        resize_to_input: bool = True,
    ) -> torch.Tensor:
        """Get segmentation predictions.

        Args:
            images: Input tensor [B, C, H, W]
            resize_to_input: Whether to resize to input size

        Returns:
            Predictions [B, H, W]
        """
        logits = self.discriminative(images)

        if resize_to_input:
            logits = torch.nn.functional.interpolate(
                logits,
                size=(images.shape[2], images.shape[3]),
                mode="bilinear",
                align_corners=False,
            )

        return logits.argmax(dim=1)

    def reset(self) -> None:
        """Reset model to initial state."""
        if self._initial_disc_state is not None:
            # Move saved state to the same device as current model
            current_device = next(self.discriminative.parameters()).device
            state_dict = {
                name: param.to(current_device)
                for name, param in self._initial_disc_state.items()
            }
            self.discriminative.load_state_dict(state_dict)
            logger.info("Discriminative model reset to initial state")

        if self.generative is not None and self._initial_gen_state is not None:
            for name, param in self.generative.transformer.named_parameters():
                if name in self._initial_gen_state:
                    # Copy to the same device as the parameter
                    param.data.copy_(self._initial_gen_state[name].to(param.device))
            logger.info("Generative model reset to initial state")

    def train(self, mode: bool = True):
        """Set training mode.

        Note: For TTA, we typically keep models in eval mode but enable gradients.

        Args:
            mode: Training mode
        """
        # Discriminative model in eval mode for TTA (use running stats)
        self.discriminative.eval()

        # Generative model in eval mode
        if self.generative is not None:
            self.generative.eval()

        return self

    def eval(self):
        """Set evaluation mode."""
        return self.train(False)

    def config_tta_grad(self) -> None:
        """Configure gradients for TTA.

        Enables gradients for parameters that should be updated during TTA.
        """
        # Discriminative model: enable gradients
        self.discriminative.requires_grad_(True)

        # Generative model: configure based on settings
        if self.generative is not None:
            update_gen = self.config.get("tta", {}).get("update_generative", True)
            self.generative.config_grad(update_gen)

        logger.info("TTA gradients configured")

    def freeze(self) -> None:
        """Freeze all parameters."""
        self.discriminative.requires_grad_(False)
        if self.generative is not None:
            self.generative.config_grad(False)

    def get_trainable_params(self):
        """Get list of trainable parameters.

        Returns:
            List of parameters with requires_grad=True
        """
        params = []

        # Discriminative params
        for param in self.discriminative.parameters():
            if param.requires_grad:
                params.append(param)

        # Generative params
        if self.generative is not None:
            for param in self.generative.transformer.parameters():
                if param.requires_grad:
                    params.append(param)

        return params
