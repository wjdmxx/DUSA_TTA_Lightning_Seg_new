"""Combined model unifying discriminative and generative models for TTA."""

from typing import Tuple, Optional, Union
from copy import deepcopy

import torch
import torch.nn as nn

from .discriminative import DiscriminativeModel
from .generative.sd3 import SD3GenerativeModel


class CombinedModel(nn.Module):
    """Combined model for test-time adaptation.
    
    Unifies the discriminative (SegFormer) and generative (SD3) models
    for joint inference and adaptation.
    """
    
    def __init__(
        self,
        discriminative: DiscriminativeModel,
        generative: Optional[SD3GenerativeModel] = None,
    ):
        """Initialize combined model.
        
        Args:
            discriminative: Discriminative model (SegFormer)
            generative: Optional generative model (SD3)
        """
        super().__init__()
        
        self.discriminative = discriminative
        self.generative = generative
        
        # Store initial state for model reset between tasks
        self._initial_state: Optional[dict] = None
    
    def save_initial_state(self) -> None:
        """Save initial model state for reset between tasks."""
        self._initial_state = deepcopy(self.state_dict())
    
    def reset_to_initial(self) -> None:
        """Reset model to initial state (for non-continual TTA)."""
        if self._initial_state is not None:
            self.load_state_dict(self._initial_state)
        else:
            raise RuntimeError(
                "Initial state not saved. Call save_initial_state() first."
            )
    
    def forward(
        self,
        images: torch.Tensor,
        forward_mode: str = "tta"
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through combined model.
        
        Args:
            images: Input images (B, 3, H, W) in [0, 255]
            forward_mode: 
                - "tta": Full TTA with discriminative + generative loss
                - "discriminative_only": Only discriminative model, no loss
                
        Returns:
            Tuple of (logits, loss):
                - logits: Segmentation logits (B, num_classes, H/4, W/4)
                - loss: Diffusion/TTA loss or None if discriminative_only
        """
        # Get discriminative predictions
        logits = self.discriminative(images)
        
        # Compute generative loss based on mode
        if forward_mode == "tta" and self.generative is not None:
            # Compute diffusion-based loss
            loss = self.generative(images, logits)
        elif forward_mode == "discriminative_only":
            # No loss for comparison experiments
            loss = None
        else:
            # No generative model or unknown mode
            loss = None
        
        return logits, loss
    
    def configure_grad(
        self,
        update_discriminative: bool = True,
        update_generative: bool = True
    ) -> None:
        """Configure gradient computation for model components.
        
        Args:
            update_discriminative: Whether to train discriminative model
            update_generative: Whether to train generative model
        """
        # Configure discriminative model
        self.discriminative.requires_grad_(update_discriminative)
        
        # Configure generative model
        if self.generative is not None:
            self.generative.configure_grad(update_transformer=update_generative)
    
    def set_eval_mode(self) -> None:
        """Set all models to eval mode (for TTA during training_step).
        
        Note: We use training_step for TTA but want eval behavior.
        """
        self.eval()
    
    def get_trainable_params(self) -> list:
        """Get list of trainable parameters.
        
        Returns:
            List of parameters with requires_grad=True
        """
        return [p for p in self.parameters() if p.requires_grad]
    
    def get_param_groups(
        self,
        discriminative_lr: float = 1e-5,
        generative_lr: float = 1e-5
    ) -> list:
        """Get parameter groups with different learning rates.
        
        Args:
            discriminative_lr: Learning rate for discriminative model
            generative_lr: Learning rate for generative model
            
        Returns:
            List of parameter group dicts
        """
        param_groups = []
        
        # Discriminative parameters
        disc_params = [
            p for p in self.discriminative.parameters() 
            if p.requires_grad
        ]
        if disc_params:
            param_groups.append({
                'params': disc_params,
                'lr': discriminative_lr,
                'name': 'discriminative'
            })
        
        # Generative parameters
        if self.generative is not None:
            gen_params = [
                p for p in self.generative.parameters()
                if p.requires_grad
            ]
            if gen_params:
                param_groups.append({
                    'params': gen_params,
                    'lr': generative_lr,
                    'name': 'generative'
                })
        
        return param_groups


def build_combined_model(
    discriminative_config: dict,
    generative_config: Optional[dict] = None,
) -> CombinedModel:
    """Build combined model from configurations.
    
    Args:
        discriminative_config: Config dict for discriminative model
        generative_config: Optional config dict for generative model
        
    Returns:
        Combined model instance
    """
    # Build discriminative model
    discriminative = DiscriminativeModel(**discriminative_config)
    
    # Build generative model if configured
    generative = None
    if generative_config is not None:
        generative = SD3GenerativeModel(**generative_config)
    
    return CombinedModel(
        discriminative=discriminative,
        generative=generative,
    )
