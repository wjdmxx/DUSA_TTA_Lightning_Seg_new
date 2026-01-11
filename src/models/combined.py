"""
Combined model for DUSA TTA.
Combines discriminative (segmentation) and generative (SD3) models.
"""

from typing import Dict, Optional, Tuple, List, Any
from copy import deepcopy
import torch
import torch.nn as nn

from .discriminative import DiscriminativeSegModel
from .generative import SD3GenerativeModel


class CombinedModel(nn.Module):
    """
    Combined model that unifies discriminative and generative models for DUSA TTA.
    
    The discriminative model performs semantic segmentation.
    The generative model provides diffusion-guided supervision for adaptation.
    """
    
    def __init__(
        self,
        discriminative_config: Dict[str, Any],
        generative_config: Dict[str, Any],
        update_discriminative: bool = True,
        update_generative: bool = True,
        update_norm_only: bool = False,
    ):
        """
        Args:
            discriminative_config: Configuration for discriminative model
            generative_config: Configuration for generative model
            update_discriminative: Whether to update discriminative model during TTA
            update_generative: Whether to update generative model during TTA
            update_norm_only: If True, only update norm layers in discriminative model
        """
        super().__init__()
        
        self.update_discriminative = update_discriminative
        self.update_generative = update_generative
        self.update_norm_only = update_norm_only
        
        # Build discriminative model
        self.discriminative = DiscriminativeSegModel(
            **discriminative_config,
            update_norm_only=update_norm_only,
        )
        
        # Build generative model
        self.generative = SD3GenerativeModel(**generative_config)
        
        # Store initial state for model reset
        self._initial_state = None
        
    def save_initial_state(self) -> None:
        """Save initial model state for later reset."""
        self._initial_state = {
            "discriminative": deepcopy(self.discriminative.model.state_dict()),
        }
        if self.update_generative:
            # Only save transformer state if we're updating it
            if hasattr(self.generative.transformer, "model"):
                self._initial_state["generative"] = deepcopy(
                    self.generative.transformer.model.state_dict()
                )
            else:
                self._initial_state["generative"] = deepcopy(
                    self.generative.transformer.state_dict()
                )
    
    def reset_model(self) -> None:
        """Reset model to initial state."""
        if self._initial_state is None:
            raise RuntimeError("Initial state not saved. Call save_initial_state() first.")
        
        # Reset discriminative model
        self.discriminative.reset_model(self._initial_state["discriminative"])
        
        # Reset generative model if it was being updated
        if self.update_generative and "generative" in self._initial_state:
            if hasattr(self.generative.transformer, "model"):
                self.generative.transformer.model.load_state_dict(
                    self._initial_state["generative"]
                )
            else:
                self.generative.transformer.load_state_dict(
                    self._initial_state["generative"]
                )
        
        # Reconfigure TTA parameters
        self.configure_tta_params()
    
    def configure_tta_params(self) -> None:
        """Configure which parameters require gradients for TTA."""
        if self.update_discriminative:
            self.discriminative.configure_tta_params()
        else:
            self.discriminative.model.requires_grad_(False)
        
        self.generative.configure_tta_params(update_transformer=self.update_generative)
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get all trainable parameters from both models."""
        params = []
        
        if self.update_discriminative:
            params.extend(self.discriminative.get_trainable_params())
        
        if self.update_generative:
            params.extend(self.generative.get_trainable_params())
        
        return params
    
    def forward(
        self,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through discriminative model only.
        
        Args:
            images: Input images of shape (B, 3, H, W) in [0, 255], RGB
            
        Returns:
            Tuple of (logits, preprocessed_size)
        """
        return self.discriminative(images, return_original_size=True)
    
    def compute_tta_loss(
        self,
        images: torch.Tensor,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute DUSA loss for TTA.
        
        Args:
            images: Input images of shape (B, 3, H, W) in [0, 255], RGB
            logits: Segmentation logits from discriminative model
            
        Returns:
            Tuple of (final_loss_tensor, accumulated_loss_value)
        """
        return self.generative(images, logits)
    
    def get_predictions(
        self,
        logits: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Get class predictions from logits.
        
        Args:
            logits: Segmentation logits
            target_size: Optional target size for upsampling
            
        Returns:
            Predictions with class indices
        """
        return self.discriminative.get_predictions(logits, target_size)
    
    def postprocess_logits(
        self,
        logits: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Upsample logits to target size.
        
        Args:
            logits: Segmentation logits
            target_size: Target size for upsampling
            
        Returns:
            Upsampled logits
        """
        return self.discriminative.postprocess_logits(logits, target_size)
    
    @property
    def num_classes(self) -> int:
        """Get number of segmentation classes."""
        return self.discriminative.num_classes
    
    def eval(self):
        """Set both models to eval mode."""
        self.discriminative.model.eval()
        self.generative.vae.eval()
        if hasattr(self.generative.transformer, "model"):
            self.generative.transformer.model.eval()
        else:
            self.generative.transformer.eval()
        return self
    
    def train(self, mode: bool = True):
        """
        Override train mode - models should stay in eval for TTA.
        This prevents accidental mode switching.
        """
        # Keep models in eval mode for TTA
        return self.eval()
