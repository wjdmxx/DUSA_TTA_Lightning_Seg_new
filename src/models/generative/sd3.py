"""SD3 Generative Model for Test-Time Adaptation."""

import logging
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig

from .sliding_window import SlidingWindowProcessor, downsample_logits_to_latent
from .text_embeddings import TextEmbeddingManager
from ..device_utils import setup_transformer_dispatch

logger = logging.getLogger(__name__)


class SD3GenerativeModel(nn.Module):
    """SD3-based generative model for TTA.

    This model uses Stable Diffusion 3 to provide a generative loss signal
    for test-time adaptation of segmentation models.
    """

    # VAE scaling factor for SD3
    VAE_SCALE_FACTOR = 1.5305
    # Latent channels
    LATENT_CHANNELS = 16

    def __init__(self, config: DictConfig):
        """Initialize SD3 generative model.

        Args:
            config: Configuration containing:
                - model_path: Path to SD3 model
                - transformer.device_map: Device mapping for multi-GPU
                - transformer.gradient_checkpointing: Enable gradient checkpointing
                - sliding_window.size: Window size
                - sliding_window.stride: Stride
                - loss.topk: TopK for class selection
                - loss.classes_threshold: Max classes
                - timestep_range: [min, max] for timestep sampling
        """
        super().__init__()
        self.config = config

        # Will be initialized in setup()
        self.vae = None
        self.transformer = None
        self.scheduler = None
        self.embedding_manager = None
        self.sliding_window = None

        # Config values
        self.model_path = config.get(
            "model_path", "stabilityai/stable-diffusion-3-medium-diffusers"
        )
        self.topk = config.get("loss", {}).get("topk", 1)
        self.classes_threshold = config.get("loss", {}).get("classes_threshold", 20)
        self.timestep_range = config.get("timestep_range", [0.25, 0.25])

        # Sliding window config
        sw_config = config.get("sliding_window", {})
        self.window_size = sw_config.get("size", 512)
        self.stride = sw_config.get("stride", 171)

        # Device config
        self.vae_device = config.get("vae", {}).get("device", "cuda:0")
        self.transformer_config = config.get("transformer", {})

    def setup(self, device: str = "cuda:0") -> None:
        """Setup the model components.

        This should be called after __init__ to load the heavy components.

        Args:
            device: Default device for components
        """
        from diffusers import StableDiffusion3Pipeline

        logger.info(f"Loading SD3 pipeline from {self.model_path}")

        # Load pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
        )

        # Setup text embeddings (do this before distributing transformer)
        self.embedding_manager = TextEmbeddingManager(
            config=self.config.get("text_embedding", {}),
            pipe=pipe,
        )
        # Pre-compute embeddings
        self.class_embeddings, self.pooled_embeddings = (
            self.embedding_manager.get_embeddings(device="cpu")
        )

        # Extract components
        self.vae = pipe.vae.to(self.vae_device)
        self.scheduler = pipe.scheduler

        # Get VAE scale factor
        if hasattr(self.vae.config, "scaling_factor"):
            self.VAE_SCALE_FACTOR = self.vae.config.scaling_factor

        # Setup transformer with multi-GPU dispatch
        device_map = self.transformer_config.get("device_map", "balanced")
        gradient_checkpointing = self.transformer_config.get(
            "gradient_checkpointing", True
        )

        self.transformer = setup_transformer_dispatch(
            pipe.transformer,
            device_map=device_map,
            input_device=self.transformer_config.get("input_device", "cuda:0"),
            output_device=self.transformer_config.get("output_device", "cuda:0"),
            gradient_checkpointing=gradient_checkpointing,
        )

        # Setup sliding window processor
        self.sliding_window = SlidingWindowProcessor(
            window_size=self.window_size,
            stride=self.stride,
        )

        # Clean up pipeline (we don't need text encoders anymore)
        del pipe
        torch.cuda.empty_cache()

        logger.info("SD3 model setup complete")

    def preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for SD3 VAE.

        Args:
            images: Input tensor [B, C, H, W] with values in [0, 1]

        Returns:
            Preprocessed tensor with values in [-1, 1]
        """
        return images * 2.0 - 1.0

    def encode_to_latent(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to latent space using VAE.

        Args:
            images: [B, C, H, W] with values in [-1, 1]

        Returns:
            Latents [B, latent_channels, H/8, W/8]
        """
        images = images.to(self.vae.device, dtype=self.vae.dtype)
        with torch.no_grad():
            latent_dist = self.vae.encode(images).latent_dist
            latent = latent_dist.mean * self.VAE_SCALE_FACTOR
        return latent

    def sample_timesteps(
        self,
        batch_size: int,
        device: str,
    ) -> torch.Tensor:
        """Sample timesteps for diffusion.

        Args:
            batch_size: Number of samples
            device: Target device

        Returns:
            Timesteps tensor [batch_size]
        """
        min_t, max_t = self.timestep_range
        # Uniform sampling in the range
        t = torch.rand(batch_size, device=device) * (max_t - min_t) + min_t
        return t

    def select_classes(
        self,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select unique classes from logits and compute probabilities.

        Args:
            logits: [B, num_classes, H, W]

        Returns:
            Tuple of:
            - probs: [B, N, H, W] softmax probabilities over N selected classes
            - unique_classes: [N] selected unique class indices
        """
        B, C, H, W = logits.shape
        device = logits.device

        # Get topk indices per pixel to find candidate classes
        _, topk_idx = torch.topk(logits, self.topk, dim=1)  # [B, K, H, W]

        # Get unique classes across all pixels and batch
        unique_classes = topk_idx.unique()

        # Limit number of classes if needed
        if unique_classes.shape[0] > self.classes_threshold:
            # Keep classes that appear in top-1 predictions
            top1_idx = logits.argmax(dim=1)  # [B, H, W]
            top1_unique = top1_idx.unique()

            if top1_unique.shape[0] > self.classes_threshold:
                # Random sample from top1 classes
                perm = torch.randperm(top1_unique.shape[0], device=device)
                unique_classes = top1_unique[perm[: self.classes_threshold]]
            else:
                # Keep all top1, fill remaining with other classes
                other_classes = unique_classes[
                    ~torch.isin(unique_classes, top1_unique)
                ]
                remaining = self.classes_threshold - top1_unique.shape[0]
                if remaining > 0 and other_classes.shape[0] > 0:
                    perm = torch.randperm(other_classes.shape[0], device=device)
                    selected_other = other_classes[perm[:remaining]]
                    unique_classes = torch.cat([top1_unique, selected_other])
                else:
                    unique_classes = top1_unique

        # Gather logits for selected classes: [B, N, H, W]
        # unique_classes: [N]
        N = unique_classes.shape[0]
        gathered_logits = logits[:, unique_classes, :, :]  # [B, N, H, W]

        # Compute softmax over the N selected classes
        probs = F.softmax(gathered_logits, dim=1)  # [B, N, H, W]

        return probs, unique_classes

    def compute_weighted_velocity(
        self,
        pred_velocity: torch.Tensor,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute probability-weighted velocity prediction.

        Args:
            pred_velocity: [B*N, C, H, W] predicted velocities for N classes
            probs: [B, N, H, W] softmax probabilities over N classes

        Returns:
            Weighted velocity [B, C, H, W]
        """
        B = probs.shape[0]
        N = probs.shape[1]
        # Reshape: [B*N, C, H, W] -> [B, N, C, H, W]
        pred_velocity = rearrange(
            pred_velocity, "(b n) c h w -> b n c h w", b=B, n=N
        )
        # Weighted sum: [B, N, H, W] x [B, N, C, H, W] -> [B, C, H, W]
        weighted = torch.einsum("b n h w, b n c h w -> b c h w", probs, pred_velocity)
        return weighted

    def forward(
        self,
        images: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass computing TTA loss.

        Args:
            images: [B, C, H, W] input images (values in [0, 1])
            logits: [B, num_classes, H/4, W/4] segmentation logits

        Returns:
            loss: Scalar loss for backpropagation
        """
        B = images.shape[0]
        device = images.device
        # Apply sliding window
        image_windows, logit_windows, num_windows = self.sliding_window.slide_batch(
            images, logits, logits_scale=4
        )

        total_loss = 0.0

        # Process each window
        for win_idx in range(num_windows):
            # Get window data
            win_images = image_windows[win_idx * B : (win_idx + 1) * B]
            win_logits = logit_windows[win_idx * B : (win_idx + 1) * B]

            # Compute loss for this window
            win_loss = self._compute_window_loss(win_images, win_logits)

            # Accumulate loss (averaged over windows)
            total_loss = total_loss + win_loss / num_windows

        return total_loss

    def _compute_window_loss(
        self,
        images: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss for a single window.

        Args:
            images: [B, C, H, W] window images
            logits: [B, num_classes, H/4, W/4] window logits

        Returns:
            Loss tensor
        """
        B = images.shape[0]
        device = images.device

        # Preprocess images for VAE
        images_preprocessed = self.preprocess_images(images)

        # Encode to latent space
        latent = self.encode_to_latent(images_preprocessed)
        latent_h, latent_w = latent.shape[2], latent.shape[3]

        # Downsample logits to latent resolution
        logits_downsampled = downsample_logits_to_latent(
            logits, (latent_h, latent_w)
        )

        # Select N unique classes and compute softmax probabilities over them
        # probs: [B, N, H, W], unique_classes: [N]
        probs, unique_classes = self.select_classes(logits_downsampled)
        N = unique_classes.shape[0]

        # Get text embeddings for selected classes
        # class_emb: [N, seq_len, hidden], pooled_emb: [N, hidden]
        class_emb = self.class_embeddings[unique_classes.to("cpu")].to(device)
        pooled_emb = self.pooled_embeddings[unique_classes.to("cpu")].to(device)

        # Expand for batch: [N, seq_len, hidden] -> [B*N, seq_len, hidden]
        class_emb = class_emb.unsqueeze(0).expand(B, -1, -1, -1)
        class_emb = rearrange(class_emb, "b n s d -> (b n) s d")

        pooled_emb = pooled_emb.unsqueeze(0).expand(B, -1, -1)
        pooled_emb = rearrange(pooled_emb, "b n d -> (b n) d")

        # Sample timesteps
        timesteps = self.sample_timesteps(B, device)

        # Sample noise
        noise = torch.randn_like(latent)

        # Compute target (flow matching target: velocity = noise - latent)
        target = (noise - latent).float()

        # Get noised latent at timestep t
        # For flow matching: x_t = (1 - t) * x_0 + t * noise
        t_expanded = timesteps.view(B, 1, 1, 1)
        noised_latent = (1 - t_expanded) * latent + t_expanded * noise

        # Expand noised latent for each class: [B, C, H, W] -> [B*N, C, H, W]
        noised_latent_expanded = noised_latent.unsqueeze(1).expand(-1, N, -1, -1, -1)
        noised_latent_expanded = rearrange(
            noised_latent_expanded, "b n c h w -> (b n) c h w"
        )

        # Expand timesteps: [B] -> [B*N]
        timesteps_expanded = timesteps.unsqueeze(1).expand(-1, N)
        timesteps_expanded = rearrange(timesteps_expanded, "b n -> (b n)")

        # Move to transformer device
        transformer_device = self._get_transformer_device()
        noised_latent_for_transformer = noised_latent_expanded.to(transformer_device)
        class_emb = class_emb.to(transformer_device)
        pooled_emb = pooled_emb.to(transformer_device)

        # Forward through transformer: output [B*N, C, H, W]
        pred_velocity = self.transformer(
            hidden_states=noised_latent_for_transformer,
            timestep=timesteps_expanded.to(transformer_device) * 1000,
            encoder_hidden_states=class_emb,
            pooled_projections=pooled_emb,
            return_dict=False,
        )[0]

        # Move output back and convert to float32
        pred_velocity = pred_velocity.to(device, dtype=torch.float32)

        # Compute weighted velocity using the probs
        # pred_velocity: [B*N, C, H, W] -> [B, N, C, H, W]
        # probs: [B, N, H, W]
        # weighted_pred: [B, C, H, W]
        weighted_pred = self.compute_weighted_velocity(pred_velocity, probs)

        # Compute MSE loss
        loss = F.mse_loss(weighted_pred, target)

        return loss

    def _get_transformer_device(self) -> str:
        """Get the device where transformer expects input."""
        # Try to get from first parameter
        try:
            param = next(self.transformer.parameters())
            return str(param.device)
        except StopIteration:
            return self.transformer_config.get("input_device", "cuda:0")

    def config_grad(self, requires_grad: bool = True) -> None:
        """Configure gradient computation for the model.

        Args:
            requires_grad: Whether to compute gradients
        """
        # VAE is always frozen
        self.vae.requires_grad_(False)

        # Transformer gradients
        if requires_grad:
            self.transformer.requires_grad_(True)
        else:
            self.transformer.requires_grad_(False)
