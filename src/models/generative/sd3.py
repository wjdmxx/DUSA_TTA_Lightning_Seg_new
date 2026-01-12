"""SD3 Generative Model for Test-Time Adaptation.

This module implements the Stable Diffusion 3 model with sliding window mechanism
for computing diffusion-based losses to assist discriminative model adaptation.
"""

from typing import Tuple, Optional, List, Union
from dataclasses import dataclass
from pathlib import Path
import hashlib
import os
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm
from diffusers import (
    StableDiffusion3Pipeline,
    AutoencoderKL,
    SD3Transformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)

from ...utils.categories import ADE_CATEGORIES


@dataclass
class SD3Config:
    """Configuration for SD3 generative model."""
    model_path: str = "stabilityai/stable-diffusion-3-medium-diffusers"
    window_size: int = 512
    stride: int = 171
    timestep_range: Tuple[float, float] = (0.25, 0.25)
    topk: int = 1
    temperature: float = 1.0
    classes_threshold: int = 20
    prompt_template: str = "a photo of a {}"
    class_names: str = "ADE_CATEGORIES"


class SD3GenerativeModel(nn.Module):
    """Stable Diffusion 3 model for TTA with sliding window.
    
    Uses SD3's flow matching objective to compute weighted losses
    based on discriminative model predictions.
    """
    
    def __init__(
        self,
        model_path: str = "stabilityai/stable-diffusion-3-medium-diffusers",
        window_size: int = 512,
        stride: int = 171,
        timestep_range: Tuple[float, float] = (0.25, 0.25),
        topk: int = 1,
        temperature: float = 1.0,
        classes_threshold: int = 20,
        prompt_template: str = "a photo of a {}",
        class_names: Union[str, Tuple[str, ...]] = "ADE_CATEGORIES",
        embedding_cache_dir: Optional[str] = None,
    ):
        """Initialize SD3 model.
        
        Args:
            model_path: HuggingFace model path for SD3
            window_size: Size of sliding window (should match short edge)
            stride: Stride for sliding window
            timestep_range: Range for sampling timesteps (min, max)
            topk: Number of top classes to consider
            temperature: Temperature for softmax
            classes_threshold: Maximum number of classes to process
            prompt_template: Template for class prompts (use {} for class name)
            class_names: Category names or key to look up
            embedding_cache_dir: Directory to cache pre-computed embeddings.
                                 If None, uses './embedding_cache'
        """
        super().__init__()
        
        self.window_size = window_size
        self.stride = stride
        self.timestep_range = timestep_range
        self.topk = topk
        self.temperature = temperature
        self.classes_threshold = classes_threshold
        self.prompt_template = prompt_template
        
        # Set up embedding cache directory
        if embedding_cache_dir is None:
            embedding_cache_dir = "./embedding_cache"
        self.embedding_cache_dir = Path(embedding_cache_dir)
        self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Resolve class names
        if isinstance(class_names, str):
            if class_names == "ADE_CATEGORIES":
                self.class_names = ADE_CATEGORIES
                self._class_names_key = class_names
            else:
                raise ValueError(f"Unknown class names key: {class_names}")
        else:
            self.class_names = class_names
            self._class_names_key = "custom"
        
        self.num_classes = len(self.class_names)
        
        # Try to load cached embeddings first (no GPU needed)
        cache_path = self._get_embedding_cache_path()
        embeddings_loaded = False
        
        if cache_path.exists():
            print(f"Loading cached embeddings from {cache_path}...")
            try:
                cached_data = torch.load(cache_path, map_location="cpu", weights_only=True)
                class_embeddings = cached_data["class_embeddings"]
                pooled_embeddings = cached_data["pooled_embeddings"]
                embeddings_loaded = True
                print(f"Successfully loaded cached embeddings for {class_embeddings.shape[0]} classes")
            except Exception as e:
                print(f"Failed to load cached embeddings: {e}")
                embeddings_loaded = False
        
        # Load pipeline and extract components
        # NOTE: Load in float32 to ensure uniform dtype for FSDP compatibility.
        # Mixed precision training is handled by PyTorch Lightning's precision setting.
        print(f"Loading SD3 from {model_path}...")
        
        if embeddings_loaded:
            # Embeddings are cached - no need to load text encoders
            # Load only VAE and Transformer directly to save memory
            print("Loading only VAE and Transformer (embeddings are cached)...")
            
            self.vae = AutoencoderKL.from_pretrained(
                model_path,
                subfolder="vae",
                torch_dtype=torch.float32,
            )
            self.transformer = SD3Transformer2DModel.from_pretrained(
                model_path,
                subfolder="transformer",
                torch_dtype=torch.float32,
            )
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                model_path,
                subfolder="scheduler",
            )
            
            # Get VAE scaling factor
            self.vae_scale_factor = (
                self.vae.config.scaling_factor 
                if hasattr(self.vae.config, 'scaling_factor') 
                else 1.5305
            )
        else:
            # Need to compute embeddings - load full pipeline with text encoders
            print("Loading full pipeline (need to compute embeddings)...")
            pipe = StableDiffusion3Pipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
            )
            
            # Extract components (keep on CPU initially)
            self.vae: AutoencoderKL = pipe.vae
            self.transformer: SD3Transformer2DModel = pipe.transformer
            self.scheduler: FlowMatchEulerDiscreteScheduler = pipe.scheduler
            
            # Get VAE scaling factor
            self.vae_scale_factor = (
                self.vae.config.scaling_factor 
                if hasattr(self.vae.config, 'scaling_factor') 
                else 1.5305
            )
            
            # Compute class embeddings on CPU
            print("Computing class embeddings on CPU (this may take a while)...")
            class_embeddings, pooled_embeddings = self._compute_class_embeddings_cpu(pipe)
            
            # Save to cache
            print(f"Saving embeddings to cache: {cache_path}")
            torch.save({
                "class_embeddings": class_embeddings,
                "pooled_embeddings": pooled_embeddings,
                "prompt_template": self.prompt_template,
                "class_names_key": self._class_names_key,
                "num_classes": self.num_classes,
            }, cache_path)
            
            # Delete text encoders from pipe to free memory (no longer needed)
            del pipe.text_encoder
            del pipe.text_encoder_2
            del pipe.text_encoder_3
            del pipe.tokenizer
            del pipe.tokenizer_2
            del pipe.tokenizer_3
            del pipe
            
            # Force garbage collection
            gc.collect()
        
        # Register embeddings as buffers
        self.register_buffer('class_embeddings', class_embeddings)
        self.register_buffer('pooled_embeddings', pooled_embeddings)
        
        # Force garbage collection and clear CUDA cache
        gc.collect()
        
        # Freeze VAE (gradients only flow through transformer and to discriminative model)
        self.vae.requires_grad_(False)
        
        print(f"SD3 initialized with {self.num_classes} classes")
    
    def _get_embedding_cache_path(self) -> Path:
        """Generate cache file path based on prompt template and class names.
        
        The filename includes a hash of the prompt template and class names
        to ensure different configurations use different cache files.
        
        Returns:
            Path to the cache file
        """
        # Create a hashable string from prompt template and class names
        config_str = f"{self.prompt_template}|{self._class_names_key}|{self.num_classes}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]
        
        # Create a human-readable prefix from the prompt template
        # Replace special characters and truncate
        prompt_prefix = self.prompt_template.replace("{}", "CLASS")
        prompt_prefix = "".join(c if c.isalnum() or c in "_ " else "_" for c in prompt_prefix)
        prompt_prefix = prompt_prefix.replace(" ", "_")[:30]
        
        filename = f"sd3_embeddings_{prompt_prefix}_{self._class_names_key}_{config_hash}.pt"
        return self.embedding_cache_dir / filename
    
    @torch.no_grad()
    def _compute_class_embeddings_cpu(
        self,
        pipe: StableDiffusion3Pipeline
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pre-compute text embeddings for all classes on CPU (avoid GPU OOM).
        
        This method computes embeddings entirely on CPU to avoid OOM errors
        during initialization when multiple GPUs are used for FSDP.
        
        Args:
            pipe: SD3 pipeline with text encoders (kept on CPU)
            
        Returns:
            Tuple of (class_embeddings, pooled_embeddings) on CPU
        """
        # Ensure pipe is on CPU
        pipe.to("cpu")
        
        prompt_embeds_list = []
        pooled_embeds_list = []
        
        for class_name in tqdm(self.class_names, desc="Encoding class prompts (CPU)"):
            prompt = self.prompt_template.format(class_name)
            
            # encode_prompt works on CPU, just slower
            prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
                prompt=prompt,
                prompt_2=None,
                prompt_3=None,
                device="cpu",
            )
            
            prompt_embeds_list.append(prompt_embeds)
            pooled_embeds_list.append(pooled_prompt_embeds)
        
        class_embeddings = torch.cat(prompt_embeds_list, dim=0)  # (num_classes, seq_len, dim)
        pooled_embeddings = torch.cat(pooled_embeds_list, dim=0)  # (num_classes, dim)
        
        return class_embeddings, pooled_embeddings
    
    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for VAE encoding.
        
        Normalizes images from [0, 255] to [-1, 1].
        
        Args:
            images: Input tensor of shape (B, 3, H, W) in range [0, 255]
            
        Returns:
            Normalized tensor in range [-1, 1]
        """
        # Normalize to [0, 1] then to [-1, 1]
        images = images / 255.0
        images = images * 2.0 - 1.0
        return images
    
    def create_sliding_windows(
        self,
        x: torch.Tensor,
        window_size: Optional[int] = None,
        stride: Optional[int] = None
    ) -> Tuple[torch.Tensor, List[Tuple[int, int, int, int]]]:
        """Create sliding windows along the long edge.
        
        Uses tensor operations (unfold) instead of for loops for efficiency.
        Windows slide along the long edge of the image.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            window_size: Size of square window (default: self.window_size)
            stride: Stride between windows (default: self.stride)
            
        Returns:
            Tuple of:
                - Windows tensor of shape (B * num_windows, C, window_size, window_size)
                - List of (y1, x1, y2, x2) coordinates for each window
        """
        if window_size is None:
            window_size = self.window_size
        if stride is None:
            stride = self.stride
        
        B, C, H, W = x.shape
        
        # Determine sliding direction based on image orientation
        # Short edge should already be window_size (512), slide along long edge
        if H <= W:
            # Landscape or square: slide horizontally (along width)
            # Height is short edge, should be window_size
            h_windows = 1
            w_windows = max(1, (W - window_size) // stride + 1)
            if W > window_size and (W - window_size) % stride != 0:
                w_windows += 1  # Add one more to cover the end
        else:
            # Portrait: slide vertically (along height)
            # Width is short edge, should be window_size
            h_windows = max(1, (H - window_size) // stride + 1)
            if H > window_size and (H - window_size) % stride != 0:
                h_windows += 1
            w_windows = 1
        
        windows = []
        coords = []
        
        for h_idx in range(h_windows):
            for w_idx in range(w_windows):
                # Calculate window position
                y1 = h_idx * stride
                x1 = w_idx * stride
                
                # Ensure window doesn't exceed image bounds
                y2 = min(y1 + window_size, H)
                x2 = min(x1 + window_size, W)
                
                # Adjust start if window would be smaller than window_size
                y1 = max(0, y2 - window_size)
                x1 = max(0, x2 - window_size)
                
                # Extract window
                window = x[:, :, y1:y2, x1:x2]
                windows.append(window)
                coords.append((y1, x1, y2, x2))
        
        # Stack windows: (num_windows, B, C, H, W) -> (B * num_windows, C, H, W)
        windows = torch.stack(windows, dim=0)  # (num_windows, B, C, H, W)
        windows = rearrange(windows, 'n b c h w -> (b n) c h w')
        
        return windows, coords
    
    def select_topk_classes(
        self,
        resized_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select top-k classes from logits with threshold limiting.
        
        Args:
            resized_logits: Logits tensor of shape (B, num_classes, H, W)
            
        Returns:
            Tuple of:
                - topk_idx_unique: Unique class indices to process
                - topk_probs: Softmax probabilities for selected classes (B, K, H, W)
                - mask: Valid pixel mask (B, 1, H, W)
        """
        B, C, H, W = resized_logits.shape
        device = resized_logits.device
        
        # Get top-k indices per pixel
        topk_logits, topk_idx = torch.topk(resized_logits, self.topk, dim=1)  # (B, topk, H, W)
        
        # Get unique class indices across all pixels
        topk_idx_unique = topk_idx.unique()
        
        # Initialize mask (all valid)
        mask = torch.ones(B, 1, H, W, device=device, dtype=resized_logits.dtype)
        
        # Limit number of classes if exceeding threshold
        if topk_idx_unique.shape[0] > self.classes_threshold:
            # Get top-1 predictions
            _, top1_idx = torch.topk(resized_logits, 1, dim=1)  # (B, 1, H, W)
            top1_idx_unique = top1_idx.unique()
            
            if top1_idx_unique.shape[0] > self.classes_threshold:
                # Too many top-1 classes, randomly sample
                perm = torch.randperm(top1_idx_unique.shape[0], device=device)
                topk_idx_unique = top1_idx_unique[perm[:self.classes_threshold]]
                
                # Mask out pixels whose top-1 class is not in selected set
                mask = torch.where(
                    torch.isin(top1_idx, topk_idx_unique),
                    torch.ones_like(mask),
                    torch.zeros_like(mask)
                )
            else:
                # Keep all top-1 classes, randomly sample from remaining
                remaining = topk_idx_unique[~torch.isin(topk_idx_unique, top1_idx_unique)]
                num_remaining = self.classes_threshold - top1_idx_unique.shape[0]
                
                if remaining.shape[0] > num_remaining:
                    perm = torch.randperm(remaining.shape[0], device=device)
                    remaining = remaining[perm[:num_remaining]]
                
                topk_idx_unique = torch.cat([top1_idx_unique, remaining], dim=0)
        
        # Rebuild topk indices using unique classes
        K = topk_idx_unique.shape[0]
        topk_idx = topk_idx_unique.view(1, K, 1, 1).expand(B, K, H, W)
        
        # Gather logits for selected classes
        topk_logits = torch.gather(resized_logits, 1, topk_idx)  # (B, K, H, W)
        
        # Compute softmax probabilities
        topk_probs = F.softmax(topk_logits / self.temperature, dim=1)  # (B, K, H, W)
        
        return topk_idx_unique, topk_probs, mask
    
    def sample_timestep(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample timesteps uniformly from the configured range.
        
        Args:
            batch_size: Number of timesteps to sample
            device: Device to create tensor on
            
        Returns:
            Timestep tensor of shape (batch_size,)
        """
        t_min, t_max = self.timestep_range
        
        if t_min == t_max:
            # Fixed timestep
            return torch.full((batch_size,), t_min, device=device, dtype=torch.float32)
        else:
            # Uniform sampling
            return torch.rand(batch_size, device=device) * (t_max - t_min) + t_min
    
    def compute_weighted_loss(
        self,
        topk_probs: torch.Tensor,
        pred_velocity: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute probability-weighted MSE loss.
        
        Args:
            topk_probs: Class probabilities (B, K, H, W)
            pred_velocity: Predicted velocity per class (B*K, C, H, W)
            target: Target velocity (B, C, H, W)
            mask: Valid pixel mask (B, 1, H, W)
            
        Returns:
            Scalar loss value
        """
        B = topk_probs.shape[0]
        
        # Reshape predictions: (B*K, C, H, W) -> (B, K, C, H, W)
        pred_velocity = rearrange(pred_velocity, '(b k) c h w -> b k c h w', b=B)
        
        # Weighted sum over classes: (B, K, H, W) * (B, K, C, H, W) -> (B, C, H, W)
        weighted_pred = torch.einsum('bkhw,bkchw->bchw', topk_probs, pred_velocity)
        
        # Apply mask
        weighted_pred = weighted_pred * mask
        target = target * mask
        
        # Compute MSE loss
        loss = F.mse_loss(weighted_pred, target)
        
        return loss
    
    def forward_single_window(
        self,
        images: torch.Tensor,
        logits: torch.Tensor
    ) -> torch.Tensor:
        """Process a single window through the diffusion model.
        
        Args:
            images: Window images (B, 3, H, W) in [0, 255]
            logits: Corresponding logits (B, num_classes, H/4, W/4)
            
        Returns:
            Loss for this window
        """
        B = images.shape[0]
        device = images.device
        
        # Preprocess images and encode to latents
        preprocessed = self.preprocess(images)
        
        # VAE encoding (no gradients through VAE)
        # Cast input to VAE's dtype for compatibility with autocast
        with torch.no_grad():
            latent = self.vae.encode(preprocessed).latent_dist.mean
            latent = latent * self.vae_scale_factor
        
        # Sample timesteps
        timesteps = self.sample_timestep(B, device)
        
        # Sample noise
        noise = torch.randn_like(latent)
        
        # Flow matching: x_t = (1-t)x_0 + t*noise
        t = timesteps.view(-1, 1, 1, 1)
        noised_latent = (1 - t) * latent + t * noise
        
        # Resize logits to match latent spatial size
        latent_size = latent.shape[2:]  # (H/8, W/8) typically
        resized_logits = F.interpolate(
            logits,
            size=latent_size,
            mode='bilinear',
            align_corners=False
        )
        
        # Select top-k classes
        topk_idx_unique, topk_probs, mask = self.select_topk_classes(resized_logits)
        
        K = topk_idx_unique.shape[0]
        
        # Get class embeddings for selected classes
        class_emb = self.class_embeddings.to(device)
        pooled_emb = self.pooled_embeddings.to(device)
        
        cond = torch.index_select(class_emb, 0, topk_idx_unique)  # (K, seq_len, dim)
        pooled_cond = torch.index_select(pooled_emb, 0, topk_idx_unique)  # (K, dim)
        
        # Expand for batch
        cond = cond.unsqueeze(0).expand(B, -1, -1, -1)  # (B, K, seq_len, dim)
        cond = rearrange(cond, 'b k s d -> (b k) s d')
        
        pooled_cond = pooled_cond.unsqueeze(0).expand(B, -1, -1)  # (B, K, dim)
        pooled_cond = rearrange(pooled_cond, 'b k d -> (b k) d')
        
        # Expand latents and timesteps for all classes
        noised_latent_expanded = noised_latent.repeat_interleave(K, dim=0)  # (B*K, C, H, W)
        timesteps_expanded = timesteps.repeat_interleave(K, dim=0)  # (B*K,)
        
        # Forward through transformer
        # Note: SD3 expects timestep * 1000
        pred_velocity = self.transformer(
            hidden_states=noised_latent_expanded,
            timestep=timesteps_expanded * 1000,
            encoder_hidden_states=cond,
            pooled_projections=pooled_cond,
            return_dict=False,
        )[0]
        
        # Target: velocity = noise - latent (for flow matching)
        target = noise - latent
        
        # Compute weighted loss
        loss = self.compute_weighted_loss(topk_probs, pred_velocity, target, mask)
        
        return loss
    
    def forward(
        self,
        images: torch.Tensor,
        logits: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with sliding window processing.
        
        Args:
            images: Input images (B, 3, H, W) in [0, 255]
            logits: Discriminative model logits (B, num_classes, H/4, W/4)
            
        Returns:
            Averaged loss across all windows
        """
        # Create sliding windows for images
        image_windows, coords = self.create_sliding_windows(images)
        
        # Scale coordinates for logits (4x downsampled)
        logit_coords = [(y1//4, x1//4, y2//4, x2//4) for (y1, x1, y2, x2) in coords]
        
        # Extract corresponding logit windows
        B = images.shape[0]
        num_windows = len(coords)
        logit_windows = []
        
        for y1, x1, y2, x2 in logit_coords:
            window_h = y2 - y1
            window_w = x2 - x1
            logit_window = logits[:, :, y1:y2, x1:x2]
            logit_windows.append(logit_window)
        
        # Stack logit windows
        logit_windows = torch.stack(logit_windows, dim=0)  # (num_windows, B, C, H, W)
        logit_windows = rearrange(logit_windows, 'n b c h w -> (b n) c h w')
        
        # Process all windows and accumulate loss
        total_loss = torch.tensor(0.0, device=images.device, dtype=torch.float32)
        
        # Process windows one at a time to save memory
        for i in range(num_windows):
            start_idx = i * B
            end_idx = (i + 1) * B
            
            window_images = image_windows[start_idx:end_idx]
            window_logits = logit_windows[start_idx:end_idx]
            
            # Clear cache before processing
            torch.cuda.empty_cache()
            
            window_loss = self.forward_single_window(window_images, window_logits)
            total_loss = total_loss + window_loss
        
        # Average over windows
        avg_loss = total_loss / num_windows
        
        return avg_loss
    
    def configure_grad(self, update_transformer: bool = True) -> None:
        """Configure which parameters require gradients.
        
        Args:
            update_transformer: Whether to train the transformer
        """
        # VAE is always frozen
        self.vae.requires_grad_(False)
        
        # Transformer can be trained
        if update_transformer:
            self.transformer.requires_grad_(True)
        else:
            self.transformer.requires_grad_(False)
    
    def convert_to_dtype(self, dtype: torch.dtype = torch.float32) -> None:
        """Convert all model parameters to specified dtype for FSDP compatibility.
        
        This ensures uniform dtype across all parameters, which is required by FSDP.
        
        Args:
            dtype: Target dtype (default: torch.float32)
        """
        self.vae = self.vae.to(dtype)
        self.transformer = self.transformer.to(dtype)
        
        # Also convert registered buffers (class embeddings)
        if hasattr(self, 'class_embeddings'):
            self.class_embeddings = self.class_embeddings.to(dtype)
        if hasattr(self, 'pooled_embeddings'):
            self.pooled_embeddings = self.pooled_embeddings.to(dtype)
