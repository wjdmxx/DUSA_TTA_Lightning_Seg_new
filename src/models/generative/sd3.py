"""SD3 Generative Model for TTA with sliding window support."""

from typing import Dict, Tuple, Optional, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from accelerate import dispatch_model
from omegaconf import DictConfig
from tqdm import tqdm
from einops import rearrange

from ...utils.categories import ADE_CATEGORIES, CITYSCAPES_CATEGORIES


class SD3GenerativeModel(nn.Module):
    """Stable Diffusion 3 generative model for TTA.

    This class handles:
    - Loading SD3 with model parallelism (transformer split across GPUs)
    - Precomputing class text embeddings
    - Sliding window processing along the long edge
    - TopK class selection and weighted loss computation
    """

    def __init__(self, cfg: DictConfig):
        """Initialize SD3 model with model parallelism.

        Args:
            cfg: Configuration containing model_path, devices, etc.
        """
        super().__init__()
        self.cfg = cfg
        self.window_size = cfg.sliding_window.window_size
        self.topk = cfg.get("topk", 1)
        self.temperature = cfg.get("temperature", 1.0)
        self.classes_threshold = cfg.get("classes_threshold", 20)
        self.timestep_range = tuple(cfg.timestep_range)

        # Get class names
        class_names_key = cfg.class_names
        if class_names_key == "ADE_CATEGORIES":
            self.class_names = ADE_CATEGORIES
        elif class_names_key == "CITYSCAPES_CATEGORIES":
            self.class_names = CITYSCAPES_CATEGORIES
        else:
            raise ValueError(f"Unknown class names: {class_names_key}")

        self.prompt_template = cfg.prompt_template

        # Device configuration
        self.vae_device = torch.device(cfg.devices.vae)
        self.text_encoder_device = torch.device(cfg.devices.text_encoders)
        self.transformer_input_device = torch.device(cfg.devices.transformer_input_device)
        self.transformer_output_device = torch.device(cfg.devices.transformer_output_device)

        # Load pipeline
        print(f"Loading SD3 Pipeline from {cfg.model_path}...")
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            cfg.model_path,
            torch_dtype=torch.bfloat16
        )

        # Precompute class embeddings before dispatching transformer
        print("Precomputing class embeddings...")
        self._precompute_embeddings()

        # Setup model parallelism
        self._setup_model_parallel()

        # Get VAE scaling factor
        self.vae_scale_factor = self.vae.config.scaling_factor if hasattr(self.vae.config, "scaling_factor") else 1.5305

    def _precompute_embeddings(self):
        """Precompute text embeddings for all classes."""
        self.pipe.to("cuda:0")

        prompt_embeds_list = []
        pooled_embeds_list = []

        for class_name in tqdm(self.class_names, desc="Encoding classes"):
            prompt = self.prompt_template.format(class_name)
            prompt_embeds, _, pooled_prompt_embeds, _ = self.pipe.encode_prompt(
                prompt=prompt, prompt_2=None, prompt_3=None
            )
            prompt_embeds_list.append(prompt_embeds.cpu())
            pooled_embeds_list.append(pooled_prompt_embeds.cpu())

        # Store as buffers
        self.register_buffer("class_embeddings", torch.cat(prompt_embeds_list, dim=0))
        self.register_buffer("pooled_embeddings", torch.cat(pooled_embeds_list, dim=0))

        # Move pipe to CPU to free memory
        self.pipe.to("cpu")
        torch.cuda.empty_cache()

    def _setup_model_parallel(self):
        """Setup model parallelism: VAE on cuda:0, transformer split across cuda:0 and cuda:1."""
        # Extract components
        self.vae = self.pipe.vae
        self.transformer = self.pipe.transformer
        self.scheduler = self.pipe.scheduler

        # Move VAE to its device
        self.vae.to(self.vae_device)
        self.vae.requires_grad_(False)

        # Setup transformer device map
        # Split transformer blocks across two GPUs
        device_map = {}
        blocks = self.transformer.transformer_blocks
        n_blocks = len(blocks)
        split = n_blocks // 2

        for i in range(n_blocks):
            device_map[f"transformer_blocks.{i}"] = 0 if i < split else 1

        # Input layers on cuda:0
        device_map.update({
            "pos_embed": 0,
            "context_embedder": 0,
            "time_text_embed": 0,
        })

        # Output layers on cuda:1
        device_map.update({
            "norm_out": 1,
            "proj_out": 1,
        })

        # Dispatch transformer
        print(f"Dispatching transformer with device_map...")
        self.transformer = dispatch_model(
            self.transformer,
            device_map=device_map,
        )

        # Clean up pipe
        del self.pipe
        torch.cuda.empty_cache()

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for SD3 VAE.

        Args:
            images: Tensor of shape (B, C, H, W) with values in [0, 255], RGB format

        Returns:
            Preprocessed tensor with values in [-1, 1]
        """
        images = images.to(self.vae_device, dtype=torch.bfloat16)

        # Scale to [0, 1]
        if images.max() > 1.0:
            images = images / 255.0

        # Map to [-1, 1]
        images = images * 2.0 - 1.0

        return images

    def encode_to_latent(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to latent space using VAE.

        Args:
            images: Preprocessed tensor with values in [-1, 1]

        Returns:
            Latent tensor
        """
        images = images.to(self.vae_device, dtype=torch.bfloat16)
        with torch.no_grad():
            latent_dist = self.vae.encode(images).latent_dist
            latent = latent_dist.mean * self.vae_scale_factor
        return latent

    def create_sliding_windows(
        self,
        tensor: torch.Tensor,
        window_size: int
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]], str]:
        """Create sliding windows along the long edge of the tensor.

        This uses tensor operations instead of for loops for efficiency.
        The window slides along the long edge (height for vertical images,
        width for horizontal images).

        Args:
            tensor: Input tensor of shape (B, C, H, W)
            window_size: Size of the square window

        Returns:
            windows: Tensor of shape (B, num_windows, C, window_size, window_size)
            positions: List of (start_h, start_w) positions for each window
            slide_direction: "vertical" or "horizontal"
        """
        b, c, h, w = tensor.shape

        # Determine slide direction based on which edge is longer
        if h > w:
            # Vertical image: slide vertically
            slide_direction = "vertical"
            slide_dim = h
            fixed_dim = w
        else:
            # Horizontal image: slide horizontally
            slide_direction = "horizontal"
            slide_dim = w
            fixed_dim = h

        # Calculate number of windows and positions
        # Windows should cover the entire long edge with minimal overlap
        if slide_dim <= window_size:
            num_windows = 1
            positions = [(0, 0)]
        else:
            # Calculate stride to cover the full length
            # We want at least 2 windows with some overlap
            num_windows = max(2, (slide_dim - 1) // (window_size // 2))
            stride = (slide_dim - window_size) / (num_windows - 1) if num_windows > 1 else 0

            positions = []
            for i in range(num_windows):
                pos = int(i * stride)
                pos = min(pos, slide_dim - window_size)  # Ensure we don't go out of bounds
                if slide_direction == "vertical":
                    positions.append((pos, 0))
                else:
                    positions.append((0, pos))

        # Extract windows
        windows_list = []
        for start_h, start_w in positions:
            end_h = start_h + window_size
            end_w = start_w + window_size

            # Handle case where window exceeds tensor size
            if slide_direction == "vertical":
                end_h = min(end_h, h)
                start_h = max(0, end_h - window_size)
                window = tensor[:, :, start_h:end_h, :]
                # Pad width if needed
                if w < window_size:
                    pad_w = window_size - w
                    window = F.pad(window, (0, pad_w, 0, 0), mode='reflect')
            else:
                end_w = min(end_w, w)
                start_w = max(0, end_w - window_size)
                window = tensor[:, :, :, start_w:end_w]
                # Pad height if needed
                if h < window_size:
                    pad_h = window_size - h
                    window = F.pad(window, (0, 0, 0, pad_h), mode='reflect')

            windows_list.append(window)

        # Stack windows: (B, num_windows, C, window_size, window_size)
        windows = torch.stack(windows_list, dim=1)

        return windows, positions, slide_direction

    def sample_timestep(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random timesteps for flow matching.

        Args:
            batch_size: Number of timesteps to sample
            device: Device to create tensor on

        Returns:
            Timesteps tensor of shape (batch_size,)
        """
        left, right = self.timestep_range
        return torch.rand(batch_size, device=device) * (right - left) + left

    def select_topk_classes(
        self,
        logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select top-k unique classes from logits with threshold.

        Args:
            logits: Logits tensor of shape (B, num_classes, H, W)

        Returns:
            topk_idx_unique: Unique class indices selected
            topk_probs: Softmax probabilities for selected classes (B, K, H, W)
            mask_metric: Mask for valid positions (B, 1, H, W)
        """
        b, k, h, w = logits.shape
        device = logits.device

        # Get top-k indices per pixel
        topk_logits, topk_idx = torch.topk(logits, self.topk, dim=1)

        # Get unique classes across all pixels
        mask_metric = torch.ones(b, 1, h, w, device=device)
        topk_idx_unique = topk_idx.unique()

        # Threshold: limit number of classes
        if topk_idx_unique.shape[0] > self.classes_threshold:
            _, top1_idx = torch.topk(logits, 1, dim=1)
            top1_idx_unique = top1_idx.unique()

            if top1_idx_unique.shape[0] > self.classes_threshold:
                # Randomly select from top1 classes
                perm = torch.randperm(top1_idx_unique.shape[0], device=device)
                topk_idx_unique = top1_idx_unique[perm[:self.classes_threshold]]
                # Mask out positions not in selected classes
                mask_metric = mask_metric * torch.isin(top1_idx, topk_idx_unique).float()
            else:
                # Keep all top1, randomly select rest from other topk
                other_idx = topk_idx_unique[~torch.isin(topk_idx_unique, top1_idx_unique)]
                if other_idx.shape[0] > 0:
                    remaining = self.classes_threshold - top1_idx_unique.shape[0]
                    perm = torch.randperm(other_idx.shape[0], device=device)
                    other_idx = other_idx[perm[:remaining]]
                    topk_idx_unique = torch.cat([top1_idx_unique, other_idx], dim=0)
                else:
                    topk_idx_unique = top1_idx_unique

        # Compute probabilities for selected classes
        # Expand topk_idx_unique to match logits shape for gathering
        k_selected = topk_idx_unique.shape[0]
        idx_expanded = topk_idx_unique.view(1, k_selected, 1, 1).expand(b, k_selected, h, w)
        selected_logits = torch.gather(logits, 1, idx_expanded)
        topk_probs = F.softmax(selected_logits, dim=1)

        return topk_idx_unique, topk_probs, mask_metric

    def forward(
        self,
        images: torch.Tensor,
        logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute TTA loss using sliding windows.

        Args:
            images: Input images of shape (B, C, H, W) in [0, 255] RGB
            logits: Segmentation logits of shape (B, num_classes, H/4, W/4)

        Returns:
            Total loss (scalar tensor)
        """
        # Preprocess images for VAE
        processed_images = self.preprocess(images)

        # Create sliding windows for images
        image_windows, positions, slide_dir = self.create_sliding_windows(
            processed_images, self.window_size
        )

        # Create sliding windows for logits (at 1/4 resolution)
        logit_window_size = self.window_size // 4
        logit_windows, _, _ = self.create_sliding_windows(
            logits.to(self.vae_device), logit_window_size
        )

        b, num_windows, c_img, h_win, w_win = image_windows.shape
        _, _, c_logit, h_logit, w_logit = logit_windows.shape

        total_loss = torch.tensor(0.0, device=self.vae_device, dtype=torch.float32)

        # Process each window
        for win_idx in range(num_windows):
            # Get current window
            img_window = image_windows[:, win_idx]  # (B, C, H, W)
            logit_window = logit_windows[:, win_idx]  # (B, num_classes, H/4, W/4)

            # Compute loss for this window
            window_loss = self._compute_window_loss(img_window, logit_window)

            # Accumulate loss
            total_loss = total_loss + window_loss / num_windows

        return total_loss

    def _compute_window_loss(
        self,
        images: torch.Tensor,
        logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss for a single window.

        Args:
            images: Window images of shape (B, C, window_size, window_size) in [-1, 1]
            logits: Window logits of shape (B, num_classes, window_size/4, window_size/4)

        Returns:
            Loss for this window (scalar tensor)
        """
        bsz = images.shape[0]

        # Encode to latent
        latent = self.encode_to_latent(images)

        # Sample timestep
        timesteps = self.sample_timestep(bsz, latent.device)

        # Sample noise
        noise = torch.randn_like(latent)

        # Resize logits to match latent size (should be same, but ensure)
        resized_logits = F.interpolate(
            logits,
            size=latent.shape[2:],
            mode='bilinear',
            align_corners=True
        )

        # Select top-k classes
        topk_idx_unique, topk_probs, mask_metric = self.select_topk_classes(resized_logits)

        # Get class embeddings for selected classes
        cond = self.class_embeddings.to(latent.device)
        cond = torch.index_select(cond, 0, topk_idx_unique)

        pooled_cond = self.pooled_embeddings.to(latent.device)
        pooled_cond = torch.index_select(pooled_cond, 0, topk_idx_unique)

        num_classes = cond.shape[0]

        # Expand for all classes
        cond = cond.unsqueeze(0).expand(bsz, -1, -1, -1)
        cond = rearrange(cond, "b k s d -> (b k) s d")

        pooled_cond = pooled_cond.unsqueeze(0).expand(bsz, -1, -1)
        pooled_cond = rearrange(pooled_cond, "b k d -> (b k) d")

        # Expand latent and timestep for all classes
        timesteps = timesteps.repeat_interleave(num_classes, dim=0)

        # Flow matching: noised_latent = (1-t)*latent + t*noise
        t = timesteps.view(-1, 1, 1, 1)
        latent_expanded = latent.repeat_interleave(num_classes, dim=0)
        noise_expanded = noise.repeat_interleave(num_classes, dim=0)
        noised_latent = (1 - t) * latent_expanded + t * noise_expanded

        # Move to transformer input device
        noised_latent = noised_latent.to(self.transformer_input_device, dtype=torch.bfloat16)
        timesteps_scaled = (timesteps * 1000).to(self.transformer_input_device, dtype=torch.bfloat16)
        cond = cond.to(self.transformer_input_device, dtype=torch.bfloat16)
        pooled_cond = pooled_cond.to(self.transformer_input_device, dtype=torch.bfloat16)

        # Forward through transformer
        pred_velocity = self.transformer(
            hidden_states=noised_latent,
            timestep=timesteps_scaled,
            encoder_hidden_states=cond,
            pooled_projections=pooled_cond,
            return_dict=False,
        )[0]

        # Move output back to VAE device and convert to float32 for loss
        pred_velocity = pred_velocity.to(self.vae_device, dtype=torch.float32)
        topk_probs = topk_probs.float()
        mask_metric = mask_metric.float()

        # Apply mask
        pred_velocity = pred_velocity * mask_metric.repeat_interleave(num_classes, dim=0)

        # Target velocity: noise - latent (for flow matching)
        target = (noise - latent).float()
        target = target * mask_metric

        # Compute weighted prediction
        pred_velocity = rearrange(pred_velocity, "(b k) c h w -> b k c h w", b=bsz)
        weighted_pred = torch.einsum("b k h w, b k c h w -> b c h w", topk_probs, pred_velocity)

        # MSE loss
        loss = F.mse_loss(weighted_pred, target)

        return loss

    def configure_tta_grad(self, update: bool = True):
        """Configure gradient settings for TTA.

        Args:
            update: Whether to update transformer parameters
        """
        self.vae.requires_grad_(False)

        if update:
            # Enable gradients for transformer
            # Convert trainable params to float32 for gradient computation
            for param in self.transformer.parameters():
                param.requires_grad = True
                if param.dtype == torch.bfloat16 or param.dtype == torch.float16:
                    param.data = param.data.float()
        else:
            self.transformer.requires_grad_(False)
