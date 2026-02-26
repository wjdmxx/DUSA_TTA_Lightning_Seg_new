"""SD3 Generative Model for Test-Time Adaptation."""

import logging
import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig
import torchvision.transforms.functional as TF

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
    VAE_SHIFT_FACTOR = 0.0609
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
        self.learnable_pooled_emb = None

        self.use_learnable_embeddings = config.get("learnable_embeddings", True)

        # Config values
        self.model_path = config.get(
            "model_path", "stabilityai/stable-diffusion-3-medium-diffusers"
        )
        self.topk = config.get("loss", {}).get("topk", 1)
        self.classes_max_num = config.get("loss", {}).get("classes_max_num", 20)
        self.classes_min_num = config.get("loss", {}).get("classes_min_num", 5)
        self.timestep_range = config.get("timestep_range", [0.25, 0.25])

        # Class selection method: "topk" or "area_k"
        self.class_select_method = config.get("loss", {}).get("class_select_method", "topk")
        # Area-K parameters (used when class_select_method == "area_k")
        self.area_k = config.get("loss", {}).get("area_k", 5)
        self.area_threshold = config.get("loss", {}).get("area_threshold", 15.0)

        # Window weighting: "none" or "entropy"
        self.window_weighting = config.get("loss", {}).get("window_weighting", "none")
        self.window_weighting_tau = config.get("loss", {}).get("window_weighting_tau", 1.0)

        # Ignore classes config
        ignore_cfg = config.get("loss", {}).get("ignore_classes", {})
        self.ignore_classes_enabled = ignore_cfg.get("enabled", False)
        self.ignore_classes = list(ignore_cfg.get("class_ids", []))

        # Short edge resize (independent from discriminative model)
        self.short_edge_size = config.get("short_edge_size", 512)

        # Sliding window config (2D)
        sw_config = config.get("sliding_window", {})
        self.window_size = sw_config.get("size", 512)
        self.stride = sw_config.get("stride", 512)

        # Sliding window mode: "grid" (existing) or "adaptive" (entropy-guided)
        self.window_mode = sw_config.get("mode", "grid")

        # Adaptive window params
        adapt_cfg = sw_config.get("adaptive", {})
        self.adapt_entropy_threshold = adapt_cfg.get("entropy_threshold", None)  # None = auto
        self.adapt_min_box_size = adapt_cfg.get("min_box_size", 128)
        self.adapt_max_box_size = adapt_cfg.get("max_box_size", 512)
        self.adapt_dilate_rounds = adapt_cfg.get("dilate_rounds", 5)
        self.adapt_min_windows = adapt_cfg.get("min_windows", 2)
        self.adapt_max_windows = adapt_cfg.get("max_windows", 8)
        self.adapt_merge_min_distance = adapt_cfg.get("merge_min_distance", 64)

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
            torch_dtype=torch.float16,
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

        if self.use_learnable_embeddings:
            # Initialize a GLOBAL shared learnable embedding (Addition / Residual) - initialized to 0
            # Shared across all classes [1, 2048] to represent global domain shift (e.g., weather, rain, fog)
            self.learnable_pooled_emb = nn.Parameter(torch.zeros(1, self.pooled_embeddings.shape[1]))

        # Extract components
        self.vae = pipe.vae.to(self.vae_device)
        self.scheduler = pipe.scheduler

        # Get VAE scale factor
        if hasattr(self.vae.config, "scaling_factor"):
            self.VAE_SCALE_FACTOR = self.vae.config.scaling_factor

        if hasattr(self.vae.config, "shift_factor"):
            self.VAE_SHIFT_FACTOR = self.vae.config.shift_factor

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

    def _resize_short_edge(self, images: torch.Tensor) -> torch.Tensor:
        """Resize images so that the short edge equals self.short_edge_size.

        Uses torchvision.transforms.functional.resize.

        Args:
            images: Input tensor [B, C, H, W]

        Returns:
            Resized tensor [B, C, H', W'] maintaining aspect ratio
        """
        return TF.resize(
            images,
            size=self.short_edge_size,
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias=True,
        )

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
            latents = latent_dist.mean
            latents = (latents - self.VAE_SHIFT_FACTOR) * self.VAE_SCALE_FACTOR
        return latents

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select unique classes from logits and compute probabilities.

        Args:
            logits: [B, num_classes, H, W]

        Returns:
            Tuple of:
            - probs: [B, N, H, W] softmax probabilities over N selected classes
            - unique_classes: [N] selected unique class indices
            - mask: [B, H, W] bool tensor, True = participate in loss, False = masked
        """
        B, C, H, W = logits.shape
        device = logits.device

        # Initialize mask: all pixels participate by default
        mask = torch.ones(B, H, W, dtype=torch.bool, device=device)

        # Get topk indices per pixel to find candidate classes
        _, topk_idx = torch.topk(logits, self.topk, dim=1)  # [B, K, H, W]

        # Get unique classes across all pixels and batch
        unique_classes = topk_idx.unique()

        # Get top-1 predictions for mask logic
        top1_idx = logits.argmax(dim=1)  # [B, H, W]
        top1_unique = top1_idx.unique()

        # Limit number of classes if needed (max logic)
        if unique_classes.shape[0] > self.classes_max_num:
            if top1_unique.shape[0] > self.classes_max_num:
                # Top1 classes exceed max: random sample and create mask
                perm = torch.randperm(top1_unique.shape[0], device=device)
                unique_classes = top1_unique[perm[: self.classes_max_num]]

                # Mask pixels whose top1 class is not in selected classes
                # For each pixel, check if top1_idx is in unique_classes
                mask = torch.isin(top1_idx, unique_classes)  # [B, H, W]
            else:
                # Keep all top1, fill remaining with other classes from topk
                other_classes = unique_classes[~torch.isin(unique_classes, top1_unique)]
                remaining = self.classes_max_num - top1_unique.shape[0]
                if remaining > 0 and other_classes.shape[0] > 0:
                    perm = torch.randperm(other_classes.shape[0], device=device)
                    selected_other = other_classes[perm[:remaining]]
                    unique_classes = torch.cat([top1_unique, selected_other])
                else:
                    unique_classes = top1_unique

        # Fill up to min if needed (min logic with weighted sampling)
        if unique_classes.shape[0] < self.classes_min_num:
            num_to_fill = self.classes_min_num - unique_classes.shape[0]

            # Get all class indices
            all_classes = torch.arange(C, device=device)

            # Find remaining classes (not yet selected)
            remaining_mask = ~torch.isin(all_classes, unique_classes)
            remaining_classes = all_classes[remaining_mask]

            if remaining_classes.shape[0] > 0:
                # Compute logits sum for each remaining class across all pixels
                # logits: [B, C, H, W] -> sum over B, H, W for remaining classes
                remaining_logits = logits[
                    :, remaining_classes, :, :
                ]  # [B, num_remaining, H, W]
                logits_sum = remaining_logits.sum(dim=(0, 2, 3))  # [num_remaining]

                # Convert to sampling probabilities using softmax
                sampling_probs = F.softmax(logits_sum, dim=0)  # [num_remaining]

                # Sample classes based on probabilities
                num_to_sample = min(num_to_fill, remaining_classes.shape[0])
                sampled_indices = torch.multinomial(
                    sampling_probs, num_to_sample, replacement=False
                )
                sampled_classes = remaining_classes[sampled_indices]

                # Add sampled classes to unique_classes
                unique_classes = torch.cat([unique_classes, sampled_classes])

        # Gather logits for selected classes: [B, N, H, W]
        # unique_classes: [N]
        N = unique_classes.shape[0]
        gathered_logits = logits[:, unique_classes, :, :]  # [B, N, H, W]

        # Compute softmax over the N selected classes
        probs = F.softmax(gathered_logits, dim=1)  # [B, N, H, W]

        return probs, unique_classes, mask

    def select_classes_area_k(
        self,
        logits: torch.Tensor,
        ori_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select classes using Area-K method based on ori_logits probabilities.

        Step 1: For each pixel, keep only the top area_k classes by probability,
                zero out the rest.
        Step 2: For each class, sum the retained probabilities across all pixels
                in the window to get score_c.
        Step 3: Select only classes where score_c > area_threshold.

        Args:
            logits: [B, num_classes, H, W] normed logits (used for final prob computation)
            ori_logits: [B, num_classes, H, W] original logits (used for area-k selection)

        Returns:
            Tuple of:
            - probs: [B, N, H, W] softmax probabilities over N selected classes
            - unique_classes: [N] selected unique class indices
            - mask: [B, H, W] bool tensor, True = participate in loss
        """
        B, C, H, W = logits.shape
        device = logits.device

        # Step 1: Compute softmax probabilities from ori_logits
        ori_probs = F.softmax(ori_logits, dim=1)  # [B, C, H, W]

        # For each pixel, keep only top-k classes, zero out the rest
        topk_vals, topk_indices = torch.topk(ori_probs, self.area_k, dim=1)  # [B, K, H, W]

        # Create a mask of retained classes per pixel
        retained = torch.zeros_like(ori_probs)  # [B, C, H, W]
        retained.scatter_(1, topk_indices, topk_vals)
        # Now retained[b, c, h, w] = ori_probs[b, c, h, w] if class c is in top-k for pixel (h,w), else 0

        # Step 2: Sum retained probabilities per class across all pixels and batch
        score_c = retained.sum(dim=(0, 2, 3))  # [C]

        # Step 3: Select classes where score_c > threshold
        selected_mask = score_c > self.area_threshold  # [C]
        unique_classes = torch.where(selected_mask)[0]  # [N]

        # Fallback: if no class exceeds threshold, take the class with highest score
        if unique_classes.shape[0] == 0:
            unique_classes = score_c.argmax(dim=0, keepdim=True)  # [1]

        logger.debug(
            f"Area-K: k={self.area_k}, threshold={self.area_threshold}, "
            f"selected {unique_classes.shape[0]} classes: {unique_classes.tolist()}, "
            f"scores: {score_c[unique_classes].tolist()}"
        )

        # All pixels participate in loss (no masking needed for area-k)
        mask = torch.ones(B, H, W, dtype=torch.bool, device=device)

        # Gather logits for selected classes and compute softmax
        N = unique_classes.shape[0]
        gathered_logits = logits[:, unique_classes, :, :]  # [B, N, H, W]
        probs = F.softmax(gathered_logits, dim=1)  # [B, N, H, W]

        return probs, unique_classes, mask

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
        pred_velocity = rearrange(pred_velocity, "(b n) c h w -> b n c h w", b=B, n=N)
        # Weighted sum: [B, N, H, W] x [B, N, C, H, W] -> [B, C, H, W]
        weighted = torch.einsum("b n h w, b n c h w -> b c h w", probs, pred_velocity)
        return weighted

    def _find_adaptive_windows(
        self,
        ori_logits: torch.Tensor,
    ) -> list:
        """Find adaptive windows guided by entropy from ori_logits.

        Steps:
          1. Compute per-pixel entropy from ori_logits
          2. Binarize via threshold (auto or manual)
          3. Morphological dilation to merge nearby hot regions
          4. Connected-component-like analysis via iterative max-pool labeling
          5. Extract bounding boxes, post-process (square, clamp size)

        Args:
            ori_logits: [B, C, H, W] original logits (un-normed)

        Returns:
            List of (y, x, h, w) tuples defining each adaptive window
        """
        device = ori_logits.device
        B, C, H, W = ori_logits.shape

        with torch.no_grad():
            # 1. Compute entropy map (average over batch)
            probs = F.softmax(ori_logits, dim=1)  # [B, C, H, W]
            ent = -(probs * probs.clamp(min=1e-10).log()).sum(dim=1)  # [B, H, W]
            entropy_map = ent.mean(dim=0)  # [H, W]

            # 2. Binarize
            if self.adapt_entropy_threshold is not None:
                threshold = self.adapt_entropy_threshold
            else:
                # Auto threshold: mean + 0.5 * std
                e_mean = entropy_map.mean()
                e_std = entropy_map.std()
                threshold = (e_mean + 0.5 * e_std).item()

            hot_mask = (entropy_map > threshold).float()  # [H, W]

            # 3. Morphological dilation (max_pool to merge nearby regions)
            mask_4d = hot_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            for _ in range(self.adapt_dilate_rounds):
                mask_4d = F.max_pool2d(mask_4d, kernel_size=3, stride=1, padding=1)
            hot_mask = mask_4d.squeeze()  # [H, W]

            # 4. Connected component extraction via iterative labeling
            #    Use a simple approach: find non-zero bounding boxes by scanning rows/cols
            hot_bool = hot_mask > 0.5

            if not hot_bool.any():
                # No high-entropy regions found, return empty
                return []

            # Label connected components using iterative flood-fill simulation
            # For efficiency on GPU: use scipy on CPU for small maps
            hot_np = hot_bool.cpu().numpy()

            try:
                from scipy import ndimage
                labeled, num_features = ndimage.label(hot_np)
            except ImportError:
                # Fallback: treat entire hot region as one component
                labeled = hot_np.astype(int)
                num_features = 1 if hot_np.any() else 0

            if num_features == 0:
                return []

            # 5. Extract bounding boxes for each component
            raw_boxes = []  # (y, x, h, w, mean_entropy)
            for comp_id in range(1, num_features + 1):
                ys, xs = (labeled == comp_id).nonzero()
                if len(ys) == 0:
                    continue
                y_min, y_max = ys.min(), ys.max()
                x_min, x_max = xs.min(), xs.max()
                box_h = y_max - y_min + 1
                box_w = x_max - x_min + 1

                # Compute mean entropy in this region for priority sorting
                region_entropy = entropy_map[y_min:y_max+1, x_min:x_max+1].mean().item()
                raw_boxes.append((y_min, x_min, box_h, box_w, region_entropy))

            # Sort by mean entropy (highest first)
            raw_boxes.sort(key=lambda b: b[4], reverse=True)

            # 6. Post-process: make square, clamp size, ensure in-bounds
            windows = []
            min_s = self.adapt_min_box_size
            max_s = self.adapt_max_box_size

            for y, x, bh, bw, _ent in raw_boxes:
                # Make square: use max(bh, bw)
                side = max(bh, bw)
                # Clamp to [min_size, max_size]
                side = max(min_s, min(side, max_s))

                # Center the square on the original box center
                cy = y + bh // 2
                cx = x + bw // 2
                y0 = max(0, cy - side // 2)
                x0 = max(0, cx - side // 2)

                # Ensure in-bounds
                if y0 + side > H:
                    y0 = max(0, H - side)
                if x0 + side > W:
                    x0 = max(0, W - side)

                # Final clamp (if image is smaller than min_size)
                actual_h = min(side, H - y0)
                actual_w = min(side, W - x0)

                # Check minimum distance to existing windows (center-to-center)
                new_cy = y0 + actual_h // 2
                new_cx = x0 + actual_w // 2
                too_close = False
                for wy, wx, wh, ww in windows:
                    exist_cy = wy + wh // 2
                    exist_cx = wx + ww // 2
                    dist = ((new_cy - exist_cy) ** 2 + (new_cx - exist_cx) ** 2) ** 0.5
                    if dist < self.adapt_merge_min_distance:
                        too_close = True
                        break

                if too_close:
                    continue

                windows.append((y0, x0, actual_h, actual_w))

                if len(windows) >= self.adapt_max_windows:
                    break

        return windows

    def _crop_and_resize_windows(
        self,
        tensor: torch.Tensor,
        windows: list,
        target_size: int,
    ) -> torch.Tensor:
        """Crop windows from tensor and resize each to target_size × target_size.

        Args:
            tensor: [B, C, H, W]
            windows: List of (y, x, h, w)
            target_size: Target square size

        Returns:
            [num_windows * B, C, target_size, target_size]
        """
        crops = []
        for y, x, h, w in windows:
            crop = tensor[:, :, y:y+h, x:x+w]  # [B, C, h, w]
            if h != target_size or w != target_size:
                crop = F.interpolate(
                    crop, size=(target_size, target_size),
                    mode="bilinear", align_corners=False,
                )
            crops.append(crop)
        return torch.cat(crops, dim=0)  # [num_windows * B, C, target_size, target_size]

    def forward(
        self,
        images: torch.Tensor,
        logits: torch.Tensor,
        ori_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass computing TTA loss.

        Internally resizes images to gen resolution, resizes disc logits
        to match, applies 2D sliding window (grid or adaptive), and computes loss per window.

        Args:
            images: [B, C, H, W] input images at original resolution (values in [0, 1])
            logits: [B, num_classes, H_d/4, W_d/4] segmentation logits from disc model (normed)
            ori_logits: [B, num_classes, H_d/4, W_d/4] original logits (un-normed), used by area_k

        Returns:
            loss: Scalar loss for backpropagation
        """
        B = images.shape[0]
        device = images.device

        # 1. Resize images to generative model resolution (short edge)
        gen_images = self._resize_short_edge(images)  # [B, 3, H_g, W_g]

        # 2. Resize disc logits to match gen image spatial dims
        gen_h, gen_w = gen_images.shape[2], gen_images.shape[3]
        logits_resized = F.interpolate(
            logits,
            size=(gen_h, gen_w),
            mode="bilinear",
            align_corners=False,
        )  # [B, C_cls, H_g, W_g]

        # 2b. Resize ori_logits if provided
        ori_logits_resized = None
        if ori_logits is not None:
            ori_logits_resized = F.interpolate(
                ori_logits,
                size=(gen_h, gen_w),
                mode="bilinear",
                align_corners=False,
            )  # [B, C_cls, H_g, W_g]

        # ── Branch by window mode ──
        if self.window_mode == "adaptive" and ori_logits_resized is not None:
            return self._forward_adaptive(
                gen_images, logits_resized, ori_logits_resized, B, device
            )
        else:
            return self._forward_grid(
                gen_images, logits_resized, ori_logits_resized, B, device
            )

    def _forward_grid(
        self,
        gen_images: torch.Tensor,
        logits_resized: torch.Tensor,
        ori_logits_resized: Optional[torch.Tensor],
        B: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Original grid-based sliding window forward."""
        # 3. Apply 2D sliding window to both images and logits
        image_windows, logit_windows, positions = self.sliding_window.slide_pair(
            gen_images, logits_resized
        )
        num_windows = len(positions)

        # Also slide ori_logits if available
        ori_logit_windows = None
        if ori_logits_resized is not None:
            _, ori_logit_windows, _ = self.sliding_window.slide_pair(
                gen_images, ori_logits_resized
            )

        # 4. Compute per-window entropy weights (detached, no gradient)
        if self.window_weighting == "entropy" and ori_logit_windows is not None:
            with torch.no_grad():
                window_entropies = []
                for win_idx in range(num_windows):
                    win_ori = ori_logit_windows[win_idx * B : (win_idx + 1) * B]
                    probs = F.softmax(win_ori, dim=1)  # [B, C, H, W]
                    ent = -(probs * probs.clamp(min=1e-8).log()).sum(dim=1)  # [B, H, W]
                    norm_ent = ent / math.log(probs.shape[1])  # normalize to [0, 1]
                    window_entropies.append(norm_ent.mean())  # scalar per window
                ent_tensor = torch.stack(window_entropies)  # [num_windows]
                weights = F.softmax(ent_tensor / self.window_weighting_tau, dim=0) * num_windows
        else:
            weights = torch.ones(num_windows, device=device)

        # 5. Process each window with entropy-based weighting
        total_loss = 0.0
        for win_idx in range(num_windows):
            win_images = image_windows[win_idx * B : (win_idx + 1) * B]
            win_logits = logit_windows[win_idx * B : (win_idx + 1) * B]
            win_ori_logits = None
            if ori_logit_windows is not None:
                win_ori_logits = ori_logit_windows[win_idx * B : (win_idx + 1) * B]

            win_loss = self._compute_window_loss(win_images, win_logits, win_ori_logits)
            total_loss = total_loss + weights[win_idx].detach() * win_loss / num_windows

        return total_loss

    def _forward_adaptive(
        self,
        gen_images: torch.Tensor,
        logits_resized: torch.Tensor,
        ori_logits_resized: torch.Tensor,
        B: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Entropy-guided adaptive window forward.

        Finds high-entropy regions, crops them, resizes to window_size,
        and computes loss on those focused regions.
        """
        # Find adaptive windows from entropy
        adaptive_windows = self._find_adaptive_windows(ori_logits_resized)

        # Fallback: if too few adaptive windows, supplement with grid windows
        if len(adaptive_windows) < self.adapt_min_windows:
            logger.info(
                f"Adaptive: only {len(adaptive_windows)} windows found, "
                f"falling back to grid mode (min={self.adapt_min_windows})"
            )
            return self._forward_grid(
                gen_images, logits_resized, ori_logits_resized, B, device
            )

        num_windows = len(adaptive_windows)
        logger.debug(
            f"Adaptive windows: {num_windows} regions, "
            f"sizes: {[(h, w) for _, _, h, w in adaptive_windows]}"
        )

        # Crop and resize all windows to standard window_size
        target_size = self.window_size
        image_crops = self._crop_and_resize_windows(
            gen_images, adaptive_windows, target_size
        )  # [num_windows * B, 3, S, S]
        logit_crops = self._crop_and_resize_windows(
            logits_resized, adaptive_windows, target_size
        )  # [num_windows * B, C_cls, S, S]
        ori_logit_crops = self._crop_and_resize_windows(
            ori_logits_resized, adaptive_windows, target_size
        )  # [num_windows * B, C_cls, S, S]

        # Compute per-window entropy weights
        if self.window_weighting == "entropy":
            with torch.no_grad():
                window_entropies = []
                for win_idx in range(num_windows):
                    win_ori = ori_logit_crops[win_idx * B : (win_idx + 1) * B]
                    probs = F.softmax(win_ori, dim=1)
                    ent = -(probs * probs.clamp(min=1e-8).log()).sum(dim=1)
                    norm_ent = ent / math.log(probs.shape[1])
                    window_entropies.append(norm_ent.mean())
                ent_tensor = torch.stack(window_entropies)
                weights = F.softmax(ent_tensor / self.window_weighting_tau, dim=0) * num_windows
        else:
            weights = torch.ones(num_windows, device=device)

        # Process each window
        total_loss = 0.0
        for win_idx in range(num_windows):
            win_images = image_crops[win_idx * B : (win_idx + 1) * B]
            win_logits = logit_crops[win_idx * B : (win_idx + 1) * B]
            win_ori_logits = ori_logit_crops[win_idx * B : (win_idx + 1) * B]

            win_loss = self._compute_window_loss(win_images, win_logits, win_ori_logits)
            total_loss = total_loss + weights[win_idx].detach() * win_loss / num_windows

        return total_loss

    def _compute_window_loss(
        self,
        images: torch.Tensor,
        logits: torch.Tensor,
        ori_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute loss for a single window.

        Args:
            images: [B, 3, S, S] window images (S = window_size)
            logits: [B, num_classes, S, S] window logits (normed, same spatial size as images)
            ori_logits: [B, num_classes, S, S] original logits (optional, for area_k)

        Returns:
            Loss tensor
        """
        B = images.shape[0]
        device = images.device

        # Preprocess images for VAE: [0,1] -> [-1,1]
        images_preprocessed = self.preprocess_images(images)

        # Encode to latent space: [B, 16, S/8, S/8]
        latent = self.encode_to_latent(images_preprocessed)
        latent_h, latent_w = latent.shape[2], latent.shape[3]

        # Downsample logits to latent resolution: [B, C, S/8, S/8]
        logits_downsampled = downsample_logits_to_latent(logits, (latent_h, latent_w))

        # Mask out configured ignore classes by setting their logits to -inf
        if self.ignore_classes_enabled and self.ignore_classes:
            logits_downsampled[:, self.ignore_classes, :, :] = -float('inf')

        # Select N unique classes based on configured method
        if self.class_select_method == "area_k" and ori_logits is not None:
            # Downsample ori_logits to latent resolution too
            ori_logits_downsampled = downsample_logits_to_latent(ori_logits, (latent_h, latent_w))
            if self.ignore_classes_enabled and self.ignore_classes:
                ori_logits_downsampled[:, self.ignore_classes, :, :] = -float('inf')
            probs, unique_classes, mask = self.select_classes_area_k(
                logits_downsampled, ori_logits_downsampled
            )
        else:
            # Default: topk method
            probs, unique_classes, mask = self.select_classes(logits_downsampled)
        N = unique_classes.shape[0]
        
        # Get text embeddings for selected classes
        # class_emb: [N, seq_len, hidden], pooled_emb: [N, hidden]
        class_emb = self.class_embeddings[unique_classes.to("cpu")].to(device)
        pooled_emb = self.pooled_embeddings[unique_classes.to("cpu")].to(device)
        
        if self.use_learnable_embeddings and self.learnable_pooled_emb is not None:
            # Add shared global domain shift to every class's pooled embedding via broadcasting
            pooled_emb = pooled_emb + self.learnable_pooled_emb.to(device)

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
        noised_latent_for_transformer = noised_latent_expanded.to(
            transformer_device, dtype=torch.float16
        )
        class_emb = class_emb.to(transformer_device, dtype=torch.float16)
        pooled_emb = pooled_emb.to(transformer_device, dtype=torch.float16)

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

        # Compute normalized L2 loss with mask
        # mask: [B, H, W] -> expand to [B, 1, H, W] for broadcasting
        mask_expanded = mask.unsqueeze(1).float()  # [B, 1, H, W]

        # # Compute element-wise squared error
        # squared_error = (weighted_pred - target) ** 2  # [B, C, H, W]

        # # Apply mask: only count masked-in pixels
        # masked_squared_error = squared_error * mask_expanded  # [B, C, H, W]

        # # Compute per-sample mean squared error
        # # Count valid elements per sample
        # num_valid_per_sample = mask_expanded.sum(dim=(2, 3)) * squared_error.shape[1]  # [B, 1]
        # num_valid_per_sample = num_valid_per_sample.squeeze(1).clamp(min=1)  # [B]

        # # Sum over C, H, W and divide by valid count
        # e = masked_squared_error.sum(dim=(1, 2, 3)) / num_valid_per_sample  # [B]

        # # Apply normalized L2 loss: norm_l2 = e / (e + c)^p
        # p, c = 0.5, 1e-3
        # norm_l2_per_sample = e / (e + c).pow(p).detach()  # [B]

        # # Average over batch
        # loss = norm_l2_per_sample.mean()

        mask_bool = mask_expanded.bool()

        valid_pred = weighted_pred.masked_select(mask_bool)
        valid_target = target.masked_select(mask_bool)

        loss = F.mse_loss(valid_pred, valid_target)

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

        # Transformer and Embedding gradients
        if requires_grad:
            if self.use_learnable_embeddings and self.learnable_pooled_emb is not None:
                self.learnable_pooled_emb.requires_grad_(True)
            self.transformer.requires_grad_(True)
            # Convert trainable params to float32 for gradient computation
            for param in self.transformer.parameters():
                if param.requires_grad and param.dtype == torch.float16:
                    param.data = param.data.float()
        else:
            if self.use_learnable_embeddings and self.learnable_pooled_emb is not None:
                self.learnable_pooled_emb.requires_grad_(False)
            self.transformer.requires_grad_(False)
