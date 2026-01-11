"""
SD3 Generative Model for DUSA TTA.

Uses Stable Diffusion 3 to compute diffusion-guided loss for test-time adaptation.
The loss is based on velocity prediction in flow matching framework.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusion3Pipeline, AutoencoderKL
from diffusers.models import SD3Transformer2DModel

from .preprocessor import GenerativePreprocessor
from .text_embeddings import prepare_class_embeddings_sd3
from src.utils.categories import ADE20K_CATEGORIES
from src.utils.slide_inference import SlidingWindowProcessor
from src.utils.device_utils import dispatch_model_to_devices, WrappedDeviceModel


class SD3GenerativeModel(nn.Module):
    """
    SD3-based generative model for DUSA TTA.
    
    Computes velocity prediction loss using Flow Matching formulation:
    L = MSE(weighted_velocity, target_velocity)
    
    where:
    - target_velocity = noise - latent
    - weighted_velocity = sum_k(p_k * v_k) where p_k is class probability
    """
    
    def __init__(
        self,
        model_path: str = "stabilityai/stable-diffusion-3-medium-diffusers",
        class_names: Tuple[str, ...] = ADE20K_CATEGORIES,
        prompt_template: str = "a photo of a {}",
        timestep: float = 0.25,
        topk: int = 1,
        classes_threshold: int = 20,
        crop_size: Tuple[int, int] = (512, 512),
        slide_stride: Tuple[int, int] = (0, 171),
        vae_device: str = "cuda:0",
        transformer_device_config: Optional[Dict[str, Any]] = None,
        gradient_checkpointing: bool = True,
        embeddings_cache_path: Optional[str] = None,
    ):
        """
        Args:
            model_path: HuggingFace model path for SD3
            class_names: Tuple of class names for text conditioning
            prompt_template: Template for class prompts
            timestep: Fixed timestep for flow matching (typically 0.25)
            topk: Number of top classes to consider per pixel
            classes_threshold: Maximum number of unique classes to process
            crop_size: Size of sliding window crop
            slide_stride: Stride for sliding window (0 means no sliding in that dimension)
            vae_device: Device for VAE
            transformer_device_config: Device configuration for transformer (supports multi-GPU)
            gradient_checkpointing: Whether to use gradient checkpointing for transformer
            embeddings_cache_path: Optional path to load/save precomputed embeddings
        """
        super().__init__()
        
        self.model_path = model_path
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.prompt_template = prompt_template
        self.timestep = timestep
        self.topk = topk
        self.classes_threshold = classes_threshold
        self.crop_size = crop_size
        self.slide_stride = slide_stride
        self.vae_device = torch.device(vae_device)
        self.transformer_device_config = transformer_device_config or {"device": "cuda:0"}
        self.gradient_checkpointing = gradient_checkpointing
        self.embeddings_cache_path = embeddings_cache_path
        
        # Initialize components
        self._load_models()
        self._setup_preprocessor()
        self._setup_sliding_window()
        self._prepare_embeddings()
        
    def _load_models(self) -> None:
        """Load SD3 components: VAE and Transformer."""
        print(f"Loading SD3 from {self.model_path}...")
        
        # Load full pipeline first to get text encoders for embedding computation
        pipe = StableDiffusion3Pipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
        )
        
        # Extract VAE and Transformer
        self.vae = pipe.vae.to(self.vae_device).float()
        self.vae.requires_grad_(False)
        self.vae.eval()
        
        # Get VAE scaling factor
        self.vae_scale_factor = self.vae.config.scaling_factor  # ~1.5305 for SD3
        
        # Setup Transformer with optional multi-GPU distribution
        transformer = pipe.transformer.float()
        
        if self.gradient_checkpointing:
            transformer.enable_gradient_checkpointing()
        
        # Dispatch transformer to device(s)
        self.transformer = dispatch_model_to_devices(
            transformer, 
            self.transformer_device_config
        )
        
        # Store pipe temporarily for embedding computation
        self._temp_pipe = pipe
        
        print("SD3 models loaded successfully.")
        
    def _setup_preprocessor(self) -> None:
        """Setup image preprocessor for generative model."""
        self.preprocessor = GenerativePreprocessor(
            target_size=self.crop_size,
            normalize_to_neg_one=True,
        )
        
    def _setup_sliding_window(self) -> None:
        """Setup sliding window processor."""
        self.slide_processor = SlidingWindowProcessor(
            crop_size=self.crop_size,
            stride=self.slide_stride,
            logits_downsample_ratio=4,
        )
        
    def _prepare_embeddings(self) -> None:
        """Prepare class text embeddings."""
        # Try to load from cache
        if self.embeddings_cache_path is not None:
            try:
                from .text_embeddings import load_embeddings
                class_embeddings, pooled_embeddings = load_embeddings(self.embeddings_cache_path)
                print(f"Loaded embeddings from {self.embeddings_cache_path}")
            except FileNotFoundError:
                class_embeddings, pooled_embeddings = self._compute_embeddings()
                from .text_embeddings import save_embeddings
                save_embeddings(class_embeddings, pooled_embeddings, self.embeddings_cache_path)
                print(f"Saved embeddings to {self.embeddings_cache_path}")
        else:
            class_embeddings, pooled_embeddings = self._compute_embeddings()
        
        # Register as buffers (not parameters, but will be moved with model)
        self.register_buffer("class_embeddings", class_embeddings)
        self.register_buffer("pooled_embeddings", pooled_embeddings)
        
        # Clean up temporary pipe
        if hasattr(self, "_temp_pipe"):
            del self._temp_pipe
            torch.cuda.empty_cache()
    
    def _compute_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute class embeddings using SD3 pipeline."""
        return prepare_class_embeddings_sd3(
            self._temp_pipe,
            self.class_names,
            self.prompt_template,
            device="cuda" if torch.cuda.is_available() else "cpu",
            show_progress=True,
        )
    
    def configure_tta_params(self, update_transformer: bool = True) -> None:
        """
        Configure which parameters require gradients for TTA.
        
        Args:
            update_transformer: Whether to update transformer parameters
        """
        # VAE is always frozen
        self.vae.requires_grad_(False)
        
        # Transformer can be optionally updated
        if hasattr(self.transformer, "model"):
            # Wrapped model
            self.transformer.model.requires_grad_(update_transformer)
        else:
            self.transformer.requires_grad_(update_transformer)
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get list of trainable parameters."""
        if hasattr(self.transformer, "model"):
            return [p for p in self.transformer.model.parameters() if p.requires_grad]
        return [p for p in self.transformer.parameters() if p.requires_grad]
    
    @torch.no_grad()
    def encode_to_latent(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent space using VAE.
        
        Args:
            images: Preprocessed images in range [-1, 1], shape (B, 3, H, W)
            
        Returns:
            Latent codes of shape (B, 4, H//8, W//8)
        """
        images = images.to(self.vae_device)
        latent_dist = self.vae.encode(images).latent_dist
        latent = latent_dist.mean * self.vae_scale_factor
        return latent
    
    def flow_matching_add_noise(
        self, 
        latent: torch.Tensor, 
        noise: torch.Tensor, 
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Add noise using flow matching interpolation.
        
        x_t = (1 - t) * x_0 + t * noise
        
        Args:
            latent: Clean latent (x_0)
            noise: Gaussian noise
            timestep: Timestep values in [0, 1]
            
        Returns:
            Noised latent (x_t)
        """
        # Expand timestep for broadcasting: (B,) -> (B, 1, 1, 1)
        t = timestep.view(-1, 1, 1, 1)
        return (1 - t) * latent + t * noise
    
    def select_topk_classes(
        self, 
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select top-k classes from logits with threshold limiting.
        
        Args:
            logits: Segmentation logits of shape (B, C, H, W)
            
        Returns:
            Tuple of:
                - topk_probs: Softmax probabilities for selected classes (B, K, H, W)
                - selected_indices: Unique class indices selected
                - mask: Valid pixel mask (B, 1, H, W)
        """
        B, C, H, W = logits.shape
        device = logits.device
        
        # Get top-k class indices per pixel
        topk_logits, topk_idx = torch.topk(logits, self.topk, dim=1)  # (B, topk, H, W)
        
        # Get unique classes across all pixels
        unique_classes = topk_idx.unique()
        
        # Initialize mask (all valid)
        mask = torch.ones(B, 1, H, W, device=device)
        
        # Limit number of classes if exceeding threshold
        if len(unique_classes) > self.classes_threshold:
            # Priority: keep top-1 classes first
            _, top1_idx = torch.topk(logits, 1, dim=1)  # (B, 1, H, W)
            top1_unique = top1_idx.unique()
            
            if len(top1_unique) > self.classes_threshold:
                # Even top-1 classes exceed threshold, randomly select
                perm = torch.randperm(len(top1_unique), device=device)
                selected_indices = top1_unique[perm[:self.classes_threshold]]
                
                # Update mask to exclude pixels with unselected top-1 classes
                mask = torch.isin(top1_idx, selected_indices).float()
            else:
                # Keep all top-1 classes, randomly select remaining from top-k
                remaining_classes = unique_classes[~torch.isin(unique_classes, top1_unique)]
                n_remaining = self.classes_threshold - len(top1_unique)
                
                if len(remaining_classes) > n_remaining:
                    perm = torch.randperm(len(remaining_classes), device=device)
                    remaining_classes = remaining_classes[perm[:n_remaining]]
                
                selected_indices = torch.cat([top1_unique, remaining_classes])
        else:
            selected_indices = unique_classes
        
        # Rebuild topk_idx using only selected classes
        # Shape: (B, num_selected, H, W)
        num_selected = len(selected_indices)
        topk_idx_new = selected_indices.view(1, -1, 1, 1).expand(B, -1, H, W)
        
        # Gather logits for selected classes
        topk_logits_new = torch.gather(
            logits, 
            dim=1, 
            index=topk_idx_new
        )  # (B, num_selected, H, W)
        
        # Apply softmax over selected classes
        topk_probs = F.softmax(topk_logits_new, dim=1)  # (B, num_selected, H, W)
        
        return topk_probs, selected_indices, mask
    
    def compute_weighted_velocity(
        self,
        probs: torch.Tensor,
        velocities: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute probability-weighted velocity.
        
        Args:
            probs: Class probabilities of shape (B, K, H, W)
            velocities: Predicted velocities of shape (B*K, C, H, W)
            
        Returns:
            Weighted velocity of shape (B, C, H, W)
        """
        B, K, H, W = probs.shape
        
        # Reshape velocities: (B*K, C, H, W) -> (B, K, C, H, W)
        velocities = velocities.view(B, K, -1, H, W)
        
        # Weighted sum: (B, K, H, W) * (B, K, C, H, W) -> (B, C, H, W)
        # Use einsum for clarity: b k h w, b k c h w -> b c h w
        weighted = torch.einsum("bkhw,bkchw->bchw", probs, velocities)
        
        return weighted
    
    def _get_transformer_device(self) -> torch.device:
        """Get the device where transformer input should be placed."""
        if hasattr(self.transformer, "device"):
            return self.transformer.device
        elif hasattr(self.transformer, "model"):
            # Wrapped model
            if hasattr(self.transformer.model, "device"):
                return self.transformer.model.device
            # Try to get device from first parameter
            try:
                return next(self.transformer.model.parameters()).device
            except StopIteration:
                pass
        # Fallback: check config
        if "device" in self.transformer_device_config:
            return torch.device(self.transformer_device_config["device"])
        return torch.device("cuda:0")
    
    def single_window_forward(
        self,
        images: torch.Tensor,
        logits: torch.Tensor,
        is_last_window: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute loss for a single sliding window.
        
        Handles multi-GPU setup where:
        - VAE is on vae_device (e.g., cuda:0)
        - Transformer is on transformer_device (e.g., cuda:1)
        - Logits are on discriminative device (e.g., cuda:0)
        
        Args:
            images: Input images of shape (B, 3, crop_h, crop_w) in [0, 255]
            logits: Segmentation logits of shape (B, C, crop_h//4, crop_w//4)
            is_last_window: If True, return combined loss; else return separate task/aux losses
            
        Returns:
            If is_last_window: single loss tensor
            Else: (task_loss, aux_loss) for gradient separation
        """
        B = images.shape[0]
        logits_device = logits.device  # Discriminative model device (for loss computation)
        transformer_device = self._get_transformer_device()  # Transformer device
        
        # 1. Preprocess images for generative model: [0,255] -> [-1,1]
        preprocessed = self.preprocessor(images.to(self.vae_device), resize=True)  # (B, 3, 512, 512)
        
        # 2. Encode to latent space (on VAE device)
        latent = self.encode_to_latent(preprocessed)  # (B, 4, 64, 64) on vae_device
        
        # 3. Sample noise and timestep (on VAE device, will be moved to transformer later)
        noise = torch.randn_like(latent)
        timestep = torch.full((B,), self.timestep, device=latent.device, dtype=latent.dtype)
        
        # 4. Add noise using flow matching (still on VAE device)
        noised_latent = self.flow_matching_add_noise(latent, noise, timestep)
        
        # 5. Resize logits to latent spatial size (stays on logits device)
        latent_h, latent_w = latent.shape[2:]
        resized_logits = F.interpolate(
            logits, 
            size=(latent_h, latent_w), 
            mode="bilinear", 
            align_corners=False
        )
        
        # 6. Select top-k classes with threshold (on logits device)
        topk_probs, selected_indices, mask = self.select_topk_classes(resized_logits)
        K = len(selected_indices)
        
        # 7. Get embeddings for selected classes and move to transformer device
        class_embeds = self.class_embeddings[selected_indices].to(transformer_device)  # (K, seq_len, dim)
        pooled_embeds = self.pooled_embeddings[selected_indices].to(transformer_device)  # (K, pooled_dim)
        
        # 8. Expand embeddings and inputs for batch processing
        # Repeat for each class: (B, ...) -> (B*K, ...)
        class_embeds = class_embeds.unsqueeze(0).expand(B, -1, -1, -1)  # (B, K, seq_len, dim)
        class_embeds = class_embeds.reshape(B * K, -1, class_embeds.shape[-1])  # (B*K, seq_len, dim)
        
        pooled_embeds = pooled_embeds.unsqueeze(0).expand(B, -1, -1)  # (B, K, pooled_dim)
        pooled_embeds = pooled_embeds.reshape(B * K, -1)  # (B*K, pooled_dim)
        
        # Move noised latent to transformer device
        noised_latent_expanded = noised_latent.to(transformer_device).repeat_interleave(K, dim=0)  # (B*K, 4, H, W)
        timestep_expanded = timestep.to(transformer_device).repeat_interleave(K, dim=0)  # (B*K,)
        
        # 9. Transformer forward pass - predict velocity (on transformer device)
        # SD3 uses timestep in [0, 1000] range
        pred_velocity = self.transformer(
            hidden_states=noised_latent_expanded.to(class_embeds.dtype),
            timestep=timestep_expanded * 1000,
            encoder_hidden_states=class_embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]  # (B*K, 4, H, W) on transformer device
        
        # 10. Move all tensors to logits device for loss computation
        # This ensures gradients flow back correctly through both paths
        pred_velocity = pred_velocity.float().to(logits_device)
        topk_probs = topk_probs.float().to(logits_device)
        
        # 11. Compute target velocity: v = noise - x_0 (move to logits device)
        target_velocity = (noise - latent).float().to(logits_device)
        
        # 12. Apply mask (move to logits device)
        mask = mask.to(logits_device)
        pred_velocity = pred_velocity * mask.repeat_interleave(K, dim=0).expand_as(pred_velocity)
        target_velocity = target_velocity * mask.expand_as(target_velocity)
        
        # 13. Compute weighted velocity and loss
        if is_last_window:
            # Single loss with all gradients flowing
            weighted_velocity = self.compute_weighted_velocity(topk_probs, pred_velocity)
            loss = F.mse_loss(weighted_velocity, target_velocity)
            return loss
        else:
            # Separate losses for gradient isolation
            # Task loss: gradients flow through probs only
            weighted_task = self.compute_weighted_velocity(topk_probs, pred_velocity.detach())
            loss_task = F.mse_loss(weighted_task, target_velocity.detach())
            
            # Aux loss: gradients flow through transformer only
            weighted_aux = self.compute_weighted_velocity(topk_probs.detach(), pred_velocity)
            loss_aux = F.mse_loss(weighted_aux, target_velocity)
            
            return loss_task, loss_aux
    
    def forward(
        self,
        images: torch.Tensor,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute DUSA loss with sliding window processing.
        
        For multiple windows, intermediate windows are backwarded immediately,
        and the last window's loss is returned for external backward.
        
        Args:
            images: Input images of shape (B, 3, H, W) in [0, 255], RGB
            logits: Segmentation logits of shape (B, C, H//4, W//4)
            
        Returns:
            Tuple of:
                - final_loss: Loss tensor for the last window (requires backward)
                - accumulated_loss_value: Float value of accumulated loss from all windows
        """
        # Get sliding windows
        windows = self.slide_processor.slide_image_and_logits(images, logits)
        num_windows = len(windows)
        
        if num_windows == 1:
            # Single window - simple case
            img_crop, logit_crop = windows[0]
            loss = self.single_window_forward(img_crop, logit_crop, is_last_window=True)
            return loss, loss.item()
        
        # Multiple windows - accumulate gradients
        accumulated_loss_value = 0.0
        final_loss = None
        
        for i, (img_crop, logit_crop) in enumerate(windows):
            is_last = (i == num_windows - 1)
            
            # Clear cache to manage memory
            torch.cuda.empty_cache()
            
            if is_last:
                # Last window: return loss normalized by num_windows
                loss = self.single_window_forward(img_crop, logit_crop, is_last_window=True)
                final_loss = loss / num_windows
                accumulated_loss_value += final_loss.item()
            else:
                # Intermediate windows: compute and backward immediately
                loss_task, loss_aux = self.single_window_forward(
                    img_crop, logit_crop, is_last_window=False
                )
                
                # Normalize by number of windows
                loss_task = loss_task / num_windows
                loss_aux = loss_aux / num_windows
                
                # Backward passes for intermediate windows
                loss_task.backward(retain_graph=True)
                loss_aux.backward()
                
                accumulated_loss_value += (loss_task.item() + loss_aux.item()) / 2
        
        return final_loss, accumulated_loss_value
