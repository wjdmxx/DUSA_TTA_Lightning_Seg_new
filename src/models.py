import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation
from diffusers import StableDiffusion3Pipeline
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch, dispatch_model
import logging

logger = logging.getLogger(__name__)

class SegformerWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        logger.info(f"Loading Segformer: {cfg.pretrained_model_name_or_path}")
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            cfg.pretrained_model_name_or_path,
            num_labels=cfg.num_labels,
            ignore_mismatched_sizes=True
        )
        
        # ImageNet Mean/Std for normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, images):
        # images: [B, C, H, W] in [0, 1]
        return (images - self.mean) / self.std

    def forward(self, images):
        # images: Raw [0, 1] tensor
        x = self.preprocess(images)
        outputs = self.model(x)
        logits = outputs.logits # [B, NumLabels, H/4, W/4]
        # Upsample to original size usually done outside or here?
        # Segformer outputs are 1/4 res.
        return logits

class SD3Wrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        logger.info(f"Loading SD3: {cfg.pretrained_model_name_or_path}")
        
        # Load Pipeline
        # We need access to components: VAE, Transformer, Scheduler, TextEncs
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            cfg.pretrained_model_name_or_path,
            torch_dtype=torch.float16 # Loading in half initially, will move to bf16
        )
        
        # Freeze components
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.text_encoder_2.requires_grad_(False) # SD3 has multiple text encoders
        self.pipe.text_encoder_3.requires_grad_(False)
        
        if cfg.dispatch_usage:
            self.dispatch_transformer()
            
        # Move VAE and Text Encoders to cuda:0 implies they are small enough or fit in first GPU
        # with Segformer.
        device_0 = torch.device("cuda:0")
        self.pipe.vae.to(device_0)
        self.pipe.text_encoder.to(device_0)
        self.pipe.text_encoder_2.to(device_0)
        self.pipe.text_encoder_3.to(device_0)
        
        # Precompute Text Embedding
        self.prompt_embeds = None
        self.pooled_prompt_embeds = None

    def dispatch_transformer(self):
        logger.info("Dispatching SD3 Transformer across GPUs...")
        # Since pipe.transformer is already loaded, we can just use dispatch_model
        # But for large models, loading empty then loading weights is better. 
        # Here we assume it fits in CPU RAM or single GPU RAM briefly to load.
        
        # Create a device map
        # SD3 Medium is ~2B params (transformer), fits heavily on one GPU?
        # Requirement: "Split transformer... dispatch model".
        
        # Use accelerate to infer map
        device_map = infer_auto_device_map(
            self.pipe.transformer,
            max_memory=self.cfg.max_memory, 
            no_split_module_classes=["TransformerBlock", "SD3Transformer2DModel"] # Avoid splitting mid-block if possible
        )
        
        # Apply dispatch
        self.pipe.transformer = dispatch_model(
            self.pipe.transformer,
            device_map=device_map
        )
        logger.info(f"SD3 Transformer Device Map: {device_map}")

    def get_text_embeddings(self, device):
        if self.prompt_embeds is None:
            # Compute once
            logger.info(f"Computing text embeddings for prompt: '{self.cfg.prompt}'")
            (
                self.prompt_embeds,
                self.negative_prompt_embeds,
                self.pooled_prompt_embeds,
                self.negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt=self.cfg.prompt,
                prompt_2=self.cfg.prompt,
                prompt_3=self.cfg.prompt,
                device=device
            )
        return self.prompt_embeds, self.pooled_prompt_embeds

    def forward_vae_encode(self, images):
        # images: [B, C, H, W] in [0, 1]
        # SD3 expect [-1, 1]
        x = 2.0 * images - 1.0
        return self.pipe.vae.encode(x).latent_dist.sample() * self.pipe.vae.config.scaling_factor

    def forward_denoise(self, latents, t, prompt_embeds, pooled_embeds):
        # latents: [B, C, H, W] (Latent space)
        # t: timestep
        
        # Transformer forward
        # transformer(hidden_states, encoder_hidden_states, pooled_projections, timestep)
        noise_pred = self.pipe.transformer(
            hidden_states=latents,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_embeds,
            timestep=t
        ).sample
        return noise_pred
