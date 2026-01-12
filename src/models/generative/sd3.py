import math
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from accelerate import dispatch_model
from diffusers import StableDiffusion3Pipeline
from einops import rearrange

from src.data.categories import ADE_CATEGORIES


class SD3Generative(nn.Module):
    """Stable Diffusion 3 auxiliary for TTA with sliding-window loss.

    - Model-parallel transformer split across cuda:0/1.
    - Caches text embeddings once, reuses per window.
    - Sliding window along long edge with window_size=512 and configurable stride.
    - bf16 autocast for speed; gradients kept in full precision for stability.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        self.model_path = cfg.get("model_path", "stabilityai/stable-diffusion-3-medium-diffusers")
        self.window_size = cfg.get("window_size", 512)
        self.stride = cfg.get("stride", 256)
        self.topk = cfg.get("topk", 1)
        self.classes_threshold = cfg.get("classes_threshold", 20)
        self.prompt_template = cfg.get("prompt", "a photo of a {}")
        self.device_primary = torch.device(cfg.get("device_primary", "cuda:0"))
        self.device_secondary = torch.device(cfg.get("device_secondary", "cuda:1"))

        pipe = StableDiffusion3Pipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
        )
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

        self.class_embeddings, self.pooled_embeddings = self._precompute_text_embeddings()
        self._split_transformer()
        self.vae = pipe.vae.to(self.device_primary)
        self.text_encoder = pipe.text_encoder.to(self.device_primary)
        self.text_encoder_2 = pipe.text_encoder_2.to(self.device_primary)
        self.text_encoder_3 = pipe.text_encoder_3.to(self.device_primary)
        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.tokenizer_3 = pipe.tokenizer_3
        self.vae_scale = getattr(pipe.vae.config, "scaling_factor", 0.18215)
        # Freeze non-trainable parts
        for p in self.vae.parameters():
            p.requires_grad = False
        for enc in [self.text_encoder, self.text_encoder_2, self.text_encoder_3]:
            for p in enc.parameters():
                p.requires_grad = False
        # Only transformer parameters are trainable
        for p in self.transformer.parameters():
            p.requires_grad = True

    @torch.no_grad()
    def _precompute_text_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        prompts = [self.prompt_template.replace("_", " ").format(name) for name in ADE_CATEGORIES]
        embeds, pooled = [], []
        prompt_embeds, _, pooled_embeds, _ = self.pipe.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            prompt_3=None,
            device=self.device_primary,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        embeds.append(prompt_embeds.detach().to(self.device_primary))
        pooled.append(pooled_embeds.detach().to(self.device_primary))
        return torch.cat(embeds, dim=0), torch.cat(pooled, dim=0)

    def _split_transformer(self):
        transformer = self.pipe.transformer
        blocks = transformer.transformer_blocks
        n_blocks = len(blocks)
        split = n_blocks // 2
        device_map = {}
        for i in range(n_blocks):
            device_map[f"transformer_blocks.{i}"] = 0 if i < split else 1
        device_map.update(
            {
                "pos_embed": 0,
                "context_embedder": 0,
                "time_text_embed": 0,
                "norm_out": 1,
                "proj_out": 1,
            }
        )
        self.transformer = dispatch_model(transformer, device_map=device_map)
        # Keep reference to manage device placement for inputs
        self.transformer_input_device = torch.device("cuda:0")

    def forward(self, images: List[torch.Tensor], logits: torch.Tensor) -> torch.Tensor:
        return self.compute_loss(images, logits)

    def compute_loss(self, images: List[torch.Tensor], logits: torch.Tensor) -> torch.Tensor:
        # images: list of CxHxW float in [0,1] on any device (batch size = 1 expected)
        stacked = torch.stack([img.to(self.device_primary) for img in images], dim=0)
        logits = logits.to(self.device_primary)
        bsz, _, h_img, w_img = stacked.shape
        assert bsz == 1, "Only batch_size=1 is supported for TTA"

        long_side = max(h_img, w_img)
        window = self.window_size
        stride = self.stride
        if long_side <= window:
            offsets = [0]
        else:
            offsets = list(range(0, long_side - window + 1, stride))
            if offsets[-1] != long_side - window:
                offsets.append(long_side - window)

        losses = []
        for offset in offsets:
            if h_img >= w_img:
                img_patch = stacked[:, :, offset : offset + window, :]
                logit_patch = logits[:, :, offset // 4 : (offset + window) // 4, :]
            else:
                img_patch = stacked[:, :, :, offset : offset + window]
                logit_patch = logits[:, :, :, offset // 4 : (offset + window) // 4]
            losses.append(self._patch_loss(img_patch, logit_patch))

        return torch.stack(losses).mean()

    def _patch_loss(self, image: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            latents = self.vae.encode(image * 2 - 1).latent_dist.sample() * self.vae_scale
        # Downsample logits to latent resolution for alignment
        logits_ds = F.interpolate(logits, size=latents.shape[-2:], mode="bilinear", align_corners=False)
        topk_probs, topk_idx, mask_metric = self._select_topk(logits_ds)

        bsz = latents.shape[0]
        t = torch.rand(bsz, device=self.device_primary)
        noise = torch.randn_like(latents)
        noised_latent = (1 - t)[:, None, None, None] * latents + t[:, None, None, None] * noise

        cond, pooled_cond = self._gather_embeddings(topk_idx, latents.device)
        classes = cond.shape[0]
        cond = cond.unsqueeze(0).repeat(bsz, 1, 1, 1)
        cond = rearrange(cond, "b k s d -> (b k) s d")
        pooled_cond = pooled_cond.unsqueeze(0).repeat(bsz, 1, 1)
        pooled_cond = rearrange(pooled_cond, "b k d -> (b k) d")
        time_steps = t.repeat_interleave(classes, dim=0) * 1000
        noised_latent = noised_latent.repeat_interleave(classes, dim=0)

        noised_latent = noised_latent.to(self.transformer_input_device)
        time_steps = time_steps.to(self.transformer_input_device)
        cond = cond.to(self.transformer_input_device)
        pooled_cond = pooled_cond.to(self.transformer_input_device)

        pred_velocity = self.transformer(
            hidden_states=noised_latent,
            timestep=time_steps,
            encoder_hidden_states=cond,
            pooled_projections=pooled_cond,
            return_dict=False,
        )[0]
        pred_velocity = pred_velocity.to(self.device_primary)

        target = (noise - latents).float()
        weighted_pred = self._weight_prediction(topk_probs.float(), pred_velocity.float(), mask_metric)
        loss = F.mse_loss(weighted_pred, target)
        return loss

    def _select_topk(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # logits: (B, C, H, W)
        topk_logits, topk_idx = torch.topk(logits, k=self.topk, dim=1)
        bsz, _, h, w = topk_logits.shape
        mask_metric = torch.ones((bsz, 1, h, w), device=logits.device, dtype=logits.dtype)
        unique_idx = torch.unique(topk_idx)
        if unique_idx.numel() > self.classes_threshold:
            _, top1_idx = torch.topk(logits, k=1, dim=1)
            top1_unique = torch.unique(top1_idx)
            if top1_unique.numel() > self.classes_threshold:
                chosen = top1_unique[torch.randperm(top1_unique.numel(), device=logits.device)[: self.classes_threshold]]
                mask_metric = mask_metric * torch.isin(top1_idx, chosen)
            else:
                remainder = unique_idx[~torch.isin(unique_idx, top1_unique)]
                need = self.classes_threshold - top1_unique.numel()
                sampled = remainder[torch.randperm(remainder.numel(), device=logits.device)[:need]]
                chosen = torch.cat([top1_unique, sampled], dim=0)
            unique_idx = chosen
        gather_idx = unique_idx.view(1, -1, 1, 1).repeat(bsz, 1, h, w)
        gathered_logits = torch.gather(logits, 1, gather_idx)
        probs = torch.softmax(gathered_logits, dim=1)
        return probs, unique_idx, mask_metric

    def _gather_embeddings(self, class_indices: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        cond = torch.index_select(self.class_embeddings, 0, class_indices.to(self.class_embeddings.device))
        pooled = torch.index_select(self.pooled_embeddings, 0, class_indices.to(self.pooled_embeddings.device))
        return cond.to(device), pooled.to(device)

    def _weight_prediction(
        self, probs: torch.Tensor, pred_velocity: torch.Tensor, mask_metric: torch.Tensor
    ) -> torch.Tensor:
        # probs: (B, K, H, W)
        # pred_velocity: (B*K, C, H, W)
        pred_velocity = rearrange(pred_velocity, "(b k) c h w -> b k c h w", b=probs.shape[0])
        weighted = torch.einsum("b k h w, b k c h w -> b c h w", probs, pred_velocity)
        weighted = torch.masked_fill(weighted, mask_metric == 0, 0.0)
        return weighted
