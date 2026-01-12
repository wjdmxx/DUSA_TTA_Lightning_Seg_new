import torch
import torch.nn.functional as F

def sliding_window_loss(image, logits, vae_encode_fn, transformer_forward_fn, prompt_embeds, pooled_embeds, window_size=512, stride=256):
    """
    Computes the TTA loss using sliding window over the image logic.
    
    Args:
        image: [B, C, H, W] - Full size image (after resize to short edge 512)
        logits: [B, NumClasses, H_out, W_out] - Segformer logits (typically 1/4 size of image)
        vae_encode_fn: Function to encode image to latents
        transformer_forward_fn: Function to run SD3 denoising
        prompt_embeds, pooled_embeds: Precomputed embeddings
        window_size: Size of the square window (512)
        stride: Stride for sliding
    """
    B, C, H, W = image.shape
    
    # 1. Prepare Latents via VAE
    # We can try encoding the whole image first?
    # Requirement: "Slide window... then VAE downsample" OR "Image slide... VAE"
    # Computing gradients through VAE for the whole 2048px image might be heavy but accurate.
    # If we slide *after* VAE, we lose some boundary correctness in VAE?
    # Usually VAE is convolutional, so safe to run on whole image if fits. 
    # Let's try running VAE on whole image to get full latents.
    
    latents = vae_encode_fn(image) # [B, 16, h, w] where h=H/8, w=W/8
    
    # 2. Add Noise (forward diffusion)
    noise = torch.randn_like(latents)
    t = torch.randint(0, 1000, (B,), device=latents.device).long() # Random timestep? Or specific for TTA?
    # For TTA usually we pick a small noise level or specific t. Let's assume random or fixed.
    # Using a fixed t for consistency/optimization might be better? Let's use random for now like training.
    
    # Sigmas? SD3 uses flow matching. 
    # Simplified: z_t = (1-t)*z_0 + t*noise (example scheme, SD3 specific scheme needed? 
    # SD3 pipeline handles this internally usually via scheduler add_noise.
    # We will assume a simple additive noise for calculation or use scheduler if available.
    # Since we don't have scheduler passed here, let's just do a simple dummy noise for the TTA generic structure
    # or rely on the `transformer_forward_fn` to handle specific flow matching loss if it wraps scheduler.
    # But usually `transformer` predicts velocity `v`.
    # Target `v` = noise - latents (rectangular flow) usually.
    
    # Let's implement rect flow target manually for simplicity:
    # t in [0, 1]
    t_float = torch.rand((B,), device=latents.device)
    # z_t = (1 - t) * x + t * 1 * noise (Rectified Flow)
    t_expanded = t_float.view(B, 1, 1, 1)
    noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
    target_v = noise - latents 
    
    # 3. Slide Window logic on Latents
    # Latent window size: 512 / 8 = 64
    latent_ws = window_size // 8
    latent_stride = stride // 8
    
    # Logits need downsampling to latent size for weighting
    # Logits: [B, K, H/4, W/4] -> interpolate to [B, K, H/8, W/8]
    model_logits = F.interpolate(logits, size=latents.shape[-2:], mode='bilinear', align_corners=False)
    # Calculate entropy/confidence for weighting
    # Weight = Entropy? Or Max Prob?
    # Logic: "Predicted velocity weighted by logits position"
    # Usually we trust high confidence regions more or less?
    # "TopK selection and Weighted loss calculation... unchanged"
    # I'll implement a basic Entropy weight map here.
    probs = torch.softmax(model_logits, dim=1)
    max_probs, _ = torch.max(probs, dim=1)
    weights = max_probs # Simple weight: value pixels where Segformer is confident?
    # Or maybe we want to optimize where it is NOT confident? 
    # "Generative model auxiliary": usually trying to make image "realistic" implies global consistency.
    # TTA logic usually: Min reconstruction error. 
    # "Weighted by logits position": I will assume we weight by confidence.
    
    # Unfold Latents, Noisy Latents, and Weights to patches
    # [B, C, H, W] -> [B, C, N_patches, PH, PW]
    # Use unfold.
    
    def get_patches(x, k, s):
        # x: [B, C, H, W]
        # unfold H
        patches = x.unfold(2, k, s).unfold(3, k, s) # [B, C, Nh, Nw, k, k]
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous() # [B, Nh, Nw, C, k, k]
        return patches.view(-1, x.shape[1], k, k) # [N_total, C, k, k]

    latent_patches = get_patches(latents, latent_ws, latent_stride)
    noisy_patches = get_patches(noisy_latents, latent_ws, latent_stride)
    target_patches = get_patches(target_v, latent_ws, latent_stride)
    weight_patches = get_patches(weights.unsqueeze(1), latent_ws, latent_stride) # [N, 1, k, k]
    
    # 4. Forward SD3 Transformer on patches
    # Note: N_patches might be large. We might need sub-batching.
    # Total patches for 2048x512 with 512 window, 256 stride: 
    # H=64, W=256. k=64, s=32.
    # Nw = (256 - 64)/32 + 1 = 192/32 + 1 = 6+1 = 7.
    # Nh = 1.
    # Total 7 patches. Fits in batch.
    
    # Prompt embeddings expansion
    # prompt_embeds: [1, Seq, Dim] -> [N, Seq, Dim]
    N = latent_patches.shape[0]
    p_emb = prompt_embeds.expand(N, -1, -1)
    pool_emb = pooled_embeds.expand(N, -1)
    t_batch = t_float.expand(N) # [N] (All patches same t)
    
    pred_v = transformer_forward_fn(noisy_patches, t_batch, p_emb, pool_emb)
    
    # 5. Calculate Weighted MSE
    loss = (pred_v - target_patches) ** 2
    loss = loss * weight_patches
    return loss.mean()
