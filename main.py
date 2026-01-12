import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import ADE20KCDataset
from src.models import SegformerWrapper, SD3Wrapper
from src.utils import sliding_window_loss
import wandb
from tqdm import tqdm
from torchmetrics import JaccardIndex
import logging
import copy

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # 1. Setup
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)
    
    # Wandb
    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )

    # 2. Data
    dataset = ADE20KCDataset(
        data_root=cfg.dataset.data_root,
        corruption=cfg.dataset.corruption,
        severity=cfg.dataset.severity,
        short_edge_size=cfg.dataset.short_edge_size
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=cfg.dataset.shuffle, num_workers=cfg.dataset.num_workers)

    # 3. Models
    # Segformer on cuda:0 by default
    seg_model = SegformerWrapper(cfg.model.segformer).to("cuda:0")
    
    # SD3 dispatched
    sd3_model = None
    if cfg.tta.forward_mode == "tta":
        sd3_model = SD3Wrapper(cfg.model.sd3) # Dispatch handled inside
        # Precompute embeds on cuda:0 or compatible
        sd3_model.get_text_embeddings("cuda:0")

    # Metrics
    iou_metric = JaccardIndex(task="multiclass", num_classes=cfg.model.segformer.num_labels, ignore_index=cfg.model.segformer.ignore_index).to("cuda:0")

    # 4. Optimization Loop
    # We might need to reset model state if continual=False
    initial_seg_state = copy.deepcopy(seg_model.state_dict())
    # SD3 usually fine-tuning or fixed? Requirement: "Update both"
    if sd3_model:
        initial_sd3_state = copy.deepcopy(sd3_model.pipe.transformer.state_dict())

    # Optimizer (re-init per image if resets, or global if continual)
    # TTA often resets per image ("Reset model between tasks" implies task level)
    # If "Task" = 1 image? Or 1 dataset? 
    # Usually TTA on C-datasets treats the whole dataset as a stream.
    # Requirement: "If not continuous test, reset between tasks".
    # Assuming "Task" here means the corruption type (handled by config run).
    # So we probably DON'T reset per image unless specified.
    # Config says `reset_model_per_task: true`.
    
    params = list(seg_model.parameters())
    if sd3_model:
        params += list(sd3_model.pipe.transformer.parameters())
        
    optimizer = optim.AdamW(params, lr=cfg.tta.lr)

    seg_model.train() # TTA usually updates stats or params? Req: "TTA models in eval state"
    # Wait, req: "TTA时设置模型均处于eval状态" (TTA set models to eval state).
    # But we need gradients? Yes, eval() affects Dropout/BN, not Grads.
    seg_model.eval()
    if sd3_model:
        sd3_model.eval()

    progress_bar = tqdm(dataloader, desc="TTA")
    
    for i, batch in enumerate(progress_bar):
        # Reset Logic per image? Usually TTA standard does Reset per image?
        # Req: "If not continuous... reset model".
        # Let's assume standard Tent/TTA style: Reset per image OR Continuous.
        # Config has `reset_model_per_task`. If False -> Continual.
        # If True -> We should reset? Or does task mean the whole dataset?
        # Usually "Task" means the domain. So reset at start (done), then continual.
        # But if standard TTA (reset per sample), we might need logic.
        # I will assume Continual for now based on "reset_model_per_task".
        
        # Move Data
        images = batch["image"].to("cuda:0") # [B, C, H, W]
        masks = batch["mask"].to("cuda:0")
        
        # TTA Steps
        if cfg.tta.forward_mode == "tta" and sd3_model:
            for step in range(cfg.tta.steps):
                optimizer.zero_grad()
                
                # 1. Disc Forward
                logits = seg_model(images) # [B, K, H/4, W/4]
                
                # 2. Gen Forward & Loss
                # Need Image on same device as SD3 inputs? SD3 wrapper handles?
                # Sliding Window Loss
                loss = sliding_window_loss(
                    image=images,
                    logits=logits,
                    vae_encode_fn=sd3_model.forward_vae_encode,
                    transformer_forward_fn=sd3_model.forward_denoise,
                    prompt_embeds=sd3_model.prompt_embeds,
                    pooled_embeds=sd3_model.pooled_prompt_embeds,
                    window_size=cfg.tta.sliding_window_size
                )
                
                # 3. Backward
                loss.backward()
                optimizer.step()
                
                # Log loss
                if cfg.wandb.enabled:
                    wandb.log({"tta_loss": loss.item()})

        # Final Inference
        with torch.no_grad():
            final_logits = seg_model(images)
            preds = torch.argmax(final_logits, dim=1)
            # Upsample prediction to mask size
            preds = torch.nn.functional.interpolate(final_logits, size=masks.shape[-2:], mode='bilinear', align_corners=False).argmax(dim=1)
            
            iou_metric.update(preds, masks)
            
        # Logging
        mIoU = iou_metric.compute()
        progress_bar.set_postfix({"mIoU": f"{mIoU:.4f}"})
        if cfg.wandb.enabled:
            wandb.log({"batch_mIoU": mIoU})

    # Final Metric
    final_mIoU = iou_metric.compute()
    logger.info(f"Final mIoU: {final_mIoU}")
    if cfg.wandb.enabled:
        wandb.log({"final_mIoU": final_mIoU})

if __name__ == "__main__":
    main()
