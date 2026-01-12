from typing import Any, Dict, List

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

from src.models.combined import CombinedModel


class TTAModule(pl.LightningModule):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.forward_mode = cfg["model"].get("forward_mode", "joint")
        self.combined = CombinedModel(
            disc_cfg=cfg["model"]["discriminative"],
            gen_cfg=cfg["model"]["generative"],
            forward_mode=self.forward_mode,
        )
        num_classes = cfg["data"].get("num_classes", 150)
        ignore_index = cfg["data"].get("ignore_index", 255)
        self.train_miou = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=num_classes, ignore_index=ignore_index
        )
        self.val_miou = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=num_classes, ignore_index=ignore_index
        )
        self.lr = cfg["trainer"].get("lr", 1e-4)

    def on_train_start(self) -> None:
        # Keep eval mode to mimic test-time behavior while still enabling grads.
        self.combined.eval()

    def forward(self, images: List[torch.Tensor]):
        return self.combined(images)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        images = batch["image"]
        masks = batch["mask"].long()
        image_list = list(images)
        logits, gen_loss = self(image_list)

        logits_up = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        preds = torch.argmax(logits_up, dim=1)

        self.train_miou.update(preds, masks)
        loss = gen_loss if gen_loss is not None else torch.zeros((), device=logits.device, requires_grad=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=images.size(0))
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        images = batch["image"]
        masks = batch["mask"].long()
        image_list = list(images)
        logits, _ = self(image_list)
        logits_up = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        preds = torch.argmax(logits_up, dim=1)
        self.val_miou.update(preds, masks)

    def on_train_epoch_end(self) -> None:
        miou = self.train_miou.compute()
        self.log("train_mIoU", miou, prog_bar=True, on_epoch=True)
        self.train_miou.reset()

    def on_validation_epoch_end(self) -> None:
        miou = self.val_miou.compute()
        self.log("val_mIoU", miou, prog_bar=True, on_epoch=True)
        self.val_miou.reset()

    def configure_optimizers(self):
        params = [p for p in self.combined.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.lr)
        return optimizer
