from typing import Dict, List, Tuple

import torch
from torch import nn

from src.models.discriminative import SegformerDiscriminative
from src.models.generative.sd3 import SD3Generative


class CombinedModel(nn.Module):
    """Unifies discriminative SegFormer and SD3 auxiliary loss."""

    def __init__(self, disc_cfg: Dict, gen_cfg: Dict, forward_mode: str = "joint"):
        super().__init__()
        self.forward_mode = forward_mode
        self.discriminative = SegformerDiscriminative(disc_cfg)
        self.generative = SD3Generative(gen_cfg)

    def forward(self, images: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.discriminative(images)
        gen_loss = None
        if self.forward_mode != "discriminative_only":
            gen_loss = self.generative(images, logits)
        return logits, gen_loss
