import torch
from torch import nn
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from typing import List, Dict, Any
import torchvision.transforms.functional as TF


class SegformerDiscriminative(nn.Module):
    """Wraps SegFormer for semantic segmentation with minimal preprocessing.

    - Resizes short edge to a fixed size in the dataloader; processor only normalizes.
    - Exposes logits at 4x downsampled resolution for downstream use.
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.model_name = cfg.get("model_name", "nvidia/segformer-b5-finetuned-ade-640-640")
        self.device = torch.device(cfg.get("device", "cuda:0"))
        self.num_classes = cfg.get("num_classes", 150)
        self.id2label = {i: str(i) for i in range(self.num_classes)}
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.processor = SegformerImageProcessor.from_pretrained(self.model_name)
        # Avoid resizing twice; dataloader already handles short-edge resize.
        self.processor.do_resize = False
        self.processor.size = {"shortest_edge": cfg.get("short_edge", 512)}
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            self.model_name,
            ignore_mismatched_sizes=True,
            num_labels=self.num_classes,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        self.model.to(self.device)

    @torch.no_grad()
    def preprocess(self, images: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Normalize images via HF processor. Expects CxHxW tensors in [0,1]."""
        # Processor is safest on CPU; move back to GPU after normalization.
        safe_images = [img.detach().to("cpu") for img in images]
        encoded = self.processor(images=list(safe_images), return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        return encoded

    def forward(self, images: List[torch.Tensor]) -> torch.Tensor:
        inputs = self.preprocess(images)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.model(**inputs)
        return outputs.logits  # (B, C, H/4, W/4)

    @torch.no_grad()
    def predict(self, images: List[torch.Tensor]) -> torch.Tensor:
        logits = self.forward(images)
        return logits
